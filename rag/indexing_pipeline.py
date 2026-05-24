"""
索引流水线 —— 文档到向量的完整流程

将「文档上传 → 解析 → 切片 → Embedding → 写入 Qdrant」串联为一条流水线。
支持全量索引和增量索引两种模式。

----------------------------------------------------------------------
## 你需要自己写的部分

索引流水线是 RAG 系统的「组装车间」——它不涉及新算法，但体现了你对整个
RAG 流程的理解。面试时可以画这条流水线来讲清楚系统架构。

学习重点：
1. 流水线编排：理解每一步的输入输出，以及如何串联
2. 增量索引：SHA256 哈希去重的原理和实现
3. 批量处理：为什么需要分批（避免一次加载太多数据到内存）
4. 状态追踪：索引了多少文档、多少向量，方便运维

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# hashlib: SHA256 哈希，用于增量索引去重
import hashlib

# datetime: 记录索引时间
from datetime import datetime

# Path: 文件路径操作
from pathlib import Path

# typing: 类型提示
from typing import List, Optional, Dict, Any

# 数据模型
from rag.models import ParsedDocument, TextChunk

# 文档加载器
from rag.loader import load_document, load_documents

# 切片器工厂
from rag.splitter import get_splitter

# 向量存储
from rag.vector_store import VectorStore

# Embedding 工厂
from core.embedding import get_embeddings

# 配置
from core.config import get_settings


class IndexingPipeline:
    """RAG 索引流水线

    将文档处理的全流程封装为可调用的流水线：
    ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
    │ 文档加载  │ → │ 条款切片  │ → │ Embedding │ → │ Qdrant   │
    │ loader   │    │ splitter │    │ embedding │    │ upsert   │
    └──────────┘    └──────────┘    └───────────┘    └──────────┘

    面试时可以说：
    "我设计了一条完整的索引流水线，支持全量索引和增量索引。
    增量索引通过 SHA256 哈希判断文档是否变更，只处理新增和修改的文件。
    每条流水线执行都记录了耗时和状态，方便排查问题。"
    """

    def __init__(
        self,
        vector_store: VectorStore,
        splitter_strategy: str = "policy_clause",
    ):
        """初始化索引流水线

        Args:
            vector_store: Qdrant 向量存储实例
            splitter_strategy: 切片策略名称
        """
        self.vector_store = vector_store
        self.splitter_strategy = splitter_strategy

        # 存储已索引文档的哈希值 {file_name: sha256_hash}
        self._indexed_hashes: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 全量索引
    # ------------------------------------------------------------------

    def index_all(self, document_dir: str) -> Dict[str, Any]:
        """全量索引：解析目录下所有文档并写入 Qdrant

        这是「一键建库」的入口——给定文档目录，自动完成全流程。
        适用于初次部署或重建索引。

        TODO(用户): 理解并手写 index_all 的编排逻辑

        流程：
        1. 扫描目录 → 收集所有 .md / .pdf / .docx 文件
        2. 逐个文档：加载 → 切片 → Embedding → 写入
        3. 记录每个文档的处理状态（成功/失败/跳过）
        4. 返回统计摘要

        Args:
            document_dir: 文档目录路径

        Returns:
            索引结果摘要:
            {
                "total_files": 10,
                "indexed": 10,
                "failed": 0,
                "total_chunks": 85,
                "total_vectors": 85,
                "cost_ms": 12345,
                "details": [...],
            }
        """
        start_time = datetime.now()
        doc_dir = Path(document_dir)

        extensions = ["*.md", "*.pdf", "*.docx", "*.txt"]
        files = []
        for ext in extensions:
            files.extend(doc_dir.glob(ext))

        # 2. 获取切片器
        splitter = get_splitter(self.splitter_strategy)

        # 3. 获取 Embedding 实例
        embeddings = get_embeddings()

        self.vector_store.create_collection(
            vector_size=embeddings.dimension,
            force_recreate=True
        )

        # 5. 逐个文档处理
        total_chunks = 0
        details = []

        for fp in files:
            try:
                # 加载文档
                doc = load_document(str(fp))
                # 切片
                chunks = splitter.split(doc)
                # 写入向量数据库（upsert）
                count = self.vector_store.upsert_chunks(chunks,embeddings)
                # 记录哈希
                self._indexed_hashes[doc.file_name] = doc.file_hash
                total_chunks += count
                details.append({
                    "file": doc.file_name,
                    "status": "ok",
                    "chunks": len(chunks),
                    "vectors": count
                })
            except Exception as e:
                details.append({
                    "file": fp.name,
                    "status": "failed",
                    "error": str(e)
                })

        # 6. 返回统计摘要
        cost_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return {
            "total_files": len(files),
            "indexed": sum(1 for d in details if d["status"] == "ok"),
            "failed": sum(1 for d in details if d["status"] == "failed"),
            "total_chunks": total_chunks,
            "total_vectors": total_chunks,
            "cost_ms": cost_ms,
            "details": details
        }
    # ------------------------------------------------------------------
    # 增量索引
    # ------------------------------------------------------------------

    def index_incremental(self, document_dir: str) -> Dict[str, Any]:
        """增量索引：只处理新增和变更的文档

        通过 SHA256 哈希比对判断文档是否需要重新索引：
        - 新文件（哈希不在记录中）→ 索引
        - 变更文件（哈希与记录不同）→ 删除旧数据 + 重新索引
        - 未变更文件 → 跳过

        这是生产环境的关键能力——不会每次都重建整个索引。

        TODO(用户): 理解并手写增量索引逻辑

        和全量索引的关键区别：
        1. 不 force_recreate Collection（保留已有数据）
        2. 索引前先比对哈希，跳过未变更文件
        3. 变更文件先 delete_by_source 再 upsert

        Args:
            document_dir: 文档目录路径

        Returns:
            索引结果摘要（格式同 index_all），额外包含 skipped 字段
        """
        start_time = datetime.now()
        doc_dir = Path(document_dir)

        extensions = ["*.md", "*.pdf", "*.docx", "*.txt"]
        files = []
        for ext in extensions:
            files.extend(doc_dir.glob(ext))

        # 2. 获取切片器
        splitter = get_splitter(self.splitter_strategy)

        # 3. 获取 Embedding
        embeddings = get_embeddings()

        # 4. 确保 Collection 存在（不 force_recreate！）
        self.vector_store.create_collection(
            vector_size=embeddings.dimension,
            force_recreate=False,  # ← 增量模式，不重建
        )

        # 5. 逐个文档处理（增加哈希比对）
        # skipped = 0
        # for fp in files:
        #     # 先计算哈希（在读文件之前）
        #     file_hash = _compute_hash(str(fp))
        #
        #     # 检查：哈希是否在记录中且未变化？
        #     if fp.name in self._indexed_hashes:
        #         if self._indexed_hashes[fp.name] == file_hash:
        #             skipped += 1
        #             continue  # ← 未变更，跳过
        #         else:
        #             # 变更了 → 先删除旧数据
        #             self.vector_store.delete_by_source(fp.name)
        #
        #     # 加载 → 切片 → 写入（和全量索引相同）
        #     doc = load_document(str(fp))
        #     chunks = splitter.split(doc)
        #     count = self.vector_store.upsert_chunks(chunks, embeddings)
        #     self._indexed_hashes[doc.file_name] = doc.file_hash
        #     ...
        #
        # return {..., "skipped": skipped, ...}
        #
        # ================================================================
        skipped = 0
        total_chunks = 0
        details = []

        for fp in files:
            try:
                # 计算哈希值
                file_hash = _compute_hash(str(fp))

                # 检查：哈希是否在记录中且未变化？
                if fp.name in self._indexed_hashes:
                    if self._indexed_hashes[fp.name] == file_hash:
                        skipped += 1
                        details.append({
                            "file": fp.name, "status": "skipped"
                        })
                        continue
                    else:
                        # 变更了 → 先删除旧数据
                        self.vector_store.delete_by_source(fp.name)

                # 加载 → 切片 → 写入
                doc = load_document(str(fp))
                chunks = splitter.split(doc)
                count = self.vector_store.upsert_chunks(chunks, embeddings)
                self._indexed_hashes[fp.name] = doc.file_hash
                total_chunks += count
                details.append({
                    "file": doc.file_name,
                    "status": "ok" if count > 0 else "empty",
                    "chunks": len(chunks),
                    "vectors": count,
                })
            except Exception as e:
                details.append({
                    "file": fp.name,
                    "status": "failed",
                    "error": str(e),
                })

        cost_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        return {
            "total_files": len(files),
            "indexed": sum(1 for d in details if d["status"] == "ok"),
            "skipped": skipped,
            "failed": sum(1 for d in details if d["status"] == "failed"),
            "total_chunks": total_chunks,
            "total_vectors": total_chunks,
            "cost_ms": cost_ms,
            "details": details,
        }

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """获取索引状态

        Returns:
            索引状态摘要
        """
        info = self.vector_store.collection_info()
        return {
            "collection": info,
            "indexed_files": len(self._indexed_hashes),
            "indexed_hashes": dict(self._indexed_hashes),  # 返回副本
            "splitter_strategy": self.splitter_strategy,
        }


# =============================================================================
# 辅助函数
# =============================================================================

def _compute_hash(file_path: str) -> str:
    """计算文件 SHA256 哈希（和 loader.py 中的实现一致）

    Args:
        file_path: 文件路径

    Returns:
        64 位十六进制哈希字符串
    """
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()
