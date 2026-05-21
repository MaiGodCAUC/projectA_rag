"""
Qdrant 向量数据库操作层

封装 Qdrant 的 CRUD 操作：创建集合、写入向量、检索查询、删除管理。

----------------------------------------------------------------------
## 你需要自己写的部分

本文件是 Qdrant 客户端的封装，属于基础设施代码。你需要理解而非逐字重写。

学习重点：
1. Qdrant 的核心概念：Collection(集合) → Point(点=向量+payload) → Search(检索)
2. upsert 的语义：insert + update = 有则更新，无则插入
3. payload 的作用：附着在向量上的元数据，检索时可过滤、可返回
4. 为什么用 Qdrant 而不用 Chroma/Milvus：Qdrant 性能好、Docker 部署简单、API 清晰

TODO(用户) 标记是你需要手写的部分。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# uuid: 生成唯一 ID，用于 upsert 时标记每个 point
import uuid

# typing: 类型提示
from typing import List, Optional, Dict, Any

# Qdrant: 向量数据库 Python 客户端
# QdrantClient: 连接 Qdrant 服务端
# models: Qdrant 的数据模型（Distance, PointStruct, Filter 等）
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,        # 距离度量方式：Cosine(余弦) / Euclid(欧氏) / Dot(内积)
    PointStruct,     # Point 结构：id + vector + payload
    Filter,          # 过滤条件
    FieldCondition,  # 字段条件
    MatchValue,      # 精确匹配
    Range,           # 范围匹配
)

# TextChunk: 切片后的文本块，写入 Qdrant 的最小单元
from rag.models import TextChunk

# 默认配置常量
from core.constants import DEFAULT_QDRANT_COLLECTION


class VectorStore:
    """Qdrant 向量存储操作层

    封装了以下操作：
    - 创建/删除 Collection
    - 批量写入向量（upsert）
    - 向量相似度检索
    - 按条件过滤检索
    - 查询 Collection 状态

    设计思想：
    - 所有操作通过 QdrantClient 完成
    - upsert 语义：同一 id 的 point 会被覆盖（实现增量更新）
    - 每个 point 携带 payload（源文件、条款编号等元数据），检索时可过滤和展示

    面试时可以说：
    "我封装了 Qdrant 操作层，支持向量检索 + payload 过滤的组合查询。
    payload 中存储了条款编号、来源文件等元数据，
    检索时可以按文档名过滤、按条款级别筛选。"
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
    ):
        """初始化 Qdrant 连接

        Args:
            host: Qdrant 服务地址（本地 Docker 默认 localhost）
            port: Qdrant HTTP 端口（默认 6333）
            collection_name: Collection 名称（相当于数据库中的"表"）
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name

        # 创建 Qdrant 客户端连接
        # prefer_grpc=False: 使用 HTTP 协议（更通用，不需要额外安装 grpc）
        self.client = QdrantClient(host=host, port=port, prefer_grpc=False)

    # ------------------------------------------------------------------
    # Collection 管理
    # ------------------------------------------------------------------

    def create_collection(self, vector_size: int, force_recreate: bool = False):
        """创建向量集合

        每个 Collection 有固定的向量维度（由 Embedding 模型决定）。
        这是写入数据前必须完成的一步。

        Args:
            vector_size: 向量维度（BGE=1024, M3E=768, Qwen=1536）
            force_recreate: 是否强制重建（True=删除旧的再创建）
        """
        # 检查是否已存在
        if self.collection_exists():
            if force_recreate:
                self.client.delete_collection(self.collection_name)
            else:
                return  # 已存在且不强制重建 → 跳过

        # 创建新集合
        # Distance.COSINE: 使用余弦相似度（最常用的语义相似度度量）
        # 余弦相似度 = 两个向量夹角的余弦值，值域 [-1, 1]，越接近 1 越相似
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "size": vector_size,
                "distance": Distance.COSINE,
            },
        )

    def collection_exists(self) -> bool:
        """检查 Collection 是否存在"""
        collections = self.client.get_collections()
        return any(
            c.name == self.collection_name
            for c in collections.collections
        )

    def collection_info(self) -> dict:
        """获取 Collection 状态信息

        Returns:
            包含名称、向量数、状态等信息的字典
        """
        if not self.collection_exists():
            return {"exists": False, "name": self.collection_name}

        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
        }

    # ------------------------------------------------------------------
    # 向量写入
    # ------------------------------------------------------------------

    # TODO(用户): 实现 upsert_chunks 方法
    #
    # 核心逻辑：
    # 1. 对每个 TextChunk，用 embedding 函数将 content 转为向量
    # 2. 将向量 + payload（元数据）一起写入 Qdrant
    # 3. 使用 batch 批量写入以提高效率
    #
    # Qdrant 的 PointStruct 结构：
    #   PointStruct(
    #       id=str(uuid.uuid4()),   # 唯一 ID
    #       vector=vector,          # 向量（float 列表）
    #       payload={               # 附加元数据（检索时可返回）
    #           "content": chunk.content,
    #           "source_file": chunk.source_file,
    #           "clause_id": chunk.clause_id,
    #           ...
    #       }
    #   )
    #
    # upsert 语义：如果 id 已存在 → 更新；如果不存在 → 插入。
    # 这意味着增量索引时可以安全地重复调用。

    def upsert_chunks(
        self,
        chunks: List[TextChunk],
        embedding_function,  # TODO(用户): 类型提示为 EmbeddingBackend
        batch_size: int = 100,
    ) -> int:
        """批量写入文本块向量

        将 TextChunk 列表转为向量并写入 Qdrant。
        这是索引流水线的最后一步。

        TODO(用户): 实现 upsert 逻辑

        Args:
            chunks: 待写入的 TextChunk 列表
            embedding_function: Embedding 实例（如 BGEBackend）
            batch_size: 每批处理的 chunk 数量

        Returns:
            成功写入的 point 数量
        """
        # ================================================================
        # TODO(用户): 从这里开始手写 —— 批量 upsert
        # ================================================================
        #
        # 参考实现：
        #
        # total = 0
        # for i in range(0, len(chunks), batch_size):
        #     batch = chunks[i : i + batch_size]
        #
        #     # 提取文本内容
        #     texts = [c.content for c in batch]
        #
        #     # 调用 embedding 函数 → 向量列表
        #     vectors = embedding_function.embed(texts)
        #
        #     # 构建 PointStruct 列表
        #     points = []
        #     for chunk, vector in zip(batch, vectors):
        #         point_id = str(uuid.uuid4())
        #         points.append(PointStruct(
        #             id=point_id,
        #             vector=vector,
        #             payload={
        #                 "chunk_id": chunk.chunk_id,
        #                 "content": chunk.content,
        #                 "source_file": chunk.source_file,
        #                 "clause_id": chunk.clause_id,
        #                 "section_title": chunk.section_title,
        #                 "chunk_index": chunk.chunk_index,
        #                 **chunk.metadata,  # 展开 metadata 字典
        #             },
        #         ))
        #
        #     # 批量写入 Qdrant
        #     self.client.upsert(
        #         collection_name=self.collection_name,
        #         points=points,
        #     )
        #     total += len(points)
        #
        # return total
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 实现 upsert_chunks —— 批量向量写入逻辑"
        )

    # ------------------------------------------------------------------
    # 向量检索
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """向量相似度检索

        给定一个 query 向量，找出 Qdrant 中与之最相似的 top_k 个 point。
        支持 payload 过滤（如只检索特定文档）。

        这是 RAG 检索的核心操作——给定用户问题的向量，
        找到最相关的文档片段。

        TODO(用户): 理解并手写 search 逻辑

        Args:
            query_vector: 查询向量（用户问题的 embedding）
            top_k: 返回最相似的 K 个结果
            filter_conditions: 可选的 payload 过滤条件
                例: {"source_file": "04-托运行李运输规定.md"}
                    {"clause_id": "第3条"}

        Returns:
            检索结果列表，每项包含:
            {
                "id": point_id,
                "score": 相似度分数,
                "content": 文本内容,
                "source_file": 来源文档,
                "clause_id": 条款编号,
                ...
            }
        """
        # ================================================================
        # TODO(用户): 从这里开始手写 —— 向量检索
        # ================================================================
        #
        # 参考实现：
        #
        # # 构建过滤条件（如果有）
        # query_filter = None
        # if filter_conditions:
        #     conditions = []
        #     for key, value in filter_conditions.items():
        #         conditions.append(
        #             FieldCondition(
        #                 key=key,
        #                 match=MatchValue(value=value),
        #             )
        #         )
        #     query_filter = Filter(must=conditions)
        #
        # # 调用 Qdrant 检索
        # results = self.client.search(
        #     collection_name=self.collection_name,
        #     query_vector=query_vector,
        #     limit=top_k,
        #     query_filter=query_filter,
        #     with_payload=True,  # 返回 payload（元数据）
        # )
        #
        # # 整理返回格式
        # return [
        #     {
        #         "id": r.id,
        #         "score": r.score,
        #         **r.payload,  # 展开所有 payload 字段
        #     }
        #     for r in results
        # ]
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 实现 search —— 向量相似度检索逻辑"
        )

    # ------------------------------------------------------------------
    # 维护操作
    # ------------------------------------------------------------------

    def delete_by_source(self, source_file: str) -> int:
        """按来源文件删除所有相关向量（用于增量更新时清除旧数据）

        Args:
            source_file: 来源文件名

        Returns:
            删除的 point 数量（近似值）
        """
        # Qdrant 的 delete 按过滤条件删除
        # 构建过滤条件：payload.source_file == source_file
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    )
                ]
            ),
        )
        return result.status.completed if result.status else 0

    def count(self) -> int:
        """获取 Collection 中的向量总数"""
        info = self.collection_info()
        return info.get("points_count", 0)
