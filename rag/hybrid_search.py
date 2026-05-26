"""
混合检索 —— 向量 + BM25 + RRF 融合

同时调用向量检索和 BM25 检索，用 RRF 算法融合两个排序列表。
这是本项目检索精度的核心保障。

----------------------------------------------------------------------
## 你需要自己写的部分

RRF（Reciprocal Rank Fusion）是混合检索的经典融合算法。

学习重点：
1. RRF 为什么比分数归一化好：不同检索器的分数尺度不同，无法直接比较
2. RRF 公式: score(doc) = Σ 1/(k + rank_i(doc))
   其中 rank_i(doc) 是 doc 在第 i 个检索器中的排名，k 是平滑常数
3. k 值的影响：k 越小，高排名结果权重越大；k 越大，排名差异越平滑

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# typing: 类型提示
from typing import List, Optional, Dict, Any

# 数据模型
from rag.models import TextChunk, RetrievalResult

# 向量存储
from rag.vector_store import VectorStore

# BM25 检索引擎
from rag.bm25 import BM25Retriever

# Embedding 工厂
from core.embedding import get_embeddings


class HybridSearcher:
    """混合检索器 —— 向量 + BM25 + RRF 融合

    检索流程:
    ┌──────────────────┐
    │  用户 Query       │
    └────────┬─────────┘
             ↓
    ┌────────┴────────┐
    │  并行执行         │
    ├─────────────────┤
    │ 向量检索   BM25  │
    │ (语义)    (关键词)│
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  RRF 融合        │
    │  score = Σ 1/(k+r)│
    └────────┬────────┘
             ↓
    ┌─────────────────┐
    │  融合结果列表     │
    └─────────────────┘

    面试话术:
    "混合检索是我系统检索精度的核心。向量检索擅长语义匹配但容易漏掉
    精确查询（如航班号），BM25 擅长精确匹配但不懂同义词。
    我用 RRF 融合两者优势：当用户查'CA1234'时 BM25 贡献精确命中，
    查'行李怎么赔'时向量贡献语义理解，两者互补。"
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        """初始化混合检索器

        Args:
            vector_store: Qdrant 向量存储（已索引）
            bm25_retriever: BM25 检索器（已索引）
            rrf_k: RRF 融合常数（默认 60，经验最优值）
        """
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # 混合检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        vector_top_k: int = 20,   # 向量检索先取更多候选
        bm25_top_k: int = 20,     # BM25 也取更多候选
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """混合检索 —— 向量 + BM25 + RRF 融合

        TODO(用户): 手写混合检索逻辑

        流程：
        1. 将 query 向量化
        2. 并行调用向量检索（取 vector_top_k 候选）和 BM25 检索（取 bm25_top_k 候选）
        3. RRF 融合两个排序列表
        4. 按融合分数降序排列，返回 Top-K

        为什么先取更多候选（20 条）再融合取 Top-K（5 条）？
        - 给 RRF 更大的候选池，让两个检索器的结果有更多交集
        - 如果只取 5 条，BM25 的第 6 名可能和向量第 1 名说的是同一件事

        Args:
            query: 用户查询文本
            top_k: 最终返回结果数
            vector_top_k: 向量检索候选数
            bm25_top_k: BM25 检索候选数
            filter_conditions: 向量检索的 payload 过滤条件

        Returns:
            RetrievalResult 列表，按 RRF 融合分数降序，source='hybrid'
        """
        # ============================================================
        # 步骤 1: 向量化 query
        # ============================================================
        # 把用户自然语言问题转为 1024 维向量
        # embed_single() 等价于 embed([query])[0]
        embeddings = get_embeddings()
        query_vector = embeddings.embed_single(query)

        # ============================================================
        # 步骤 2: 并行执行两路检索
        # ============================================================
        # 向量检索：在 Qdrant 中用余弦相似度找语义相近的文档
        #   返回 list[dict]，每项含 id/score/content/source_file/clause_id 等
        vector_results = self.vector_store.search(
            query_vector, top_k=vector_top_k,
            filter_conditions=filter_conditions,
        )

        # BM25 检索：基于 jieba 分词 + 倒排索引的关键词精确匹配
        #   返回 list[(TextChunk, float)]，每项含 chunk 对象和 BM25 分数
        bm25_results = self.bm25.search(query, top_k=bm25_top_k)

        # ============================================================
        # 步骤 3: RRF (Reciprocal Rank Fusion) 融合
        # ============================================================
        # 为什么用 RRF 而不是直接加权分数？
        #   BM25 分数范围 [0, +∞)，向量检索分数范围 [0, 1]
        #   两者不在同一个尺度上——直接加权会让 BM25 大分数碾压向量小分数
        #
        # RRF 的核心思想：不看分数绝对值，只看排名
        #   每个文档的 RRF 分数 = Σ 1/(k + rank_i)
        #   其中 rank_i 是该文档在第 i 个检索器中的排名
        #   k=60 是经验最优常数，用于平滑排名差异
        #
        # 关键洞察：
        #   同一文档可能在两路检索中都出现（排在各自的不同位置）
        #   → RRF 累加两路贡献 → 两路都靠前 = 总分最高 = 最可能是正确结果
        #   某文档只在一路出现 → 只有该路的贡献 → 总分较低
        #
        # fused 数据结构: {chunk_id → {"score": 累加RRF分, "chunk": TextChunk, ...}}
        #   - 键 chunk_id: 全局唯一标识，用于判断「同一文档在两路是否都出现」
        #   - 值 score:    RRF 公式的累加和（两路贡献合并）
        #   - 值 chunk:    TextChunk 对象（BM25 直接有，向量检索需从 payload 重建）
        fused = {}

        # ---- 3a: 向量检索结果融入 RRF ----
        # enumerate(r, start=1) → rank 从 1 开始，rank 越小贡献越大
        for rank, r in enumerate(vector_results, start=1):
            # chunk_id 在 vector_store 返回格式中可能叫 "chunk_id" 或 "id"
            cid = r.get("chunk_id") or r.get("id", "")

            # 如果 cid 之前没见过 → 初始化一个分数容器
            # 如果 cid 已经在 fused 中（说明 BM25 也命中了）→ 复用已有容器
            if cid not in fused:
                fused[cid] = {"score": 0.0}
                # 向量结果没有 TextChunk 对象，需要保留 payload 字段
                # 供步骤 4 重建 TextChunk
                fused[cid]["content"] = r.get("content", "")
                fused[cid]["source_file"] = r.get("source_file", "")
                fused[cid]["clause_id"] = r.get("clause_id")
                fused[cid]["section_title"] = r.get("section_title")

            # RRF 核心公式: score += 1/(k + rank)
            # rank=1 → 1/61≈0.0164, rank=20 → 1/80≈0.0125
            # k 越大排名差异越小，k=60 让第1名比第20名只多约 30%
            fused[cid]["score"] += 1.0 / (self.rrf_k + rank)
            fused[cid]["vector_rank"] = rank  # 记录原始排名（调试用）

        # ---- 3b: BM25 检索结果融入 RRF ----
        # BM25 返回 (TextChunk, float)，其中 chunk 对象是完整的
        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id  # BM25 的 chunk 自带完整 chunk_id

            # 和 3a 相同的逻辑：检查 cid 是否已存在
            if cid not in fused:
                fused[cid] = {"score": 0.0}

            # RRF 累加: 如果这个 chunk 也在向量结果中 → 分数叠加
            fused[cid]["score"] += 1.0 / (self.rrf_k + rank)
            fused[cid]["chunk"] = chunk      # TextChunk 对象，可直接用
            fused[cid]["bm25_rank"] = rank   # 记录原始排名

        # ============================================================
        # 步骤 4: 按 RRF 分数降序排序 → 返回 Top-K
        # ============================================================
        # fused.values() 包含所有候选文档（两路检索的并集）
        # 按 RRF 分数从高到低排列：
        #   分数最高的 = 在两路检索中都排得靠前 = 最相关
        sorted_items = sorted(
            fused.values(),
            key=lambda x: x["score"],  # 按 RRF 分数排序
            reverse=True,               # 降序: 高分在前
        )

        # 构建 RetrievalResult 列表（取 Top-K）
        results = []
        for item in sorted_items[:top_k]:
            # 获取或重建 TextChunk 对象
            # 情况 A: 有 chunk 对象（来自 BM25，或两路都命中）→ 直接使用
            # 情况 B: 仅向量检索命中（无 chunk 对象）→ 从 payload 字段重建
            if "chunk" in item:
                chunk = item["chunk"]
            else:
                # 从 Qdrant 返回的 payload 字段重建 TextChunk
                chunk = TextChunk(
                    chunk_id="",
                    content=item.get("content", ""),
                    source_file=item.get("source_file", ""),
                    clause_id=item.get("clause_id"),
                    section_title=item.get("section_title"),
                )

            results.append(RetrievalResult(
                chunk=chunk,
                score=item["score"],
                source="hybrid",  # 标记来源为混合检索
            ))

        return results


    # ------------------------------------------------------------------
    # 单路检索（用于对比实验）
    # ------------------------------------------------------------------

    def search_vector_only(
        self, query: str, top_k: int = 5
    ) -> List[RetrievalResult]:
        """纯向量检索（对比实验用）"""
        embeddings = get_embeddings()
        query_vector = embeddings.embed_single(query)
        raw = self.vector_store.search(query_vector, top_k=top_k)
        return [
            RetrievalResult(
                chunk=TextChunk(
                    chunk_id=r.get("chunk_id", ""),
                    content=r.get("content", ""),
                    source_file=r.get("source_file", ""),
                    clause_id=r.get("clause_id"),
                    section_title=r.get("section_title"),
                ),
                score=r["score"],
                source="vector",
            )
            for r in raw
        ]

    def search_bm25_only(
        self, query: str, top_k: int = 5
    ) -> List[RetrievalResult]:
        """纯 BM25 检索（对比实验用）"""
        raw = self.bm25.search(query, top_k=top_k)
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                source="bm25",
            )
            for chunk, score in raw
        ]
