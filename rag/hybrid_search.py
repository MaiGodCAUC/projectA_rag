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
        # ================================================================
        # TODO(用户): 手写混合检索逻辑
        # ================================================================
        #
        # 实现参考:
        #
        # # 1. 向量化 query
        # embeddings = get_embeddings()
        # query_vector = embeddings.embed_single(query)
        #
        # # 2. 并行检索
        # # 向量检索
        # vector_results = self.vector_store.search(
        #     query_vector, top_k=vector_top_k,
        #     filter_conditions=filter_conditions,
        # )
        #
        # # BM25 检索
        # bm25_results = self.bm25.search(query, top_k=bm25_top_k)
        #
        # # 3. RRF 融合
        # # 为每个 chunk 计算 RRF 分数
        # # chunk_id → RRF 分数累加 + 原始结果记录
        # fused = {}  # {chunk_id: {"score": rrf_score, "chunk": TextChunk, ...}}
        #
        # for rank, r in enumerate(vector_results, start=1):
        #     cid = r.get("chunk_id", r.get("id", ""))
        #     fused[cid] = fused.get(cid, {"score": 0, ...})
        #     fused[cid]["score"] += 1.0 / (self.rrf_k + rank)
        #     fused[cid]["chunk"] = ...  # 保留 chunk 信息
        #     fused[cid]["vector_rank"] = rank
        #
        # for rank, (chunk, score) in enumerate(bm25_results, start=1):
        #     cid = chunk.chunk_id
        #     fused[cid] = fused.get(cid, {"score": 0, ...})
        #     fused[cid]["score"] += 1.0 / (self.rrf_k + rank)
        #     fused[cid]["chunk"] = chunk
        #     fused[cid]["bm25_rank"] = rank
        #
        # # 4. 按 RRF 分数降序排序 → 取 Top-K
        # sorted_items = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        # return [
        #     RetrievalResult(
        #         chunk=item["chunk"],
        #         score=item["score"],
        #         source="hybrid",
        #     )
        #     for item in sorted_items[:top_k]
        # ]
        #
        # ================================================================

        # 1. 向量化 query
        embeddings = get_embeddings()
        query_vector = embeddings.embed_single(query)

        # 2. 并行检索
        # 向量检索
        vector_results = self.vector_store.search(
            query_vector, top_k=vector_top_k,
            filter_conditions=filter_conditions,
        )

        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=bm25_top_k)

        # 3. RRF 融合
        fused = {}

        for rank, r in enumerate(vector_results, start=1):
            cid = r.get("chunk_id", r.get("id", ""))
            if cid not in fused:
                fused[cid] = {"score": 0.0}
            fused[cid]["score"] += 1.0 / (self.rrf_k + rank)
            fused[cid]["vector_rank"] = rank
            # 存储 payload 字段用于构造 chunk
            fused[cid]["content"] = r.get("content", "")
            fused[cid]["source_file"] = r.get("source_file", "")
            fused[cid]["clause_id"] = r.get("clause_id")
            fused[cid]["section_title"] = r.get("section_title")
            fused[cid]["metadata"] = r

        for rank, (chunk, _) in enumerate(bm25_results, start=1):
            cid = chunk.chunk_id
            if cid not in fused:
                fused[cid] = {"score": 0.0}
            fused[cid]["score"] += 1.0 / (self.rrf_k + rank)
            fused[cid]["bm25_rank"] = rank
            fused[cid]["chunk"] = chunk

        # 4. 按 RRF 分数降序 → Top-K
        sorted_items = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

        results = []
        for item in sorted_items[:top_k]:
            if "chunk" in item:
                chunk = item["chunk"]
            else:
                # 仅出现在向量结果中，从 payload 重建 TextChunk
                chunk = TextChunk(
                    chunk_id=item.get("metadata", {}).get("chunk_id", ""),
                    content=item.get("content", ""),
                    source_file=item.get("source_file", ""),
                    clause_id=item.get("clause_id"),
                    section_title=item.get("section_title"),
                    metadata={"strategy": "hybrid_vector_only"},
                )
            results.append(RetrievalResult(
                chunk=chunk,
                score=item["score"],
                source="hybrid",
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
