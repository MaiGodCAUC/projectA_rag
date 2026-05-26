"""
重排序模块 —— 对检索结果精排

在混合检索返回 Top-N 结果后，用 Cross-Encoder 模型对 query-document 对
做联合编码，输出更精确的相关性分数。

----------------------------------------------------------------------
## 你需要自己写的部分

重排序（Reranking）是 RAG 系统的「精加工」环节。

学习重点：
1. Bi-Encoder vs Cross-Encoder 的区别：
   - Bi-Encoder: 分别编码 query 和 doc，速度快但精度低（向量检索用这个）
   - Cross-Encoder: 联合编码 query+doc，精度高但速度慢（重排序用这个）
2. 重排序策略: 粗排（Top-20）→ 精排（Cross-Encoder）→ 取 Top-5
3. bge-reranker-v2-m3: BAAI 发布的中文 Cross-Encoder，MTEB 榜单前列

面试话术:
"我用 bge-reranker-v2-m3 做重排序。向量检索和 BM25 都是 Bi-Encoder 方式，
对 query 和 doc 分别编码后算相似度——速度快但损失了交互信息。
Cross-Encoder 把 query 和 doc 拼接后一起编码，能捕捉到更细粒度的语义匹配，
比如'逾重行李'和'超重行李费'这种近义表达。代价是速度慢，所以只在 Top-20 上做。"

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# typing: 类型提示
from typing import List

# 数据模型
from rag.models import RetrievalResult


class Reranker:
    """重排序器 —— 对候选结果精排

    使用 FlagEmbedding 的 FlagReranker（bge-reranker-v2-m3 的官方封装）。
    Cross-Encoder 联合编码 query-doc 对，输出相关性分数。

    架构:
    粗排（混合检索 Top-20）→ 精排（Cross-Encoder）→ 最终 Top-5

    面试时可以说:
    "重排序解决的是'粗排结果中哪个最相关'的问题。
    混合检索用 Bi-Encoder 速度快但精度不够，重排序用 Cross-Encoder
    弥补精度损失。这是典型的 precision-recall trade-off 工程实践。"
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ):
        """初始化重排序器

        懒加载 FlagReranker，不影响不依赖重排序的模块。

        Args:
            model_name: 重排序模型名/路径
            use_fp16: 是否使用半精度（减少显存，加速推理）
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None  # 懒加载

    def _lazy_init(self):
        """懒加载 FlagReranker 模型"""
        if self._model is not None:
            return
        from FlagEmbedding import FlagReranker
        self._model = FlagReranker(
            self.model_name,
            use_fp16=self.use_fp16,
        )

    # ------------------------------------------------------------------
    # 重排序
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """对候选检索结果进行 Cross-Encoder 重排序

        TODO(用户): 手写重排序逻辑

        流程：
        1. 构建 query-doc 对列表 [(query, doc_content), ...]
        2. 调用 FlagReranker.compute_score(pairs) 批量打分
        3. 按 Cross-Encoder 分数降序排列
        4. 更新每个结果的 score 为 rerank_score, source 含 'rerank' 标记

        Args:
            query: 用户查询
            candidates: 粗排结果列表（混合检索的输出）
            top_k: 最终返回的精排结果数

        Returns:
            重排序后的结果列表，按 rerank 分数降序
        """
        self._lazy_init()

        if not candidates:
            return []

        # ================================================================
        # TODO(用户): 手写重排序逻辑（参考下面的逐行解释）
        # ================================================================
        #
        # ---- 步骤 1: 构建 query-doc 对 ----
        # 列表推导式: 对每个候选结果，生成 [query, doc_content] 二元列表
        # 例: [["逾重行李费怎么算", "逾重行李费的计算方法..."],
        #      ["逾重行李费怎么算", "行李托运基本规定..."], ...]
        # Cross-Encoder 需要 query 和 doc 成对输入，才能做联合编码
        #
        # ---- 步骤 2: 批量计算 Cross-Encoder 分数 ----
        # compute_score(pairs) 内部流程:
        #   1. tokenizer 将 query + doc 拼接为 [CLS] query [SEP] doc [SEP]
        #   2. 通过 Transformer 编码（所有 token 之间做 Cross-Attention）
        #   3. 取 [CLS] 位置的输出 → 线性层 → 单个分数
        # 返回每个 pair 的相关性分数（分数越高 = 越匹配）
        #
        # ---- 步骤 2b: 处理单条/批量返回格式差异 ----
        # compute_score 单条时返回 float, 批量时返回 list[float]
        # 统一转换为 list 方便后续 zip 操作
        #
        # ---- 步骤 3: 按 Cross-Encoder 分数降序排列 ----
        # zip(candidates, scores) → [(candidate_1, score_1), ...]
        # sort(key=lambda x: x[1]) → 按分数（元组第二个元素）排序
        # reverse=True → 降序（分数高在前）
        #
        # ---- 步骤 4: 更新结果并返回 Top-K ----
        # 把 Cross-Encoder 分数写回 candidate.score（覆盖 Bi-Encoder 原始分）
        # source 标记为 "rerank(hybrid)" 表明经过了重排序
        # 方便后续分析：哪些结果的排名被重排序改变了
        #
        # ================================================================

        # 构建 query-doc 对
        pairs = [[query, c.chunk.content] for c in candidates]

        # 批量计算 Cross-Encoder 分数
        scores = self._model.compute_score(pairs)

        # compute_score 返回 list[float] 或 float（单条时）
        if isinstance(scores, float):
            scores = [scores]

        # 按Cross - Encoder分数降序排列
        scored = list(zip(candidates,scores))
        scored.sort(key=lambda x:x[1], reverse=True)

        # 更新结果并返回 Top-K
        results = []
        for candidate, rerank_score in scored[:top_k]:
            candidate.score = rerank_score
            candidate.source = f"rerank({candidate.source})"
            results.append(candidate)
        return results