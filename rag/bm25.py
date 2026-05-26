"""
BM25 关键词检索引擎 —— 精确匹配的锚

BM25 是 TF-IDF 的改进版，专门用于关键词级精确匹配。
在民航知识库中，它解决了向量检索的一个盲区：
  用户问 "CA1234 退票费率" → 向量可能找不到精确的航班号
  BM25 直接匹配 "CA1234" → 精准命中

----------------------------------------------------------------------
## 你需要自己写的部分

BM25 是信息检索的经典算法，面试高频考点。

学习重点：
1. BM25 公式的三个核心参数：k1（词频饱和）、b（长度归一化）、avgdl（平均文档长度）
2. jieba 中文分词在搜索引擎中的应用
3. 为什么 BM25 和向量检索互补：精确匹配 vs 语义理解

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# re: 正则表达式，用于简单的关键词提取
import re

# typing: 类型提示
from typing import List, Tuple, Optional, Dict

# jieba: 中文分词库，BM25 需要分词才能建立倒排索引
# 安装: pip install jieba
import jieba

# rank_bm25: BM25 算法的 Python 实现
# 安装: pip install rank-bm25
from rank_bm25 import BM25Okapi

# TextChunk: 切片后的文本块，BM25 检索的最小单元
from rag.models import TextChunk


class BM25Retriever:
    """BM25 关键词检索器

    原理：对文档集合建立倒排索引，查询时计算每个文档与 query 的相关性分数。
    分数 = 各查询词在文档中的 IDF × TF 的加权和。

    BM25 和 TF-IDF 的关键区别：
    - TF-IDF: TF 线性增长（出现 10 次 = 出现 1 次 × 10）
    - BM25:   TF 饱和增长（出现 10 次 ≈ 出现 3 次 × 1.5）
              → 避免长文档因词频高而霸榜

    面试话术：
    "我用 BM25 作为关键词精确匹配的锚。民航员工经常查询具体的航班号、
    条款编号、舱位代码，这些向量检索容易遗漏但 BM25 能精准命中。
    比如'第12条'这种查询，向量检索可能返回各种条款，但 BM25 能精确定位。"
    """

    def __init__(self):
        """初始化 BM25 检索器

        jieba 分词配置:
        - 默认精确模式（jieba.cut），适合搜索引擎
        - 全模式（jieba.cut_all）会把所有可能的词都切出来，适合关键词提取
        """
        self._docs: List[TextChunk] = []           # 文档列表（TextChunk 对象）
        self._tokenized_corpus: List[List[str]] = []  # 分词后的语料
        self._bm25: Optional[BM25Okapi] = None        # BM25 模型实例

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def index(self, chunks: List[TextChunk]):
        """建立 BM25 索引

        将 TextChunk 列表分词后构建 BM25Okapi 模型。

        TODO(用户): 手写索引逻辑

        流程：
        1. 对每个 chunk.content 用 jieba 分词
        2. 将分词结果存入 self._tokenized_corpus
        3. 用 BM25Okapi 构建模型

        注意事项：
        - jieba.cut() 返回生成器，需要转为 list
        - 分词前可以用正则清理标点符号（提高匹配精度）
        - BM25Okapi 构造函数接收 list[list[str]]

        Args:
            chunks: 待索引的 TextChunk 列表
        """
        # ================================================================
        # TODO(用户): 手写 BM25 索引逻辑
        # ================================================================
        #
        # 实现参考:
        #
        # self._docs = chunks
        # self._tokenized_corpus = []
        # for chunk in chunks:
        #     # jieba 分词 + 去标点
        #     text = re.sub(r'[^一-鿿\w]', ' ', chunk.content)
        #     tokens = [w.strip() for w in jieba.cut(text) if w.strip()]
        #     self._tokenized_corpus.append(tokens)
        # self._bm25 = BM25Okapi(self._tokenized_corpus)
        #
        # ================================================================
        self._docs = chunks
        self._tokenized_corpus = []
        for chunk in chunks:
            # jieba分词 + 去标点
            text = re.sub(r'[^一-鿿\w]', ' ', chunk.content)
            tokens = [w.strip() for w in jieba.cut(text) if w.strip()]
            self._tokenized_corpus.append(tokens)
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def add_documents(self, new_chunks: List[TextChunk]):
        """增量添加文档到已有索引

        注意：BM25Okapi 不支持增量更新，需要重建整个索引。
        对于大量文档的场景，可以考虑定期批量重建。

        Args:
            new_chunks: 新增的 TextChunk 列表
        """
        all_chunks = self._docs + new_chunks
        self.index(all_chunks)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[TextChunk, float]]:
        """BM25 关键词检索

        对 query 分词后，计算每个文档的 BM25 分数，返回 Top-K。

        TODO(用户): 手写检索逻辑

        流程：
        1. 对 query 用 jieba 分词
        2. 调用 self._bm25.get_scores(tokenized_query)
        3. 取 Top-K 最高分的文档索引
        4. 返回 (TextChunk, score) 列表

        Args:
            query: 查询文本（如 "经济舱免费行李额"）
            top_k: 返回结果数

        Returns:
            [(TextChunk, score), ...] 按分数降序排列
            score 是 BM25 原始分数（非归一化，仅在同索引内可比）
        """
        if not self._bm25:
            return []

        # ================================================================
        # TODO(用户): 手写 BM25 检索逻辑
        # ================================================================
        #
        # 实现参考:
        #
        # # jieba 分词
        # tokens = [w.strip() for w in jieba.cut(query) if w.strip()]
        #
        # # 计算所有文档的 BM25 分数
        # scores = self._bm25.get_scores(tokens)
        #
        # # 取 Top-K（argsort 降序）
        # top_indices = sorted(
        #     range(len(scores)),
        #     key=lambda i: scores[i],
        #     reverse=True
        # )[:top_k]
        #
        # return [
        #     (self._docs[idx], scores[idx])
        #     for idx in top_indices
        #     if scores[idx] > 0  # 过滤零分结果
        # ]
        #
        # ================================================================

        # jieba 分词
        tokens = [w.strip() for w in jieba.cut(query) if w.strip()]

        # 计算 BM25 分数
        scores = self._bm25.get_scores(tokens)

        # 获取 Top-K 索引（按分数降序排列）
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        # 返回 (TextChunk, score) 列表，过滤零分结果
        return [
            (self._docs[idx], scores[idx])
            for idx in top_indices
            if scores[idx] > 0
        ]

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def doc_count(self) -> int:
        """已索引的文档数"""
        return len(self._docs)

    def is_ready(self) -> bool:
        """索引是否就绪"""
        return self._bm25 is not None
