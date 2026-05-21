"""
Embedding 工厂 —— 多模型热切换

支持 3 种 Embedding 模型，统一接口，通过 .env 切换。

----------------------------------------------------------------------
## 你需要自己写的部分

Embedding 是 RAG 的核心基础设施，直接影响检索质量。

学习重点：
1. 理解 Embedding 的本质：将文本映射到高维向量空间，语义相近 → 向量距离近
2. 理解不同模型的差异：维度、中文效果、部署方式、调用成本
3. 理解为什么需要 Embedding 工厂：后续做对比实验时热切换模型，不需要改代码

TODO(用户) 标记的部分是你需要手写的核心逻辑：
- _embed_bge(): 本地 BGE 模型的调用方式
- _embed_m3e(): 本地 M3E 模型的调用方式
- _embed_qwen(): 通义千问 Embedding API 的调用方式
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# os: 读取环境变量（如 EMBEDDING_API_KEY）
import os

# typing: 类型提示
from typing import List, Optional

# 导入配置单例和常量
# get_settings(): 获取全局配置，读取 .env 中的 EMBEDDING_PROVIDER 等字段
# EMBEDDING_PROVIDERS: 支持的 provider 列表 ["bge", "m3e", "qwen"]
# DEFAULT_EMBEDDING_PROVIDER: 默认 provider = "bge"
# LOCAL_EMBEDDING_PROVIDERS: 本地运行的 provider ["bge", "m3e"]
# EMBEDDING_DEFAULT_MODELS: 各 provider 默认模型名映射
from core.constants import (
    EMBEDDING_PROVIDERS,
    DEFAULT_EMBEDDING_PROVIDER,
    LOCAL_EMBEDDING_PROVIDERS,
    EMBEDDING_DEFAULT_MODELS,
)
from core.config import get_settings


# =============================================================================
# Embedding 工厂函数 —— 统一入口
# =============================================================================

def get_embeddings(provider: Optional[str] = None, model: Optional[str] = None):
    """获取 Embedding 实例（统一入口）

    这是整个项目获取 Embedding 的唯一入口。
    后续的索引流水线、检索器、SemanticSplitter 都通过这个函数获取 Embedding。

    设计思想——工厂模式：
    - 调用者不需要知道用的是 bge 还是 qwen
    - 改 .env 一个变量就切换模型
    - 新增模型只需要在这里加一个 elif 分支

    Args:
        provider: "bge" | "m3e" | "qwen"，不传则从 .env 读取
        model: 模型名，不传则使用该 provider 的默认模型

    Returns:
        一个可调用的 Embedding 对象，统一接口：embed(texts) → list[list[float]]

    Usage:
        embeddings = get_embeddings()           # 使用 .env 配置的默认模型
        embeddings = get_embeddings("bge")       # 强制使用 BGE
        vectors = embeddings.embed(["行李限额", "退票规则"])
    """
    # 从配置中获取 provider 和 model
    settings = get_settings()
    provider = provider or settings.embedding_provider or DEFAULT_EMBEDDING_PROVIDER
    model = model or settings.embedding_model or EMBEDDING_DEFAULT_MODELS.get(provider, "")

    # 校验 provider
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"不支持的 Embedding provider: {provider}。"
            f"支持: {EMBEDDING_PROVIDERS}"
        )

    # 分发到具体实现
    if provider == "bge":
        return BGEBackend(model)
    elif provider == "m3e":
        return M3EBackend(model)
    elif provider == "qwen":
        return QwenBackend(model, settings.embedding_api_key)

    raise ValueError(f"未知的 Embedding provider: {provider}")


# =============================================================================
# Backend 基类 —— 定义统一接口
# =============================================================================

class EmbeddingBackend:
    """Embedding 后端抽象基类

    所有具体的 Embedding 实现都继承这个类，必须实现 _embed 方法。
    对外暴露 embed() 和 embed_single() 两个方法。

    为什么需要基类？
    - 确保所有后端有相同的接口（embed / embed_single）
    - 后续做对比实验时，可以遍历所有后端，统一调用
    """

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension    # 向量维度，如 BGE=1024, M3E=768

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化

        Args:
            texts: 文本列表，如 ["行李限额", "退票规则"]

        Returns:
            向量列表，每个向量是 float 列表，长度 = dimension
        """
        return self._embed(texts)

    def embed_single(self, text: str) -> List[float]:
        """单条文本向量化"""
        return self.embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """实际调用 Embedding 模型（子类必须实现）"""
        raise NotImplementedError


# =============================================================================
# Backend 1: BGE (BAAI/bge-large-zh-v1.5) —— 本地运行
# =============================================================================

class BGEBackend(EmbeddingBackend):
    """BGE Embedding 后端 —— FlagEmbedding 库

    BGE 是 BAAI 发布的中文 Embedding 模型，当前中文效果最好的开源模型之一。
    bge-large-zh-v1.5 输出 1024 维向量，支持 max_length=512 tokens。

    特点：
    - 优点：中文语义理解强，民航术语表示质量高
    - 缺点：需要本地 GPU/CPU 推理，模型文件约 1.3GB
    - 依赖：pip install FlagEmbedding（需要 PyTorch >= 2.4）
    """

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """初始化 BGE 后端

        TODO(用户): 你需要理解并手写 BGE 模型的加载和调用逻辑

        关键 API（FlagEmbedding）:
        - from FlagEmbedding import FlagModel
        - model = FlagModel(model_name, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        - vectors = model.encode(texts)  # 返回 numpy array

        BGE 的特殊之处：query_instruction_for_retrieval
        - BGE 模型在训练时用了 instruction 前缀
        - 检索 query 需要加前缀，文档 content 不需要
        - 这是 BGE 比其他模型效果更好的原因之一
        """
        # ================================================================
        # TODO(用户): 从这里开始手写 BGE 的加载和调用逻辑
        # ================================================================
        #
        # 参考实现：
        #
        # from FlagEmbedding import FlagModel
        #
        # self._model = FlagModel(
        #     model_name,
        #     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        #     use_fp16=True,  # 使用半精度，减少显存占用
        # )
        #
        # def _embed(self, texts):
        #     vectors = self._model.encode(texts)
        #     return vectors.tolist()  # numpy → list
        #
        # ================================================================
        super().__init__(model_name, dimension=1024)
        # self._model = None  # TODO(用户): 取消注释并实现加载逻辑

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """调用 BGE 模型进行向量化"""
        # TODO(用户): 实现 BGE 的向量化调用
        raise NotImplementedError("TODO(用户): 实现 BGE Embedding 调用逻辑")


# =============================================================================
# Backend 2: M3E (moka-ai/m3e-base) —— 本地轻量
# =============================================================================

class M3EBackend(EmbeddingBackend):
    """M3E Embedding 后端 —— sentence-transformers 库

    M3E 是 moka-ai 发布的中文 Embedding 模型，比 BGE 更轻量。
    m3e-base 输出 768 维向量，模型文件约 400MB。

    特点：
    - 优点：轻量，CPU 也可运行，对民航短文本效果不错
    - 缺点：长文本语义理解不如 BGE
    - 依赖：pip install sentence-transformers（需要 PyTorch）
    """

    def __init__(self, model_name: str = "moka-ai/m3e-base"):
        """初始化 M3E 后端

        TODO(用户): 手写 M3E 的加载和调用逻辑

        关键 API（sentence-transformers）:
        - from sentence_transformers import SentenceTransformer
        - model = SentenceTransformer(model_name)
        - vectors = model.encode(texts, normalize_embeddings=True)
          normalize_embeddings=True → L2 归一化，余弦相似度 = 内积

        和 BGE 的关键区别：
        - M3E 不需要 query instruction 前缀
        - M3E 维度 768 vs BGE 1024（维度越低，存储越小，但表达能力越弱）
        """
        # ================================================================
        # TODO(用户): 从这里开始手写 M3E 的加载和调用逻辑
        # ================================================================
        #
        # 参考实现：
        #
        # from sentence_transformers import SentenceTransformer
        #
        # self._model = SentenceTransformer(model_name)
        #
        # def _embed(self, texts):
        #     vectors = self._model.encode(
        #         texts,
        #         normalize_embeddings=True,  # L2 归一化
        #         show_progress_bar=False,
        #     )
        #     return vectors.tolist()
        #
        # ================================================================
        super().__init__(model_name, dimension=768)
        # self._model = None  # TODO(用户): 取消注释并实现加载逻辑

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """调用 M3E 模型进行向量化"""
        raise NotImplementedError("TODO(用户): 实现 M3E Embedding 调用逻辑")


# =============================================================================
# Backend 3: 通义千问 text-embedding-v3 —— API 调用
# =============================================================================

class QwenBackend(EmbeddingBackend):
    """通义千问 Embedding 后端 —— OpenAI 兼容 API

    使用阿里云 DashScope 的 text-embedding-v3 模型。
    输出 1536 维向量，通过 HTTP API 调用，不需要本地 GPU。

    特点：
    - 优点：免部署，中文效果优秀，维度高表达能力最强
    - 缺点：有 API 调用成本，网络延迟，批量处理较慢
    - 依赖：langchain-openai（OpenAI 兼容接口）
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v3",
        api_key: Optional[str] = None,
    ):
        """初始化 Qwen Embedding 后端

        TODO(用户): 手写 Qwen Embedding API 的调用逻辑

        关键 API（langchain-openai, OpenAI 兼容）:
        - from langchain_openai import OpenAIEmbeddings
        - embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
          )
        - vectors = embeddings.embed_documents(texts)

        为什么用 langchain-openai 而非直接调 DashScope API？
        - DashScope 提供了 OpenAI 兼容接口
        - 统一代码风格，方便后续切换
        """
        # ================================================================
        # TODO(用户): 从这里开始手写 Qwen API 调用逻辑
        # ================================================================
        #
        # 参考实现：
        #
        # from langchain_openai import OpenAIEmbeddings
        #
        # self._client = OpenAIEmbeddings(
        #     model=model_name,
        #     api_key=api_key or os.getenv("EMBEDDING_API_KEY", ""),
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # )
        #
        # def _embed(self, texts):
        #     return self._client.embed_documents(texts)
        #
        # 注意：embed_documents 内部会自动处理批量切分（API 单次限制），
        # 不需要手动分批。
        #
        # ================================================================
        super().__init__(model_name, dimension=1536)
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
        # self._client = None  # TODO(用户): 取消注释并实现加载逻辑

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """调用 Qwen Embedding API 进行向量化"""
        raise NotImplementedError("TODO(用户): 实现 Qwen Embedding API 调用逻辑")


# =============================================================================
# Embedding 对比实验框架 —— 选型决策工具
# =============================================================================

def run_embedding_comparison(
    queries: List[str],
    reference_texts: List[str],
    providers: Optional[List[str]] = None,
):
    """Embedding 模型对比实验

    对同一批 query × reference 组合，测试不同 Embedding 模型的检索效果。
    这是面试时可以讲的"我用数据支撑了模型选型决策"。

    TODO(用户): 理解实验设计并手写

    实验设计思路：
    1. 准备 15 条民航员工典型 query（如"经济舱免费行李额多少公斤"）
    2. 准备对应的正确答案文本（从文档中提取）
    3. 用每种 Embedding 分别对 query 和 reference 做向量化
    4. 计算余弦相似度，取 Top-5
    5. 统计命中率（正确答案是否在 Top-5 内）

    评估指标：
    - Precision@5: Top-5 中正确结果的比例
    - MRR (Mean Reciprocal Rank): 第一个正确答案的排名倒数

    Args:
        queries: Query 文本列表
        reference_texts: 参考文本（文档片段）列表
        providers: 要对比的 provider 列表，默认全部

    Returns:
        对比结果 dict，格式如:
        {
            "bge": {"precision@5": 0.87, "mrr": 0.72, "avg_latency_ms": 15},
            "m3e": {"precision@5": 0.80, "mrr": 0.65, "avg_latency_ms": 8},
            "qwen": {"precision@5": 0.85, "mrr": 0.70, "avg_latency_ms": 120},
        }
    """
    # ================================================================
    # TODO(用户): 实现 Embedding 对比实验逻辑
    # ================================================================
    #
    # 参考实现框架：
    #
    # import time
    # import numpy as np
    #
    # results = {}
    # for provider_name in providers:
    #     embeddings = get_embeddings(provider_name)
    #
    #     # 向量化 query
    #     query_vecs = embeddings.embed(queries)
    #     # 向量化 reference
    #     ref_vecs = embeddings.embed(reference_texts)
    #
    #     # 计算余弦相似度矩阵
    #     # cosine_sim[i][j] = query_i 和 reference_j 的相似度
    #     query_arr = np.array(query_vecs)
    #     ref_arr = np.array(ref_vecs)
    #
    #     # 归一化后内积 = 余弦相似度
    #     query_norm = query_arr / np.linalg.norm(query_arr, axis=1, keepdims=True)
    #     ref_norm = ref_arr / np.linalg.norm(ref_arr, axis=1, keepdims=True)
    #     cosine_sim = np.dot(query_norm, ref_norm.T)
    #
    #     # 对每个 query，取 Top-5
    #     top5_indices = np.argsort(-cosine_sim, axis=1)[:, :5]
    #
    #     # 统计命中率（假设正确答案在 reference_texts 中索引已知）
    #     # ...
    #
    #     results[provider_name] = {...}
    #
    # return results
    #
    # ================================================================
    raise NotImplementedError(
        "TODO(用户): 实现 Embedding 对比实验逻辑"
    )
