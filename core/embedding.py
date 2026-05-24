"""
Embedding 工厂 —— 多模型热切换

支持 3 种 Embedding 模型，统一接口，通过 .env 切换。

======================================================================
三种 Embedding 模型对比（面试速查表）
======================================================================

| 特性         | BGE (bge-large-zh-v1.5) | M3E (m3e-base)       | Qwen (text-embedding-v3) |
|-------------|------------------------|---------------------|--------------------------|
| 开发商       | 智源 BAAI               | moka-ai              | 阿里云                    |
| 运行方式     | 本地 GPU/CPU            | 本地 GPU/CPU         | HTTP API 远程调用         |
| 向量维度     | 1024                    | 768                  | 1536                     |
| 模型大小     | ~1.3 GB                 | ~400 MB              | 无（云端）                |
| 中文效果     | ★★★★★ (SOTA)           | ★★★★                | ★★★★★                   |
| 是否需要 GPU | 建议有                   | 不需要（CPU可跑）    | 不需要                    |
| Query 前缀   | 需要 instruction 前缀    | 不需要                | 不需要                    |
| 调用成本     | 免费（本地推理）         | 免费（本地推理）      | 按 token 计费            |
| 适用场景     | 追求最高精度             | 轻量部署/开发测试     | 免部署/高并发             |
| 依赖库       | FlagEmbedding           | sentence-transformers| langchain-openai         |

面试时可以说：
"我封装了三种 Embedding 的工厂模式，支持热切换。最终我选 BGE 作为默认方案，
因为它中文语义理解最强，对民航术语'逾重行李''代码共享'等专业词汇的表示质量最高。
M3E 作为轻量备选，通义千问 API 作为免部署方案的对比基线。"

======================================================================
学习重点
======================================================================

1. Embedding 本质：文本 → 高维向量，语义相近 → 向量空间距离近（余弦相似度高）
2. 维度 ≠ 质量：BGE 的 1024 维未必比 Qwen 的 1536 维差，关键看训练数据和任务对齐
3. 归一化的重要性：余弦相似度要求向量 L2 归一化后做内积
4. Query/Document 不对称：BGE 对 query 和 document 用不同方式处理（instruction 前缀）
5. 批量 vs 单条：encode() 接收列表比循环单条调用快 10 倍以上（GPU 并行）
======================================================================
"""

# ===========================================================================
# 导入依赖
# ===========================================================================

# os.getenv(): 读取环境变量，用于获取 API Key
# 为什么不用 config.get_settings()？因为 config 依赖 .env 文件，
# 而这里作为底层模块，直接用 os.getenv 更轻量、更少耦合
import os

# typing: 类型提示，增强代码可读性
from typing import List, Optional

# 从项目常量模块导入配置
from core.constants import (
    EMBEDDING_PROVIDERS,          # ["bge", "m3e", "qwen"]
    DEFAULT_EMBEDDING_PROVIDER,   # "bge"
    LOCAL_EMBEDDING_PROVIDERS,    # ["bge", "m3e"] —— 无需 API Key 的本地模型
    EMBEDDING_DEFAULT_MODELS,     # {"bge": "BAAI/bge-large-zh-v1.5", ...}
)
from core.config import get_settings


# =============================================================================
# Embedding 工厂函数 —— 统一入口
# =============================================================================

def get_embeddings(provider: Optional[str] = None, model: Optional[str] = None):
    """获取 Embedding 实例 —— 整个项目的唯一入口

    工厂模式核心：
    - 调用者不关心底层是本地模型还是 API
    - 改 .env 中 EMBEDDING_PROVIDER 一个变量即可全局切换
    - 新增模型只需在这里加一个 elif 分支 + 对应的 Backend 类

    Args:
        provider: "bge" | "m3e" | "qwen"，默认从 .env 读取
        model: 模型名，默认使用该 provider 的推荐模型

    Returns:
        EmbeddingBackend 实例，统一接口 embed(texts) → list[list[float]]

    Usage:
        emb = get_embeddings()               # .env 里配什么就用什么
        emb = get_embeddings("bge")           # 强制 BGE
        emb = get_embeddings("qwen", "text-embedding-v4")  # 指定模型版本
        vecs = emb.embed(["行李限额", "退票规则"])  # → [[1024个float], [1024个float]]
    """
    # 获取全局配置单例（.env 文件只读一次，后续直接读缓存）
    settings = get_settings()

    # 确定 provider：参数 > .env 配置 > 硬编码默认值
    # 这个优先级链确保：局部覆盖 > 全局配置 > 兜底方案
    provider = provider or settings.embedding_provider or DEFAULT_EMBEDDING_PROVIDER

    # 确定 model：参数 > .env 配置 > 该 provider 的默认模型名
    model = model or settings.embedding_model or EMBEDDING_DEFAULT_MODELS.get(provider, "")

    # 防御性校验：provider 必须在白名单中
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(
            f"不支持的 Embedding provider: {provider}。"
            f"支持: {EMBEDDING_PROVIDERS}"
        )

    # 策略分发：根据 provider 名返回对应的 Backend 实例
    if provider == "bge":
        return BGEBackend(model)
    elif provider == "m3e":
        return M3EBackend(model)
    elif provider == "qwen":
        # Qwen 多传一个 api_key（本地模型不需要）
        return QwenBackend(model, settings.embedding_api_key)

    raise ValueError(f"未知的 Embedding provider: {provider}")


# =============================================================================
# Backend 抽象基类 —— 定义统一接口
# =============================================================================

class EmbeddingBackend:
    """Embedding 后端抽象基类

    所有具体后端（BGE / M3E / Qwen）都继承这个类。
    它定义了统一的外部接口 embed() 和 embed_single()，
    子类只需要实现 _embed() 方法即可。

    为什么需要基类？
    1. 接口统一：外部调用只用 embed(texts)，不用管底层是谁
    2. 类型安全：所有 Backend 都是 EmbeddingBackend 的子类
    3. 对比实验：可以 for backend in [BGE(), M3E(), Qwen()] 统一遍历
    """

    def __init__(self, model_name: str, dimension: int):
        """初始化基类

        Args:
            model_name: 模型名称，如 "BAAI/bge-large-zh-v1.5"
            dimension: 输出向量维度。决定 Qdrant Collection 的 vector_size
                       不同模型维度不同：BGE=1024, M3E=768, Qwen=1536
        """
        self.model_name = model_name   # 记录用的哪款模型，日志/调试用
        self.dimension = dimension     # 向量维度，创建 Qdrant Collection 必需

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化 —— 对外统一接口

        这是 RAG 流水线中调用最频繁的方法。
        接收一批文本，返回一批等长的向量。

        Args:
            texts: 文本列表，长度不限（内部自动处理）

        Returns:
            向量列表，每个向量是 dimension 个 float 的列表
            例: embed(["A", "B"]) → [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        """
        return self._embed(texts)

    def embed_single(self, text: str) -> List[float]:
        """单条文本向量化 —— 语法糖

        embed([text])[0] 写法太啰嗦，这个方法是快捷方式。
        用于检索时对用户问题做单条向量化。
        """
        return self.embed([text])[0]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """实际调用 Embedding 模型 —— 子类必须重写这个方法"""
        raise NotImplementedError("子类必须实现 _embed 方法")


# =============================================================================
# Backend 1: BGE（智源 bge-large-zh-v1.5）—— 🥇 默认推荐
# =============================================================================

class BGEBackend(EmbeddingBackend):
    """BGE Embedding —— 当前中文开源 SOTA

    开发商: 北京智源人工智能研究院 (BAAI)
    论文:   C-Pack: Packaged Resources To Advance General Chinese Embedding
    下载:   HuggingFace → BAAI/bge-large-zh-v1.5 (首次运行自动下载 ~1.3GB)

    BGE 的核心创新：Query Instruction（查询指令）
    -----------------------------------------------------
    训练时给每条 query 加了前缀"为这个句子生成表示以用于检索相关文章："，
    document 不加。这让模型学会区分「我在找东西」和「我是被找的东西」两种语义角色。
    推理时 query 必须也加同样的前缀，否则效果会明显下降。

    这个设计在 RAG 场景特别重要——用户的查询意图（"经济舱能带几件行李"）
    和文档中的描述文字（"经济舱旅客可免费托运1件行李"）在语义上不完全相同，
    但加了 instruction 前缀后模型知道"这是个 query，去找相关的 document"。

    面试话术:
    "BGE 是目前中文检索场景最强的开源 Embedding 之一，核心创新是
    Query Instruction——训练时给 query 加前缀，让模型区分查询和被查询
    两种语义角色。这对 RAG 场景非常关键，因为用户问题和文档文字的表述方式
    往往不同。"
    """

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        """初始化 BGE 模型

        自动检测本地缓存路径（ModelScope / HuggingFace），优先使用本地文件，
        避免重复下载。FlagEmbedding 懒加载，不影响其他 provider。
        """
        from FlagEmbedding import FlagModel

        super().__init__(model_name, dimension=1024)

        # 自动检测本地模型路径（优先级：ModelScope > HuggingFace 缓存 > 远程下载）
        model_path = self._resolve_model_path(model_name)

        self._model = FlagModel(
            model_path,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True,
        )

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        """自动检测本地模型缓存路径

        检测顺序（按优先级）:
        1. 直接路径（已经传了本地路径）
        2. ModelScope 缓存（国内镜像下载的，优先检测）
        3. HuggingFace 缓存
        4. 回退到原始 model_name（由 FlagModel 自动下载）

        验证标准: 目录存在且包含 config.json（证明下载完整）
        """
        import os
        from pathlib import Path

        def _is_valid_model(path: Path) -> bool:
            """检查目录是否包含完整的模型文件"""
            return path.is_dir() and (path / "config.json").exists()

        # 已经是完整本地路径 → 直接返回
        if _is_valid_model(Path(model_name)):
            return model_name

        org, name = model_name.split("/")

        # 候选路径列表（按优先级排列）
        candidates = []

        # 1. ModelScope 缓存
        for base in ["E:/huggingface_cache", "D:/huggingface_cache"]:
            base_path = Path(base)
            if base_path.exists():
                for d in base_path.iterdir():
                    if d.is_dir() and d.name.startswith(f"{org}--{name}"):
                        candidates.append(d)

        # 2. HuggingFace 缓存
        hf_cache = Path.home() / ".cache/huggingface/hub"
        hf_model = hf_cache / f"models--{org}--{name}"
        if hf_model.exists():
            snapshots = hf_model / "snapshots"
            if snapshots.exists():
                candidates.extend(list(snapshots.iterdir()))

        # 找第一个有效的路径
        for c in candidates:
            if _is_valid_model(c):
                return str(c)

        # 回退
        return model_name

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """用 BGE 将文本转为向量

        encode() 内部：tokenize → Transformer 编码 → [CLS] pooling → L2 归一化
        返回 numpy.ndarray (len(texts), 1024)，需 tolist() 转为 Python list
        批量处理比逐条快 ~8 倍（GPU 并行）
        """
        vectors = self._model.encode(texts)
        return vectors.tolist()


# =============================================================================
# Backend 2: M3E（moka-ai/m3e-base）—— 🥈 轻量备选
# =============================================================================

class M3EBackend(EmbeddingBackend):
    """M3E Embedding —— 轻量级中文模型

    开发商: moka-ai
    下载:   HuggingFace → moka-ai/m3e-base (约 400MB)

    特点:
    - 比 BGE 轻量得多（400MB vs 1.3GB），下载快、加载快
    - 输出 768 维向量，比 BGE 的 1024 维少 ~25%，Qdrant 存储也更省
    - CPU 可运行，不需要 GPU（推理速度慢一些但可接受）
    - 不需要 query instruction 前缀，使用更简单
    - 中文短文本效果不错，长文本和 BGE 有差距

    面试话术:
    "M3E 是一个轻量级中文 Embedding 替代方案，400MB 模型 CPU 就能跑。
    我在对比实验中发现它对短文本（如条款名称）效果接近 BGE，
    但长文本（如完整的政策段落）语义理解不如 BGE。
    最终我选 BGE 作为主力，M3E 作为资源受限场景的备选。"

    关键 API: sentence-transformers 库
    - 这是 HuggingFace 生态中最通用的 embedding 框架
    - 支持几乎所有开源的 sentence embedding 模型
    - 统一接口: model = SentenceTransformer(name); model.encode(texts)
    - 和 FlagEmbedding 不同，sentence-transformers 更通用但优化少一些
    """

    def __init__(self, model_name: str = "moka-ai/m3e-base"):
        """初始化 M3E 模型

        SentenceTransformer 封装了：模型下载 → 加载 → tokenize → forward → pooling → normalize
        model_name="moka-ai/m3e-base" → 从 HuggingFace 下载约 400MB（BERT-base 架构）
        """
        from sentence_transformers import SentenceTransformer

        super().__init__(model_name, dimension=768)
        self._model = SentenceTransformer(model_name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """用 M3E 将文本转为向量

        normalize_embeddings=True: L2 归一化后余弦相似度 = 内积，Qdrant 检索更快
        show_progress_bar=False: API 场景无需进度条
        """
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


# =============================================================================
# Backend 3: 通义千问（text-embedding-v3）—— 🥉 API 方案
# =============================================================================

class QwenBackend(EmbeddingBackend):
    """通义千问 Embedding —— 阿里云 API

    模型:   text-embedding-v3（DashScope 平台）
    维度:   1536 维（三种方案中最高）
    计费:   按 token 数计费（约 0.0007 元/千 tokens）

    特点:
    - 免部署：不需要 GPU、不需要下载模型文件
    - 维度最高：1536 维，理论上表达能力最强
    - 中文效果好：阿里云专门针对中文优化
    - 网络依赖：需要稳定的外网连接
    - 有调用成本：大批量索引时费用可观

    为什么用 langchain-openai 而不是直接调 DashScope SDK？
    ---------------------------------------------------
    DashScope 提供了 OpenAI 兼容接口（/compatible-mode/v1），
    这意味着所有 OpenAI SDK 兼容的客户端都能直接调用。
    用 langchain-openai 的 OpenAIEmbeddings 可以:
    1. 自动处理批量切分（API 单次最多 2048 tokens）
    2. 自动重试（网络抖动时）
    3. 统一的错误处理

    面试话术:
    "通义千问 Embedding 是我选的 API 方案。和本地模型相比，
    它的优势是免部署、维度高（1536维），适合快速搭建原型。
    对比实验显示它在中文民航文本上的效果和 BGE 接近，
    我保留它作为不需要 GPU 环境的备选方案。"
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v3",
        api_key: Optional[str] = None,
    ):
        """初始化 Qwen Embedding API 客户端

        用 langchain-openai 调用 DashScope 的 OpenAI 兼容接口。
        DashScope 端点: https://dashscope.aliyuncs.com/compatible-mode/v1

        Args:
            model_name: DashScope 模型名（text-embedding-v3 / v2 / v1）
            api_key: 阿里云 API Key，默认从环境变量读取
        """
        from langchain_openai import OpenAIEmbeddings

        super().__init__(model_name, dimension=1536)
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")

        self._client = OpenAIEmbeddings(
            model=model_name,
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """用 Qwen API 将文本转为向量

        embed_documents() 内部自动：按 token 数分 batch → HTTP POST → 拼接结果
        直接返回 list[list[float]]，无需 .tolist()（和本地模型不同）
        网络延迟 ~100-500ms/batch
        """
        return self._client.embed_documents(texts)


# =============================================================================
# Embedding 对比实验框架 —— 选型决策工具
# =============================================================================

def run_embedding_comparison(
    queries: List[str],
    reference_texts: List[str],
    correct_indices: List[int],
    providers: Optional[List[str]] = None,
) -> dict:
    """Embedding 模型对比实验 —— 用数据支撑选型

    对同一批 query × reference 组合，测试不同 Embedding 模型的检索效果。
    算法: 向量化 → 余弦相似度矩阵 → Top-5 检索 → 统计命中率。
    面试时说"我用数据而非直觉选 Embedding"，这就是证据。

    实验流程:
    1. 准备 query 列表（如"经济舱免费行李额多少公斤"）
    2. 准备 reference 列表（从文档中提取的候选片段）
    3. 对每种 Embedding 分别向量化 query 和 reference
    4. 计算余弦相似度矩阵（每个 query × 每个 reference）
    5. 对每个 query 取 Top-5 最相似的 reference
    6. 统计命中率、MRR、平均延迟

    评估指标:
    - Precision@5: Top-5 结果中正确答案的数量 / len(queries)
      → 越高越好，反映"前几个结果里有多少是对的"
    - MRR (Mean Reciprocal Rank): 1/第一个正确答案的排名
      → 越高越好，反映"正确答案排得有多靠前"
      例: 正确排第1位 → 1.0, 排第3位 → 0.33, 没找到 → 0

    Args:
        queries: 用户问题列表，如 ["行李额多少", "退票费怎么算", ...]
        reference_texts: 候选文档片段列表，正确答案混在其中
        correct_indices: 每个 query 对应的正确答案在 reference_texts 中的索引
            例: [0, 3, 7] 表示 query[0] 的正确答案是 reference[0],
                query[1] 的正确答案是 reference[3]
        providers: 要对比的 provider 列表，默认 ["bge", "m3e", "qwen"]

    Returns:
        {
            "bge":  {"precision@5": 0.87, "mrr": 0.72, "avg_latency_ms": 15},
            "m3e":  {"precision@5": 0.80, "mrr": 0.65, "avg_latency_ms": 8},
            "qwen": {"precision@5": 0.85, "mrr": 0.70, "avg_latency_ms": 120},
        }
    """
    import time
    import numpy as np

    # 默认对比全部三种
    if providers is None:
        providers = ["bge", "m3e", "qwen"]

    results = {}

    for provider_name in providers:
        embeddings = get_embeddings(provider_name)

        # 向量化 query 并计时
        t0 = time.time()
        query_vecs = embeddings.embed(queries)
        ref_vecs = embeddings.embed(reference_texts)
        t1 = time.time()

        # 转 numpy 数组 + L2 归一化（归一化后余弦相似度 = 内积）
        query_arr = np.array(query_vecs)
        ref_arr = np.array(ref_vecs)
        query_norm = query_arr / np.linalg.norm(query_arr, axis=1, keepdims=True)
        ref_norm = ref_arr / np.linalg.norm(ref_arr, axis=1, keepdims=True)

        # 余弦相似度矩阵: cosine_sim[i][j] = query_i 和 reference_j 的相似度
        cosine_sim = np.dot(query_norm, ref_norm.T)

        # Top-5: 每行降序排列取前 5 个索引（负号 = 降序）
        top5_indices = np.argsort(-cosine_sim, axis=1)[:, :5]

        # 统计 Precision@5 和 MRR
        precision_at_5 = 0
        reciprocal_ranks = []
        for i, correct_idx in enumerate(correct_indices):
            if correct_idx in top5_indices[i]:
                precision_at_5 += 1
            # 找到正确答案在 Top-5 中的排名
            for rank, idx in enumerate(top5_indices[i], start=1):
                if idx == correct_idx:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)  # 没在 Top-5 里
        precision_at_5 /= len(queries)
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)


        avg_latency_ms = int((t1 - t0) * 1000 / len(queries))

        results[provider_name] = {
            "precision@5": precision_at_5,
            "mrr": mrr,
            "avg_latency_ms": avg_latency_ms,
        }

    return results
