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

        注意：FlagEmbedding 的 import 写在函数内部（懒加载），
        这样不调用 BGE 时不会加载这个重量级库，不影响其他 provider 的使用。
        这和 pdf_loader 里 lazy import pdfplumber 是同样的工程考量。
        """
        # =================================================================
        # from FlagEmbedding import FlagModel
        # =================================================================
        # FlagEmbedding 是 BGE 模型的官方 Python SDK
        # FlagModel 封装了模型加载 + tokenizer + 前向推理的全流程
        # 第一次导入时会检查 PyTorch 版本（需要 >= 2.4）
        from FlagEmbedding import FlagModel

        # =================================================================
        # super().__init__(model_name, dimension=1024)
        # =================================================================
        # 调用父类 __init__ 保存 model_name 和 dimension
        # dimension=1024 是 bge-large-zh-v1.5 的固定输出维度
        # 这个值会影响 Qdrant Collection 的 vector_size 设置
        super().__init__(model_name, dimension=1024)

        # =================================================================
        # self._model = FlagModel(
        #     model_name,                        # 模型名/路径
        #     query_instruction_for_retrieval=   # ★ BGE 核心参数
        #         "为这个句子生成表示以用于检索相关文章：",
        #     use_fp16=True,                     # 半精度推理
        # )
        # =================================================================
        # 参数详解:
        #
        # model_name:
        #   "BAAI/bge-large-zh-v1.5" 是 HuggingFace 上的标准路径
        #   第一次运行会从 HF 下载约 1.3GB 模型文件到本地缓存
        #   也可以传本地路径如 "./models/bge-large-zh-v1.5"
        #
        # query_instruction_for_retrieval:
        #   这是 BGE 的「咒语」—— 训练时 query 加了这个前缀，推理时必须一致
        #   只对 query（用户问题）生效，对 document（文档内容）不加
        #   FlagModel 内部：查询时自动加前缀，编码文档时不加
        #   面试时可以强调："我理解 embedding 模型不是黑盒——
        #   instruction 前缀对 BGE 的检索精度有显著影响"
        #
        # use_fp16=True:
        #   使用半精度浮点数（float16 替代 float32）
        #   好处: 显存占用减半(~1.3GB→~650MB)，推理速度提升 ~40%
        #   代价: 精度损失微乎其微（对余弦相似度排序几乎无影响）
        #   有 GPU 时建议开启，纯 CPU 可以关掉
        self._model = FlagModel(
            model_name,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=True,
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """用 BGE 将文本转为向量

        encode() 返回 numpy.ndarray，形状 (len(texts), 1024)
        tolist() 转为 Python 原生 list，因为 Qdrant 客户端不认 numpy array
        """
        # =================================================================
        # vectors = self._model.encode(texts)
        # =================================================================
        # encode() 是 FlagModel 的核心方法，一次处理整个列表
        # 内部流程:
        #   1. tokenizer 将文本转为 token IDs（BPE 分词）
        #   2. 通过 Transformer 编码器（24层，1024维 hidden size）
        #   3. 取最后一层 [CLS] token 的向量（或 mean pooling）
        #   4. L2 归一化 → 返回 numpy array
        #
        # 批量处理的优势:
        #   GPU 可以并行计算多个文本，8 条一起处理比逐条处理快 ~8 倍
        #   CPU 也有 batch 优化（矩阵乘法库的向量化）
        vectors = self._model.encode(texts)

        # =================================================================
        # return vectors.tolist()
        # =================================================================
        # numpy array → Python list
        # 例: ndarray([[0.1, 0.2,...], [0.3, 0.4,...]])
        #   → [[0.1, 0.2,...], [0.3, 0.4,...]]
        # 必须转，因为 Qdrant 的 PointStruct.vector 期望 list[float] 类型
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
        """初始化 M3E 模型"""
        # =================================================================
        # from sentence_transformers import SentenceTransformer
        # =================================================================
        # SentenceTransformer 是 sentence-transformers 库的核心类
        # 它封装了: 模型下载 → 加载 → tokenize → forward → pooling → normalize
        # 和 FlagEmbedding 的 FlagModel 是竞品关系，API 类似
        from sentence_transformers import SentenceTransformer

        # 调用父类，声明 768 维输出
        super().__init__(model_name, dimension=768)

        # =================================================================
        # self._model = SentenceTransformer(model_name)
        # =================================================================
        # model_name="moka-ai/m3e-base" → 从 HuggingFace 下载约 400MB
        # SentenceTransformer 内部:
        #   1. 加载 BERT-base-chinese 架构的模型权重
        #   2. 使用 mean pooling（对最后一层所有 token 取平均）
        #   3. 支持 CPU 和 GPU 推理（自动检测）
        self._model = SentenceTransformer(model_name)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """用 M3E 将文本转为向量"""
        # =================================================================
        # vectors = self._model.encode(
        #     texts,
        #     normalize_embeddings=True,   # ← L2 归一化
        #     show_progress_bar=False,      # ← 不显示进度条（API 调用场景）
        # )
        # =================================================================
        # 参数详解:
        #
        # texts: 文本列表，和 BGE 的 encode 一样支持批量
        #
        # normalize_embeddings=True:
        #   L2 归一化: 将向量长度缩放为 1
        #   L2_norm(v) = v / sqrt(sum(v[i]^2))
        #   归一化后，两个向量的余弦相似度 = 它们的内积
        #   这样 Qdrant 用 DOT_PRODUCT 距离就可以等价于 COSINE 距离
        #   计算更快（省去每次检索时的归一化开销）
        #
        # show_progress_bar=False:
        #   不打印 tqdm 进度条，因为我们不是在 Jupyter 里交互使用
        #   作为 API 调用，打印进度条会干扰日志输出
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # numpy → python list，和 BGE 同样的原因
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

        Args:
            model_name: DashScope 的模型名
            api_key: 阿里云 API Key，默认从环境变量 EMBEDDING_API_KEY 读取
        """
        # =================================================================
        # from langchain_openai import OpenAIEmbeddings
        # =================================================================
        # OpenAIEmbeddings 是 langchain-openai 提供的 Embedding 客户端
        # 它原本是为 OpenAI API 设计的，但因为 DashScope 提供了
        # OpenAI 兼容接口，改一下 base_url 就能直接调用通义千问
        # 这就是「OpenAI 兼容」的价值——生态复用
        from langchain_openai import OpenAIEmbeddings

        # 调用父类，声明 1536 维输出
        super().__init__(model_name, dimension=1536)

        # =================================================================
        # self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
        # =================================================================
        # API Key 的优先级: 参数传入 > 环境变量 > 空字符串（会报错）
        # os.getenv 直接从系统环境变量读取，不依赖 .env 文件
        # 第二个参数 "" 是默认值——没设环境变量也不报 None
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")

        # =================================================================
        # self._client = OpenAIEmbeddings(
        #     model=model_name,        # "text-embedding-v3"
        #     api_key=self.api_key,    # 阿里云 API Key
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # )
        # =================================================================
        # 参数详解:
        #
        # model: DashScope 上的模型名。可选:
        #   "text-embedding-v3" —— 最新版，1536 维，推荐
        #   "text-embedding-v2" —— 旧版，1536 维
        #   "text-embedding-v1" —— 最早版，1024 维（已不推荐）
        #
        # api_key: 从阿里云 DashScope 控制台获取
        #   https://dashscope.console.aliyun.com/
        #
        # base_url:
        #   这是 DashScope 的 OpenAI 兼容端点
        #   路径 /compatible-mode/v1 让 OpenAI SDK 认为在调 OpenAI
        #   实际上是阿里云后端在响应
        #   如果不设 base_url，默认调 api.openai.com（会报错）
        self._client = OpenAIEmbeddings(
            model=model_name,
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """用 Qwen API 将文本转为向量"""
        # =================================================================
        # return self._client.embed_documents(texts)
        # =================================================================
        # embed_documents() 内部做了三件事:
        # 1. 将 texts 按 token 数切分为多个 batch（单次 API 限制 ~2048 tokens）
        # 2. 对每个 batch 发送 HTTP POST 到 DashScope
        # 3. 拼接所有 batch 的结果返回
        #
        # 和本地模型不同，API 调用有网络延迟（~100-500ms/batch），
        # 批量处理 100 条文本可能需要数秒到数十秒
        #
        # 对比 BGE/M3E 的 numpy tolist:
        #   embed_documents 直接返回 list[list[float]]，不需要 .tolist()
        return self._client.embed_documents(texts)


# =============================================================================
# Embedding 对比实验框架 —— 选型决策工具
# =============================================================================

def run_embedding_comparison(
    queries: List[str],
    reference_texts: List[str],
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
    - Precision@5: Top-5 结果中正确答案的数量 / 5
      → 越高越好，反映"前几个结果里有多少是对的"
    - MRR (Mean Reciprocal Rank): 1/第一个正确答案的排名
      → 越高越好，反映"正确答案排得有多靠前"
      例: 正确排第1位 → 1.0, 排第3位 → 0.33, 没找到 → 0

    Args:
        queries: 用户问题列表，如 ["行李额多少", "退票费怎么算", ...]
        reference_texts: 候选文档片段列表，正确答案混在其中
        providers: 要对比的 provider 列表，默认 ["bge", "m3e", "qwen"]

    Returns:
        {
            "bge":  {"precision@5": 0.87, "mrr": 0.72, "avg_latency_ms": 15},
            "m3e":  {"precision@5": 0.80, "mrr": 0.65, "avg_latency_ms": 8},
            "qwen": {"precision@5": 0.85, "mrr": 0.70, "avg_latency_ms": 120},
        }
    """
    # =================================================================
    # 延迟导入重型依赖
    # =================================================================
    # time: 用于测量每种 Embedding 的延迟（毫秒级）
    # numpy: 用于高效的矩阵运算（余弦相似度 = 归一化内积）
    # 这两个只在对比实验时用，不进文件顶部
    import time
    import numpy as np

    # 默认对比全部三种
    if providers is None:
        providers = ["bge", "m3e", "qwen"]

    results = {}

    for provider_name in providers:
        # ---- 获取 Embedding 实例 ----
        embeddings = get_embeddings(provider_name)

        # ---- 计时：向量化 query ----
        t0 = time.time()
        query_vecs = embeddings.embed(queries)
        # query_vecs: [[1024个float], [1024个float], ...] 共 len(queries) 个

        # ---- 计时：向量化 reference ----
        ref_vecs = embeddings.embed(reference_texts)
        # ref_vecs: [[1024个float], ...] 共 len(reference_texts) 个
        t1 = time.time()

        # ---- 转为 numpy 数组以进行矩阵运算 ----
        # query_arr.shape = (len(queries), dimension)
        # ref_arr.shape   = (len(reference_texts), dimension)
        query_arr = np.array(query_vecs)
        ref_arr = np.array(ref_vecs)

        # ---- L2 归一化 ----
        # np.linalg.norm(arr, axis=1, keepdims=True):
        #   计算每个向量(axis=1)的 L2 范数，keepdims 保持二维形状以便广播除法
        #   例如: [1024维向量] → 归一化 → [单位向量，长度为1]
        # 归一化后，余弦相似度 = 内积 = np.dot(query_norm, ref_norm.T)
        query_norm = query_arr / np.linalg.norm(query_arr, axis=1, keepdims=True)
        ref_norm = ref_arr / np.linalg.norm(ref_arr, axis=1, keepdims=True)

        # ---- 计算余弦相似度矩阵 ----
        # cosine_sim.shape = (len(queries), len(reference_texts))
        # cosine_sim[i][j] = query_i 和 reference_j 的余弦相似度
        # 值域: [-1, 1]，1 表示完全相同，-1 表示完全相反
        # .T 是转置: ref_norm 从 (N, D) 变为 (D, N) 才能和 query_norm (Q, D) 做内积
        cosine_sim = np.dot(query_norm, ref_norm.T)

        # ---- 对每个 query 取 Top-5 ----
        # np.argsort(-cosine_sim, axis=1):
        #   负号实现降序排列（argsort 默认升序，加负号 = 降序）
        #   axis=1 表示按行排序（每行 = 一个 query 的所有 reference 相似度）
        # [:, :5] 取每行前 5 个（相似度最高的 5 个 reference 的索引）
        top5_indices = np.argsort(-cosine_sim, axis=1)[:, :5]

        # ---- 统计指标 ----
        # TODO(用户): 这里需要你知道每个 query 的正确答案在 reference_texts 中的索引
        # 需要额外参数 correct_indices: List[int]，标注每个 query 对应的正确答案位置
        #
        # 参考实现:
        #
        # 如果你有 correct_indices（长度 = len(queries)，每个值是 0~len(references)-1）:
        #
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
            "top5_indices": top5_indices,  # 仅供调试查看
            "avg_latency_ms": avg_latency_ms,
            # TODO(用户): 取消下面注释并填入计算好的值
            # "precision@5": precision_at_5,
            # "mrr": mrr,
        }

    return results
