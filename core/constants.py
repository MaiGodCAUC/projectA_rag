"""
项目常量定义

集中管理 Provider 名称、默认模型、默认 URL 等字符串常量，
避免在 config.py / llm.py / embedding.py 中多处硬编码。
"""

# ================================================================
# LLM Provider
# ================================================================
QWEN = "qwen"
DEEPSEEK = "deepseek"

LLM_PROVIDERS: list[str] = [QWEN, DEEPSEEK]
DEFAULT_LLM_PROVIDER: str = DEEPSEEK

# 各 Provider 默认 Base URL
LLM_BASE_URLS: dict[str, str] = {
    QWEN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    DEEPSEEK: "https://api.deepseek.com/v1",
}

# 各 Provider 默认模型名（切换 provider 时自动兜底）
LLM_DEFAULT_MODELS: dict[str, str] = {
    QWEN: "qwen-plus",
    DEEPSEEK: "deepseek-chat",
}

# ================================================================
# Embedding Provider
# ================================================================
BGE = "bge"
M3E = "m3e"
QWEN_EMB = "qwen"  # 通义千问 Embedding API

EMBEDDING_PROVIDERS: list[str] = [BGE, M3E, QWEN_EMB]
DEFAULT_EMBEDDING_PROVIDER: str = BGE

# 本地 Embedding 模型（不需要 API Key）
LOCAL_EMBEDDING_PROVIDERS: list[str] = [BGE, M3E]

# 各 Provider 默认模型名
EMBEDDING_DEFAULT_MODELS: dict[str, str] = {
    BGE: "BAAI/bge-large-zh-v1.5",
    M3E: "moka-ai/m3e-base",
    QWEN_EMB: "text-embedding-v3",
}

# ================================================================
# Qdrant
# ================================================================
DEFAULT_QDRANT_COLLECTION = "airchina_knowledge_base"

# ================================================================
# LangSmith
# ================================================================
DEFAULT_LANGSMITH_PROJECT = "airchina-rag"

# ================================================================
# RAG 通用参数
# ================================================================
DEFAULT_RAG_TEMPERATURE = 0.1      # RAG 场景需要准确性
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RRF_K = 60                 # RRF 融合参数
