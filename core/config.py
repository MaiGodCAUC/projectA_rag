"""
核心配置管理模块

================================================================================
【学习要点总结】
  1. pydantic-settings 工作原理：类字段名自动映射为环境变量名（大小写不敏感）
     - 例如 llm_provider 字段 → LLM_PROVIDER 环境变量
     - 读取优先级：环境变量 > .env 文件 > Field(default=...) 默认值
  2. model_config["extra"] 控制对未知环境变量的处理策略
     - "ignore"：忽略未定义字段，静默处理（生产环境推荐）
     - "forbid"：遇到未知字段直接报错（更严格）
     - "allow"：允许未知字段通过
  3. @field_validator 是 Pydantic v2 的字段校验装饰器
     - 在 Settings 对象实例化时自动触发
     - mode="before" 表示在校验前拦截原始值
  4. 全局单例模式：整个应用共享一个 Settings 对象，避免重复读取 .env 文件
================================================================================

使用 pydantic-settings 实现：
- 从 .env 文件读取配置
- 启动时自动校验必填项
- 支持多 LLM Provider 热切换
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.constants import (
    DEFAULT_LLM_PROVIDER,
    LLM_PROVIDERS,
    LLM_DEFAULT_MODELS,
    DEFAULT_EMBEDDING_PROVIDER,
    EMBEDDING_PROVIDERS,
    LOCAL_EMBEDDING_PROVIDERS,
    EMBEDDING_DEFAULT_MODELS,
    DEFAULT_QDRANT_COLLECTION,
    DEFAULT_LANGSMITH_PROJECT,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_RRF_K,
)


# =============================================================================
# 项目根目录计算
# 当前文件路径：core/config.py → 向上一级是 core/ → 再向上一级是项目根目录
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Settings 配置类
# =============================================================================

class Settings(BaseSettings):
    """应用全局配置

    【核心机制】pydantic-settings 自动映射规则：
      类字段名 llm_provider 会自动从以下来源按优先级查找值：
        1. 系统环境变量 LLM_PROVIDER（最高优先级）
        2. .env 文件中的 LLM_PROVIDER=xxx
        3. Field(default=...) 中指定的默认值（最低优先级）
      字段名会按字母大写 + 保留大小写各匹配一次，所以 llm_provider、LLM_PROVIDER、Llm_Provider 都能对应。
      注意：LLM 和 PROVIDER 之间的分隔需要匹配，LLMPROVIDER（无下划线）不会匹配。
    """

    # -------------------------------------------------------------------------
    # model_config：Pydantic v2 的配置字典
    # -------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        # env_file：指定 .env 文件路径，程序启动时自动读取为环境变量
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        # extra="ignore"：告诉 Pydantic 忽略 .env 中定义但 Settings 类中没有对应字段的变量
        # 例如 .env 中有 REDIS_URL=xxx，但 Settings 没有 redis_url 字段
        #  - "ignore"（推荐）：静默忽略，生产环境中 .env 常有其他服务变量，不应报错
        #  - "forbid"：严格模式，遇到未定义变量抛出 ValidationError
        #  - "allow"：宽松模式，未定义变量也保留在实例中（通过 extra 属性访问）
        extra="ignore",
    )

    # =========================================================================
    # LLM 配置 —— 控制对话模型的连接参数
    # =========================================================================
    llm_provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,            # 默认用 DeepSeek
        description=f"LLM 提供商：{' / '.join(LLM_PROVIDERS)}",
    )
    llm_api_key: str = Field(
        default="sk-placeholder",                 # 占位符，启动时校验会拦截
        description="LLM API Key",
    )
    llm_base_url: Optional[str] = Field(
        default=None,                             # None 表示使用 provider 默认地址
        description="LLM API Base URL（不填则使用 provider 默认地址）",
    )
    llm_model: str = Field(
        default=LLM_DEFAULT_MODELS[DEFAULT_LLM_PROVIDER],  # 默认模型，如 "deepseek-chat"
        description="LLM 模型名称",
    )

    # =========================================================================
    # Embedding 配置 —— 控制向量化模型的连接参数
    # =========================================================================
    embedding_provider: str = Field(
        default=DEFAULT_EMBEDDING_PROVIDER,
        description=f"Embedding 提供商：{' / '.join(EMBEDDING_PROVIDERS)}",
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="Embedding API Key（仅 qwen 需要，bge/m3e 本地模型不需要）",
    )
    embedding_model: str = Field(
        default=EMBEDDING_DEFAULT_MODELS[DEFAULT_EMBEDDING_PROVIDER],
        description="Embedding 模型名称",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Embedding 推理设备（bge/m3e 本地模型使用）：cpu / cuda",
    )

    # =========================================================================
    # Qdrant 配置 —— 向量数据库连接参数
    # =========================================================================
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant 主机地址",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant HTTP 端口",
    )
    qdrant_collection: str = Field(
        default=DEFAULT_QDRANT_COLLECTION,
        description="Qdrant 集合名称",
    )

    # =========================================================================
    # 服务配置 —— FastAPI 启动参数
    # =========================================================================
    host: str = Field(default="0.0.0.0", description="API 监听地址")
    port: int = Field(default=8000, description="API 监听端口")
    debug: bool = Field(default=False, description="调试模式")

    # =========================================================================
    # LangSmith 配置 —— 可观测性 / LLM 调用追踪
    # =========================================================================
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API Key",
    )
    langsmith_project: str = Field(
        default=DEFAULT_LANGSMITH_PROJECT,
        description="LangSmith 项目名",
    )

    # =========================================================================
    # 检索配置 —— RAG 检索阶段参数
    # =========================================================================
    retrieval_top_k: int = Field(
        default=DEFAULT_RETRIEVAL_TOP_K,
        description="检索返回 Top-K",
    )
    retrieval_score_threshold: float = Field(
        default=0.0, description="检索最低相似度阈值"
    )
    hybrid_rrf_k: int = Field(
        default=DEFAULT_RRF_K,
        description="RRF 融合参数 k",
    )

    # =========================================================================
    # 字段校验器（@field_validator）
    #
    # 【执行时机】Settings() 实例化时自动按顺序执行
    #   - 先收集所有字段值（env / .env / default）
    #   - 再依次调用所有 @field_validator 方法
    #   - 任一校验失败抛出 ValidationError，整个实例化失败
    #
    # 【装饰器语法】@field_validator("字段名") 绑定到指定字段
    #   - 第一个参数 v 是当前字段被收集到的值
    #   - 方法需要返回校验通过后的值（可修改）
    #   - @classmethod 是 Pydantic v2 的强制要求
    # =========================================================================

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """校验 LLM Provider 合法性

        执行时机：Settings() 实例化时，在 llm_provider 字段赋值后立即调用。
        作用：防止用户填写了不支持的 provider 名称（如 "openai"、"gpt" 等），
              在启动阶段就发现配置错误，而非等到实际调用 LLM 时才报错。
        """
        if v not in LLM_PROVIDERS:
            raise ValueError(
                f"不支持的 LLM Provider: '{v}'。"
                f"当前支持：{', '.join(LLM_PROVIDERS)}"
            )
        return v

    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key(cls, v: str) -> str:
        """确保 LLM API Key 已配置（非占位符）

        执行时机：Settings() 实例化时，llm_provider 校验通过后。
        作用：防止用户忘记修改 .env 文件中的占位符 key 就启动服务。
        """
        if v == "sk-placeholder" or not v:
            raise ValueError(
                "LLM_API_KEY 未配置！请在 .env 文件中设置有效的 API Key。\n"
                "DeepSeek：https://platform.deepseek.com/api_keys\n"
                "通义千问：https://dashscope.console.aliyun.com/apiKey"
            )
        return v

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        """校验 Embedding Provider 合法性

        与 validate_llm_provider 类似的防御性校验，确保 embedding provider 在支持列表中。
        """
        if v not in EMBEDDING_PROVIDERS:
            raise ValueError(
                f"不支持的 Embedding Provider: '{v}'。"
                f"当前支持：{', '.join(EMBEDDING_PROVIDERS)}"
            )
        return v

    @field_validator("embedding_api_key", mode="before")
    @classmethod
    def default_embedding_key(cls, v, info):
        """
        Embedding API Key 智能处理

        【mode="before" 含义】在 Pydantic 的类型转换之前执行此校验器。
          普通校验器（无 mode）在类型转换后执行：env "None" → Python None → 校验
          mode="before" 在类型转换前执行：直接拿到 env 原始值 "None" → 校验
          这里用 mode="before" 是为了在 Optional[str] 把 "" 转为 None 之前就拦截处理。

        【info 参数】ValidationInfo 对象，包含：
          - info.data：当前已验证过的字段值字典（按声明顺序）
          - info.field_name：当前正在校验的字段名

        【处理逻辑】
          - bge / m3e 本地模型：不需要 API Key，直接返回空字符串
          - qwen API 模型：需要 Key，如果用户没填则复用 llm_api_key
            （很多用户对 embedding 和 chat 用同一个 Key）
        """
        emb_provider = info.data.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
        if emb_provider in LOCAL_EMBEDDING_PROVIDERS:
            return ""  # 本地模型不需要 API Key
        if v is None or v == "":
            return info.data.get("llm_api_key", "")
        return v


# =============================================================================
# 全局单例模式
#
# 【什么是单例】
#   确保整个程序只有一个 Settings 对象，所有模块通过 get_settings() 获取同一个实例。
#
# 【为什么需要单例】
#   1. Settings() 实例化时读取 .env 文件和校验配置，是"有成本"的操作
#   2. 多个模块（llm.py, embedding.py, main.py）都需要 settings 时，
#      如果各自 new Settings()，会重复读取 .env 和校验
#   3. 单例保证了配置一致性：运行时不可能出现"两个模块读到不同配置"的 bug
#
# 【延迟初始化（Lazy Initialization）】
#   _settings 初始为 None，第一次调用 get_settings() 时才创建实例。
#   好处：import config 时不会触发 .env 读取（避免循环导入问题），
#         只有真正需要配置时才加载。
#
# 【线程安全说明】
#   本项目的简单全局单例在 Python GIL 保护下是安全的。
#   在多 worker 场景（如 gunicorn -w 4）中每个 worker 进程各有自己的实例，
#   这不影响正确性。
# =============================================================================
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例

    返回同一个 Settings 对象，第一次调用时初始化。
    """
    global _settings
    if _settings is None:
        _settings = Settings()       # 这里触发 .env 读取 + 所有 @field_validator 校验
    return _settings
