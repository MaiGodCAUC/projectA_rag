"""
核心配置管理模块

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


# 项目根目录（从 core/config.py 向上两级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    """应用全局配置"""

    model_config = SettingsConfigDict(
        env_file=os.path.join(PROJECT_ROOT, ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========== LLM 配置 ==========
    llm_provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        description=f"LLM 提供商：{' / '.join(LLM_PROVIDERS)}",
    )
    llm_api_key: str = Field(
        default="sk-placeholder",
        description="LLM API Key",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="LLM API Base URL（不填则使用 provider 默认地址）",
    )
    llm_model: str = Field(
        default=LLM_DEFAULT_MODELS[DEFAULT_LLM_PROVIDER],
        description="LLM 模型名称",
    )

    # ========== Embedding 配置 ==========
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

    # ========== Qdrant 配置 ==========
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

    # ========== 服务配置 ==========
    host: str = Field(default="0.0.0.0", description="API 监听地址")
    port: int = Field(default=8000, description="API 监听端口")
    debug: bool = Field(default=False, description="调试模式")

    # ========== LangSmith 配置 ==========
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API Key",
    )
    langsmith_project: str = Field(
        default=DEFAULT_LANGSMITH_PROJECT,
        description="LangSmith 项目名",
    )

    # ========== 检索配置 ==========
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

    # ========== 校验 ==========

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """校验 LLM Provider 合法性"""
        if v not in LLM_PROVIDERS:
            raise ValueError(
                f"不支持的 LLM Provider: '{v}'。"
                f"当前支持：{', '.join(LLM_PROVIDERS)}"
            )
        return v

    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key(cls, v: str) -> str:
        """确保 LLM API Key 已配置（非占位符）"""
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
        """校验 Embedding Provider 合法性"""
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
        Embedding API Key 处理：
        - bge / m3e 本地模型不需要 API Key
        - qwen Embedding API 需要 Key，未填时复用 llm_api_key
        """
        emb_provider = info.data.get("embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
        if emb_provider in LOCAL_EMBEDDING_PROVIDERS:
            return ""  # 本地模型不需要 API Key
        if v is None or v == "":
            return info.data.get("llm_api_key", "")
        return v


# 全局单例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
