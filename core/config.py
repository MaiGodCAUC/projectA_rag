"""
核心配置管理模块

使用 pydantic-settings 实现：
- 从 .env 文件读取配置
- 启动时自动校验必填项
- 支持多 LLM Provider 热切换
"""

import os
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    llm_provider: Literal["qwen", "deepseek"] = Field(
        default="qwen",
        description="LLM 提供商：qwen（通义千问）/ deepseek（DeepSeek）",
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
        default="qwen-plus",
        description="LLM 模型名称。通义千问：qwen-plus / qwen-max / qwen-turbo；DeepSeek：deepseek-chat / deepseek-reasoner",
    )

    # ========== Embedding 配置 ==========
    embedding_provider: Literal["bge", "m3e", "qwen"] = Field(
        default="bge",
        description="Embedding 提供商：bge（BAAI/bge-large-zh-v1.5，本地）/ m3e（moka-ai/m3e-base，本地）/ qwen（通义千问 Embedding，API）",
    )
    embedding_api_key: Optional[str] = Field(
        default=None,
        description="Embedding API Key（仅 qwen 需要，bge/m3e 本地模型不需要）",
    )
    embedding_model: str = Field(
        default="BAAI/bge-large-zh-v1.5",
        description="Embedding 模型名称。bge: BAAI/bge-large-zh-v1.5；m3e: moka-ai/m3e-base；qwen: text-embedding-v3",
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
        default="airchina_knowledge_base",
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
        default="airchina-rag",
        description="LangSmith 项目名",
    )

    # ========== 检索配置 ==========
    retrieval_top_k: int = Field(default=5, description="检索返回 Top-K")
    retrieval_score_threshold: float = Field(
        default=0.0, description="检索最低相似度阈值"
    )
    hybrid_rrf_k: int = Field(default=60, description="RRF 融合参数 k")

    # ========== 校验 ==========

    @field_validator("llm_api_key")
    @classmethod
    def validate_llm_api_key(cls, v: str) -> str:
        """确保 LLM API Key 已配置（非占位符）"""
        if v == "sk-placeholder" or not v:
            raise ValueError(
                "LLM_API_KEY 未配置！请在 .env 文件中设置有效的 API Key。\n"
                "通义千问：https://dashscope.console.aliyun.com/apiKey\n"
                "DeepSeek：https://platform.deepseek.com/api_keys"
            )
        return v

    @field_validator("embedding_api_key", mode="before")
    @classmethod
    def default_embedding_key(cls, v, info):
        """
        Embedding API Key 处理：
        - bge / m3e 本地模型不需要 API Key，返回空字符串即可
        - qwen Embedding API 需要 Key，未填时复用 llm_api_key
        """
        emb_provider = info.data.get("embedding_provider", "bge")
        if emb_provider in ("bge", "m3e"):
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
