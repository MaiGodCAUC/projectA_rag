"""
LLM 工厂模块

统一封装不同 LLM Provider 的调用接口，返回 LangChain BaseChatModel 实例。

支持国内模型：
- DeepSeek: 通过 OpenAI 兼容接口调用 DeepSeek API
- 通义千问: 通过 OpenAI 兼容接口调用 DashScope

两者均使用 langchain_openai.ChatOpenAI，仅 base_url 不同。
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config import Settings
from core.constants import (
    LLM_BASE_URLS,
    LLM_DEFAULT_MODELS,
    DEFAULT_RAG_TEMPERATURE,
)


def get_llm(config: Settings) -> BaseChatModel:
    """
    根据配置返回对应的 LLM 实例。

    支持两种国内 provider：
    - deepseek: DeepSeek，默认 base_url=https://api.deepseek.com/v1
    - qwen: 通义千问，默认 base_url=https://dashscope.aliyuncs.com/compatible-mode/v1

    两者均兼容 OpenAI SDK 协议，统一使用 ChatOpenAI 调用。

    Args:
        config: Settings 对象，提供 llm_provider, llm_api_key, llm_base_url, llm_model

    Returns:
        LangChain BaseChatModel 实例

    Raises:
        ValueError: 不支持的 provider
    """
    provider = config.llm_provider

    # 确定 base_url：用户自定义优先，否则取 provider 默认值
    base_url = config.llm_base_url or LLM_BASE_URLS.get(provider)
    if not base_url:
        raise ValueError(
            f"不支持的 LLM Provider: '{provider}'。"
            f"当前支持：{', '.join(LLM_BASE_URLS.keys())}"
        )

    # 确定 model：用户自定义优先，否则取 provider 默认 model 兜底
    model = config.llm_model or LLM_DEFAULT_MODELS.get(provider, "")

    return ChatOpenAI(
        model=model,
        api_key=config.llm_api_key,
        base_url=base_url,
        temperature=DEFAULT_RAG_TEMPERATURE,
    )
