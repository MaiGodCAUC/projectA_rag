"""
LLM 工厂模块

统一封装不同 LLM Provider 的调用接口，返回 LangChain BaseChatModel 实例。

支持国内模型：
- 通义千问: 通过 OpenAI 兼容接口调用 DashScope
- DeepSeek: 通过 OpenAI 兼容接口调用 DeepSeek API

两者均使用 langchain_openai.ChatOpenAI，仅 base_url 不同。

⚠️ TODO(用户)：你需要实现 get_llm() 中的多 provider 切换逻辑。
"""

from langchain_core.language_models import BaseChatModel

from core.config import Settings


# Provider 默认 Base URL
DEFAULT_BASE_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


def get_llm(config: Settings) -> BaseChatModel:
    """
    根据配置返回对应的 LLM 实例。

    支持两种国内 provider：
    - qwen: 通义千问，默认 base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
    - deepseek: DeepSeek，默认 base_url=https://api.deepseek.com/v1

    两者均兼容 OpenAI SDK 协议，统一使用 ChatOpenAI 调用。

    Args:
        config: Settings 对象，提供 llm_provider, llm_api_key, llm_base_url, llm_model

    Returns:
        LangChain BaseChatModel 实例

    Raises:
        ValueError: 不支持的 provider
    """
    # ================================================================
    # TODO(用户)：在此实现多 provider 切换逻辑
    #
    # 提示：
    # 1. 两个 provider 都通过 langchain_openai.ChatOpenAI 兼容调用
    # 2. 关键参数：
    #    - model: config.llm_model
    #    - api_key: config.llm_api_key
    #    - base_url: config.llm_base_url 或从 DEFAULT_BASE_URLS 取默认值
    #    - temperature: 建议 0.1（RAG 场景需要准确性）
    #
    # 示例框架（取消注释并完善）：
    # from langchain_openai import ChatOpenAI
    #
    # provider = config.llm_provider
    # base_url = config.llm_base_url or DEFAULT_BASE_URLS.get(provider)
    #
    # if not base_url:
    #     raise ValueError(f"不支持的 LLM Provider: {provider}")
    #
    # return ChatOpenAI(
    #     model=config.llm_model,
    #     api_key=config.llm_api_key,
    #     base_url=base_url,
    #     temperature=0.1,
    # )
    # ================================================================
    raise NotImplementedError(
        "TODO(用户)：请在 core/llm.py 的 get_llm() 中实现多 provider 切换逻辑。\n"
        "支持 qwen（通义千问）和 deepseek（DeepSeek）两种国内模型。\n"
        "参见代码中的注释和示例框架。"
    )
