"""
LLM 工厂模块

统一封装不同 LLM Provider 的调用接口，返回 LangChain BaseChatModel 实例。

⚠️ TODO(用户)：你需要实现 get_llm() 中的多 provider 切换逻辑。
    LangChain 已提供各 provider 的集成：
    - OpenAI: langchain_openai.ChatOpenAI
    - 通义千问: langchain_community.chat_models.ChatTongyi (或 ChatOpenAI 兼容模式)
    - DeepSeek: langchain_openai.ChatOpenAI (兼容 OpenAI SDK，指定 base_url)
"""

from langchain_core.language_models import BaseChatModel

from core.config import Settings


def get_llm(config: Settings) -> BaseChatModel:
    """
    根据配置返回对应的 LLM 实例。

    支持三种 provider：
    - openai: 使用 ChatOpenAI，base_url 默认 https://api.openai.com/v1
    - qwen: 通义千问，base_url 默认 https://dashscope.aliyuncs.com/compatible-mode/v1
    - deepseek: DeepSeek，base_url 默认 https://api.deepseek.com/v1

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
    # 1. 三个 provider 都可以通过 langchain_openai.ChatOpenAI 兼容调用
    #    （通义千问和 DeepSeek 都兼容 OpenAI SDK 格式）
    # 2. 关键参数：
    #    - model: config.llm_model
    #    - api_key: config.llm_api_key
    #    - base_url: config.llm_base_url 或 provider 默认值
    #    - temperature: 建议默认 0.1（RAG 场景需要准确性）
    # 3. 建议为每个 provider 设置默认的 base_url 兜底
    #
    # 示例框架：
    # from langchain_openai import ChatOpenAI
    #
    # provider = config.llm_provider
    #
    # if provider == "openai":
    #     base_url = config.llm_base_url or "https://api.openai.com/v1"
    # elif provider == "qwen":
    #     base_url = config.llm_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # elif provider == "deepseek":
    #     base_url = config.llm_base_url or "https://api.deepseek.com/v1"
    # else:
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
        "参见代码中的注释和示例框架。"
    )
