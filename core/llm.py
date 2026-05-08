"""
LLM 工厂模块

================================================================================
【学习要点总结】
  1. "Provider"（提供商）概念
     - Provider = 大模型服务供应商，如 DeepSeek、通义千问（阿里云 DashScope）
     - 每个 Provider 有自己的 API 端点（base_url）和认证方式（api_key）
     - 本项目通过切 provider 实现"一套代码对接多家模型"，无须改业务逻辑
  2. Provider 与大模型的关系
     - Provider 是"谁提供 API 服务"（公司/平台层）
     - Model 是"具体用哪个模型"（能力层），如 deepseek-chat、qwen-plus
     - 一个 Provider 下通常有多个 model 可选
  3. DEFAULT_BASE_URLS 的作用
     - 为每个 provider 预设默认 API 地址
     - 用户如果不填 llm_base_url 就用默认值，填了就用自定义值
     - 支持公司内网代理 / 第三方 API 中转等场景
  4. LLM_DEFAULT_MODELS 兜底逻辑
     - 用户切换 provider 时如果没同步改 model，自动使用新 provider 的推荐模型
     - 防止"切到 deepseek 但 model 还填着 qwen-plus"导致的调用失败
================================================================================

【本文件职责边界】
  本文件是"LLM 工厂"——根据配置参数创建并返回 LangChain ChatModel 实例。
  职责范围：
    - 根据 provider 确定 base_url 和 model
    - 创建 langchain_openai.ChatOpenAI 对象
    - 不负责：Prompt 拼接、对话历史管理、RAG 检索逻辑、流式输出
  调用方：main.py（直接对话接口）、rag/generator.py（RAG 生成器）
================================================================================

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


# =============================================================================
# LLM 工厂函数
# =============================================================================

def get_llm(config: Settings) -> BaseChatModel:
    """
    根据配置返回对应的 LLM 实例。

    【Provider 切换流程】
      用户在 .env 中修改 LLM_PROVIDER=deepseek 或 LLM_PROVIDER=qwen，
      本函数自动选择对应的 base_url 并创建 ChatOpenAI 实例。
      DeepSeek 和通义千问都兼容 OpenAI SDK 协议，因此无需引入额外的 SDK 包。

    【base_url 确定策略（两阶段）】
      第一阶段：检查用户是否在 .env 中手动指定了 LLM_BASE_URL
        → 有：使用用户自定义地址（如公司内网代理 http://proxy.internal:8080/v1）
        → 没有：进入第二阶段
      第二阶段：从 LLM_BASE_URLS 字典取 provider 对应的默认地址
        → deepseek → https://api.deepseek.com/v1
        → qwen → https://dashscope.aliyuncs.com/compatible-mode/v1
        → 都不匹配 → 抛出 ValueError

    【model 确定策略（兜底设计）】
      同样两级优先级：
        1. config.llm_model（用户在 .env 中指定的模型名）
        2. LLM_DEFAULT_MODELS[provider]（provider 对应的推荐默认模型）

      兜底设计的意图：
        - 用户只改 LLM_PROVIDER 忘记改 LLM_MODEL 时不会崩溃
        - 例如：从 deepseek 切到 qwen，如果 model 还是 "deepseek-chat"，
          LLM_DEFAULT_MODELS 不会生效（因为 config.llm_model 已经有值），
          此时会直接用 "deepseek-chat" 去请求 qwen 的 API
          → 注意：这不是自动纠错，只是当用户清空 LLM_MODEL 时才自动选对模型
        - 最佳实践：切换 provider 时同时修改 model 字段

    【temperature 说明】
      固定使用 DEFAULT_RAG_TEMPERATURE = 0.1，偏低以保证 RAG 回答的准确性。
      与创造性写作场景（temperature≈0.8）不同，知识问答需要事实一致性。

    Args:
        config: Settings 对象，提供 llm_provider, llm_api_key, llm_base_url, llm_model

    Returns:
        LangChain BaseChatModel 实例，可直接调用 .invoke(prompt)

    Raises:
        ValueError: 不支持的 provider（base_url 查找失败时抛出）
    """
    provider = config.llm_provider

    # ---- 确定 base_url：用户自定义优先，否则取 provider 默认值 ----
    base_url = config.llm_base_url or LLM_BASE_URLS.get(provider)
    if not base_url:
        raise ValueError(
            f"不支持的 LLM Provider: '{provider}'。"
            f"当前支持：{', '.join(LLM_BASE_URLS.keys())}"
        )

    # ---- 确定 model：用户自定义优先，否则取 provider 默认 model 兜底 ----
    model = config.llm_model or LLM_DEFAULT_MODELS.get(provider, "")

    # ---- 创建 ChatOpenAI 实例 ----
    # ChatOpenAI 是 LangChain 对 OpenAI 兼容协议的封装
    # 只要 API 端点兼容 OpenAI 的 /v1/chat/completions 格式，都可以用这个类调用
    return ChatOpenAI(
        model=model,
        api_key=config.llm_api_key,
        base_url=base_url,
        temperature=DEFAULT_RAG_TEMPERATURE,
    )
