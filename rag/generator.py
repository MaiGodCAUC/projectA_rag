"""
RAG 生成引擎 —— 检索结果 → LLM 回答 + 引用溯源

将检索结果注入 Prompt，调用 LLM 生成带条款级引用的专业回答。
支持流式输出（SSE Serve-Sent Events）和多轮对话。

整体数据流:
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ 用户 Query   │ →   │ 混合检索     │ →   │ RAGGenerator  │ →   │ CitedAnswer │
│              │     │ Top-K 结果   │     │ .generate()   │     │ (含引用)     │
└─────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                    内部 6 步流水线（含可观测性）:
                                    ① 上下文窗口管理: 截断超长检索结果
                                    ② 构建上下文:   检索结果 → 格式化文本块
                                    ③ 组装消息:     SystemPrompt + 历史 + 上下文 + Query
                                    ④ 调用 LLM:     .invoke() 或 .astream()
                                    ⑤ 提取引用:     正则 [来源:...] → Citation 列表
                                    ⑥ 包装返回:     CitedAnswer(含 trace_id + cost_ms)

----------------------------------------------------------------------
## 你需要自己写的部分

RAG 生成是「检索」和「用户」之间的桥梁。这是整个系统对外呈现的
最终环节——前面的文档解析、切片、检索都在为这一步服务。

学习重点：
1. Prompt 工程：如何让 LLM 输出格式化的带引用回答
2. 上下文窗口管理：token 超限时的截断策略
3. 引用提取：用正则从 LLM 回答中提取 [来源: ...] 标记
4. 流式输出：用 async generator 实现 SSE
5. 可观测性集成：用 RAGTraceCallback 给每个环节插桩

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ===========================================================================
# 1. 导入依赖
# ===========================================================================

# json —— Python 标准库，用于 JSON 序列化
# 用途: generate_stream() 末尾把 Citation 列表序列化为 JSON，
#       嵌入到 SSE 流的特殊标记 <!--CITATIONS:...--> 中传给前端
import json

# re (Regular Expression) —— 正则表达式模块
# 用途: _extract_citations() 中扫描 LLM 回答，
#       找到所有 [来源: XXX] 标记并提取括号内的来源文本
# 关键函数:
#   re.finditer(pattern, text) → 返回迭代器，逐个给出匹配位置对象
#   m.group(1)                 → 捕获第一个括号 () 内的子串
import re

# time —— 时间模块
# 用途: generate() 中记录开始/结束时间，计算全链路耗时 cost_ms
# 注意: time.time() 返回 Unix 时间戳(墙上时钟)，精度到微秒
#       int((end-start)*1000) → 毫秒级整数
import time

# typing —— 类型提示模块
# List[Citation]             → 元素类型的列表泛型
# Optional[str]              → "str 或 None" 的简写
# AsyncIterator[str]         → 异步生成器，每次 yield 一个 str
# Dict, Any                  → 灵活的字典类型
from typing import List, Optional, AsyncIterator, Dict, Any

# ---- 本项目内部模块 ----

# Pydantic 数据模型（来自 rag/models.py）
# RetrievalResult: 单条检索结果 = TextChunk + score + source("vector"/"bm25"/"hybrid")
# CitedAnswer:     带引用的回答 = answer_text + citations[] + cost_ms + trace_id
# Citation:        单条引用   = doc_name + clause_id + section_title + original_text
from rag.models import RetrievalResult, CitedAnswer, Citation

# 可观测性回调（来自 rag/callbacks.py）
# RAGTraceCallback: LangChain BaseCallbackHandler 子类
#   在 RAG 流程的关键节点插桩，记录每个环节的耗时和状态
#   用法: callback = RAGTraceCallback()
#         callback.start_node("vector_search")
#         ...执行检索...
#         callback.end_node({"hit_count": 20})
#         最后 callback.get_report() 获取完整耗时报告
from rag.callbacks import RAGTraceCallback

# 配置管理（来自 core/config.py）
# get_settings() 返回全局 Settings 单例:
#   settings.llm_provider → "openai" / "qwen" / "deepseek"
#   settings.openai_api_key / settings.qwen_api_key / ...
from core.config import get_settings

# LLM 工厂（来自 core/llm.py）
# get_llm(settings) 根据 settings.llm_provider 返回对应的 LangChain ChatModel
# 内部逻辑: if provider=="openai" → ChatOpenAI(...)
#          elif provider=="qwen" → ChatTongyi(...)
#          elif provider=="deepseek" → ChatDeepSeek(...)
from core.llm import get_llm

# 常量（来自 core/constants.py）
# DEFAULT_RAG_TEMPERATURE = 0.1  → RAG 需要确定性输出
# DEFAULT_RETRIEVAL_TOP_K = 5    → 默认注入 Prompt 的检索结果数
from core.constants import DEFAULT_RAG_TEMPERATURE, DEFAULT_RETRIEVAL_TOP_K

# LangChain 消息类型
# SystemMessage: role="system"，用于设定 AI 行为规则（System Prompt）
# HumanMessage:  role="user"，  用于用户问题和上下文注入
# 注意: LangChain 的 .invoke() 和 .astream() 只接受这两种消息对象，
#       不接受原始 Python dict。需要先转换。
from langchain_core.messages import HumanMessage, SystemMessage


# =============================================================================
# 2. System Prompt —— 国航内部业务支持助手
# =============================================================================
# 这是 RAG 质量的决定性因素之一。
# Prompt 写不好 → LLM 瞎编、格式乱、引用不准确
# Prompt 写得好 → LLM 按要求输出，引用可追溯
#
# 面试时可以讲的设计思路:
# ┌─────────────────────────────────────────────────────────────┐
# │ 1. 角色设定: 「国航内部业务支持助手」                       │
# │    → 让 LLM 知道服务对象是「员工」而非「旅客」              │
# │    → 回答风格: 专业、简洁、可操作                           │
# │                                                             │
# │ 2. 输出格式: 结论先行 → 政策依据 → 操作指引                │
# │    → 一线员工最关心「答案是什么」「依据是什么」「我该怎么     │
# │       做」，这个格式直接对应他们的工作流                    │
# │                                                             │
# │ 3. 引用规范: [来源: 文档名 条款编号]                        │
# │    → 这是企业 RAG 的核心——每条结论必须可追溯               │
# │    → 不是 "来自文档A"，而是精确到条款级                     │
# │                                                             │
# │ 4. 红线: 不确定时标注 ⚠️，不得编造                          │
# │    → RAG 最大的风险是 LLM 幻觉，用 Prompt 约束是最低成本     │
# │       的缓解手段                                            │
# │                                                             │
# │ 5. 语气: 专业、准确、简洁 + 正式但不生硬                    │
# │    → 匹配企业内部的沟通文化                                 │
# └─────────────────────────────────────────────────────────────┘

SYSTEM_PROMPT = """你是中国国际航空股份有限公司（国航）内部业务支持助手。
你的职责是帮助一线员工（客服坐席、值机柜台、登机口、行李查询等）
快速准确地查找公司政策、规定和操作流程。

## 回答格式

1. **结论先行**：第一句话直接回答问题
2. **政策依据**：引用具体的文档名称和条款编号
3. **操作指引**：如有必要，给出下一步操作建议
4. **引用标记**：每个事实性陈述后标注 [来源: 文档名 条款编号]

## 引用格式

文中使用 [来源: 文档名 条款编号] 标记信息来源。例如：
- [来源: 托运行李运输规定 第1条]
- [来源: 国内旅客运输总条件 第3.2条]

## 红线

- 如果检索到的信息不足以给出确定答案，必须明确标注：
  「⚠️ 以下信息可能需要进一步核实，建议查阅原文」
- 不得编造任何政策条款、数字、费率
- 如果有多个条款涉及同一个问题，列出所有相关条款
- 法律和赔偿相关问题，建议同时咨询法务部门

## 语气

- 专业、准确、简洁
- 使用正式但不生硬的商务中文
- 面对员工不熟悉的复杂问题时，耐心解释术语
"""


# =============================================================================
# 3. RAGGenerator 类 —— RAG 生成引擎核心
# =============================================================================

class RAGGenerator:
    """RAG 生成器 —— 检索结果 → LLM 回答 + 引用溯源

    核心流水线（6 步）：
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ ① 窗口   │ → │ ② 构建   │ → │ ③ 组装   │ → │ ④ LLM   │ → │ ⑤ 引用   │ → │ ⑥ 包装   │
    │   管理    │    │   上下文  │    │   消息    │    │   生成   │    │   提取   │    │   返回   │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘

    每个步骤都通过 RAGTraceCallback 记录了耗时，可在 LangSmith 中查看完整的调用链。

    面试话术:
    "我把检索结果注入到精心设计的 System Prompt 中，让 LLM 以
    国航业务支持助手的角色回答。关键是引用的可追溯性——
    每个结论必须标注 [来源: 文档名 条款编号]，
    内部员工可以点击跳转到原文，而不是盲信 LLM 的回答。
    同时，通过 RAGTraceCallback 我给每个环节都插了桩，
    能在 LangSmith 中看到完整的耗时拆解。"
    """

    # ------------------------------------------------------------------
    # __init__: 构造函数 —— 只在创建 RAGGenerator 实例时执行一次
    # ------------------------------------------------------------------

    def __init__(self, temperature: float = DEFAULT_RAG_TEMPERATURE):
        """初始化 RAG 生成器

        做了两件事:
        ① 调用 get_llm() 拿到 LangChain ChatModel 实例
           —— 具体是 GPT-4o / 通义千问 / DeepSeek，由 .env 中的 LLM_PROVIDER 决定
        ② 设置 temperature = 0.1
           —— 极低温度 = 接近确定性输出
           —— 政策问答不能有随机性，今天和明天的答案必须一致
           —— 为什么不是 0？某些模型厂商在 temperature=0 时有特殊处理（如缓存），
              0.1 是「近乎确定但保留极小灵活性」的经验值

        Args:
            temperature: LLM 温度参数，值域 [0, 2]
                         0 = 贪婪解码（完全确定性）
                         2 = 最大随机性（不适合 RAG）
        """
        # 存为实例属性（目前只用于初始化，后续可能用于动态调整）
        self.temperature = temperature

        # get_llm(get_settings()) → LangChain BaseChatModel 子类实例
        # 这个对象后续用 .invoke([messages]) 做同步调用
        #             用 .astream([messages]) 做异步流式调用
        self.llm = get_llm(get_settings())

        # 在 LLM 实例上设置 temperature
        # 注意: LangChain ChatModel 的 temperature 属性是直接赋值的
        self.llm.temperature = temperature

    # ==================================================================
    # generate() —— 非流式生成（一次性返回完整回答）
    # ==================================================================

    def generate(
        self,
        query: str,                                         # 用户问题字符串
        retrieval_results: List[RetrievalResult],            # 检索结果列表
        chat_history: Optional[List[Dict[str, str]]] = None, # 可选的多轮对话历史
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,                # 注入 Prompt 的检索结果数
    ) -> CitedAnswer:
        """基于检索结果生成带引用的回答（同步/非流式）

        完整流程 6 步 ——

        步骤① 上下文窗口管理:
          如果检索结果太多导致 token 超限，用三层截断策略裁剪。
          调用 _manage_context_window() —— 按分数排序 → 条款去重 → 字符截断

        步骤② 构建上下文 (context):
          把每个 RetrievalResult 翻译成 LLM 可读的格式化文本块。
          格式: [1] 来源：文档名/条款编号/章节标题\n原文内容...

        步骤③ 组装消息 (messages):
          把 System Prompt、历史对话、检索上下文、用户问题拼成消息列表。
          结构: [SystemMessage, (历史...), HumanMessage(上下文+问题)]

        步骤④ 调用 LLM:
          self.llm.invoke(lc_messages) → AIMessage
          response.content → 回答文本（内嵌 [来源: ...] 标记）
          通过 config={"callbacks": [callback]} 传递回调

        步骤⑤ 提取引用:
          self._extract_citations(response_text, trimmed_results)
          → 正则扫描 → 解析 doc_name/clause_id → 回查原文 → Citation 列表

        步骤⑥ 包装返回:
          CitedAnswer(answer_text=..., citations=..., cost_ms=..., trace_id=...)

        Args:
            query: 用户原始问题，如 "旅客行李箱摔坏了怎么赔?"
            retrieval_results: 来自 hybrid_search.search() 的检索结果
            chat_history: 历史对话，格式 [{"role":"user","content":"..."},
                         {"role":"assistant","content":"..."}, ...]
            top_k: 往 Prompt 注入几条检索结果（默认 5）

        Returns:
            CitedAnswer 对象:
            - answer_text: 内嵌 [来源: ...] 标记的完整回答
            - citations:   结构化引用列表，前端做点击弹窗用
            - cost_ms:     从函数入口到返回的总耗时（毫秒）
            - trace_id:    LangSmith 追踪 ID，可据此在后台查找完整调用链
        """
        # ---- 创建回调实例，用于全链路插桩 ----
        # 每个 generate() 调用创建一个新的 RAGTraceCallback，
        # trace_id 唯一标识这次请求
        callback = RAGTraceCallback()

        # ---- 记录开始时间，用于最终计算 cost_ms ----
        start_time = time.time()

        # ================================================================
        # 步骤①: 上下文窗口管理 —— Token 超限时智能截断
        # ================================================================
        # 先截断再构建上下文，因为截断后的结果才是真正注入 Prompt 的
        callback.start_node("context_manage")
        trimmed_results = self._manage_context_window(retrieval_results[:top_k])
        callback.end_node({
            "input_count": len(retrieval_results[:top_k]),
            "output_count": len(trimmed_results),
        })

        # ================================================================
        # 步骤②: 构建上下文 —— 检索结果 → 格式化文本块
        # ================================================================
        # 为什么要格式化？
        #   检索结果是 Pydantic 对象列表，LLM 只接受字符串
        callback.start_node("build_context")

        # context_parts: 存放每个检索结果的格式化文本
        context_parts = []

        # enumerate(seq, start=1): i 从 1 开始（而非 0），让 LLM 看到 [1] [2] [3]
        for i, r in enumerate(trimmed_results, 1):
            chunk = r.chunk               # r 是 RetrievalResult，r.chunk 是 TextChunk

            # 构建头部行: "[1] 来源：托运行李运输规定.md/第3.2条/破损赔偿"
            header = f"[{i}] 来源：{chunk.source_file}"
            if chunk.clause_id:
                header += f"/ {chunk.clause_id}"        # 追加条款编号
            if chunk.section_title:
                header += f"/ {chunk.section_title}"     # 追加章节标题

            # 拼接: 头部 + 换行 + 原文正文
            context_parts.append(f"{header}\n{chunk.content}")

        # "\n\n---\n\n".join(...): 用分隔线连接各参考块
        context = "\n\n---\n\n".join(context_parts)
        callback.end_node({"context_chars": len(context)})

        # ================================================================
        # 步骤③: 组装消息列表 —— dict → LangChain 消息对象
        # ================================================================
        callback.start_node("assemble_messages")

        # 消息列表结构:
        #   [0] SystemMessage  = 行为规则（角色、格式、红线）
        #   [1...n-1] 历史对话 = 多轮上下文（可选）
        #   [n] HumanMessage   = 参考资料 + 用户问题（当前交互）

        messages: List[Dict[str, str]] = []

        # 第一条: System Prompt —— 设定 LLM 的角色和输出格式
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # 中间: 历史对话（如果有多轮上下文）
        if chat_history:
            # chat_history[-6:] → 只保留最近 6 条消息（= 最近 3 轮对话）
            messages.extend(chat_history[-6:])

        # 最后: 当前问题 + 参考资料（打包成一条 HumanMessage）
        user_message = (
            f"请根据以下参考资料回答员工的问题。\n\n"
            f"## 参考资料\n{context}\n\n"
            f"## 员工问题\n{query}"
        )
        messages.append({"role": "user", "content": user_message})
        callback.end_node()

        # ================================================================
        # 步骤④: 调用 LLM —— 同步调用，阻塞等待完整回答
        # ================================================================
        # 把 Python dict 消息列表转为 LangChain 能识别的消息对象列表
        lc_messages = []
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            else:
                lc_messages.append(HumanMessage(content=m["content"]))

        # self.llm.invoke([messages], config={"callbacks": [callback]})
        # config={"callbacks": [callback]} 是关键：
        #   把我们的 RAGTraceCallback 传给 LangChain
        #   LangChain 会在调用 LLM 前后自动触发 callback.on_llm_start() 和
        #   callback.on_llm_end()，记录 LLM 调用的 token 消耗和耗时
        response = self.llm.invoke(
            lc_messages,
            config={"callbacks": [callback]},
        )
        response_text = response.content
        # 此时 response_text 的内容类似:
        # "根据行李运输规定，托运行李损坏的赔偿标准为每公斤不超过人民币100元
        #  [来源: 托运行李运输规定 第3.2条]。旅客需在航班到达后7日内
        #  以书面形式提出索赔 [来源: 旅客投诉处理规范 第5.1条]。"
        #
        # 回答中已经嵌入了 [来源: ...] 标记——
        # 这是 SYSTEM_PROMPT 中「引用格式」指令的效果！

        # ================================================================
        # 步骤⑤: 提取引用 —— 正则扫描 [来源: ...] 标记
        # ================================================================
        callback.start_node("citation_extract")
        citations = self._extract_citations(response_text, trimmed_results)
        callback.end_node({"citation_count": len(citations)})

        # ================================================================
        # 步骤⑥: 包装返回结果
        # ================================================================
        # time.time() - start_time → 浮点数秒数（如 1.523 秒）
        # * 1000 → 毫秒（如 1523.0）
        # int(...) → 整数（1523）
        cost_ms = int((time.time() - start_time) * 1000)

        # 构造并返回 CitedAnswer 对象
        return CitedAnswer(
            answer_text=response_text,       # LLM 生成的完整回答（含 [来源:...]）
            citations=citations,              # 结构化引用列表
            cost_ms=cost_ms,                  # 全链路耗时（面试时可以说 "P95 < 3s"）
            trace_id=callback.trace_id,       # LangSmith 追踪 ID
        )

    # ==================================================================
    # generate_stream() —— 流式生成（SSE 推送，逐 token 输出）
    # ==================================================================

    async def generate_stream(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> AsyncIterator[str]:
        """流式生成 —— 逐 token 输出，适合 SSE 推送

        和非流式 generate() 的核心区别:

        ┌───────────────────────┬──────────────────────────────┐
        │  generate()           │  generate_stream()           │
        ├───────────────────────┼──────────────────────────────┤
        │ self.llm.invoke()     │ self.llm.astream()           │
        │ 阻塞等待全部 token     │ 异步逐个 yield token          │
        │ 返回 CitedAnswer      │ yield 字符串 → SSE data      │
        │ HTTP 普通 JSON 响应   │ HTTP SSE(text/event-stream)  │
        └───────────────────────┴──────────────────────────────┘

        流式输出的时序（以生成 "根据规定，行李损坏赔偿标准为..." 为例）:
        t=0ms    async for 开始
        t=50ms   chunk.content = "根据"     → yield "根据"
        t=80ms   chunk.content = "规定"     → yield "规定"
        t=110ms  chunk.content = "，"       → yield "，"
        ...      ...
        t=2000ms chunk.content = ""         → LLM 生成结束
                 full_answer 收集完毕
                 → _extract_citations() 提取引用
                 → yield "<!--CITATIONS:[{...}]-->"  ← 特殊标记

        Args:
            query: 用户问题
            retrieval_results: 检索结果
            chat_history: 历史对话
            top_k: 注入的检索结果数

        Yields:
            每次 yield 一个 str:
            - 大多数 yield 的是单个 token（一个或几个汉字）
            - 第二个 yield 是 <!--TRACE_ID:xxx--> 追踪标记
            - 最后一个 yield 是 <!--CITATIONS:[...]--> 引用数据
        """
        # ---- 创建回调实例 ----
        callback = RAGTraceCallback()
        start_time = time.time()

        # ---- 步骤①: 上下文窗口管理 ----
        callback.start_node("context_manage")
        trimmed_results = self._manage_context_window(retrieval_results[:top_k])
        callback.end_node({
            "input_count": len(retrieval_results[:top_k]),
            "output_count": len(trimmed_results),
        })

        # ---- 步骤②: 构建上下文（逻辑同 generate()） ----
        callback.start_node("build_context")
        context_parts = []
        for i, r in enumerate(trimmed_results, 1):
            chunk = r.chunk
            header = f"[{i}] 来源：{chunk.source_file}"
            if chunk.clause_id:
                header += f"/ {chunk.clause_id}"
            if chunk.section_title:
                header += f"/ {chunk.section_title}"
            context_parts.append(f"{header}\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)
        callback.end_node({"context_chars": len(context)})

        # ---- 步骤③: 组装消息（逻辑同 generate()） ----
        callback.start_node("assemble_messages")
        messages: List[Dict[str, str]] = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        if chat_history:
            # 只保留最近 3 轮对话（6 条消息）
            messages.extend(chat_history[-6:])

        user_message = (
            f"请根据以下参考资料回答员工的问题。\n\n"
            f"## 参考资料\n{context}\n\n"
            f"## 员工问题\n{query}"
        )
        messages.append({"role": "user", "content": user_message})
        callback.end_node()

        # ---- 步骤④: 流式调用 LLM ----
        # 用列表推导式把 dict 消息转为 LangChain 消息对象
        lc_messages = [
            SystemMessage(content=m["content"]) if m["role"] == "system"
            else HumanMessage(content=m["content"])
            for m in messages
        ]

        # 先 yield trace_id，让前端知道这次请求的追踪 ID
        yield f"<!--TRACE_ID:{callback.trace_id}-->\n"

        # full_answer: 累加所有 token，最后用于提取引用
        full_answer = ""

        # self.llm.astream(lc_messages, config={"callbacks": [callback]})
        # config 中的 callback 会触发 on_llm_start/on_llm_end/on_llm_error
        async for chunk in self.llm.astream(
            lc_messages,
            config={"callbacks": [callback]},
        ):
            token = chunk.content       # AIMessageChunk.content = 这一步的 token 字符串

            # 如果 token 为 None 或空字符串 → 跳过
            if not token:
                continue

            full_answer += token        # 累加到完整回答
            yield token                 # 实时推送给前端（SSE data）

        # ---- 步骤⑤: 流结束后提取引用 ----
        callback.start_node("citation_extract")
        citations = self._extract_citations(full_answer, trimmed_results)
        callback.end_node({"citation_count": len(citations)})

        # 通过特殊 HTML 注释标记把引用数据传给前端
        # 格式: <!--CITATIONS:[{...}, {...}]-->
        #
        # c.model_dump() → Pydantic v2 的内置方法:
        #   把 Citation 对象递归转为 Python 字典
        #
        # json.dumps(..., ensure_ascii=False):
        #   ensure_ascii=False → 中文不会被转成 \uXXXX 编码，直接输出可读中文
        #
        # 前端解析逻辑（伪代码）:
        #   if chunk.startsWith("<!--CITATIONS:"):
        #       citations = JSON.parse(chunk.match(/<!--CITATIONS:(.*?)-->/)[1])
        yield f"\n\n<!--CITATIONS:{json.dumps([c.model_dump() for c in citations], ensure_ascii=False)}-->"

    # ==================================================================
    # _extract_citations() —— 从 LLM 回答中提取引用
    # ==================================================================

    def _extract_citations(
        self,
        answer_text: str,                              # LLM 生成的完整回答
        retrieval_results: List[RetrievalResult],       # 检索结果（用于回查原文）
    ) -> List[Citation]:
        r"""从 LLM 回答中提取 [来源: ...] 引用标记，生成 Citation 列表

        这是「引用可追溯性」的工程实现核心。
        面试官会问: 你如何保证 LLM 生成的引用是真实的？
        → 答案: 我们用检索结果回查验证——每个引用都对应到实际检索到的原文片段

        完整算法流程:

        ┌────────────────────────────────────────────────────────────────┐
        │ 输入 answer_text:                                              │
        │ "赔偿标准为每公斤不超过100元[来源: 托运行李运输规定 第3.2条]。" │
        └──────────────┬─────────────────────────────────────────────────┘
                       │
                       ▼  ① 正则扫描
        ┌────────────────────────────────────────────────────────────────┐
        │ pattern = r'\[来源:\s*([^\]]+)\]'                              │
        │                                                                 │
        │ 匹配: group(1) = "托运行李运输规定 第3.2条"                     │
        └──────────────┬─────────────────────────────────────────────────┘
                       │
                       ▼  ② 解析来源文本
        ┌────────────────────────────────────────────────────────────────┐
        │ "托运行李运输规定 第3.2条"  → .split() → ["托运行李运输规定",   │
        │                                              "第3.2条"]        │
        │ → doc_name  = "托运行李运输规定" (parts[0])                     │
        │ → clause_id = "第3.2条" (匹配到 第?\d+ 模式)                    │
        └──────────────┬─────────────────────────────────────────────────┘
                       │
                       ▼  ③ 去重检查
        ┌────────────────────────────────────────────────────────────────┐
        │ key = "托运行李运输规定:第3.2条"                                │
        │ 如果 key 在 seen 中 → 跳过（同一引用被 LLM 重复写了多次）       │
        │ 否则 → 加入 seen，继续处理                                      │
        └──────────────┬─────────────────────────────────────────────────┘
                       │
                       ▼  ④ 回查检索结果
        ┌────────────────────────────────────────────────────────────────┐
        │ 在 retrieval_results 中找 source_file 包含 doc_name 的那条     │
        │ → original_text = chunk.content[:200] (前200字原文片段)        │
        │ → section_title = chunk.section_title                          │
        └──────────────┬─────────────────────────────────────────────────┘
                       │
                       ▼  ⑤ 构建 Citation
        ┌────────────────────────────────────────────────────────────────┐
        │ Citation(                                                      │
        │   doc_name="托运行李运输规定",                                  │
        │   clause_id="第3.2条",                                         │
        │   section_title="破损赔偿",                                     │
        │   original_text="第3.2条 托运行李损坏赔偿标准\n1. ..."          │
        │ )                                                              │
        └────────────────────────────────────────────────────────────────┘

        Args:
            answer_text: LLM 的完整回答，内嵌 [来源: XXX] 标记
            retrieval_results: 用于回查原文的检索结果列表

        Returns:
            Citation 列表，和 answer_text 中 [来源: ...] 标记一一对应
        """
        # 结果列表: 每个 Citation 对应一个 [来源: ...] 标记
        citations: List[Citation] = []

        # seen 集合: 用于去重
        # 为什么需要去重?
        #   LLM 可能在回答的不同位置多次引用同一条款
        #   例如: "第3.2条规定赔偿100元/公斤...此外根据第3.2条，还需填写申请表"
        #   我们只保留第一个引用，避免前端展示重复卡片
        seen: set = set()

        # ================================================================
        # 正则解释: r'\[来源:\s*([^\]]+)\]'
        #
        # \[       → 匹配左方括号 "["（需要转义，因为 [] 是正则元字符）
        # 来源:    → 匹配字面文本「来源:」
        # \s*      → 匹配 0 个或多个空白字符（空格、tab）
        # (        → 开始捕获组——这部分会被 group(1) 提取
        #   [^\]]+ → 匹配 1 个或多个非 "]" 字符（即右方括号前的全部内容）
        # )        → 结束捕获组
        # \]       → 匹配右方括号 "]"
        #
        # 测试用例:
        #   "[来源: 行李规定 第3.2条]"  → group(1) = "行李规定 第3.2条"
        #   "[来源:  行李规定]"          → group(1) = "行李规定" (多个空格被 \s* 吃掉)
        #   "[来源: 客规 / 第5.1条]"    → group(1) = "客规 / 第5.1条"
        # ================================================================
        pattern = r'\[来源:\s*([^\]]+)\]'

        # re.finditer(pattern, answer_text) → 迭代器
        # 逐个 yield Match 对象，每个 Match 对象代表一次匹配
        # 用 finditer 而非 findall，因为我们需要 match 对象来获取位置信息（调试用）
        matches = re.finditer(pattern, answer_text)

        for m in matches:
            # m.group(1) → 捕获组捕获到的内容
            # .strip()   → 去除首尾空白（防御性编程:
            #               防止 LLM 生成 "[来源:  行李规定  ]" 这种带空格的情况）
            source_text = m.group(1).strip()

            # 防御性跳过: 如果 LLM 生成 "[来源: ]" 空标记，直接跳过
            if not source_text:
                continue

            # ================================================================
            # 解析 source_text → doc_name + clause_id
            #
            # source_text 的可能格式:
            #   "托运行李运输规定 第3.2条"
            #   "旅客投诉处理规范 / 第5.1条 / 索赔时限"
            #   "凤凰知音会员章程 第8条 权益细则"
            #
            # 解析策略: replace(" / ", " ") → 统一用空格分隔 → .split()
            # ================================================================
            parts = source_text.replace(" / ", " ").split()
            # parts: ["托运行李运输规定", "第3.2条"]
            #        ["旅客投诉处理规范", "第5.1条", "索赔时限"]
            #        ["凤凰知音会员章程", "第8条", "权益细则"]

            # 取第一个词作为文档名（启发式假设——大部分情况下文档名在最前面）
            # 如果 parts 为空（理论上不会），退而用整个 source_text
            doc_name = parts[0] if parts else source_text

            # 在 parts 中寻找条款编号
            # 条款编号的特征: 以"第"开头 + 数字，或以数字开头
            # re.match(r'第?\d+', p):
            #   r'第?\d+' → "第" 出现 0 或 1 次 + 1 个或多个数字
            #   匹配: "第3.2条"(匹配"第3"), "5条"(匹配"5"), "第8条"(匹配"第8")
            clause_id = None
            for p in parts:
                if re.match(r'第?\d+', p):   # 找到第一个匹配条款编号模式的值
                    clause_id = p
                    break                    # 找到就停（通常只有一个条款编号）

            # ================================================================
            # 去重: 同一条款只保留一次
            #
            # key = "托运行李运输规定:第3.2条"
            # 如果 LLM 在回答中两次引用 [来源: 托运行李运输规定 第3.2条]
            # → 第二次匹配时 key 已经在 seen 中 → continue 跳过
            # ================================================================
            key = f"{doc_name}:{clause_id}"
            if key in seen:
                continue
            seen.add(key)

            # ================================================================
            # 回查检索结果: 找到 doc_name 对应的原文
            #
            # 这是防止 LLM 幻觉的关键步骤:
            #   如果 LLM 编造了一个不存在的文档名 → 循环结束后 original_text 仍为 ""
            #   前端看到 original_text 为空 → 知道这条引用可能不可靠
            # ================================================================
            original_text = ""
            section_title = None

            for r in retrieval_results:
                chunk = r.chunk
                # 子串匹配: doc_name 是否出现在 source_file 中
                # 例: doc_name="行李规定", chunk.source_file="托运行李运输规定.md"
                #     "行李规定" in "托运行李运输规定.md" → True
                if chunk.source_file and doc_name in chunk.source_file:
                    # 取原文前 200 个字符
                    # 为什么 200 字？弹窗显示时足够了，太多影响 API 响应大小
                    original_text = chunk.content[:200]
                    # 同时获取章节标题
                    section_title = chunk.section_title
                    break   # 找到第一个匹配的 chunk 就停止

            # 构建 Citation 对象并加入结果列表
            # ⚠️ 注意: 这行必须在 for r in retrieval_results 循环之外！
            #    如果在循环内部，每遍历一条不匹配的结果都会创建重复的 Citation
            citations.append(Citation(
                doc_name=doc_name,            # "托运行李运输规定"
                clause_id=clause_id,          # "第3.2条" 或 None
                section_title=section_title,  # "破损赔偿" 或 None
                original_text=original_text,  # 原文前200字 或 ""
            ))

        # 返回所有提取到的引用
        return citations

    # ==================================================================
    # _manage_context_window() —— 上下文窗口管理
    # ==================================================================

    def _manage_context_window(
        self,
        retrieval_results: List[RetrievalResult],
        max_tokens: int = 3000,
    ) -> List[RetrievalResult]:
        """上下文窗口管理 —— Token 超限时智能截断

        为什么需要这个函数？
        → LLM 的上下文窗口有限（GPT-4o 是 128K token，但要留空间给回答）
        → 检索可能返回大量长文档，全部塞入 Prompt 会导致:
          ① API 报错 (context length exceeded)
          ② Token 成本暴增
          ③ LLM 注意力分散，答案质量下降
        → 需要在注入 Prompt 之前做智能截断

        三层截断策略（按优先级执行）:

        ┌─────────────────────────────────────────────────────────────────┐
        │ 策略 1: 按分数降序排列                                           │
        │   retrieval_results → sorted(by score, descending)              │
        │   效果: 最高分的检索结果排在最前面，低分结果先被淘汰              │
        └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 策略 2: 同一条款去重（只保留最高分 chunk）                        │
        │   场景: 同一条款被切成 3 个 chunk，检索返回了这 3 个              │
        │         chunk A (score=0.85), chunk B (score=0.72),              │
        │         chunk C (score=0.61)                                    │
        │   → 只保留 chunk A（最高分），B 和 C 跳过                        │
        │   效果: 去除信息冗余，节省 token 预算                             │
        └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────────┐
        │ 策略 3: 字符数预算截断                                           │
        │   max_chars = max_tokens × 2 ≈ 6000 字符                        │
        │   逐个累加 chunk 的 content 长度                                  │
        │   超出预算时: 截断最后一个 chunk，丢弃后续                        │
        │   效果: 确保总字符数不超过 LLM 上下文窗口限制                     │
        └─────────────────────────────────────────────────────────────────┘

        面试话术:
        "当检索结果太多导致 token 超限时，我设计了三层截断策略:
        优先保留高分结果、同一条款去重、字符预算截断。
        保证 LLM 看到的是最有价值、不重复、不超限的上下文。"

        Args:
            retrieval_results: 可能很长的检索结果列表
            max_tokens: 最大允许的 token 数，默认 3000
                        为什么是 3000?
                        → 留足检索结果空间，同时给 LLM 回答留够余量
                        → 3000 token ≈ 6000 中文字符 = 一篇中等长度的报告

        Returns:
            截断后的检索结果列表（条数 ≤ 输入条数）
        """
        # 粗略换算: 1 中文字符 ≈ 0.5 token (保守估算)
        # 实际 1 个中文字符在 GPT tokenizer 中约 1~2 token
        # 用 ×2 是偏向于「少塞」的保守策略——宁可少给，不要超限
        max_chars = max_tokens * 2

        # ================================================================
        # 策略 1: 按 score 降序排列
        # sorted(..., key=lambda r: r.score, reverse=True)
        #   key=lambda r: r.score → 用检索结果的 score 字段作为排序依据
        #   reverse=True → 降序: 分数最高的排在最前面
        # ================================================================
        sorted_result = sorted(
            retrieval_results,
            key=lambda r: r.score,    # 取每个 RetrievalResult 的 score 字段
            reverse=True,              # True = 降序 = 高分优先
        )

        # ================================================================
        # 策略 2: 同一 clause_id 只保留最高分 chunk
        # 核心逻辑:
        #   - 有 clause_id 的 chunk → 检查是否已见过 → 见过则跳过
        #   - 无 clause_id 的 chunk → 说明不是条款文本，直接保留
        # ⚠️ 修复: 原代码缺少 `if cid` 判断，导致所有无条款编号的 chunk
        #    都被错误跳过。正确逻辑: cid 为 None 时不检查去重，直接保留。
        # ================================================================
        seen_clauses: set = set()     # 已见过的条款编号集合
        deduped: List[RetrievalResult] = []   # 去重后的结果

        for r in sorted_result:
            cid = r.chunk.clause_id

            # 情况 A: chunk 无条款编号（如纯叙述性文本）→ 不做去重，直接保留
            if not cid:
                deduped.append(r)
                continue

            # 情况 B: chunk 有条款编号，且之前已见过（有更高分的同条款 chunk）
            #         → 跳过（这是较低分的重复）
            if cid in seen_clauses:
                continue

            # 情况 C: 首次遇到这个条款编号 → 保留，并标记已见
            seen_clauses.add(cid)
            deduped.append(r)

        # ================================================================
        # 策略 3: 字符数预算截断
        # 逐个累加 chunk.content 的长度，超出 max_chars 则截断或丢弃
        # ================================================================
        selected: List[RetrievalResult] = []    # 最终选中的结果
        total_chars = 0                          # 已累加的字符数

        for r in deduped:
            content_len = len(r.chunk.content)   # 当前 chunk 的文本长度

            # 情况 A: 加上这个 chunk 还不超预算 → 全量保留
            if total_chars + content_len <= max_chars:
                selected.append(r)
                total_chars += content_len

            else:
                # 情况 B: 预算不够放完整 chunk → 截断
                remaining = max_chars - total_chars    # 还剩多少字符额度

                # 至少剩 100 字符才有截断的价值
                # 如果只剩 20 个字符，截断后基本是碎片，无意义
                if remaining > 100:
                    # 截取前 remaining 个字符，尾部加 "..." 表示被截断
                    r.chunk.content = r.chunk.content[:remaining] + "..."
                    selected.append(r)

                # 预算用完了，停止循环（后面的 chunk 全部丢弃）
                break

        return selected


# =============================================================================
# 4. 示例：RAGGenerator 完整流程演示
# =============================================================================
# 以下是一个概念性示例，展示从检索结果到最终回答的完整链路。
# 由于需要 LLM API Key 和 Qdrant 服务，实际运行需要配置 .env 文件。
#
# 建议: 在完成 FastAPI 封装后，通过 Swagger UI 调用 /rag/chat 接口
#       来实际体验完整流程，比看代码示例更直观。
#
#
# === RAG Generator 全流程示例 ===
#
# 假设场景:
#   一线员工问"旅客行李箱被摔坏了，赔偿标准是什么？"
#
# ----------------------------------------------------------------------
# 步骤 0: 前置条件 —— 检索结果
# ----------------------------------------------------------------------
#
# 来自 hybrid_search.search("旅客行李箱摔坏赔偿标准", top_k=5):
#
# retrieval_results = [
#     RetrievalResult(
#         chunk=TextChunk(
#             chunk_id="行李运输规定_第3条",
#             content="第3.2条 托运行李损坏赔偿标准\n"
#                     "1. 承运人对旅客托运行李的毁灭、遗失或者损坏承担责任。\n"
#                     "2. 赔偿金额按每公斤不超过人民币100元计算。\n"
#                     "3. 旅客需在航班到达后7日内以书面形式提出。",
#             source_file="托运行李运输规定.md",
#             clause_id="第3.2条",
#             section_title="破损赔偿"
#         ),
#         score=0.85,
#         source="hybrid"
#     ),
#     RetrievalResult(
#         chunk=TextChunk(
#             chunk_id="投诉处理规范_第5条",
#             content="第5.1条 索赔时限\n"
#                     "旅客应在航班到达后7日内以书面形式向承运人提出索赔，逾期不予受理。",
#             source_file="旅客投诉处理规范.md",
#             clause_id="第5.1条",
#             section_title="索赔时限"
#         ),
#         score=0.78,
#         source="hybrid"
#     ),
#     RetrievalResult(
#         chunk=TextChunk(
#             chunk_id="会员章程_第8条",
#             content="第8条 会员权益\n"
#                     "头等舱会员托运行李损坏赔偿上限为经济舱的2倍。",
#             source_file="凤凰知音会员章程.md",
#             clause_id="第8条",
#             section_title="权益细则"
#         ),
#         score=0.72,
#         source="vector"
#     ),
# ]
#
# ----------------------------------------------------------------------
# 步骤 1: 上下文窗口管理 (_manage_context_window)
# ----------------------------------------------------------------------
#
# trimmed = generator._manage_context_window(retrieval_results)
# → 3条结果，总字符约500，远未到3000token限制，不截断
# → 无重复条款，不去重
# → 结果: 保持3条
#
# ----------------------------------------------------------------------
# 步骤 2: 构建上下文 (context)
# ----------------------------------------------------------------------
#
# generator = RAGGenerator(temperature=0.1)
#
# 遍历 trimmed[:5] → 格式化:
#
#   context = "[1] 来源：托运行李运输规定.md/第3.2条/破损赔偿\n"
#             "第3.2条 托运行李损坏赔偿标准\n1. ...\n\n"
#             "---\n\n"
#             "[2] 来源：旅客投诉处理规范.md/第5.1条/索赔时限\n"
#             "第5.1条 索赔时限\n...\n\n"
#             "---\n\n"
#             "[3] 来源：凤凰知音会员章程.md/第8条/权益细则\n"
#             "第8条 会员权益\n..."
#
# ----------------------------------------------------------------------
# 步骤 3: 组装消息 (messages)
# ----------------------------------------------------------------------
#
# messages = [
#     {"role": "system", "content": "你是中国国际航空股份有限公司..."},
#     {"role": "user", "content":
#         "请根据以下参考资料回答员工的问题。\n\n"
#         "## 参考资料\n" + context + "\n\n"
#         "## 员工问题\n旅客行李箱被摔坏了，赔偿标准是什么？"
#     }
# ]
#
# ----------------------------------------------------------------------
# 步骤 4: LLM 生成回答 (self.llm.invoke() 返回的内容)
# ----------------------------------------------------------------------
#
# response_text =
#   "根据公司规定，旅客托运行李损坏的赔偿标准如下：\n"
#   "\n"
#   "托运行李损坏按每公斤不超过人民币100元进行赔偿"
#   "[来源: 托运行李运输规定 第3.2条]。\n"
#   "\n"
#   "旅客需在航班到达后7日内以书面形式向承运人提出索赔，"
#   "逾期将不予受理 [来源: 旅客投诉处理规范 第5.1条]。\n"
#   "\n"
#   "若为头等舱会员，赔偿上限为经济舱的2倍"
#   "[来源: 凤凰知音会员章程 第8条]。\n"
#   "\n"
#   "操作指引：\n"
#   "1. 请旅客填写《行李赔偿申请表》\n"
#   "2. 拍照留存行李箱损坏情况\n"
#   "3. 提交至行李查询柜台处理"
#
# ----------------------------------------------------------------------
# 步骤 5: 提取引用 (_extract_citations)
# ----------------------------------------------------------------------
#
# 正则扫描 response_text → 找到 3 个匹配:
#
#   匹配 1: "托运行李运输规定 第3.2条"
#     → doc_name="托运行李运输规定", clause_id="第3.2条"
#     → 回查 → original_text=content[:200], section_title="破损赔偿"
#
#   匹配 2: "旅客投诉处理规范 第5.1条"
#     → doc_name="旅客投诉处理规范", clause_id="第5.1条"
#
#   匹配 3: "凤凰知音会员章程 第8条"
#     → doc_name="凤凰知音会员章程", clause_id="第8条"
#
# ----------------------------------------------------------------------
# 步骤 6: 最终输出 (CitedAnswer)
# ----------------------------------------------------------------------
#
# CitedAnswer(
#     answer_text = "根据公司规定，旅客托运行李损坏的赔偿标准如下...",
#     citations = [
#         Citation(doc_name="托运行李运输规定", clause_id="第3.2条",
#                  section_title="破损赔偿", original_text="第3.2条 托运行李..."),
#         Citation(doc_name="旅客投诉处理规范", clause_id="第5.1条",
#                  section_title="索赔时限", original_text="第5.1条 索赔时限..."),
#         Citation(doc_name="凤凰知音会员章程", clause_id="第8条",
#                  section_title="权益细则", original_text="第8条 会员权益..."),
#     ],
#     cost_ms = 1523,      # 全链路 1.5 秒
#     trace_id = "a1b2c3d4" # LangSmith 追踪 ID
# )
#
# ----------------------------------------------------------------------
# 前端渲染效果（概念图）
# ----------------------------------------------------------------------
#
#   🤖 国航业务支持助手                    Trace: a1b2c3d4
#   ┌──────────────────────────────────────────────────────────────┐
#   │ 根据公司规定，旅客托运行李损坏的赔偿标准如下：                │
#   │                                                              │
#   │ 托运行李损坏按每公斤不超过人民币100元进行赔偿                │
#   │ [📎 托运行李运输规定 第3.2条]  ← 点击弹出原文弹窗            │
#   │                                                              │
#   │ 旅客需在航班到达后7日内以书面形式提出索赔，                   │
#   │ 逾期不予受理                                                 │
#   │ [📎 旅客投诉处理规范 第5.1条]                                │
#   │                                                              │
#   │ 若为头等舱会员，赔偿上限为经济舱的2倍                         │
#   │ [📎 凤凰知音会员章程 第8条]                                   │
#   │                                                              │
#   │ 📋 操作指引:                                                 │
#   │ 1. 请旅客填写《行李赔偿申请表》                                │
#   │ 2. 拍照留存行李箱损坏情况                                      │
#   │ 3. 提交至行李查询柜台处理                                      │
#   └──────────────────────────────────────────────────────────────┘
#
#   ⏱️ 响应耗时: 1523ms | 📊 引用来源: 3 条
#
# === 示例结束 ===
