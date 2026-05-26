"""
RAG 生成引擎 —— 检索结果 → LLM 回答 + 引用溯源

将检索结果注入 Prompt，调用 LLM 生成带条款级引用的专业回答。
支持流式输出和多轮对话。

----------------------------------------------------------------------
## 你需要自己写的部分

RAG 生成是「检索」和「用户」之间的桥梁。这是整个系统对外呈现的
最终环节——前面的文档解析、切片、检索都在为这一步服务。

学习重点：
1. Prompt 工程：如何让 LLM 输出格式化的带引用回答
2. 上下文窗口管理：token 超限时的截断策略
3. 引用提取：用正则从 LLM 回答中提取 [来源: ...] 标记
4. 流式输出：用 async generator 实现 SSE

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# re: 正则表达式，用于从 LLM 回答中提取 [来源: ...] 引用标记
import re

# typing: 类型提示
from typing import List, Optional, AsyncIterator, Dict, Any

# 数据模型
from rag.models import RetrievalResult, CitedAnswer, Citation

# 配置和 LLM 工厂
from core.config import get_settings
from core.llm import get_llm
from core.constants import DEFAULT_RAG_TEMPERATURE, DEFAULT_RETRIEVAL_TOP_K


# =============================================================================
# System Prompt —— 国航内部业务支持助手
# =============================================================================

# TODO(用户): 面试时如果你能讲清楚这个 Prompt 的设计思路，
# 证明你理解「RAG 不只是检索+生成，Prompt 决定最终质量」

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


class RAGGenerator:
    """RAG 生成器 —— 检索结果 → LLM 回答 + 引用

    核心流水线:
    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ 检索结果  │ → │ Prompt   │ → │ LLM      │ → │ 引用提取  │ → CitedAnswer
    │ Top-K    │    │ 组装     │    │ 生成     │    │ + 格式化  │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘

    面试话术:
    "我把检索结果注入到精心设计的 System Prompt 中，让 LLM 以
    国航业务支持助手的角色回答。关键是引用的可追溯性——
    每个结论必须标注 [来源: 文档名 条款编号]，
    内部员工可以点击跳转到原文，而不是盲信 LLM 的回答。"
    """

    def __init__(self, temperature: float = DEFAULT_RAG_TEMPERATURE):
        """初始化 RAG 生成器

        Args:
            temperature: LLM 温度参数（0.1 = 接近确定性输出，适合政策问答）
        """
        self.temperature = temperature
        self.llm = get_llm(get_settings())
        # 设置较低温度确保回答稳定一致
        self.llm.temperature = temperature

    # ------------------------------------------------------------------
    # 生成回答（非流式）
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> CitedAnswer:
        """基于检索结果生成带引用的回答

        TODO(用户): 手写 generate 逻辑

        流程：
        1. 构建上下文（检索结果 → 格式化的文本块）
        2. 组装消息列表（System Prompt + 历史对话 + 上下文 + 用户问题）
        3. 调用 LLM 生成回答
        4. 从回答中提取引用标记 [来源: ...] → Citation 列表
        5. 包装为 CitedAnswer 返回

        Args:
            query: 用户问题
            retrieval_results: 检索结果列表（混合检索或纯向量/BM25 的输出）
            chat_history: 历史对话 [{"role": "user/assistant", "content": "..."}, ...]
            top_k: 注入 Prompt 的检索结果数

        Returns:
            CitedAnswer（含 answer_text + citations 列表）
        """
        import time
        start_time = time.time()

        # ================================================================
        # TODO(用户): 从这里开始手写 —— RAG 生成逻辑
        # ================================================================
        #
        # 实现参考:
        #
        # ---- 步骤 1: 构建上下文 ----
        # 将检索结果格式化为 LLM 可读的上下文块
        # 每个结果包含: 文档名、条款编号、章节标题、文本内容
        #
        # context_parts = []
        # for i, r in enumerate(retrieval_results[:top_k], 1):
        #     chunk = r.chunk
        #     # 构建带编号的上下文块
        #     # [1] 来源: 文档名 / 条款编号 / 章节标题
        #     # 原文内容...
        #     header = f"[{i}] 来源: {chunk.source_file}"
        #     if chunk.clause_id:
        #         header += f" / {chunk.clause_id}"
        #     if chunk.section_title:
        #         header += f" / {chunk.section_title}"
        #     context_parts.append(f"{header}\n{chunk.content}")
        #
        # context = "\n\n---\n\n".join(context_parts)
        #
        # ---- 步骤 2: 组装消息 ----
        # messages = [
        #     {"role": "system", "content": SYSTEM_PROMPT},
        # ]
        #
        # # 加入历史对话（最近 N 轮）
        # if chat_history:
        #     messages.extend(chat_history[-6:])  # 保留最近 3 轮对话
        #
        # # 加入当前问题 + 上下文
        # user_message = (
        #     f"请根据以下参考资料回答员工的问题。\n\n"
        #     f"## 参考资料\n{context}\n\n"
        #     f"## 员工问题\n{query}"
        # )
        # messages.append({"role": "user", "content": user_message})
        #
        # ---- 步骤 3: 调用 LLM ----
        # from langchain_core.messages import HumanMessage, SystemMessage
        # lc_messages = []
        # for m in messages:
        #     if m["role"] == "system":
        #         lc_messages.append(SystemMessage(content=m["content"]))
        #     else:
        #         lc_messages.append(HumanMessage(content=m["content"]))
        # response = self.llm.invoke(lc_messages)
        # answer_text = response.content
        #
        # ---- 步骤 4: 提取引用 ----
        # citations = self._extract_citations(answer_text, retrieval_results[:top_k])
        #
        # ---- 步骤 5: 包装结果 ----
        # cost_ms = int((time.time() - start_time) * 1000)
        # return CitedAnswer(
        #     answer_text=answer_text,
        #     citations=citations,
        #     cost_ms=cost_ms,
        # )
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 参考上面的注释实现 RAG 生成逻辑"
        )

    # ------------------------------------------------------------------
    # 流式生成
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> AsyncIterator[str]:
        """流式生成 —— 逐 token 输出，适合 SSE 推送

        TODO(用户): 手写流式生成逻辑

        和非流式的区别：
        - 使用 self.llm.astream() 代替 self.llm.invoke()
        - 用 async for 逐 token yield
        - yield 的内容可以直接作为 SSE data 发送

        流程：
        1. 构建上下文和消息（同 generate）
        2. 调用 self.llm.astream(lc_messages)
        3. async for chunk in stream: yield chunk.content

        Args:
            query: 用户问题
            retrieval_results: 检索结果
            chat_history: 历史对话
            top_k: 注入的结果数

        Yields:
            每次 yield 一个 token 字符串
        """
        import time
        start_time = time.time()

        # ================================================================
        # TODO(用户): 手写流式生成逻辑
        # ================================================================
        #
        # 实现参考:
        #
        # ---- 步骤 1-2: 同 generate(), 构建上下文和消息 ----
        # context_parts = []
        # for i, r in enumerate(retrieval_results[:top_k], 1):
        #     ...
        # context = "\n\n---\n\n".join(context_parts)
        #
        # messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # if chat_history:
        #     messages.extend(chat_history[-6:])
        # user_message = f"请根据以下参考资料回答...\n\n## 参考资料\n{context}\n\n## 员工问题\n{query}"
        # messages.append({"role": "user", "content": user_message})
        #
        # ---- 步骤 3: 流式调用 LLM ----
        # from langchain_core.messages import HumanMessage, SystemMessage
        # lc_messages = [
        #     SystemMessage(content=m["content"]) if m["role"] == "system"
        #     else HumanMessage(content=m["content"])
        #     for m in messages
        # ]
        #
        # full_answer = ""
        # async for chunk in self.llm.astream(lc_messages):
        #     token = chunk.content
        #     full_answer += token
        #     yield token
        #
        # ---- 步骤 4: 流结束后提取引用 ----
        # citations = self._extract_citations(full_answer, retrieval_results[:top_k])
        # # 引用信息通过特殊标记 yield（前端解析用）
        # yield f"\n\n<!--CITATIONS:{json.dumps([c.model_dump() for c in citations])}-->"
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 参考上面的注释实现流式生成逻辑"
        )

    # ------------------------------------------------------------------
    # 引用提取
    # ------------------------------------------------------------------

    def _extract_citations(
        self,
        answer_text: str,
        retrieval_results: List[RetrievalResult],
    ) -> List[Citation]:
        """从 LLM 回答中提取 [来源: ...] 引用标记，生成 Citation 列表

        算法:
        1. 用正则找出所有 [来源: 文档名 条款编号] 标记
        2. 对每个标记，在检索结果中找到匹配的 chunk
        3. 构建 Citation 对象（含 doc_name, clause_id, section_title, original_text）

        TODO(用户): 手写引用提取逻辑

        正则解析思路:
        - 模式: \[来源:\s*([^\]]+)\]
        - 提取括号内容 → 按空格/斜杠拆分为文档名、条款编号
        - 在 retrieval_results 中模糊匹配

        Args:
            answer_text: LLM 生成的回答全文
            retrieval_results: 用于查找原文的检索结果

        Returns:
            Citation 列表（和 answer_text 中的 [来源: ...] 一一对应）
        """
        # ================================================================
        # TODO(用户): 手写引用提取逻辑
        # ================================================================
        #
        # 实现参考:
        #
        # citations = []
        # seen = set()  # 去重
        #
        # # 正则: 匹配 [来源: ...] 格式
        # pattern = r'\[来源:\s*([^\]]+)\]'
        # matches = re.finditer(pattern, answer_text)
        #
        # for m in matches:
        #     source_text = m.group(1).strip()  # "托运行李运输规定 第1条"
        #
        #     # 解析来源文本 → doc_name + clause_id + section_title
        #     parts = source_text.replace(" / ", " ").split()
        #     doc_name = parts[0] if parts else source_text
        #     clause_id = None
        #     for p in parts:
        #         if re.match(r'第?\d+', p):  # 匹配条款编号
        #             clause_id = p
        #             break
        #
        #     # 去重
        #     key = f"{doc_name}:{clause_id}"
        #     if key in seen:
        #         continue
        #     seen.add(key)
        #
        #     # 在检索结果中找匹配的原文片段
        #     original_text = ""
        #     section_title = None
        #     for r in retrieval_results:
        #         chunk = r.chunk
        #         if chunk.source_file and doc_name in chunk.source_file:
        #             original_text = chunk.content[:200]
        #             section_title = chunk.section_title
        #             break
        #
        #     citations.append(Citation(
        #         doc_name=doc_name,
        #         clause_id=clause_id,
        #         section_title=section_title,
        #         original_text=original_text,
        #     ))
        #
        # return citations
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 参考上面的注释实现引用提取逻辑"
        )

    # ------------------------------------------------------------------
    # 上下文窗口管理
    # ------------------------------------------------------------------

    def _manage_context_window(
        self,
        retrieval_results: List[RetrievalResult],
        max_tokens: int = 3000,
    ) -> List[RetrievalResult]:
        """上下文窗口管理 —— Token 超限时智能截断

        策略（按优先级）：
        1. 保留高分段（分数阈值过滤）
        2. 截断低分文档的内容（只取前 N 字）
        3. 按 chunk.clause_id 去重（同一条款只保留最优 chunk）

        TODO(用户): 手写上下文窗口管理逻辑

        面试话术:
        "当检索结果太多导致 token 超限时，我设计了三层截断策略：
        优先保留高分结果、截断低分结果、同一条款去重。
        保证 LLM 看到的是最有价值的上下文。"

        Args:
            retrieval_results: 检索结果列表
            max_tokens: 最大允许的 token 数（中文粗略按 1 字符 ≈ 0.5 token）

        Returns:
            截断后的检索结果列表
        """
        # ================================================================
        # TODO(用户): 手写上下文窗口管理逻辑
        # ================================================================
        #
        # 实现参考（粗估: 1 中文字符 ≈ 0.5 token）:
        #
        # max_chars = max_tokens * 2  # 粗略换算
        #
        # # 策略1: 按分数排序，取高分
        # sorted_results = sorted(retrieval_results, key=lambda r: r.score, reverse=True)
        #
        # # 策略2: 同一 clause_id 只保留最高分 chunk
        # seen_clauses = set()
        # deduped = []
        # for r in sorted_results:
        #     cid = r.chunk.clause_id
        #     if cid and cid in seen_clauses:
        #         continue
        #     if cid:
        #         seen_clauses.add(cid)
        #     deduped.append(r)
        #
        # # 策略3: 累加直到超过 max_chars，超出的截断
        # selected = []
        # total_chars = 0
        # for r in deduped:
        #     content_len = len(r.chunk.content)
        #     if total_chars + content_len <= max_chars:
        #         selected.append(r)
        #         total_chars += content_len
        #     else:
        #         # 截断最后一个 chunk 的内容
        #         remaining = max_chars - total_chars
        #         if remaining > 100:  # 至少 100 字才有意义
        #             r.chunk.content = r.chunk.content[:remaining] + "..."
        #             selected.append(r)
        #         break
        #
        # return selected
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 参考上面的注释实现上下文窗口管理逻辑"
        )
