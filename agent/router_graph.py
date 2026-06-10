"""
LangGraph 智能检索路由 Agent

在 RAG 基础上叠加 Agent 决策能力：
  - 意图分类：政策查询 / 操作指引 / 应急 / 不确定
  - 置信度评估：检索结果是否够好？
  - Self-Reflection：不够好 → 改写 Query 重试（最多 2 次）
  - 降级策略：2 次还不够 → 诚实告知 + 建议联系主管

----------------------------------------------------------------------
LangGraph 核心概念在本文件中的体现:

1. StateGraph: Agent 的骨架，定义「有哪些节点」「节点间如何跳转」
2. State (TypedDict): 在节点间流转的数据，每个节点读取/修改 State
3. Node: 处理函数 async def xxx(state: AgentState) -> dict，返回 State 更新
4. ConditionalEdge: 根据 State 中的值决定下一步跳转到哪个节点
5. Self-Reflection Loop: evaluate → rewrite → retrieve → evaluate 形成闭环

面试话术:
"我在 RAG 基础上用 LangGraph 加了 Agent 路由层。核心是三个决策点——
意图分类（不同问题走不同检索策略）、置信度评估（检索不够好就改写重试）、
降级兜底（两次重试还不够就诚实告知）。这比纯 RAG 多了「自我判断」和
「自我纠错」的能力。"
----------------------------------------------------------------------

你需要自己写的部分:
1. _classify_intent()    —— LLM Prompt 设计，判断用户意图
2. _evaluate_confidence()—— 评估检索结果质量
3. _rewrite_query()      —— LLM Prompt 设计，改写 Query

TODO(用户) 标记的部分是你需要手写的核心逻辑。
"""

# =============================================================================
# 导入依赖
# =============================================================================

# LangGraph 核心
# StateGraph: 状态图容器，add_node / add_edge / add_conditional_edges
# END: 特殊节点，表示图的终止
from langgraph.graph import StateGraph, END

# TypedDict: 定义 State 的类型结构（LangGraph 用 dict 传递状态）
from typing import TypedDict, List, Dict, Any

# RAG 管线组件（Agent 会调用这些组件）
from rag.hybrid_search import HybridSearcher
from rag.vector_store import VectorStore
from rag.bm25 import BM25Retriever
from rag.generator import RAGGenerator
from rag.models import RetrievalResult, CitedAnswer, TextChunk

# LLM（用于意图分类和 Query 改写）
from core.llm import get_llm
from core.config import get_settings


# =============================================================================
# AgentState —— Agent 的状态定义
# =============================================================================
# LangGraph 的状态就是一份在节点间流转的 dict。
# 用 TypedDict 定义结构后，LangSmith 可以自动追踪每个字段的变化。
#
# Annotated[List, operator.add] 的含义:
#   - 普通字段更新 = 覆盖（后写的值替代先写的值）
#   - Annotated[List, operator.add] = 追加（新值拼接到列表尾部）
#   这在 Self-Reflection 循环中很重要：
#   rewrite_count 每次 +1（覆盖），retrieval_history 每次追加（不丢失历史）

class AgentState(TypedDict):
    """Agent 状态 —— 在 5 个 Node 之间流转的全部数据"""

    # ---- 输入 ----
    query: str                          # 用户原始问题
    top_k: int                          # 检索数量（默认 5）

    # ---- 意图分类 ----
    intent: str                         # 意图分类结果
                                        # "policy_query" / "operation_guide" / "emergency" / "uncertain"
    intent_reason: str                  # 分类理由（LLM 输出，用于调试）

    # ---- 检索 ----
    retrieval_results: List[Dict]       # 检索结果列表（RetrievalResult 序列化后的 dict）
    rewrite_count: int                  # Query 改写次数（0 → 1 → 2 → 放弃）
    rewritten_query: str                # 改写后的 Query（空字符串 = 未改写）

    # ---- 置信度评估 ----
    confidence: float                   # 检索置信度（0.0 ~ 1.0）
    confidence_reason: str              # 置信度判断理由

    # ---- 输出 ----
    answer: str                         # 最终回答
    citations: List[Dict]               # 引用列表
    trace_id: str                       # 追踪 ID

    # ---- 错误 ----
    error: str                          # 错误信息（空 = 无错误）


# =============================================================================
# 常量
# =============================================================================

# 最大 Query 改写次数
MAX_REWRITES = 2

# 置信度阈值：低于此值触发 Query 改写
CONFIDENCE_THRESHOLD = 0.5

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         共享工具函数                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝


def _serialize_results(results: List[RetrievalResult]) -> List[Dict]:
    """RetrievalResult 对象 → dict 列表（存入 State，LangGraph 不支持 Pydantic）

    把两层的 Pydantic 对象（RetrievalResult + 内嵌的 TextChunk）完全展开为纯 dict。
    """
    return [
        {
            # RetrievalResult 自身字段
            "score": r.score,
            "source": r.source,
            # TextChunk 字段（r.chunk）
            "chunk_id": r.chunk.chunk_id,
            "content": r.chunk.content,
            "source_file": r.chunk.source_file,
            "chunk_index": r.chunk.chunk_index,
            "clause_id": r.chunk.clause_id,
            "section_title": r.chunk.section_title,
            "doc_name": r.chunk.source_file,  # 前端习惯用 doc_name，取 source_file
        }
        for r in results
    ]


def _deserialize_results(dicts: List[Dict], top_k: int = 5) -> List[RetrievalResult]:
    """dict 列表 → RetrievalResult 对象列表（从 State 还原，用于 generator）"""
    return [
        RetrievalResult(
            score=r.get("score", 0),
            source=r.get("source", "hybrid"),
            chunk=TextChunk(
                chunk_id=r.get("chunk_id", ""),
                content=r.get("content", ""),
                source_file=r.get("source_file", ""),
                clause_id=r.get("clause_id"),
                section_title=r.get("section_title"),
                chunk_index=r.get("chunk_index", 0),
            ),
        )
        for r in dicts[:top_k]
    ]


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      TODO(用户): 核心 LLM 交互逻辑                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# 以下三个函数是 Agent 的「大脑」——它们调用 LLM 做决策。
# 你需要设计 Prompt、解析 LLM 输出、处理边界情况。
#
# 写完这些函数后，Agent 就能完整运行了。
# 参考实现写在注释中，取消注释并理解后重写。


def _classify_intent(query: str) -> Dict[str, str]:
    """用 LLM 判断用户 Query 的意图类别

    TODO(用户): 设计意图分类 Prompt + 解析 LLM 输出

    输入: "旅客行李箱摔坏了怎么赔？"
    输出: {"intent": "policy_query", "reason": "...短理由..."}

    四种意图:
    ┌──────────────────┬──────────────────────────────────────┐
    │ 意图             │ 典型 Query                            │
    ├──────────────────┼──────────────────────────────────────┤
    │ policy_query     │ "金卡能带人进休息室吗？"              │
    │                  │ "Y 舱退票手续费多少？"                │
    │ operation_guide  │ "值机系统怎么给旅客升舱？"            │
    │                  │ "登机口广播模板是什么？"              │
    │ emergency        │ "旅客突发疾病怎么办？"                │
    │                  │ "航班上发现可疑物品怎么处理？"        │
    │ uncertain        │ "今天天气怎么样？"                    │
    │                  │ 与工作无关或无法分类的问题            │
    └──────────────────┴──────────────────────────────────────┘

    实现要点:
    1. 构建 System Prompt，告诉 LLM 它是国航员工助手，需要分类意图
    2. 把 query 放入 User Message
    3. 调用 LLM.invoke() 获取分类结果
    4. 解析 LLM 输出（要求 LLM 按固定格式输出，如 "intent: policy_query"）
    5. 如果 LLM 输出不符合预期 → 返回 "policy_query" 作为默认值
    """
    # ═══════════════════════════════════════════════════════════════
    # 参考实现（取消注释并逐行理解后重写）
    # ═══════════════════════════════════════════════════════════════
    #
    settings = get_settings()
    llm = get_llm(settings)
    system_prompt = """你是国航内部员工智能助手的前置分类器。
    你的任务：根据员工的提问，判断问题属于哪一类。

    类别定义：
    - policy_query: 查询公司政策、规定、条款（行李/退改签/会员/赔偿/证件等）
    - operation_guide: 询问操作步骤、系统使用方法、工作流程
    - emergency: 涉及紧急情况、安全威胁、旅客突发状况
    - uncertain: 以上三类都不是，或信息不足无法判断

    输出格式（严格按此格式，不要加额外内容）：
    intent: <类别>
    reason: <一句话理由>"""

    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"员工提问：{query}")
    ])
    # # 解析 LLM 输出
    text = response.content.strip()
    intent = "policy_query"
    reason = ""
    for line in text.split("\n"):
        if line.startswith("intent:"):
            raw = line.replace("intent:", "").strip()
            if raw in ("policy_query", "operation_guide", "emergency", "uncertain"):
                intent = raw
        elif line.startswith("reason:"):
            reason = line.replace("reason:", "").strip()

    return {"intent":intent, "reason": reason}


def _evaluate_confidence(
    query: str,
    retrieval_results: List[Dict],
    intent: str,
) -> Dict[str, Any]:
    """评估检索结果的置信度

    TODO(用户): 设计置信度评估逻辑

    输入:
    - query: 用户问题
    - retrieval_results: 检索结果列表
    - intent: 意图类别

    输出: {"confidence": 0.72, "reason": "Top-1 结果直接提到了行李赔偿条款"}

    评估维度（选一种或组合使用）:
    ┌────────────────────┬─────────────────────────────────────┐
    │ 方法               │ 说明                                │
    ├────────────────────┼─────────────────────────────────────┤
    │ 分数阈值法         │ 如果 Top-1 score > 某阈值 → 高置信  │
    │                    │ 优点: 快，缺点: 分数不一定可比较     │
    │ 关键词匹配法       │ 如果 query 的关键词出现在结果中     │
    │                    │ → 高置信                             │
    │ LLM 评估法         │ 让 LLM 判断「这些检索结果能回答吗？」 │
    │                    │ 优点: 准确，缺点: 多一次 LLM 调用    │
    │ 混合法（推荐）     │ 先用分数/关键词快速判断，边界情况    │
    │                    │ 才用 LLM                             │
    └────────────────────┴─────────────────────────────────────┘

    实现建议（从简单开始）:
    1. 如果没有检索结果 → confidence=0.0
    2. 如果 Top-1 score > 0.3 → confidence=0.8
    3. 如果结果数量 >= 3 → confidence 适当提高
    4. 否则 → confidence=0.3（触发改写）
    """
    if not retrieval_results:
        return {"confidence": 0.0, "reason": "检索无结果"}

    # 取 Top-1 分数
    top_score = retrieval_results[0].get("score",0)
    result_count = len(retrieval_results)

    # 简单启发式评估
    if top_score > 0.5:
        confidence = 0.9
        reason = f"Top-1 分数{top_score:.2f} 很高"
    elif result_count > 3:
        confidence = 0.5
        reason = f"分数偏低但命中{result_count} 条结果"
    else:
        confidence = 0.2
        reason = f"分数{top_score:.2f} 偏低且仅{result_count}条结果"

    return {"confidence": confidence, "reason": reason}

def _rewrite_query(query: str, intent: str) -> str:
    """用 LLM 改写 Query，尝试换一种表述方式重新检索

    TODO(用户): 设计 Query 改写 Prompt

    输入:
    - query: 原始问题
    - intent: 意图类别

    输出: 改写后的问题字符串

    改写策略:
    1. 保留原意，换一种说法
    2. 提取核心关键词（去掉口语化表达）
    3. 补充可能的同义词/缩写（如 "金卡" → "金卡 贵宾卡 会员卡"）

    举例:
    原始: "箱子摔烂了怎么办？"
    改写: "托运行李损坏 赔偿标准 索赔流程"

    实现要点:
    1. Prompt 告诉 LLM：你是 Query 改写器，把口语化查询转为更精准的检索 Query
    2. 保留原意，添加关键词变体
    3. 输出直接是改写后的 Query，不要多余解释
    """
    settings = get_settings()
    llm = get_llm(settings)

    system_prompt = """你是查询改写助手。把员工口语化的提问改写为更精准的检索查询。

    规则：
    1. 保留原意，不要添加原问题没有的信息
    2. 提取核心关键词，去掉"怎么办""怎么查""请问"等口语化表达
    3. 补充同义词扩展（用空格分隔关键词）
    4. 输出只有改写后的查询文本，不要加任何前缀、解释、引号"""

    from langchain_core.messages import HumanMessage, SystemMessage
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"原始查询：{query}")
    ])

    return response.content.strip()




# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      LangGraph Node 函数                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# 每个 Node 函数签名为 (state: AgentState) -> dict
# 返回的 dict 会被「部分更新」到 State 中（覆盖同名 key）
#
# LangGraph 执行流程:
#   1. 从入口节点开始
#   2. 执行节点函数，拿到返回的 dict
#   3. 把 dict 合并到当前 State
#   4. 根据边（Edge/ConditionalEdge）决定下一个节点
#   5. 如果下一个节点是 END → 停止，返回最终 State
#
# 面试话术:
# "我把 Agent 的决策拆成 5 个独立的节点，每个节点职责单一。
# 意图分类只管判断意图，检索只管查，评估只管打分——
# 节点之间通过 State 传递数据，用条件边实现路由。
# 这样做的好处是每个节点可以独立测试、独立优化。"


# =============================================================================
# RAG 组件懒加载（和 chat.py 同样的模式）
# =============================================================================

_rag_components = None


def _get_rag_components():
    """懒加载 RAG 管线组件"""
    global _rag_components
    if _rag_components is not None:
        return _rag_components
    try:
        vector_store = VectorStore()
        bm25 = BM25Retriever()
        hybrid_searcher = HybridSearcher(vector_store=vector_store, bm25_retriever=bm25)
        generator = RAGGenerator()
        _rag_components = {"hybrid_searcher": hybrid_searcher, "generator": generator}
        return _rag_components
    except Exception as e:
        print(f"[Agent] RAG 组件初始化失败: {e}")
        return None


# =============================================================================
# Node 1: intent_classify —— 意图分类
# =============================================================================

async def node_intent_classify(state: AgentState) -> dict:
    """意图分类节点 —— Agent 的「第一道关卡」

    判断员工的提问属于哪种类型，后续节点根据类型选择不同的处理策略。

    Input:  state["query"]
    Output: state["intent"], state["intent_reason"]
    """
    query = state.get("query", "")

    # 调用 LLM 做意图分类
    # 注意：_classify_intent() 是同步函数，但 LangGraph 节点可以是 async
    result = _classify_intent(query)

    # result 可能是 dict，也可能因为 TODO 未实现而返回 None
    if not result:
        # 兜底：默认当作政策查询处理
        return {"intent": "policy_query", "intent_reason": "默认分类（分类器未就绪）"}

    return {
        "intent": result.get("intent", "policy_query"),
        "intent_reason": result.get("reason", ""),
    }


# =============================================================================
# Conditional Edge: route_by_intent —— 根据意图决定下一节点
# =============================================================================

def route_by_intent(state: AgentState) -> str:
    """条件路由：根据意图分类结果决定下一步

    这个函数不是 Node，而是 ConditionalEdge 的路由函数。
    签名必须是 (state: AgentState) -> str，返回值是「下一个节点的名字」。

    路由规则:
    - policy_query     → "retrieve"（走标准 RAG 检索）
    - operation_guide  → "retrieve"（走标准 RAG 检索）
    - emergency        → "retrieve"（走标准 RAG 检索，但检索参数可调整）
    - uncertain        → END（不检索，直接生成兜底回答）
    """
    intent = state.get("intent", "policy_query")

    if intent == "uncertain":
        # 不确定的问题 → 跳过检索，直接给兜底回答
        return "generate_fallback"
    else:
        # 政策查询 / 操作指引 / 应急 → 统一走检索流程
        return "retrieve"


# =============================================================================
# Node 2: retrieve —— 检索执行
# =============================================================================

async def node_retrieve(state: AgentState) -> dict:
    """检索执行节点 —— 调用混合检索器搜索相关文档

    Input:  state["query"] (或 state["rewritten_query"]), state["top_k"]
    Output: state["retrieval_results"]
    """
    # 如果有改写后的 query，优先用它
    query = state.get("rewritten_query", "") or state.get("query", "")
    top_k = state.get("top_k", 5)
    intent = state.get("intent", "policy_query")

    components = _get_rag_components()
    if not components:
        return {"error": "RAG 组件未就绪", "retrieval_results": []}

    searcher = components["hybrid_searcher"]

    # 应急场景 → 多取一些结果（提高覆盖面）
    actual_top_k = top_k * 2 if intent == "emergency" else top_k

    try:
        results = searcher.search(query=query, top_k=actual_top_k)
    except Exception as e:
        return {"error": f"检索失败: {e}", "retrieval_results": []}

    return {"retrieval_results": _serialize_results(results)}


# =============================================================================
# Node 3: evaluate_confidence —— 置信度评估
# =============================================================================

async def node_evaluate_confidence(state: AgentState) -> dict:
    """置信度评估节点 —— 判断检索结果是否足够好

    Input:  state["rewritten_query"] 或 state["query"], state["retrieval_results"], state["intent"]
    Output: state["confidence"], state["confidence_reason"]

    这是 Self-Reflection 的关键节点：
    - confidence >= CONFIDENCE_THRESHOLD → 去生成回答
    - confidence < CONFIDENCE_THRESHOLD  → 去改写 Query（如果还没超过最大次数）
    - confidence < CONFIDENCE_THRESHOLD 且改写次数已满 → 去生成兜底回答

    为什么用 rewritten_query 优先？
    第二次检索是基于改写后的 query 搜的，评估时自然也该用改写后的 query。
    否则评估依据（原始 query）和实际检索依据（改写 query）不一致，
    会导致「改写后搜到了好东西，但用原始 query 评估觉得不相关」的误判。

    举例:
    原始 query: "箱子摔烂了怎么办？"
    改写 query: "托运行李损坏 赔偿标准 索赔流程"
    检索结果: 命中了「托运行李运输规定 第3.2条 破损赔偿」
    → 用原始 query 评估: "箱子摔烂了" 和 "托运行李运输规定" 匹配度一般
    → 用改写 query 评估: 关键词完全匹配！置信度应该高
    """
    query = state.get("rewritten_query", "") or state.get("query", "")
    retrieval_results = state.get("retrieval_results", [])
    intent = state.get("intent", "policy_query")

    result = _evaluate_confidence(query, retrieval_results, intent)

    if not result:
        # 兜底：有结果就是高置信度
        return {
            "confidence": 0.8 if retrieval_results else 0.0,
            "confidence_reason": "默认评估（评估器未就绪）",
        }

    return {
        "confidence": result.get("confidence", 0.5),
        "confidence_reason": result.get("reason", ""),
    }


# =============================================================================
# Conditional Edge: route_by_confidence —— 根据置信度决定下一节点
# =============================================================================

def route_by_confidence(state: AgentState) -> str:
    """条件路由：根据置信度和改写次数决定下一步

    这是 Self-Reflection 的「路由控制中心」：

    置信度高 ∧ 有结果 → "generate"（直接生成回答）
    置信度低 ∧ 改写次数 < MAX → "rewrite_query"（改写重试）
    置信度低 ∧ 改写次数 >= MAX → "generate_fallback"（降级兜底）
    无检索结果 → "generate_fallback"
    """
    confidence = state.get("confidence", 0.0)
    rewrite_count = state.get("rewrite_count", 0)
    retrieval_results = state.get("retrieval_results", [])

    # 没有检索结果 → 直接降级
    if not retrieval_results:
        return "generate_fallback"

    # 高置信度 → 生成回答
    if confidence >= CONFIDENCE_THRESHOLD:
        return "generate"

    # 低置信度但还可以再试试 → Query 改写
    if rewrite_count < MAX_REWRITES:
        return "rewrite_query"

    # 低置信度且已经试够了 → 降级兜底
    return "generate_fallback"


# =============================================================================
# Node 4: rewrite_query —— Query 改写（Self-Reflection 核心）
# =============================================================================

async def node_rewrite_query(state: AgentState) -> dict:
    """Query 改写节点 —— Self-Reflection 的执行者

    当检索置信度不足时，这个节点用 LLM 改写 Query，换一种说法重新检索。

    Input:  state["query"], state["intent"], state["rewrite_count"]
    Output: state["rewritten_query"], state["rewrite_count"] (+1)

    面试话术:
    "Self-Reflection 是我 Agent 的核心亮点。当首次检索结果不理想时，
    Agent 不是直接放弃，而是反思 '是不是 Query 表述有问题'，
    然后用 LLM 改写成更精准的检索词重新检索。最多 2 轮，
    2 轮后仍然不行就诚实告知用户问题太复杂，建议联系主管。
    这避免了无限循环和强行编造答案的问题。"
    """
    query = state.get("query", "")
    intent = state.get("intent", "policy_query")
    rewrite_count = state.get("rewrite_count", 0)

    # 调用 LLM 改写 Query
    rewritten = _rewrite_query(query, intent)

    if not rewritten:
        # 改写失败 → 保留原 Query
        rewritten = query

    return {
        "rewritten_query": rewritten,
        "rewrite_count": rewrite_count + 1,
    }


# =============================================================================
# Node 5: generate —— 生成回答（正常路径）
# =============================================================================

async def node_generate(state: AgentState) -> dict:
    """生成回答节点 —— 检索结果良好时的正常生成路径

    Input:  state["query"], state["retrieval_results"], state["top_k"]
    Output: state["answer"], state["citations"], state["trace_id"]
    """
    query = state.get("query", "")
    retrieval_results_dicts = state.get("retrieval_results", [])
    top_k = state.get("top_k", 5)
    rewrite_count = state.get("rewrite_count", 0)

    components = _get_rag_components()
    if not components:
        return {"error": "RAG 组件未就绪", "answer": "系统未就绪，请稍后重试"}

    generator = components["generator"]

    retrieval_results = _deserialize_results(retrieval_results_dicts, top_k)

    try:
        cited_answer: CitedAnswer = generator.generate(
            query=query,
            retrieval_results=retrieval_results,
            top_k=top_k,
        )
    except Exception as e:
        return {"error": f"LLM 生成失败: {e}", "answer": "回答生成失败，请重试"}

    citations = [
        {
            "doc_name": c.doc_name,
            "clause_id": c.clause_id,
            "section_title": c.section_title,
            "original_text": c.original_text,
        }
        for c in cited_answer.citations
    ]

    # 如果经过 Query 改写，在回答前加说明
    prefix = ""
    if rewrite_count > 0:
        prefix = f"> 💡 经过 {rewrite_count} 次查询优化后找到以下信息：\n\n"

    return {
        "answer": prefix + cited_answer.answer_text,
        "citations": citations,
        "trace_id": cited_answer.trace_id,
    }


# =============================================================================
# Node 6: generate_fallback —— 降级兜底
# =============================================================================

async def node_generate_fallback(state: AgentState) -> dict:
    """降级兜底节点 —— 检索不足时的安全处理

    触发条件:
    1. 意图分类为 "uncertain"（非工作相关问题）
    2. 检索无结果
    3. 两次 Query 改写后置信度仍然不足

    策略：诚实告知，不强行编造答案。这是 RAG 系统的安全红线。
    """
    intent = state.get("intent", "policy_query")
    rewrite_count = state.get("rewrite_count", 0)
    retrieval_results = state.get("retrieval_results", [])

    # 根据不同情况给出不同的兜底回答
    if intent == "uncertain":
        answer = (
            "您的问题不属于国航内部业务范畴，我暂时无法提供准确答案。\n\n"
            "我是国航内部员工知识助手，擅长回答以下类型的问题：\n"
            "- 客规运价（退改签规则、舱位代码）\n"
            "- 行李规定（免费行李额、逾重费、特殊行李）\n"
            "- 会员权益（凤凰知音、里程累积与兑换）\n"
            "- 特殊旅客服务（无陪儿童、轮椅旅客等）\n"
            "- 航班不正常处置（延误赔偿、改签规则）\n"
            "- 证件签证要求\n\n"
            "如果您的问题确实与工作相关，请尝试换一种方式描述。"
        )
    elif not retrieval_results:
        answer = (
            "抱歉，我在知识库中未找到与您问题相关的信息。\n\n"
            "建议：\n"
            "- 尝试使用更具体的术语重新描述问题\n"
            "- 查阅相关原文手册获取完整信息\n"
            "- 如确需立即处理，请联系主管部门获取人工支持"
        )
    else:
        answer = (
            f"经过 {rewrite_count} 次查询优化，我仍未能找到足够可靠的信息来回答您的问题。\n\n"
            "为避免提供不准确的信息影响您的判断，建议您：\n"
            "- 查阅原始政策文档获取完整内容\n"
            "- 联系主管部门确认最新规定\n"
            "- 尝试用更具体的条款编号或关键词重新查询"
        )

    return {
        "answer": answer,
        "citations": [],
        "trace_id": "",
    }


# =============================================================================
# 构建 Agent 状态图
# =============================================================================

def build_agent_graph():
    """构建 LangGraph 状态图 —— 把所有节点和边组装成完整的 Agent

    Returns:
        编译好的 LangGraph StateGraph（可调用 .ainvoke() 执行）

    图结构（ASCII 可视化）:

                        ┌──────────┐
                        │  START   │
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │  intent  │──── uncertain ────┐
                        │ classify │                    │
                        └────┬─────┘                    │
                             │ policy/op/emergency      │
                        ┌────▼─────┐                    │
                        │ retrieve │                    │
                        └────┬─────┘                    │
                             │                          │
                        ┌────▼──────┐                   │
                        │ evaluate  │                   │
                        │ confidence│                   │
                        └────┬──────┘                   │
                             │                          │
              ┌──────────────┼──────────────┐           │
              │ high         │ low + retry   │ low+max   │
         ┌────▼────┐   ┌─────▼──────┐   ┌───▼───────────▼──┐
         │ generate│   │  rewrite   │   │ generate_fallback │
         └────┬────┘   │  query     │   └────────┬─────────┘
              │        └─────┬──────┘            │
              │              │                   │
              │        ┌─────▼──────┐            │
              │        │  retrieve  │ (loop)     │
              │        └─────┬──────┘            │
              │              │                   │
              │        ┌─────▼──────┐            │
              │        │ evaluate   │            │
              │        │ confidence │            │
              │        └─────┬──────┘            │
              │              │                   │
              └──────────────┼───────────────────┘
                             │
                        ┌────▼─────┐
                        │   END    │
                        └──────────┘

    Self-Reflection 循环说明:
    rewrite_query → retrieve → evaluate_confidence → (generate 或 rewrite 或 fallback)
    这个循环最多执行 MAX_REWRITES=2 次，之后强制走 fallback。
    """

    # ---- 步骤 1: 创建状态图 ----
    # StateGraph(AgentState): AgentState 是状态的类型定义
    # 所有节点读写的数据都必须符合 AgentState 的结构
    workflow = StateGraph(AgentState)

    # ---- 步骤 2: 添加节点 ----
    # add_node("节点名", 处理函数)
    # 节点名用于 in ConditionalEdge 的路由返回值
    workflow.add_node("intent_classify", node_intent_classify)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("evaluate_confidence", node_evaluate_confidence)
    workflow.add_node("rewrite_query", node_rewrite_query)
    workflow.add_node("generate", node_generate)
    workflow.add_node("generate_fallback", node_generate_fallback)

    # ---- 步骤 3: 添加边 ----
    # set_entry_point: 图的入口节点
    workflow.set_entry_point("intent_classify")

    # ---- 步骤 4: 添加条件边 ----
    # add_conditional_edges("源节点", 路由函数, {返回值: "目标节点"})
    #   1. 执行 intent_classify 节点
    #   2. 调用 route_by_intent(state) 获取返回值
    #   3. 在 mapping 中查找返回值对应的目标节点
    #   4. 跳转到目标节点
    workflow.add_conditional_edges(
        "intent_classify",
        route_by_intent,
        {
            "retrieve": "retrieve",
            "generate_fallback": "generate_fallback",
        },
    )

    # retrieve → evaluate_confidence（固定边）
    workflow.add_edge("retrieve", "evaluate_confidence")

    # evaluate_confidence → 三个方向（条件边）
    workflow.add_conditional_edges(
        "evaluate_confidence",
        route_by_confidence,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "generate_fallback": "generate_fallback",
        },
    )

    # rewrite_query → retrieve → evaluate_confidence → ...（Self-Reflection 循环）
    # 这三步形成一个闭环，通过 evaluate_confidence 的条件边来决定是继续循环还是跳出
    workflow.add_edge("rewrite_query", "retrieve")

    # generate → END
    workflow.add_edge("generate", END)

    # generate_fallback → END
    workflow.add_edge("generate_fallback", END)

    # ---- 步骤 5: 编译 ----
    # compile() 把图编译为可执行对象
    # 编译后可以调用 .ainvoke(initial_state) 执行
    return workflow.compile()


# =============================================================================
# 便捷调用函数
# =============================================================================

# 模块级缓存：编译好的 Agent Graph（单例）
_agent_graph = None


def get_agent_graph():
    """获取编译好的 Agent Graph 单例"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


async def run_agent(
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """运行 Agent —— 一次完整的「意图分类 → 检索 → 评估 → 生成」流程

    Args:
        query: 员工提问
        top_k: 检索数量

    Returns:
        最终 AgentState 的 dict，包含 answer、citations、intent 等
    """
    graph = get_agent_graph()

    # 构建初始状态
    initial_state: AgentState = {
        "query": query,
        "top_k": top_k,
        "intent": "",
        "intent_reason": "",
        "retrieval_results": [],
        "rewrite_count": 0,
        "rewritten_query": "",
        "confidence": 0.0,
        "confidence_reason": "",
        "answer": "",
        "citations": [],
        "trace_id": "",
        "error": "",
    }

    # ainvoke: async invoke —— 异步执行整个图
    # LangGraph 会按边定义的顺序依次执行节点
    # 最终返回所有节点执行完毕后的完整 State
    final_state = await graph.ainvoke(initial_state)
    return final_state


# =============================================================================
# 状态图可视化
# =============================================================================

def get_graph_mermaid() -> str:
    """获取 Mermaid 格式的状态图

    可以在 https://mermaid.live 上粘贴查看，或放在文档中展示。
    """
    graph = get_agent_graph()
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        return "Mermaid 生成失败，请检查 langgraph 版本"


def save_graph_png(filepath: str = "doc/agent_graph.png"):
    """保存状态图为 PNG 图片

    需要安装: pip install pygraphviz 或 pip install grandalf
    """
    graph = get_agent_graph()
    try:
        graph.get_graph().draw_png(output_file_path=filepath)
        print(f"[Agent] 状态图已保存到 {filepath}")
    except ImportError:
        print("[Agent] 需要安装 pygraphviz 才能生成 PNG，尝试用 Mermaid 替代...")
        mermaid = get_graph_mermaid()
        print(f"\nMermaid 代码（复制到 https://mermaid.live 查看）：\n{mermaid}")
    except Exception as e:
        print(f"[Agent] 状态图生成失败: {e}")
