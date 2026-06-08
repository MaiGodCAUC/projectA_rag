"""
Agent 路由对话接口 —— LangGraph 智能路由 + RAG 问答

POST /agent/chat         → Agent 路由对话（非流式）
POST /agent/chat/stream  → Agent 路由对话（SSE 流式）

和 /chat 的区别:
- /chat: 直接检索 → 生成，所有 Query 走同一条链路
- /agent/chat: 意图分类 → 条件路由 → 置信度评估 → 可能需要改写 → 生成
  不是简单的"检索+生成"，而是先做决策再行动

这是 Day 10 的核心交付物——把 LangGraph Agent 封装为 HTTP API。
"""

import time
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.models import APIResponse, ChatRequest, ChatData, ErrorCode
from core.observability import get_collector

router = APIRouter(tags=["Agent对话"])


# =============================================================================
# POST /agent/chat —— Agent 路由对话（非流式）
# =============================================================================

@router.post("/agent/chat")
async def agent_chat(req: ChatRequest):
    """Agent 智能路由对话（非流式）

    和 /chat 的区别：请求先经过 LangGraph Agent 做意图分类和条件路由，
    而不是直接走检索+生成。

    请求示例:
        POST /agent/chat
        {
          "message": "旅客行李箱摔坏了怎么赔？",
          "top_k": 5,
          "stream": false
        }

    响应示例（同 /chat 格式）:
        {
          "code": 0,
          "data": {
            "question": "旅客行李箱摔坏了怎么赔？",
            "answer": "根据...",
            "citations": [...],
            "retrieval_count": 5,
            "agent_intent": "policy_query",
            "agent_rewrites": 0,
            "agent_confidence": 0.9
          },
          "trace_id": "a1b2c3d4",
          "cost_ms": 1850
        }
    """
    # ---- 参数校验 ----
    if not req.message.strip():
        return APIResponse.fail(code=ErrorCode.PARAM_ERROR, detail="消息不能为空")

    start_time = time.time()

    # ---- 运行 Agent ----
    try:
        from agent.router_graph import run_agent
        # run_agent() 返回最终的 AgentState dict
        final_state = await run_agent(query=req.message, top_k=req.top_k)
    except Exception as e:
        return APIResponse.fail(code=ErrorCode.UNKNOWN_ERROR, detail=f"Agent 执行失败: {e}")

    # ---- 提取结果 ----
    answer = final_state.get("answer", "")
    citations = final_state.get("citations", [])
    retrieval_count = len(final_state.get("retrieval_results", []))
    intent = final_state.get("intent", "")
    rewrites = final_state.get("rewrite_count", 0)
    confidence = final_state.get("confidence", 0.0)
    trace_id = final_state.get("trace_id", "")
    error = final_state.get("error", "")

    if error:
        return APIResponse.fail(code=ErrorCode.SEARCH_ERROR, detail=error)

    # ---- 上报可观测性指标 ----
    cost_ms = int((time.time() - start_time) * 1000)
    get_collector().record_request({
        "trace_id": trace_id,
        "total_ms": cost_ms,
        "tokens": 0,
        "nodes": {},
        "query": req.message,
        "success": True,
    })

    # ---- 构建响应 ----
    return APIResponse.ok(
        data={
            "question": req.message,
            "answer": answer,
            "citations": citations,
            "retrieval_count": retrieval_count,
            # Agent 特有字段：让前端/面试官看到 Agent 的决策过程
            "agent_intent": intent,
            "agent_rewrites": rewrites,
            "agent_confidence": round(confidence, 2),
        },
        trace_id=trace_id,
        cost_ms=cost_ms,
    )


# =============================================================================
# POST /agent/chat/stream —— Agent 路由对话（SSE 流式）
# =============================================================================

@router.post("/agent/chat/stream")
async def agent_chat_stream(req: ChatRequest):
    """Agent 智能路由对话（SSE 流式输出）

    流式版本的 Agent 对话。先执行 Agent 的检索决策流程，
    然后用流式方式输出最终的回答。

    SSE 事件格式:
      data: 根\n\n
      data: 据\n\n
      ...
      data: <!--CITATIONS:[...]-->\n\n
    """
    # ---- 参数校验 ----
    if not req.message.strip():
        return APIResponse.fail(code=ErrorCode.PARAM_ERROR, detail="消息不能为空")

    # ---- 先运行 Agent 检索决策（这部分必须等待完成） ----
    try:
        from agent.router_graph import run_agent, _get_rag_components
        final_state = await run_agent(query=req.message, top_k=req.top_k)
    except Exception as e:
        return APIResponse.fail(code=ErrorCode.UNKNOWN_ERROR, detail=f"Agent 执行失败: {e}")

    # 如果 Agent 已经走到了 fallback（没有检索结果），直接返回非流式
    retrieval_results = final_state.get("retrieval_results", [])
    answer = final_state.get("answer", "")
    rewrites = final_state.get("rewrite_count", 0)

    if not retrieval_results or rewrites >= 2:
        # fallback 场景：降级回答不需要流式
        return APIResponse.ok(data={
            "question": req.message,
            "answer": answer,
            "citations": [],
            "retrieval_count": 0,
        })

    # ---- 流式生成（检索结果转 RetrievalResult 对象） ----
    from rag.models import RetrievalResult

    retrieval_objects = [
        RetrievalResult(
            score=r.get("score", 0),
            chunk={
                "content": r.get("content", ""),
                "doc_name": r.get("doc_name", ""),
                "clause_id": r.get("clause_id", ""),
                "section_title": r.get("section_title", ""),
            },
        )
        for r in retrieval_results[:req.top_k]
    ]

    async def event_generator():
        """SSE 事件流生成器"""
        components = _get_rag_components()
        if not components:
            yield f"data: 系统未就绪\n\n"
            return

        async for token in components["generator"].generate_stream(
            query=req.message,
            retrieval_results=retrieval_objects,
            top_k=req.top_k,
        ):
            yield f"data: {token}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
