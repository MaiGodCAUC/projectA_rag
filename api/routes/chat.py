"""
对话接口 —— RAG 问答（普通 + SSE 流式）

POST /chat         → 非流式 RAG 对话
POST /chat/stream  → SSE 流式 RAG 对话

----------------------------------------------------------------------
## 你需要自己写的部分

把 RAG 管线接入 FastAPI 是工程化的关键一步。之前的 rag/ 模块
都是"库"，这里才是"服务"——真正对外提供 HTTP 接口。

学习重点:
1. FastAPI 中 async endpoint 和同步函数的混用
2. StreamingResponse + SSE 的实现方式
3. 统一 APIResponse 格式的包装
4. 错误处理和降级策略

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.models import (
    APIResponse, ChatRequest, ChatData, ErrorCode,
)
from rag.hybrid_search import HybridSearcher
from rag.vector_store import VectorStore
from rag.bm25 import BM25Retriever
from rag.generator import RAGGenerator
from core.config import get_settings

router = APIRouter(tags=["对话"])


# =============================================================================
# Lazy init —— 首次请求时加载 RAG 管线组件
# =============================================================================
# 不在模块导入时初始化（避免 Qdrant 没启动就报错），
# 而是在第一次请求时懒加载。

_rag_components = None           # 单例缓存


def _get_rag_components():
    """懒加载 RAG 管线组件（单例模式）

    只在第一次请求时初始化，后续复用同一实例。
    如果 Qdrant 没启动或没有索引数据，返回 None。
    """
    global _rag_components
    if _rag_components is not None:
        return _rag_components

    try:
        settings = get_settings()
        vector_store = VectorStore()
        bm25 = BM25Retriever()
        hybrid_searcher = HybridSearcher(
            vector_store=vector_store,
            bm25_retriever=bm25,
        )
        generator = RAGGenerator()

        _rag_components = {
            "hybrid_searcher": hybrid_searcher,
            "generator": generator,
        }
        return _rag_components
    except Exception as e:
        print(f"[警告] RAG 组件初始化失败: {e}")
        return None


# =============================================================================
# POST /chat —— 非流式 RAG 对话
# =============================================================================

@router.post("/chat")
async def rag_chat(req: ChatRequest):
    """RAG 知识库问答（非流式）

    完整的 RAG 流程：检索 → 生成 → 返回带引用的回答。

    请求示例:
        POST /chat
        {
          "message": "旅客行李箱摔坏了怎么赔？",
          "top_k": 5,
          "stream": false
        }

    响应示例:
        {
          "code": 0,
          "data": {
            "question": "旅客行李箱摔坏了怎么赔？",
            "answer": "根据公司规定，托运行李损坏的赔偿...",
            "citations": [{...}],
            "retrieval_count": 5
          },
          "trace_id": "a1b2c3d4",
          "cost_ms": 1523
        }
    """
    # ================================================================
    # TODO(用户): 手写 RAG 对话接口逻辑
    # ================================================================
    #
    # 实现参考:
    #

    # ---- 步骤 1: 参数校验 ----
    # if not req.message.strip():
    #     return APIResponse.fail(code=ErrorCode.PARAM_ERROR, detail="消息不能为空")
    #
    # ---- 步骤 2: 加载 RAG 管线 ----
    # components = _get_rag_components()
    # if components is None:
    #     return APIResponse.fail(
    #         code=ErrorCode.DOC_NOT_INDEXED,
    #         detail="RAG 组件未就绪，请检查 Qdrant 是否启动且已完成文档索引"
    #     )
    #
    # hybrid_searcher = components["hybrid_searcher"]
    # generator = components["generator"]
    #
    # ---- 步骤 3: 检索 ----
    # start_time = time.time()
    # try:
    #     retrieval_results = hybrid_searcher.search(
    #         query=req.message,
    #         top_k=req.top_k,
    #     )
    # except Exception as e:
    #     return APIResponse.fail(
    #         code=ErrorCode.SEARCH_ERROR,
    #         detail=f"检索失败: {str(e)}"
    #     )
    #
    # ---- 步骤 4: 生成 ----
    # try:
    #     cited_answer = generator.generate(
    #         query=req.message,
    #         retrieval_results=retrieval_results,
    #         top_k=req.top_k,
    #     )
    # except Exception as e:
    #     return APIResponse.fail(
    #         code=ErrorCode.LLM_ERROR,
    #         detail=f"LLM 生成失败: {str(e)}"
    #     )
    #
    # ---- 步骤 5: 包装响应 ----
    # cost_ms = int((time.time() - start_time) * 1000)
    # citations_list = [
    #     {
    #         "doc_name": c.doc_name,
    #         "clause_id": c.clause_id,
    #         "section_title": c.section_title,
    #         "original_text": c.original_text,
    #     }
    #     for c in cited_answer.citations
    # ]
    #
    # chat_data = ChatData(
    #     question=req.message,
    #     answer=cited_answer.answer_text,
    #     citations=citations_list,
    #     retrieval_count=len(retrieval_results),
    # )
    #
    # return APIResponse.ok(
    #     data=chat_data.model_dump(),
    #     trace_id=cited_answer.trace_id,
    #     cost_ms=cost_ms,
    # )

    # ================================================================
    if not req.message.strip():
        return APIResponse.fail(
            code=ErrorCode.PARAM_ERROR, detail="消息不能为空",
            trace_id=req.trace_id if hasattr(req, 'trace_id') else ""
        )

    components = _get_rag_components()
    if not components:
        return APIResponse.fail(
            code=ErrorCode.DOC_NOT_INDEXED,
            detail="知识库尚未就绪，请先上传文档并完成索引"
        )

    hybrid_searcher = components["hybrid_searcher"]
    generator = components["generator"]

    start_time = time.time()

    try:
        retrieval_results = hybrid_searcher.search(query=req.message, top_k=req.top_k)
    except Exception as e:
        return APIResponse.fail(code=ErrorCode.SEARCH_ERROR, detail=f"检索失败: {e}")

    try:
        cited_answer = generator.generate(
            query=req.message,
            retrieval_results=retrieval_results,
            top_k=req.top_k
        )
    except Exception as e:
        return APIResponse.fail(code=ErrorCode.LLM_ERROR, detail=f"LLM生成失败: {e}")

    cost_ms = int((time.time() - start_time) * 1000)
    citations_list = [
        {
            "doc_name": c.doc_name,
            "clause_id": c.clause_id,
            "section_title": c.section_title,
            "original_text": c.original_text,
        }
        for c in cited_answer.citations
    ]

    return APIResponse.ok(
        data=ChatData(
            question=req.message,
            answer=cited_answer.answer_text,
            citations=citations_list,
            retrieval_count=len(retrieval_results)
        ).model_dump(),
        trace_id=cited_answer.trace_id,
        cost_ms=cost_ms
    )


# =============================================================================
# POST /chat/stream —— SSE 流式 RAG 对话
# =============================================================================

@router.post("/chat/stream")
async def rag_chat_stream(req: ChatRequest):
    """RAG 知识库问答（SSE 流式输出）

    TODO(用户): 手写 SSE 流式对话接口逻辑

    和 /chat 的区别：
    - 用 generator.generate_stream() 代替 generator.generate()
    - 返回 StreamingResponse 代替 JSONResponse
    - 前端用 EventSource 逐 token 消费

    请求头要求:
        Content-Type: application/json
        Accept: text/event-stream

    流式输出格式:
        data: 根
        data: 据
        data: 规
        data: 定
        ...
        data: <!--TRACE_ID:a1b2c3d4-->
        data: <!--CITATIONS:[{...}]-->

    面试话术:
    "我用 SSE 实现流式输出。选 SSE 而不是 WebSocket 的原因是：
    RAG 对话是单向的（服务端 → 客户端推送回答），不需要双向通信。
    SSE 是基于 HTTP 的轻量协议，浏览器原生支持 EventSource API，
    比 WebSocket 更简单、更容易部署（不需要特殊代理配置）。"
    """
    # ================================================================
    # TODO(用户): 手写 SSE 流式对话接口逻辑
    # ================================================================
    #
    # 实现参考:
    #
    # if not req.message.strip():
    #     return APIResponse.fail(code=ErrorCode.PARAM_ERROR, detail="消息不能为空")
    #
    # components = _get_rag_components()
    # if not components:
    #     return APIResponse.fail(code=ErrorCode.DOC_NOT_INDEXED, detail="知识库未就绪")
    #
    # hybrid_searcher = components["hybrid_searcher"]
    # generator = components["generator"]
    #
    # # 先检索
    # retrieval_results = hybrid_searcher.search(query=req.message, top_k=req.top_k)
    #
    # # SSE 流式生成器函数
    # async def event_generator():
    #     async for token in generator.generate_stream(
    #         query=req.message,
    #         retrieval_results=retrieval_results,
    #         top_k=req.top_k
    #     ):
    #         # SSE 格式: "data: <内容>\n\n"
    #         # 双换行是 SSE 协议的分隔符
    #         yield f"data: {token}\n\n"
    #
    # return StreamingResponse(
    #     event_generator(),
    #     media_type="text/event-stream",
    #     headers={
    #         "Cache-Control": "no-cache",
    #         "Connection": "keep-alive",
    #         "X-Accel-Buffering": "no",       # 禁用 Nginx 缓冲
    #     }
    # )

    # ================================================================
    if not req.message.strip():
        return APIResponse.fail(code=ErrorCode.PARAM_ERROR, detail="消息不能为空")

    components = _get_rag_components()
    if not components:
        return APIResponse.fail(code=ErrorCode.DOC_NOT_INDEXED, detail="知识库未就绪")

    retrieval_results = components["hybrid_searcher"].search(query=req.message, top_k=req.top_k)

    async def event_generator():
        async for token in components["generator"].generate_stream(
            query=req.message,
            retrieval_results=retrieval_results,
            top_k=req.top_k
        ):
            yield f"data: {token}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
