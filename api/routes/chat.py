"""
对话接口 —— RAG 问答（普通 + SSE 流式）

POST /chat         → 非流式 RAG 对话
POST /chat/stream  → SSE 流式 RAG 对话

----------------------------------------------------------------------
FastAPI 核心用法在本文件中的体现:

1. 请求体解析: async def rag_chat(req: ChatRequest)
   FastAPI 看到参数类型是 Pydantic BaseModel 子类 → 自动读取 JSON body 并校验

2. 返回类型: 返回 APIResponse 对象 → FastAPI 自动序列化为 JSON

3. StreamingResponse: 返回流式数据给客户端 → 用于 SSE 逐字输出

4. 懒加载单例: 不在模块加载时初始化 RAG 管线（Qdrant 可能没启动）
   而是在第一次请求时初始化，后续复用

----------------------------------------------------------------------
你需要自己写的部分:

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

# APIRouter: 路由分组器
# HTTPException: 抛 FastAPI 的标准 HTTP 异常（本项目统一用 APIResponse.fail 代替）
from fastapi import APIRouter

# StreamingResponse: FastAPI 的流式响应类
#   和普通 JSON 响应的区别：
#   普通: return dict → FastAPI 一次性序列化为 JSON → 发送 → 连接关闭
#   流式: return StreamingResponse(generator) → FastAPI 逐个 yield → 发送 → 保持连接
# ================================================================
# StreamingResponse 的三个参数:
#   1. content: 可迭代对象（生成器/迭代器），每次 yield 一个字符串
#   2. media_type: "text/event-stream" 是 SSE 协议规定的 MIME 类型
#   3. headers: 额外的 HTTP 响应头
# ================================================================
from fastapi.responses import StreamingResponse

from api.models import APIResponse, ChatRequest, ChatData, ErrorCode

# RAG 管线组件
from rag.hybrid_search import HybridSearcher
from rag.vector_store import VectorStore
from rag.bm25 import BM25Retriever
from rag.generator import RAGGenerator
from core.config import get_settings

# 创建路由器
router = APIRouter(tags=["对话"])


# =============================================================================
# Lazy init —— 首次请求时加载 RAG 管线组件
# =============================================================================
# 为什么不是模块导入时就初始化？
# 模块导入发生在 python main.py 启动时。如果 Qdrant Docker 容器还没启动，
# 或者没有索引数据，VectorStore() 会连接失败 → 整个进程启动失败。
#
# 懒加载 = 启动时不初始化，等第一个请求来了再试
# 如果请求时还失败 → 返回错误信息，不影响服务进程存活

# _rag_components: 模块级变量，存储单例实例
# None = 还没初始化过
# dict = 初始化成功，存了 hybrid_searcher 和 generator
_rag_components = None


def _get_rag_components():
    """懒加载 RAG 管线组件（单例模式）

    只在第一次请求时初始化，后续复用同一实例。
    如果 Qdrant 没启动或没有索引数据，返回 None。

    单例模式的好处:
    1. 只初始化一次 —— Qdrant 连接复用，避免每次请求都建立新连接
    2. 懒加载 —— 服务启动不依赖 Qdrant，提高启动成功率
    3. 全局共享 —— 所有请求共用同一个 HybridSearcher 和 RAGGenerator
    """
    # global 声明：告诉 Python "我要修改模块级变量 _rag_components"
    global _rag_components

    # 如果已经初始化过（不是 None），直接返回缓存的实例
    if _rag_components is not None:
        return _rag_components

    try:
        # 加载配置（.env 中的 API Key、Qdrant 地址等）
        settings = get_settings()

        # 初始化 RAG 管线的三个核心组件
        # VectorStore: 连接 Qdrant，负责向量存储和检索
        vector_store = VectorStore()
        # BM25Retriever: 关键词检索，基于 jieba 分词
        bm25 = BM25Retriever()
        # HybridSearcher: 融合向量检索和 BM25 的混合检索器
        hybrid_searcher = HybridSearcher(
            vector_store=vector_store,
            bm25_retriever=bm25,
        )
        # RAGGenerator: 将检索结果注入 Prompt，调用 LLM 生成回答
        generator = RAGGenerator()

        # 存入全局缓存
        _rag_components = {
            "hybrid_searcher": hybrid_searcher,
            "generator": generator,
        }
        return _rag_components

    except Exception as e:
        # 初始化失败（如 Qdrant 未启动）→ 打印警告，返回 None
        # 调用方检查到 None 后返回友好错误信息，而不是崩溃
        print(f"[警告] RAG 组件初始化失败: {e}")
        return None


# =============================================================================
# POST /chat —— 非流式 RAG 对话
# =============================================================================

# ================================================================
# @router.post("/chat") 的工作原理:
#
# 1. 注册路由: 当客户端 POST /chat 时触发此函数
#
# 2. 参数解析: async def rag_chat(req: ChatRequest)
#    FastAPI 看到 req 类型是 ChatRequest（Pydantic BaseModel 子类）
#    → 自动从请求体读取 JSON → 按 ChatRequest 字段定义做校验
#    → 校验通过: 实例化 ChatRequest 对象 → 传入函数
#    → 校验失败: 返回 422 Unprocessable Entity（HTTP 层面，不经过我们的代码）
#
# 3. 返回序列化: return APIResponse(...)
#    FastAPI 看到返回值是 Pydantic BaseModel 子类
#    → 自动调用 .model_dump() 转 dict → 序列化为 JSON → 发送
# ================================================================
@router.post("/chat")
async def rag_chat(req: ChatRequest):
    """RAG 知识库问答（非流式）—— 检索 → 生成 → 返回完整回答

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
    # ================================================================
    # 学习要点——这段代码的每个步骤:
    #
    # 步骤 1: 参数二次校验
    #   为什么 FastAPI 已经做了 Pydantic 校验（min_length=1），
    #   这里还要检查 req.message.strip()？
    #   → min_length=1 只检查长度，但 "   "（全是空格）长度为 3，校验通过
    #   → .strip() 去掉首尾空格后再检查，防止纯空格消息进入 LLM
    #
    # 步骤 2: 加载 RAG 管线
    #   _get_rag_components() 返回 None = RAG 未就绪
    #   用 APIResponse.fail 返回友好错误而非抛异常
    #
    # 步骤 3: 检索
    #   hybrid_searcher.search() 是同步函数（不是 async）
    #   内部流程: 向量化 query → Qdrant 检索 → BM25 检索 → RRF 融合
    #   耗时通常在 50~200ms
    #
    # 步骤 4: 生成
    #   generator.generate() 也是同步函数
    #   内部流程: 构建 Prompt → 调用 LLM API → 提取引用
    #   耗时通常在 800~2000ms（主要是 LLM API 调用）
    #
    # 步骤 5: 包装响应
    #   model_dump() = Pydantic v2 的序列化方法，把对象转 dict
    #   APIResponse.ok() = 快捷构造 code=0 的成功响应
    # ================================================================

    # ---- 步骤 1: 参数二次校验（防止纯空格消息） ----
    # .strip() 去除字符串首尾的空格、tab、换行
    # 如果 strip 后为空 → 消息内容全是空白字符 → 拒绝
    if not req.message.strip():
        # APIResponse.fail() 返回统一格式的错误响应
        # code=ErrorCode.PARAM_ERROR = 1001
        return APIResponse.fail(
            code=ErrorCode.PARAM_ERROR,
            detail="消息不能为空"
        )

    # ---- 步骤 2: 加载 RAG 管线 ----
    components = _get_rag_components()
    if not components:
        # RAG 未就绪（Qdrant 未启动 / 知识库为空）
        return APIResponse.fail(
            code=ErrorCode.DOC_NOT_INDEXED,
            detail="知识库尚未就绪，请先上传文档并完成索引"
        )

    # 从单例缓存中取出两个核心组件
    hybrid_searcher = components["hybrid_searcher"]  # 混合检索器
    generator = components["generator"]               # RAG 生成器

    # ---- 步骤 3: 检索 + ---- 步骤 4: 生成 ----
    start_time = time.time()   # 记录开始时间，最后用于计算 cost_ms

    # 检索：调用混合检索器搜索 TOP-K 最相关文档片段
    # query=req.message → 用户的原始问题
    # top_k=req.top_k → 前端传的检索数量（默认 5）
    try:
        retrieval_results = hybrid_searcher.search(
            query=req.message,
            top_k=req.top_k
        )
    except Exception as e:
        # 检索失败（如 Qdrant 连接断开、超时）
        return APIResponse.fail(
            code=ErrorCode.SEARCH_ERROR,
            detail=f"检索失败: {e}"
        )

    # 生成：把检索结果注入 Prompt，调用 LLM 生成带引用的回答
    try:
        # cited_answer 是 CitedAnswer 对象:
        #   .answer_text → LLM 的回答文本（含 [来源: ...] 标记）
        #   .citations → Citation 对象列表（doc_name, clause_id, ...）
        #   .trace_id → LangSmith 追踪 ID
        cited_answer = generator.generate(
            query=req.message,
            retrieval_results=retrieval_results,
            top_k=req.top_k
        )
    except Exception as e:
        # LLM 调用失败（如 API Key 无效、调用超时）
        return APIResponse.fail(
            code=ErrorCode.LLM_ERROR,
            detail=f"LLM生成失败: {e}"
        )

    # ---- 步骤 5: 包装响应 ----
    # 计算总耗时（毫秒）
    cost_ms = int((time.time() - start_time) * 1000)

    # 把 Citation 对象列表转为 dict 列表（前端需要 JSON 可序列化的格式）
    citations_list = [
        {
            "doc_name": c.doc_name,           # 文档名，如 "托运行李运输规定"
            "clause_id": c.clause_id,          # 条款编号，如 "第3.2条"
            "section_title": c.section_title,  # 章节标题，如 "破损赔偿"
            "original_text": c.original_text,  # 原文片段（前 200 字）
        }
        for c in cited_answer.citations
    ]

    # 构建 ChatData 子模型 → 放入 APIResponse 的 data 字段
    # ChatData.model_dump() 将 Pydantic 对象转为 dict
    return APIResponse.ok(
        data=ChatData(
            question=req.message,
            answer=cited_answer.answer_text,
            citations=citations_list,
            retrieval_count=len(retrieval_results)
        ).model_dump(),
        trace_id=cited_answer.trace_id,   # 把 LangSmith trace_id 传给前端
        cost_ms=cost_ms                    # 全链路耗时
    )


# =============================================================================
# POST /chat/stream —— SSE 流式 RAG 对话
# =============================================================================

@router.post("/chat/stream")
async def rag_chat_stream(req: ChatRequest):
    """RAG 知识库问答（SSE 流式输出）

    TODO(用户): 手写 SSE 流式对话接口逻辑

    ================================================================
    SSE (Server-Sent Events) 详解:

    SSE 是一种基于 HTTP 的服务器推送技术。和普通 HTTP 的区别：

    普通 HTTP 请求:
      客户端发送请求 → 服务端处理 → 返回完整结果 → 连接关闭
      客户端必须等待所有数据准备好才能看到

    SSE:
      客户端发送请求 → 服务端返回流 → 逐个推送 token → 流结束关闭
      客户端边收边显示，不需要等待完整结果

    SSE vs WebSocket:
      SSE: 单向（服务端 → 客户端），基于 HTTP，浏览器原生支持 EventSource
      WebSocket: 双向，独立协议，需要专门的代理/负载均衡配置
      RAG 对话只需要单向推送 → 选 SSE 更合适

    SSE 数据格式:
      data: <内容>\n\n      ← 每个事件以 "data: " 开头，以 "\n\n" 结尾
      data: 根\n\n           ← 空行是事件分隔符
      data: 据\n\n
      data: 规\n\n
      data: 定\n\n
      ...
      data: <!--CITATIONS:[{...}]-->\n\n   ← 特殊标记传出引用数据

    面试话术:
    "我用 SSE 实现流式输出。选 SSE 而不是 WebSocket 的原因是：
    RAG 对话是单向的（服务端 → 客户端推送回答），不需要双向通信。
    SSE 是基于 HTTP 的轻量协议，浏览器原生支持 EventSource API，
    比 WebSocket 更简单、更容易部署（不需要特殊代理配置）。"
    ================================================================
    """
    # ================================================================
    # TODO(用户): 手写 SSE 流式对话接口逻辑
    # ================================================================
    # ================================================================
    # 学习要点:
    #
    # 1. StreamingResponse 的三个关键参数:
    #    content: 异步生成器函数 event_generator()
    #      每 yield 一次，FastAPI 就向客户端推送一个 SSE 事件
    #    media_type: "text/event-stream"
    #      告诉浏览器"这是 SSE 流"，EventSource API 才能正确解析
    #    headers: SSE 配套的 HTTP 响应头
    #      Cache-Control: no-cache → 不让代理/CDN 缓存 SSE 数据
    #      Connection: keep-alive → 保持 TCP 连接不关闭
    #      X-Accel-Buffering: no → 禁用 Nginx 缓冲（生产环境关键）
    #
    # 2. async def event_generator() —— 异步生成器
    #    async for token in generator.generate_stream(...):
    #      逐个接收 LLM 生成的 token
    #    yield f"data: {token}\n\n"
    #      按 SSE 格式包装并推送
    #
    # 3. SSE 协议的 "\n\n" 双换行
    #    单换行 = data 内容的一部分
    #    双换行 = 一个事件的结束标记
    #
    # 4. 和 /chat 的代码结构区别
    #    /chat:
    #      → hybrid_searcher.search() → generator.generate()
    #      → return APIResponse(...)
    #    /chat/stream:
    #      → hybrid_searcher.search() → generator.generate_stream()
    #      → return StreamingResponse(event_generator())
    #    前半段（检索）完全一样，后半段（生成）用流式版本
    # ================================================================

    # ---- 参数校验（同 /chat） ----
    if not req.message.strip():
        return APIResponse.fail(code=ErrorCode.PARAM_ERROR, detail="消息不能为空")

    # ---- 加载管线 ----
    components = _get_rag_components()
    if not components:
        return APIResponse.fail(code=ErrorCode.DOC_NOT_INDEXED, detail="知识库未就绪")

    # ---- 检索 ----
    retrieval_results = components["hybrid_searcher"].search(
        query=req.message, top_k=req.top_k
    )

    # ---- 流式生成 ----
    # ================================================================
    # async def event_generator() 是一个异步生成器函数
    # 调用它不会立即执行，而是返回一个 async generator 对象
    # FastAPI 拿到这个 generator 后，用 async for 逐个消费
    #
    # generate_stream() 的内部逻辑（见 rag/generator.py）:
    #   1. 构建 Prompt
    #   2. async for chunk in self.llm.astream(messages):
    #        token = chunk.content
    #        yield token          ← 每生成一个 token 就 yield 出去
    #   3. yield "<!--CITATIONS:[...]-->"  ← 最后 yield 引用数据
    #
    # 所以这里 async for token in ...:
    #   每次 LLM 生成一个 token → yield 通过 SSE 推给前端
    # ================================================================
    async def event_generator():
        """SSE 事件流生成器 —— 逐个 yield 数据给前端"""
        # generator.generate_stream() 返回 AsyncIterator[str]
        # async for: 异步迭代，每次 LLM 生成一个 token 就收到一个
        async for token in components["generator"].generate_stream(
            query=req.message,
            retrieval_results=retrieval_results,
            top_k=req.top_k
        ):
            # 按 SSE 协议格式包装并推送
            # f"data: {token}\n\n"
            #   例: token = "根" → 前端收到: "data: 根\n\n"
            #   例: token = "<!--CITATIONS:[...]-->" → 前端收到引用数据
            yield f"data: {token}\n\n"

    # ================================================================
    # StreamingResponse 返回后，FastAPI 会:
    #   1. 设置响应头（media_type + headers）
    #   2. 开始异步消费 event_generator()
    #   3. 每收到一个 yield，立即写入 HTTP 响应流
    #   4. 客户端（浏览器）通过 EventSource.onmessage 逐个收到
    # ================================================================
    return StreamingResponse(
        event_generator(),                     # 异步生成器对象
        media_type="text/event-stream",        # SSE 协议规定的 MIME 类型
        headers={
            # Cache-Control: 不让任何中间环节缓存 SSE 数据
            # no-cache 不是"不缓存"，而是"每次使用前必须重新验证"
            "Cache-Control": "no-cache",
            # Connection: 保持 TCP 连接不关闭（SSE 需要长连接）
            "Connection": "keep-alive",
            # X-Accel-Buffering: 只对 Nginx 有效
            # Nginx 默认会缓冲响应数据（攒够一定量再发），
            # 这会导致 SSE 的实时性被破坏——token 被 Nginx 攒起来不放给客户端
            # "no" = 告诉 Nginx 不要缓冲这个请求的响应
            "X-Accel-Buffering": "no",
        }
    )
