"""
国航内部员工智能知识助手 - API 入口

================================================================================
【学习要点总结】
  1. FastAPI 应用结构
     - Lifespan（应用生命周期）：通过 @asynccontextmanager 实现，控制启动/关闭行为
     - 中间件（Middleware）：CORS + 自定义 trace_id 注入，在请求到达路由前/后执行
     - 路由（Routes）：@app.get/post 装饰器定义，每个路由是一个独立的请求处理函数
  2. 请求/响应模型（Pydantic）
     - 请求体用 BaseModel 定义，FastAPI 自动做 JSON 解析 + 校验
     - 响应体同样用 BaseModel 定义，自动序列化为 JSON
     - Field(...) 表示必填字段，Field(default=...) 表示可选字段
  3. 中间件执行顺序（洋葱模型）
     请求进来 → CORS 中间件 → trace_id 中间件 → 路由处理函数 → trace_id 中间件 → CORS 中间件 → 响应返回
  4. SSE 流式输出预留设计
     - 当前 /chat 使用 llm.invoke()（一次性返回完整结果）
     - 若要改为流式输出，只需将 invoke 替换为 stream，响应改为 StreamingResponse
     - LLM 的 ChatOpenAI 本身支持 stream 方法，改动量很小

启动方式：
    python main.py
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import get_settings
from core.llm import get_llm

# ================================================================
# 注意：以下 import 在用户完成 core/llm.py 实现后启用
# ================================================================
# from core.llm import get_llm


# =============================================================================
# 请求/响应模型
#
# 【为什么用 Pydantic BaseModel 而不是 dict】
#   1. 自动校验：请求体不符合定义时 FastAPI 返回 422 而非 500，用户体验更好
#   2. 自动文档：FastAPI 从字段的 type/description 生成 Swagger 文档
#   3. IDE 友好：调用方有完整的类型提示
# =============================================================================

class ChatRequest(BaseModel):
    """对话请求体

    FastAPI 自动从 POST 请求的 JSON body 中解析字段。
    前端只需发送：{"message": "飞机上能带充电宝吗？"}
    """
    message: str = Field(
        ...,                                # "..." 是 Pydantic 的"必填"标记，等价于 required=True
        min_length=1,                       # 消息不能为空
        max_length=5000,                    # 限制最大长度，防止滥用
        description="用户消息"
    )


class ChatResponse(BaseModel):
    """对话响应体

    统一响应格式，所有接口用同一结构，方便前端统一处理。
    """
    code: int = Field(
        default=0,                          # 0 = 成功，非 0 = 失败（参照业界 API 规范）
        description="状态码：0 成功，非 0 失败"
    )
    data: dict = Field(
        default_factory=dict,               # 使用 default_factory 避免多个实例共享同一个 dict
        description="响应数据"
    )
    trace_id: str = Field(
        ...,                                # 必填，每个请求都有唯一 ID，方便日志追踪
        description="请求追踪 ID"
    )
    cost_ms: int = Field(
        default=0,
        description="处理耗时（毫秒）"
    )


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str = "0.1.0"


# =============================================================================
# 应用生命周期（Lifespan）
#
# 【什么是 Lifespan】
#   FastAPI 用 lifespan 代替了旧的 @app.on_event("startup") / @app.on_event("shutdown")。
#   它是 Python 3.7+ 的异步上下文管理器（async context manager）。
#
# 【执行流程】
#   yield 之前的代码 → 服务启动时执行
#   yield 之后的代码 → 服务关闭时执行（Ctrl+C 触发）
#
# 【实际应用场景】
#   - 启动时：初始化数据库连接池、预热模型、检查依赖服务可用性
#   - 关闭时：关闭数据库连接、清理临时文件、上报关闭事件
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子"""
    settings = get_settings()                           # 这里触发 .env 读取和校验
    print(f"[启动] 国航内部员工智能知识助手 v0.1.0")
    print(f"[配置] LLM Provider: {settings.llm_provider}")
    print(f"[配置] LLM Model: {settings.llm_model}")
    print(f"[配置] Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    yield                                                # ← 应用运行期间
    print("[关闭] 服务已停止")


# =============================================================================
# 创建 FastAPI 应用实例
# =============================================================================

settings = get_settings()                               # 模块加载时获取全局单例

app = FastAPI(
    title="国航内部员工智能知识助手",
    description="基于 RAG 的民航内部员工知识库问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# =============================================================================
# CORS 中间件
#
# 【什么是 CORS】
#   浏览器安全策略禁止页面从 domain-a.com 向 domain-b.com 发起 AJAX 请求。
#   开发阶段前端（localhost:8501）和后端（localhost:8000）是不同"域"，
#   需要后端显式允许跨域。
#
# 【当前配置说明】
#   allow_origins=["*"]：允许任意域名访问（开发/内网环境 OK，生产环境应改为具体域名）
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                                # 生产环境应改为前端实际域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 自定义中间件 —— 请求追踪
#
# 【中间件机制】@app.middleware("http") 注册的是"HTTP 级别中间件"
#   它拦截所有 HTTP 请求，在路由处理前后都可以插入逻辑。
#
# 【异步中间件执行顺序】
#   请求 → CORS → trace_id 中间件 → /chat 路由函数 → 中间件收尾 → 返回
#           ↑ call_next(request) 调用处是"控制权的交接点"
#
# 【trace_id 的作用】
#   - 每个请求生成唯一的 8 位 ID（uuid 前 8 位），写入响应头 X-Trace-ID
#   - 前端拿到后如果报错，带 trace_id 来找后端排查日志
#   - 在微服务架构中还会透传给下游服务
#
# 【cost_ms 的作用】
#   记录从中间件收到请求到返回响应的总耗时，包含 LLM 调用时间。
#   可用于性能监控和慢请求告警。
# =============================================================================

@app.middleware("http")
async def add_trace_id(request, call_next):
    """为每个请求添加 trace_id 和耗时记录"""
    trace_id = str(uuid.uuid4())[:8]                    # uuid4 → "a1b2c3d4-e5f6-..." → 取前 8 位
    start_time = time.time()

    response = await call_next(request)                 # ← 控制权交给下一个中间件或路由处理函数

    cost_ms = int((time.time() - start_time) * 1000)    # 秒转毫秒
    response.headers["X-Trace-ID"] = trace_id           # X- 前缀表示自定义 HTTP 头
    response.headers["X-Cost-Ms"] = str(cost_ms)

    return response


# =============================================================================
# 路由：健康检查
#
# GET /health → 返回服务状态
# 用途：Docker healthcheck、Kubernetes liveness probe、负载均衡器探测
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查

    简单返回 {"status": "ok", "version": "0.1.0"}。
    更完善的健康检查可以在这里检查数据库连接、LLM API 连通性等。
    """
    return HealthResponse(status="ok")


# =============================================================================
# 路由：对话接口
#
# POST /chat → 接收用户消息，调用 LLM，返回回答
#
# 【请求流程】
#   1. FastAPI 解析 JSON body → ChatRequest 对象（自动校验 message 长度）
#   2. 生成 trace_id
#   3. 调用 get_llm(settings) 获取 ChatOpenAI 实例
#   4. llm.invoke(message) 发送请求到大模型 API
#   5. 提取 response.content，封装为 ChatResponse 返回
#
# 【关于报错 307 重定向】
#   如果出现 307，检查 .env 中 LLM_BASE_URL 是否缺少 /v1 后缀：
#   - 正确：https://api.deepseek.com/v1
#   - 错误：https://api.deepseek.com（缺少 /v1，API 会返回 307 重定向到 /v1）
# =============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通对话接口（不含 RAG）

    当前实现：直接调用 LLM，不检索知识库。
    后续 RAG 版本会在此加入检索 → 增强 prompt → 生成回答的完整链路。

    【SSE 流式输出预留设计】
      当前用 llm.invoke() 一次性返回完整结果。
      若改为流式（逐字输出，提升用户体验），改动方案：
        1. 将 llm.invoke(request.message) 替换为 llm.stream(request.message)
        2. 导入 from fastapi.responses import StreamingResponse
        3. 返回 StreamingResponse(生成器, media_type="text/event-stream")
        4. 前端改为 EventSource 读取 SSE 事件流

      因为 ChatOpenAI 本身支持 .stream()，改动范围仅限本函数和前端。
    """
    trace_id = str(uuid.uuid4())[:8]                     # 路由内部也生成 trace_id，
                                                          # 优先于中间件的 trace_id（更精确）
    start_time = time.time()

    try:
        llm = get_llm(settings)                           # 从工厂获取 LLM 实例
        response = llm.invoke(request.message)            # 同步调用，阻塞直到 LLM 返回完整结果
        content = response.content                        # AIMessage.content 是字符串
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")
    return ChatResponse(
            data={"content": content},
            trace_id=trace_id,
            cost_ms=int((time.time() - start_time) * 1000)
        )


# =============================================================================
# 入口
#
# 【if __name__ == "__main__" 的作用】
#   直接运行 python main.py 时执行 uvicorn.run()，
#   被 import 时不执行（其他模块可以 import main.app 而不启动服务）。
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",                         # "模块名:FastAPI实例名"
        host=settings.host,                 # 0.0.0.0 → 监听所有网络接口
        port=settings.port,                 # 默认 8000
        reload=settings.debug,              # debug=True 时启用热重载（代码修改自动重启）
    )
