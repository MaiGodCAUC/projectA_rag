"""
国航内部员工智能知识助手 - API 入口

启动方式：
    python main.py
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

访问 Swagger 文档：
    http://localhost:8000/docs

----------------------------------------------------------------------
FastAPI 应用组装流程（本文件的核心）：

    main.py 是 "组装车间" —— 把各个独立的模块组合成完整的 Web 应用

    1. 创建 FastAPI() 实例（app）
    2. 注册 Lifespan 钩子（启动/关闭时的操作）
    3. 挂载中间件（CORS → 异常处理 → 请求日志）
    4. 挂载子路由（health + chat + document + eval）
    5. uvicorn.run() 启动 HTTP 服务器

    模块分工：
    - api/models.py:    定义数据结构（请求/响应/错误码）
    - api/middleware.py: 定义中间件（异常处理/请求日志）
    - api/routes/*.py:   定义路由处理函数（按业务模块划分）
    - main.py:           组装 + 启动
----------------------------------------------------------------------
"""

# ================================================================
# asynccontextmanager: Python 标准库的异步上下文管理器装饰器
# 用法:
#   @asynccontextmanager
#   async def lifespan(app):
#       # yield 之前 = 启动时执行
#       yield
#       # yield 之后 = 关闭时执行
# 这是 FastAPI 推荐的生命周期管理方式（替代旧版的 @app.on_event()）
# ================================================================
from contextlib import asynccontextmanager

# ================================================================
# FastAPI: 框架核心类
#   app = FastAPI(...) 创建一个 ASGI 应用实例
#   这个实例就是给 uvicorn 运行的入口
#
# FastAPI 构造函数参数:
#   title: Swagger 文档的标题
#   description: Swagger 文档的详细说明（支持 Markdown）
#   version: API 版本号
#   lifespan: 应用生命周期钩子函数
# ================================================================
from fastapi import FastAPI

# ================================================================
# CORSMiddleware: FastAPI 内置的跨域中间件
# 来自 Starlette（FastAPI 的底层框架）
# CORS = Cross-Origin Resource Sharing（跨源资源共享）
# 没有这个中间件，浏览器会拦截来自不同域/端口的 AJAX 请求
# ================================================================
from fastapi.middleware.cors import CORSMiddleware

# 配置管理
from core.config import get_settings

# 自定义中间件
from api.middleware import exception_handler_middleware, request_logging_middleware

# 子路由模块
# eval as eval_routes: eval 是 Python 内置函数名，用别名避免冲突
from api.routes import health, chat, document, eval as eval_routes, observability


# =============================================================================
# 应用生命周期（Lifespan）
# =============================================================================
# FastAPI 用 lifespan 管理启动/关闭逻辑，替代了旧版的 @app.on_event("startup/shutdown")
# 它是一个异步上下文管理器 (async context manager)
#
# 执行流程:
#   1. 启动 uvicorn → 触发 lifespan 函数
#   2. 执行 yield 之前的代码（启动逻辑）
#   3. yield → 暂停在这里，应用正常运行
#   4. 收到关闭信号（Ctrl+C / SIGTERM）→ 从 yield 恢复
#   5. 执行 yield 之后的代码（关闭逻辑）

@asynccontextmanager                 # 装饰器：把 async generator 转为 async context manager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子

    yield 之前 = 启动时执行（初始化、预热、检查依赖）
    yield 之后 = 关闭时执行（清理连接、保存状态）
    """

    # ---- 启动逻辑 ----
    # get_settings() 在这里触发 .env 文件的读取和 Pydantic 校验
    # 如果 .env 中 LLM_API_KEY 缺失或格式错误，这里就会报错——服务不会启动
    settings = get_settings()

    print(f"[启动] 国航内部员工智能知识助手 v0.1.0")
    print(f"[配置] LLM Provider: {settings.llm_provider}")
    print(f"[配置] LLM Model: {settings.llm_model}")
    print(f"[配置] Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"[文档] Swagger UI → http://{settings.host}:{settings.port}/docs")
    # 这里可以加更多启动逻辑:
    # - 预热 Embedding 模型（首次调用会卡很久，提前加载）
    # - 检查 Qdrant 连接是否正常
    # - 初始化数据库连接池

    yield   # ← 应用运行期间暂停在此处

    # ---- 关闭逻辑 ----
    print("[关闭] 服务已停止")
    # 这里可以加更多关闭逻辑:
    # - 关闭数据库连接
    # - 清理临时文件
    # - 上报关闭事件到监控系统


# =============================================================================
# 创建 FastAPI 应用实例
# =============================================================================
# 这行代码创建了整个 Web 应用的入口对象
# 所有路由、中间件、生命周期都挂在这个 app 实例上
# uvicorn 启动时读取这个实例，开始监听 HTTP 请求

app = FastAPI(
    # title: 显示在 Swagger 文档顶部
    title="国航内部员工智能知识助手",

    # description: Swagger 文档首页的说明（支持 Markdown 渲染）
    # 这里的 Markdown 会在 http://localhost:8000/docs 的首页展示
    description="""基于 RAG 的民航内部员工知识库问答系统。

## 功能模块

- **对话**: RAG 增强问答（普通 + SSE 流式）
- **文档管理**: 上传、列表、索引、删除
- **评估**: RAGAS 质量评估
- **健康检查**: 服务状态探测

## 统一响应格式

```json
{
  "code": 0,
  "data": {...},
  "trace_id": "a1b2c3d4",
  "cost_ms": 123
}
```

## 错误码

| 错误码 | 含义 |
|--------|------|
| 0 | 成功 |
| 1001 | 参数校验失败 |
| 1002 | 文档解析失败 |
| 1003 | 检索失败 |
| 1004 | LLM 生成失败 |
| 1005 | 文档未索引 |
| 9999 | 未知错误 |
""",

    # version: API 版本号，显示在 Swagger 文档中
    version="0.1.0",

    # lifespan: 生命周期钩子
    lifespan=lifespan,
)


# =============================================================================
# 中间件注册
# =============================================================================
# 中间件的执行顺序由注册顺序决定（先注册 = 外层/先执行）

# ---- add_middleware: 注册第三方中间件（类形态） ----
# CORSMiddleware 是 Starlette 提供的类，不是纯函数
# add_middleware 会用类实例的 __call__ 方法作为中间件处理函数
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"]: 允许任何域名的前端访问
    # "*" 是最宽松配置，适合开发/内网环境
    # 生产环境应改为具体域名，如 ["https://airchina-kb.internal.cn"]
    allow_origins=["*"],

    # allow_credentials=True: 允许跨域请求携带 Cookie
    allow_credentials=True,

    # allow_methods=["*"]: 允许所有 HTTP 方法（GET/POST/DELETE/PUT/PATCH）
    allow_methods=["*"],

    # allow_headers=["*"]: 允许所有请求头
    allow_headers=["*"],
)

# ---- app.middleware("http"): 注册纯函数中间件 ----
# middleware("http") 返回一个装饰器，用法: @app.middleware("http")
# 也可以显式调用: app.middleware("http")(your_function)
# "http" 表示这是 HTTP 级别的中间件（还有 "websocket" 级别）
#
# 注意：这两个中间件的注册顺序很重要！
# exception_handler 先注册 → 在最外层（能捕获所有异常）
# request_logging 后注册 → 在内层（先看到请求、后看到响应）
app.middleware("http")(exception_handler_middleware)
app.middleware("http")(request_logging_middleware)


# =============================================================================
# 挂载子路由
# =============================================================================
# include_router: 把 APIRouter 实例的路由合并到 app 中
# 每个子模块的 router 里用 @router.get/post/delete 定义的路由
# 都会被 include_router 添加到 app 的路由表里
#
# 等效于在 main.py 里直接写 @app.get("/chat")，但这样模块化更好维护

app.include_router(health.router)       # GET  /health
app.include_router(chat.router)         # POST /chat, POST /chat/stream
app.include_router(document.router)     # GET/POST/DELETE /documents/*
app.include_router(eval_routes.router)      # POST /eval/run, GET /eval/report
app.include_router(observability.router)    # GET /metrics, /metrics/nodes, ...


# =============================================================================
# 启动入口
# =============================================================================
# if __name__ == "__main__" 的含义:
#   当直接运行 python main.py 时 → __name__ = "__main__" → 执行 uvicorn.run()
#   当被其他模块 import 时 → __name__ = "main" → 不执行 uvicorn.run()
# 这样可以 import main.app 用于测试（如 TestClient）而不启动服务器

if __name__ == "__main__":
    # uvicorn: Python 的 ASGI 服务器（FastAPI 推荐的生产/开发服务器）
    # 比 Flask 自带的开发服务器快得多，支持异步、WebSocket、热重载
    import uvicorn

    settings = get_settings()
    # uvicorn.run() 启动 HTTP 服务器，开始监听请求
    uvicorn.run(
        # "main:app": "模块名:FastAPI实例名"
        # uvicorn 会 import main 模块，找到 app 变量
        "main:app",

        # host: 监听的网络接口
        # "0.0.0.0": 监听所有网络接口（外部可访问）
        # "127.0.0.1": 仅本地访问
        host=settings.host,

        # port: 监听的端口
        port=settings.port,

        # reload: 热重载开关
        # True 时，代码修改后自动重启（开发环境推荐）
        # 由 .env 中 DEBUG 变量控制
        reload=settings.debug,
    )
