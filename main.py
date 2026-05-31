"""
国航内部员工智能知识助手 - API 入口

启动方式：
    python main.py
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

访问 Swagger 文档：
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from api.middleware import exception_handler_middleware, request_logging_middleware
from api.routes import health, chat, document, eval as eval_routes


# =============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子"""
    settings = get_settings()
    print(f"[启动] 国航内部员工智能知识助手 v0.1.0")
    print(f"[配置] LLM Provider: {settings.llm_provider}")
    print(f"[配置] LLM Model: {settings.llm_model}")
    print(f"[配置] Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    print(f"[文档] Swagger UI → http://{settings.host}:{settings.port}/docs")
    yield
    print("[关闭] 服务已停止")


# =============================================================================
# 创建 FastAPI 应用
# =============================================================================

app = FastAPI(
    title="国航内部员工智能知识助手",
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
    version="0.1.0",
    lifespan=lifespan,
)

# =============================================================================
# 中间件（洋葱模型 —— 先注册的先执行外层）
# =============================================================================

# CORS: 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP 级别中间件：异常处理 + 请求日志
app.middleware("http")(exception_handler_middleware)
app.middleware("http")(request_logging_middleware)


# =============================================================================
# 挂载子路由
# =============================================================================

app.include_router(health.router)      # /health
app.include_router(chat.router)        # /chat, /chat/stream
app.include_router(document.router)    # /documents/*
app.include_router(eval_routes.router) # /eval/*


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
