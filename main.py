"""
国航内部员工智能知识助手 - API 入口

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

# ================================================================
# 注意：以下 import 在用户完成 core/llm.py 实现后启用
# ================================================================
# from core.llm import get_llm


# ---------- 请求/响应模型 ----------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="用户消息")


class ChatResponse(BaseModel):
    code: int = Field(default=0, description="状态码：0 成功，非 0 失败")
    data: dict = Field(default_factory=dict)
    trace_id: str = Field(..., description="请求追踪 ID")
    cost_ms: int = Field(default=0, description="处理耗时（毫秒）")


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


# ---------- 应用生命周期 ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子"""
    settings = get_settings()
    print(f"[启动] 国航内部员工智能知识助手 v0.1.0")
    print(f"[配置] LLM Provider: {settings.llm_provider}")
    print(f"[配置] LLM Model: {settings.llm_model}")
    print(f"[配置] Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    yield
    print("[关闭] 服务已停止")


# ---------- 创建应用 ----------

settings = get_settings()

app = FastAPI(
    title="国航内部员工智能知识助手",
    description="基于 RAG 的民航内部员工知识库问答系统",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 中间件 ----------

@app.middleware("http")
async def add_trace_id(request, call_next):
    """为每个请求添加 trace_id 和耗时记录"""
    trace_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    response = await call_next(request)

    cost_ms = int((time.time() - start_time) * 1000)
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Cost-Ms"] = str(cost_ms)

    return response


# ---------- 路由 ----------

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    普通对话接口（不含 RAG）

    ⚠️ TODO(用户)：需要先完成 core/llm.py 的 get_llm() 实现，然后取消下方注释。
    """
    trace_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # ================================================================
    # TODO(用户)：取消下方注释，完成 LLM 调用逻辑
    #
    # from core.llm import get_llm
    #
    # try:
    #     llm = get_llm(settings)
    #     response = llm.invoke(request.message)
    #     content = response.content
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")
    #
    # return ChatResponse(
    #     data={"content": content},
    #     trace_id=trace_id,
    #     cost_ms=int((time.time() - start_time) * 1000),
    # )
    # ================================================================

    raise HTTPException(
        status_code=501,
        detail="LLM 调用尚未实现。请先完成 core/llm.py 中的 get_llm() 函数。",
    )


# ---------- 入口 ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
