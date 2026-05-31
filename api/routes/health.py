"""
健康检查接口

返回服务运行状态，用于 Docker healthcheck、K8s liveness probe、
负载均衡器健康探测。
"""

from fastapi import APIRouter

router = APIRouter(tags=["健康检查"])


@router.get("/health")
async def health():
    """服务健康检查

    Docker Compose 中配置:
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s
          timeout: 3s
          retries: 3
    """
    return {
        "status": "ok",
        "version": "0.1.0",
        "service": "国航内部员工智能知识助手",
    }
