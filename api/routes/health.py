"""
健康检查接口

返回服务运行状态，用于 Docker healthcheck、K8s liveness probe、
负载均衡器健康探测。

----------------------------------------------------------------------
FastAPI APIRouter 用法详解:

APIRouter 的作用是把路由组织成独立模块，避免所有路由挤在 main.py 里。
它就像一个"迷你 FastAPI app"——可以用 @router.get/post 定义路由，
最后在 main.py 用 app.include_router(router) 挂载。

关键用法:
1. 创建: router = APIRouter(tags=["标签名"])
   tags 参数控制这个路由在 Swagger 文档中的分组

2. 定义路由: @router.get("/path")  / @router.post("/path")
   语法和 @app.get() 完全一样，只是挂在 router 而非 app 上

3. 挂载: app.include_router(router)
   在 main.py 中把 router 的路由合并到 app 中
----------------------------------------------------------------------
"""

# APIRouter: FastAPI 的路由分组器
#   把同一个业务模块的路由放在一个 router 中，类似于 Flask 的 Blueprint
from fastapi import APIRouter

# 创建路由器实例
# tags=["健康检查"] → Swagger 文档中这个组叫 "健康检查"
# 所有用 @router 定义的路由都会出现在这个分组下
router = APIRouter(tags=["健康检查"])


# @router.get("/health")
# 这行代码做了三件事:
#   1. 注册 GET /health 路由（当客户端 GET /health 时触发此函数）
#   2. 自动生成 Swagger 文档条目（路径、参数、响应格式）
#   3. 把函数的返回值自动序列化为 JSON（因为默认 media_type=application/json）
#
# async def: 异步函数
#   虽然这里没有 await 操作，但 FastAPI 推荐用 async——如果以后要加 await 逻辑
#   （如检查数据库连接），不用改函数签名
@router.get("/health")
async def health():
    """服务健康检查

    当客户端访问 GET /health 时：
    1. FastAPI 收到 HTTP 请求
    2. 匹配到 @router.get("/health")
    3. 调用 health() 函数
    4. 把返回值 {"status": "ok", ...} 序列化为 JSON
    5. 作为 HTTP 响应返回（Content-Type: application/json）

    Docker Compose 健康检查配置:
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
          interval: 30s               # 每 30 秒检查一次
          timeout: 3s                 # 超过 3 秒没响应 = 不健康
          retries: 3                  # 连续 3 次失败 = 标记为 unhealthy
    """

    # FastAPI 会把这个 dict 自动转为 JSON 字符串返回
    # 不需要手动 json.dumps()——FastAPI 内部做了序列化
    return {
        "status": "ok",
        "version": "0.1.0",
        "service": "国航内部员工智能知识助手",
    }
