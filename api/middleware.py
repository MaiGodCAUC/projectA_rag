"""
API 中间件 —— 异常处理 + 请求日志

所有中间件按注册顺序形成"洋葱模型"：
    请求 → CORS → 异常处理 → 请求日志 → 路由处理 → 请求日志 → 异常处理 → CORS → 响应

----------------------------------------------------------------------
## 你需要自己写的部分

中间件是 FastAPI 的"拦截器"机制。请求在到达路由之前，
会经过所有中间件；响应在返回客户端之前，也会经过所有中间件。

学习重点:
1. @app.middleware("http") 装饰器的作用范围
2. await call_next(request) 是"控制权交接点"
3. 异常处理中间件捕获所有未处理的异常，统一返回 JSON

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

import time
import uuid
from fastapi import Request
from fastapi.responses import JSONResponse
from api.models import APIResponse, ErrorCode


# =============================================================================
# 1. 异常处理中间件 —— 捕获所有未处理异常
# =============================================================================

async def exception_handler_middleware(request: Request, call_next):
    """全局异常处理 —— 防止未捕获的异常导致 500 HTML 页面

    TODO(用户): 理解异常处理中间件的工作原理

    面试话术:
    "我用中间件捕获所有未处理的异常，统一包装成 JSON 返回。
    这样前端不需要处理 HTML 错误页面，也不需要区分 500/503/422 等
    不同的状态码格式——全部走统一的 APIResponse.fail()。"

    流程:
    ┌──────────────────────────────────────────────────┐
    │ try:                                             │
    │     response = await call_next(request)          │
    │     return response                              │
    │ except Exception as e:                           │
    │     return JSONResponse(                         │
    │         content=APIResponse.fail(...).dict(),    │
    │         status_code=200  ← 业务错误也返回200,     │
    │     )                     错误信息在code字段里    │
    └──────────────────────────────────────────────────┘
    """
    # ================================================================
    # TODO(用户): 手写异常处理中间件逻辑
    # ================================================================
    #
    # 实现参考:
    #
    # try:
    #     # call_next(request) 将请求交给下一个中间件或路由处理函数
    #     # 这是"控制权的交接点"——之后的所有处理都在这行代码中完成
    #     response = await call_next(request)
    #     return response
    #
    # except Exception as e:
    #     # 打印错误日志（生产环境应改为 logging.error）
    #     print(f"[ERROR] {request.method} {request.url.path} -> {e}")
    #
    #     trace_id = str(uuid.uuid4())[:8]
    #
    #     # 返回统一格式的错误响应
    #     # 注意：HTTP 状态码仍为 200，业务错误码在 code 字段中
    #     # 这样前端不需要判断 HTTP 状态码，只看 code
    #     return JSONResponse(
    #         content=APIResponse.fail(
    #             code=ErrorCode.UNKNOWN_ERROR,
    #             detail=f"服务器内部错误: {str(e)}",
    #             trace_id=trace_id,
    #         ).model_dump(),
    #         status_code=200,
    #     )
    #
    # ================================================================
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        trace_id = str(uuid.uuid4())[:8]
        return JSONResponse(
            content=APIResponse.fail(
                code=ErrorCode.UNKNOWN_ERROR,
                detail=f"服务器内部错误: {str(e)}",
                trace_id=trace_id,
            ).model_dump(),
            status_code=200,
        )


# =============================================================================
# 2. 请求日志中间件 —— 记录每个请求的路径、耗时、状态码
# =============================================================================

async def request_logging_middleware(request: Request, call_next):
    """请求日志 —— 记录请求路径、方法、耗时、状态码

    TODO(用户): 理解请求日志中间件的工作原理

    日志格式示例:
        [2024-01-15 14:30:22] POST /chat -> 200 (1523ms) trace=a1b2c3d4

    面试话术:
    "每个请求进来我都记录路径、耗时和状态码。这看起来简单，
    但生产环境中这是排查问题的第一手数据——'哪个接口慢'
    '哪个时间段报错多'，都靠它。"
    """
    # ================================================================
    # TODO(用户): 手写请求日志中间件逻辑
    # ================================================================
    #
    # 实现参考:
    #
    # start_time = time.time()
    #
    # # call_next(request) 是行"分界线"：
    # #   这行之前的代码在请求到达时执行（日志前置）
    # #   这行之后的代码在响应返回时执行（日志后置）
    # response = await call_next(request)
    #
    # cost_ms = int((time.time() - start_time) * 1000)
    #
    # # 打印日志（生产环境应改为 logging.info）
    # # 日志格式: [时间] 方法 路径 -> 状态码 (耗时)
    # import datetime
    # now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"[{now}] {request.method} {request.url.path} -> "
    #       f"{response.status_code} ({cost_ms}ms)")
    #
    # return response
    #
    # ================================================================
    import datetime
    start_time = time.time()
    response = await call_next(request)
    cost_ms = int((time.time() - start_time) * 1000)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {request.method} {request.url.path} -> "
          f"{response.status_code} ({cost_ms}ms)")
    return response
