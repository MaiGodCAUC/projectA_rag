"""
API 中间件 —— 异常处理 + 请求日志

所有中间件按注册顺序形成"洋葱模型"：
    请求 → CORS → 异常处理 → 请求日志 → 路由处理 → 请求日志 → 异常处理 → CORS → 响应

----------------------------------------------------------------------
FastAPI 中间件的工作原理:

1. 注册方式（两种）：
   方式 A: @app.middleware("http") 装饰器
     用于在 main.py 里直接写的中间件

   方式 B: app.add_middleware(SomeMiddleware, ...)
     用于第三方中间件（如 CORSMiddleware），它们有类形态的参数配置

   方式 C: app.middleware("http")(your_function)
     注册一个纯函数作为 HTTP 中间件（本项目用这种）

2. 执行顺序：
   - add_middleware 先添加的先执行（外层）
   - middleware("http") 后注册的反而先执行
   - 但不管哪种，都是「先注册 = 外层」，按洋葱模型嵌套

3. call_next:
   - 不是网络请求，是"调用下一个中间件或路由函数"的函数指针
   - await call_next(request) = "我把请求传下去，我在原地等响应回来"

----------------------------------------------------------------------
你需要自己写的部分:

中间件是 FastAPI 的"拦截器"机制。请求在到达路由之前，
会经过所有中间件；响应在返回客户端之前，也会经过所有中间件。

学习重点:
1. @app.middleware("http") 装饰器的作用范围
2. await call_next(request) 是"控制权交接点"
3. 异常处理中间件捕获所有未处理的异常，统一返回 JSON

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# time: 用于计算请求耗时
import time

# uuid: 用于生成 trace_id（异常处理中间件中用到）
import uuid

# datetime: 用于日志时间格式化
import datetime

# Request: FastAPI 的请求对象
#   可以从中获取 method（GET/POST）、url.path（路径）、headers（请求头）等
# call_next 的参数和返回值都是这个
from fastapi import Request

# JSONResponse: FastAPI 的 JSON 响应类
#   和普通返回 dict 的区别：可以自定义 HTTP 状态码、响应头
#   用法: return JSONResponse(content={...}, status_code=200, headers={...})
from fastapi.responses import JSONResponse

# 统一响应模型
from api.models import APIResponse, ErrorCode


# =============================================================================
# 1. 异常处理中间件 —— 捕获所有未处理异常，防止返回 500 HTML 页面
# =============================================================================
# 为什么需要这个？
# 默认情况下，路由函数里的未捕获异常 → FastAPI 返回 500 Internal Server Error（HTML）
# 前端收到的不是 JSON → 解析失败 → 用户看到的是空白页或崩溃
# 这个中间件把异常统一包装成 {"code": 9999, "data": {"detail": "..."}} 的 JSON 格式

async def exception_handler_middleware(request: Request, call_next):
    """全局异常处理 —— 捕获所有未处理异常，统一返回 JSON

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
    │         status_code=200                           │
    │     )                                            │
    └──────────────────────────────────────────────────┘
    """
    # ================================================================
    # TODO(用户): 手写异常处理中间件逻辑
    # ================================================================
    # ================================================================
    # 学习要点详解:
    #
    # 1. try/except 的包裹范围:
    #    call_next(request) 内部会执行「请求日志中间件 → 路由函数」
    #    这中间任何地方抛异常都会被这里的 except 捕获
    #
    # 2. 为什么不区分异常类型?
    #    所有未预期的异常都按 UNKNOWN_ERROR 处理。
    #    如果你能预料到的错误（如参数错误），应该在路由函数里自己 catch
    #    并返回 APIResponse.fail(PARAM_ERROR, ...)，不会走到这里。
    #    走到这里的 = "真的出 bug 了"
    #
    # 3. status_code=200 而不是 500:
    #    业务错误通过 code 字段表达，HTTP 始终返回 200。
    #    好处: 前端只需检查 res.json().code，不需要根据 HTTP 状态码
    #    写 switch/case。Nginx/负载均衡的 5xx 告警也不会被业务错误触发。
    #    争议: 有人认为 RESTful 就应该用 4xx/5xx。两者各有优劣，
    #    内部系统用 code 字段更简单。
    # ================================================================
    try:
        # -----------------------------------------------
        # call_next(request):
        #   - request 是 FastAPI 的 Request 对象，包含 method、url、headers 等
        #   - call_next 是 "下一个处理者" 的函数指针
        #   - await = 我在这里等着，下一个处理者（可能是请求日志中间件，
        #     也可能是路由函数）处理完后返回 response
        #   - 如果下一个处理者抛了异常，await 会把异常重新抛出来
        # -----------------------------------------------
        response = await call_next(request)
        return response

    except Exception as e:
        # 异常被捕获了！

        # print 打印错误日志到终端（生产环境应改为 logging.error + 上报到监控系统）
        # request.method: "GET" / "POST" / "DELETE"
        # request.url.path: "/chat" / "/documents" 等
        print(f"[ERROR] {request.method} {request.url.path} -> {e}")

        # 生成一个新的 trace_id，用于标记这次错误
        trace_id = str(uuid.uuid4())[:8]

        # 构造统一格式的错误 JSON 响应
        # APIResponse.fail():
        #   code=ErrorCode.UNKNOWN_ERROR = 9999
        #   detail 放入 data.detail 字段
        #   trace_id 供前端展示或报 Bug 时提供
        # .model_dump():
        #   Pydantic v2 方法，把对象递归转为 Python dict
        #   旧版叫 .dict()，v2 改名为 .model_dump()
        return JSONResponse(
            content=APIResponse.fail(
                code=ErrorCode.UNKNOWN_ERROR,
                detail=f"服务器内部错误：{str(e)}",
                trace_id=trace_id
            ).model_dump(),
            status_code=200              # HTTP 层面返回 200，错误信息在 JSON 的 code 字段里
        )


# =============================================================================
# 2. 请求日志中间件 —— 记录每个请求的路径、耗时、状态码
# =============================================================================
# 为什么需要这个？
# 没有日志的接口 = 黑盒，出了问题只能猜
# 这个中间件给每个请求打印一行日志，形成完整的请求审计记录

async def request_logging_middleware(request: Request, call_next):
    """请求日志 —— 记录请求路径、方法、耗时、状态码

    TODO(用户): 理解请求日志中间件的工作原理

    日志格式示例:
        [2024-01-15 14:30:22] POST /chat -> 200 (1523ms)

    面试话术:
    "每个请求进来我都记录路径、耗时和状态码。这看起来简单，
    但生产环境中这是排查问题的第一手数据——'哪个接口慢'
    '哪个时间段报错多'，都靠它。"
    """
    # ================================================================
    # TODO(用户): 手写请求日志中间件逻辑
    # ================================================================
    # ================================================================
    # 学习要点详解:
    #
    # 1. call_next(request) 是 "分界线":
    #    - 之前的代码 = 前置逻辑（请求进来时执行）
    #    - 之后的代码 = 后置逻辑（响应返回时执行）
    #
    #    例子:
    #      start_time = time.time()       # ← 前置：记录开始时间
    #      response = await call_next(request)  # ← 分界线
    #      cost_ms = int((time.time() - start_time) * 1000)  # ← 后置：算耗时
    #
    # 2. cost_ms 的计算:
    #    time.time() 返回 Unix 时间戳（秒，浮点数）
    #    两个 time.time() 相减 = 经过的秒数
    #    * 1000 = 毫秒数
    #    int() = 截断小数部分
    #
    # 3. status_code:
    #    路由函数通过 return APIResponse(...) 返回响应时，
    #    FastAPI 将 Pydantic 对象序列化为 JSON，status_code 默认 200
    #    如果抛了异常且被上面的 exception_handler_middleware 捕获，
    #    也返回 200（异常中间件设置的）
    # ================================================================
    start_time = time.time()                # 记录请求到达的时间戳

    # ============================================================
    # await call_next(request) —— 控制权交接点
    # 这行代码的含义："我把请求传给下一个处理者，在原地等它处理完"
    # 下一个处理者可能是：
    #   - 另一个中间件（如果还有没执行的）
    #   - 路由函数（如 rag_chat()）
    # 等它们全部处理完，返回 response 对象
    # ============================================================
    response = await call_next(request)

    # 算耗时
    cost_ms = int((time.time() - start_time) * 1000)

    # 打印日志到终端
    # datetime.datetime.now(): 当前本地时间 → "2026-05-31 14:30:22.123456"
    # .strftime("%Y-%m-%d %H:%M:%S"): 格式化为 "2026-05-31 14:30:22"
    # 生产环境应改为 logging.info() 或结构化日志（如 structlog）
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 日志格式: [时间] 方法 路径 -> 状态码 (耗时)
    # 例: [2026-05-31 14:30:22] POST /chat -> 200 (1523ms)
    print(f"[{now}] {request.method} {request.url.path} -> "
          f"{response.status_code} ({cost_ms}ms)")

    return response
