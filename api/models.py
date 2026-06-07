"""
API 统一请求/响应模型

所有接口使用同一个响应格式：
    {"code": 0, "data": {...}, "trace_id": "a1b2c3d4", "cost_ms": 123}

----------------------------------------------------------------------
FastAPI 的 Pydantic 模型是怎么工作的？

FastAPI 用 Pydantic 做两件事：
1. 请求校验——客户端发来的 JSON 自动转成 Python 对象，字段类型不符
   直接返回 422 错误，不会让错误数据进入业务逻辑。
2. 响应序列化——return 一个 Pydantic 对象，FastAPI 自动转成 JSON 字符串。

核心概念：
- BaseModel: 所有模型的基类，继承它就有了自动校验 + 序列化能力
- Field(): 给字段加约束和元信息，这些信息会出现在 Swagger 文档里
- model_dump(): Pydantic v2 的方法，把模型转成 Python dict（准备序列化为 JSON）

面试话术:
"我设计了一套统一的 API 响应规范，所有接口返回相同的外层结构。
前端只需写一套解析逻辑。code=0 表示成功，非 0 查看错误码表。
每个响应都带 trace_id 和 cost_ms，方便排查慢请求。"
----------------------------------------------------------------------
"""

# uuid: Python 标准库，用于生成全局唯一 ID
# uuid4() 基于随机数生成，如 "a1b2c3d4-e5f6-4321-1234-567890abcdef"
import uuid

# typing: Python 类型提示模块
# Any: 任意类型，用于 data 字段（不同接口返回不同类型的数据）
# Optional[str]: 等价于 str | None，表示"字符串或 None"
from typing import Any

# pydantic: FastAPI 的数据建模库
# BaseModel: 所有请求/响应模型的基类
#   - 继承后自动获得 JSON → Python 对象 的解析能力
#   - 字段类型不匹配时自动抛出 ValidationError（FastAPI 转成 422 响应）
#   - def __init__(self, **data) 自动生成构造函数
# Field: 给字段附加元信息的函数
#   - default: 默认值
#   - default_factory: 默认值的工厂函数（每次实例化时调用，避免可变默认值问题）
#   - min_length/max_length: 字符串长度约束
#   - ge/le: 数值范围约束（greater than or equal / less than or equal）
#   - description: Swagger 文档中的字段说明
from pydantic import BaseModel, Field


# =============================================================================
# APIResponse —— 统一响应外层（所有接口都用它包一层）
# =============================================================================

class APIResponse(BaseModel):
    """统一 API 响应 —— 所有接口共用的外层包装

    为什么需要统一响应格式？
    前端只需要写一套解析逻辑：
        const res = await fetch('/chat', ...)
        const data = await res.json()
        if (data.code === 0) {
            handleSuccess(data.data)
        } else {
            handleError(data.code, data.data.detail)
        }
    不会因为不同接口返回不同字段名而需要写多套解析代码。

    示例:
        APIResponse(
            code=0,
            data={"answer": "根据行李运输规定..."},
            trace_id="a1b2c3d4",
            cost_ms=1523,
        )
    """

    # code: 业务状态码（不是 HTTP 状态码！）
    # 为什么 0 表示成功？业界惯例：0 = 无错误，非 0 = 有错误
    # 这样前端不需要区分 HTTP 200/400/500，只需要看 code 字段
    code: int = Field(
        default=0,                         # 默认成功
        description="状态码。0=成功，1001=参数错误，1002=文档解析失败，"
                    "1003=检索失败，1004=生成失败，9999=未知错误",
    )

    # data: 实际业务数据，类型因接口而异
    # Any 类型 = 不限制，可以是 dict / list / str / None
    # 不同接口的 data 字段结构不同（见下方的 ChatData / DocumentListData 等子模型）
    data: Any = Field(
        default=None,
        description="响应数据，类型因接口而异",
    )

    # trace_id: 请求追踪 ID
    # default_factory 的作用：每次创建 APIResponse 实例时，都会执行 lambda 生成新 ID
    # 为什么不用 default=str(uuid.uuid4())[:8]？
    #   → 那是模块加载时执行一次，所有实例共享同一个值（Python 的经典陷阱）
    # default_factory 确保每次实例化都重新调用
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        # uuid4() → "a1b2c3d4-e5f6-4321-1234-567890abcdef"
        # str()[:8] → "a1b2c3d4"（只取前 8 位，够用了）
        description="请求追踪 ID，可据此在 LangSmith 中查找完整调用链",
    )

    # cost_ms: 请求处理耗时
    # 每个接口在处理完后记录从收到请求到返回响应的毫秒数
    cost_ms: int = Field(
        default=0,
        description="请求处理耗时（毫秒）",
    )

    # =========================================================================
    # @classmethod 快捷工厂方法
    # =========================================================================
    # 为什么需要工厂方法？
    # 避免每个路由函数都要写一长串 APIResponse(code=0, data=..., trace_id=...)
    # 用 APIResponse.ok(data) 一行搞定

    @classmethod
    def ok(cls, data: Any = None, trace_id: str = "", cost_ms: int = 0) -> "APIResponse":
        """快捷构造成功响应

        使用方式:
            return APIResponse.ok(data={"answer": "..."}, trace_id="abc")

        等价于:
            return APIResponse(code=0, data={"answer": "..."}, trace_id="abc")
        """
        # cls 在这里就是 APIResponse 类本身
        # cls(code=0, ...) = APIResponse(code=0, ...)
        return cls(code=0, data=data, trace_id=trace_id, cost_ms=cost_ms)

    @classmethod
    def fail(cls, code: int, detail: str, trace_id: str = "") -> "APIResponse":
        """快捷构造失败响应

        使用方式:
            return APIResponse.fail(code=ErrorCode.SEARCH_ERROR, detail="Qdrant 连接超时")

        等价于:
            return APIResponse(
                code=1003,
                data={"detail": "Qdrant 连接超时"},
                trace_id="abc"
            )
        """
        return cls(
            code=code,
            data={"detail": detail},               # 错误信息统一放在 data.detail 里
            trace_id=trace_id or str(uuid.uuid4())[:8],
            # trace_id 为空时自动生成一个新的
        )


# =============================================================================
# 请求模型 —— 定义客户端需要发送的 JSON 结构
# =============================================================================

class ChatRequest(BaseModel):
    """RAG 对话请求

    FastAPI 在路由函数中声明参数类型为 ChatRequest 时：
    1. 自动读取请求体的 JSON
    2. 按 ChatRequest 的字段定义做校验
    3. 校验通过后实例化为 ChatRequest 对象，传给路由函数
    4. 校验失败 → 自动返回 422，附带 "message 长度不能超过 5000" 等详细信息

    示例:
        {
          "message": "旅客行李箱摔坏了怎么赔？",
          "top_k": 5,
          "stream": false
        }
    """

    # Field(...) 中的 "..."（省略号）是 Pydantic 的特殊标记，表示"必填"
    # 等价于 Field(required=True)，但没有显式默认值
    message: str = Field(
        ...,                                           # ← 必填字段
        min_length=1,                                  # 最短 1 个字符（不能是空字符串）
        max_length=5000,                               # 最长 5000 字符（防止滥用）
        description="员工提问内容",
    )

    # ge=1, le=20: Pydantic 的数值范围校验
    # ge = greater than or equal（大于等于）
    # le = less than or equal（小于等于）
    top_k: int = Field(
        default=5,                                     # 默认返回 Top-5
        ge=1, le=20,                                   # 限制 1~20 之间
        description="注入 Prompt 的检索结果数量（1-20）",
    )

    stream: bool = Field(
        default=False,                                 # 默认非流式
        description="是否使用 SSE 流式输出",
    )


# =============================================================================
# 响应 data 子模型 —— 定义 response.data 字段的内部结构
# =============================================================================
# 为什么需要这些子模型？
# 父模型 APIResponse 的 data 字段是 Any 类型（太宽泛了）
# 子模型让 Swagger 文档能展示每个接口的具体响应结构
# 也方便路由函数用 Pydantic 校验自己构造的返回数据

class ChatData(BaseModel):
    """/chat 响应的 data 字段 —— 告诉前端这次对话的完整信息"""
    question: str = Field(
        description="用户原始问题"                       # 把原始问题回传，方便前端展示
    )
    answer: str = Field(
        description="RAG 增强回答（含 [来源: ...] 引用标记）"
    )
    citations: list[dict] = Field(
        default_factory=list,                          # 默认空列表
        # default_factory=list 而不是 default=[]
        # 原因和前面的 uuid 一样：避免所有实例共享同一个 list 对象
        description="引用溯源列表 [{doc_name, clause_id, section_title, original_text}]",
    )
    retrieval_count: int = Field(
        default=0,
        description="检索命中数",                       # 用于监控检索是否正常工作
    )


class DocumentInfo(BaseModel):
    """单条文档信息 —— /documents 接口返回的列表中每个元素的结构"""
    id: str = Field(
        description="文档 ID（文件名 MD5 前 8 位）"      # 用于前端做删除/查看操作
    )
    file_name: str = Field(
        description="文件名（含扩展名）"
    )
    file_size: int = Field(
        description="文件大小（字节）"                   # 前端可格式化为 KB/MB 显示
    )
    indexed: bool = Field(
        description="是否已索引"                         # 已索引 = 可以被 RAG 检索到
    )
    chunk_count: int = Field(
        default=0,
        description="切块数量"                           # 索引后将文档切成了多少片段
    )


class DocumentListData(BaseModel):
    """/documents 响应的 data 字段"""
    total: int = Field(
        description="文档总数"
    )
    documents: list[dict] = Field(
        default_factory=list,
        description="文档信息列表（DocumentInfo 对象的 dict）"
    )


class EvalData(BaseModel):
    """评估报告数据 —— /eval/report 响应的 data 字段"""
    total_samples: int = Field(description="评估样本数")
    avg_faithfulness: float = Field(description="平均忠实度（0~1）")
    avg_answer_relevancy: float = Field(description="平均回答相关性（0~1）")
    avg_context_precision: float = Field(description="平均上下文精确度（0~1）")
    avg_context_recall: float = Field(description="平均上下文召回率（0~1）")


# =============================================================================
# 错误码常量类 —— 统一管理所有业务错误码
# =============================================================================
# 为什么用类而不是模块级常量？
# 1. IDE 有自动补全（输入 ErrorCode. 就会列出所有错误码）
# 2. 命名空间隔离（不会和模块中其他变量冲突）
# 3. 面试时能讲"我设计了错误码体系"而不是"我定义了几个常量"

class ErrorCode:
    """统一错误码 —— 所有业务错误通过 code 字段而非 HTTP 状态码表达"""
    SUCCESS = 0                  # 成功——HTTP 200
    PARAM_ERROR = 1001           # 参数校验失败——如 message 为空、top_k 超范围
    DOC_PARSE_ERROR = 1002       # 文档解析失败——如 PDF 文件损坏
    SEARCH_ERROR = 1003          # 检索超时/失败——如 Qdrant 连接断开
    LLM_ERROR = 1004             # LLM 生成失败——如 API Key 无效、调用超时
    DOC_NOT_INDEXED = 1005       # 文档未索引——如知识库为空时发起查询
    UNKNOWN_ERROR = 9999         # 未知错误——catch 到未预期的异常时用
