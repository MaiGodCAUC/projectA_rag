"""
API 统一请求/响应模型

所有接口使用同一个响应格式：
    {"code": 0, "data": {...}, "trace_id": "a1b2c3d4", "cost_ms": 123}

面试话术:
"我设计了一套统一的 API 响应规范，所有接口返回相同的外层结构。
前端只需写一套解析逻辑。code=0 表示成功，非 0 查看错误码表。
每个响应都带 trace_id 和 cost_ms，方便排查慢请求。"
"""

import uuid
from typing import Any, Optional
from pydantic import BaseModel, Field


# =============================================================================
# 统一响应外层
# =============================================================================

class APIResponse(BaseModel):
    """统一 API 响应 —— 所有接口共用的外层包装

    示例:
        APIResponse(
            code=0,
            data={"answer": "根据行李运输规定..."},
            trace_id="a1b2c3d4",
            cost_ms=1523,
        )
    """
    code: int = Field(
        default=0,
        description="状态码。0=成功，1001=参数错误，1002=文档解析失败，1003=检索失败，1004=生成失败，9999=未知错误",
    )
    data: Any = Field(
        default=None,
        description="响应数据，类型因接口而异",
    )
    trace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="请求追踪 ID，可据此在 LangSmith 中查找完整调用链",
    )
    cost_ms: int = Field(
        default=0,
        description="请求处理耗时（毫秒）",
    )

    @classmethod
    def ok(cls, data: Any = None, trace_id: str = "", cost_ms: int = 0) -> "APIResponse":
        """快捷构造成功响应"""
        return cls(code=0, data=data, trace_id=trace_id, cost_ms=cost_ms)

    @classmethod
    def fail(cls, code: int, detail: str, trace_id: str = "") -> "APIResponse":
        """快捷构造失败响应"""
        return cls(
            code=code,
            data={"detail": detail},
            trace_id=trace_id or str(uuid.uuid4())[:8],
        )


# =============================================================================
# 请求模型
# =============================================================================

class ChatRequest(BaseModel):
    """RAG 对话请求

    示例:
        {
          "message": "旅客行李箱摔坏了怎么赔？",
          "top_k": 5,
          "stream": false
        }
    """
    message: str = Field(
        ...,                           # "..." = 必填
        min_length=1,
        max_length=5000,
        description="员工提问内容",
    )
    top_k: int = Field(
        default=5,
        ge=1, le=20,                   # ge = greater than or equal, le = less than or equal
        description="注入 Prompt 的检索结果数量（1-20）",
    )
    stream: bool = Field(
        default=False,
        description="是否使用 SSE 流式输出",
    )


class DocumentUploadRequest(BaseModel):
    """文档上传请求（通过 multipart/form-data，非 JSON body）

    实际文件通过 File Upload 控件上传，
    此模型用于 Swagger 文档展示预期字段。
    """
    file_name: Optional[str] = Field(
        default=None,
        description="上传的文件名（自动从原始文件名获取）",
    )


# =============================================================================
# 响应 data 子模型
# =============================================================================

class ChatData(BaseModel):
    """/chat 响应的 data 字段"""
    question: str = Field(description="用户原始问题")
    answer: str = Field(description="RAG 增强回答（含 [来源: ...] 引用标记）")
    citations: list = Field(
        default_factory=list,
        description="引用溯源列表 [{doc_name, clause_id, section_title, original_text}]",
    )
    retrieval_count: int = Field(
        default=0,
        description="检索命中数",
    )


class DocumentInfo(BaseModel):
    """文档信息"""
    id: str = Field(description="文档 ID（文件名 MD5）")
    file_name: str = Field(description="文件名（含扩展名）")
    file_size: int = Field(description="文件大小（字节）")
    indexed: bool = Field(description="是否已索引")
    chunk_count: int = Field(default=0, description="切块数量")


class DocumentListData(BaseModel):
    """/documents 响应"""
    total: int = Field(description="文档总数")
    documents: list = Field(default_factory=list, description="文档信息列表")


class IndexStatusData(BaseModel):
    """索引状态"""
    indexed_count: int = Field(description="已索引文档数")
    total_chunks: int = Field(description="总切块数")
    last_indexed_at: Optional[str] = Field(default=None, description="最近一次索引时间")


class EvalData(BaseModel):
    """评估报告数据"""
    total_samples: int = Field(description="评估样本数")
    avg_faithfulness: float = Field(description="平均忠实度")
    avg_answer_relevancy: float = Field(description="平均回答相关性")
    avg_context_precision: float = Field(description="平均上下文精确度")
    avg_context_recall: float = Field(description="平均上下文召回率")


# =============================================================================
# 错误码常量
# =============================================================================

class ErrorCode:
    """统一错误码"""
    SUCCESS = 0
    PARAM_ERROR = 1001              # 参数校验失败
    DOC_PARSE_ERROR = 1002          # 文档解析失败
    SEARCH_ERROR = 1003             # 检索超时/失败
    LLM_ERROR = 1004                # LLM 生成失败
    DOC_NOT_INDEXED = 1005          # 文档未索引
    UNKNOWN_ERROR = 9999            # 未知错误
