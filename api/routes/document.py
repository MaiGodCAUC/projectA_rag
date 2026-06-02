"""
文档管理接口 —— 上传、列表、索引、删除

GET    /documents           → 文档列表
POST   /documents/upload    → 上传文档 + 自动索引
DELETE /documents/{doc_id}  → 删除文档 + 清理索引
POST   /documents/index     → 触发重新索引
GET    /documents/status    → 索引状态

----------------------------------------------------------------------
FastAPI 核心用法在本文件中的体现:

1. UploadFile: FastAPI 处理文件上传的方式
   - 自动解析 multipart/form-data 中的文件
   - 支持 await file.read() 异步读取内容
   - 不需要像 Flask 那样手动处理 request.files

2. 路径参数: DELETE /documents/{doc_id}
   {doc_id} 是路径参数 —— FastAPI 自动从 URL 中提取并传给函数

3. 中间件配合: 文件上传/解析/索引中任何异常
   → 被 exception_handler_middleware 捕获 → 统一 JSON 错误返回

----------------------------------------------------------------------
你需要自己写的部分:

文档管理 API 是 RAG 系统的"数据入口"——没有文档就没有知识库。
这里需要处理文件上传、格式校验、解析、索引串联。

学习重点:
1. FastAPI 的 UploadFile 处理 multipart/form-data
2. 文档解析 → 切片 → 索引的完整流水线串联
3. 索引状态管理（哪些文档已索引、哪些未索引）

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

import os
import hashlib
from typing import List

# ================================================================
# FastAPI 导入详解:
#
# APIRouter: 路由分组器（同 health.py 中的用法）
#
# UploadFile: FastAPI 的文件上传处理类
#   - 当参数类型是 UploadFile 时，FastAPI 自动解析 multipart/form-data
#   - file.filename: 原始文件名
#   - file.content_type: MIME 类型
#   - await file.read(): 异步读取文件内容（返回 bytes）
#   - file.size: 文件大小（字节）
#
# File: FastAPI 的文件声明标记
#   - File(...) 告诉 FastAPI "这是文件参数，从 form-data 中取"
#   - File(...) = 必填（"..." 是 Pydantic 的必填标记）
#   - File(default=None) = 可选文件
#   - File() 和 Form()、Body() 类似，都是 FastAPI 的参数类型标记
# ================================================================
from fastapi import APIRouter, UploadFile, File

from api.models import (
    APIResponse, DocumentInfo, DocumentListData,
    ErrorCode,
)

router = APIRouter(tags=["文档管理"])


# =============================================================================
# 配置常量
# =============================================================================

# 允许上传的文件后缀
# set 类型——in 操作是 O(1)，比 list 的 O(n) 快
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

# UPLOAD_DIR: 文件存储目录的绝对路径
# __file__ = api/routes/document.py 的路径
# os.path.dirname 三次向上 = 项目根目录
# 最终路径: <项目根>/data/documents/
UPLOAD_DIR = os.path.join(
    os.path.dirname(                               # 第三层: api/
        os.path.dirname(                            # 第二层: routes/
            os.path.dirname(os.path.abspath(__file__))  # 第一层: document.py 所在目录
        )
    ),
    "data", "documents"                            # 拼接 data/documents/
)


# =============================================================================
# GET /documents —— 文档列表 + 索引状态
# =============================================================================

# @router.get("/documents")
# 当客户端 GET /documents 时触发
# 不需要参数 → 直接返回所有文档的列表
@router.get("/documents")
async def list_documents():
    """获取所有文档列表

    遍历 data/documents/ 目录，返回每个文件的：
    - 文件名、大小
    - 索引状态（是否已索引、切块数量）

    响应示例:
        {
          "code": 0,
          "data": {
            "total": 10,
            "documents": [
              {"id": "abc123", "file_name": "托运行李运输规定.md", ...},
              ...
            ]
          }
        }
    """
    # ================================================================
    # TODO(用户): 手写文档列表 + 索引状体查询逻辑
    # ================================================================
    # ================================================================
    # 学习要点:
    #
    # 1. os.listdir(UPLOAD_DIR): 列出目录下所有文件名
    #    sorted() 排序后输出，保证每次返回顺序一致
    #
    # 2. hashlib.md5(filename.encode()).hexdigest()[:8]:
    #    filename.encode() → 把字符串转为字节（md5 只能处理 bytes）
    #    hashlib.md5(...) → 创建 MD5 哈希对象
    #    .hexdigest() → 转为 32 字符的十六进制字符串（如 "a1b2c3d4e5f6..."）
    #    [:8] → 取前 8 字符作为短 ID
    #
    # 3. ANY vs ALL:
    #    any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    #    → filename 只要匹配任何一个允许的扩展名就返回 True
    #    all(...) 的话需要匹配所有扩展名（不可能），所以用 any
    #
    # 4. DocumentInfo(...).model_dump():
    #    构造 Pydantic 对象后立即转为 dict
    #    原因: 前端只需要 JSON，不需要 Pydantic 对象的额外能力
    # ================================================================

    docs = []   # 存放 DocumentInfo dict 的列表

    # os.path.isdir(): 检查路径是否存在且为目录
    if os.path.isdir(UPLOAD_DIR):
        # 遍历文件（sorted 保证顺序一致）
        for filename in sorted(os.listdir(UPLOAD_DIR)):
            # 过滤掉不支持的文件格式（如 .py、.json）
            # any(): 任意一个条件满足就返回 True
            if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue

            # 拼接完整路径，获取文件大小
            filepath = os.path.join(UPLOAD_DIR, filename)
            file_size = os.path.getsize(filepath)    # 单位: 字节

            # 生成文档 ID（取文件名 MD5 的前 8 位）
            doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]

            # 构造 DocumentInfo 对象并转为 dict
            # indexed=False, chunk_count=0 是占位——索引状态需要查 Qdrant 才能确定
            docs.append(DocumentInfo(
                id=doc_id,
                file_name=filename,
                file_size=file_size,
                indexed=False,         # TODO: 从索引状态文件/Qdrant 查询真实状态
                chunk_count=0          # TODO: 同上
            ).model_dump())

    # 返回统一格式
    # DocumentListData(...).model_dump() → {"total": n, "documents": [...]}
    return APIResponse.ok(
        data=DocumentListData(total=len(docs), documents=docs).model_dump()
    )


# =============================================================================
# POST /documents/upload —— 上传文档
# =============================================================================

# ================================================================
# async def upload_document(file: UploadFile = File(...)):
#
# UploadFile: FastAPI 的异步文件处理类
#   - 当请求是 multipart/form-data 时，FastAPI 自动把文件字段
#     解析为 UploadFile 对象
#   - await file.read() 异步读取内容（不阻塞事件循环）
#   - file.filename: 客户端上传的原始文件名
#
# File(...): FastAPI 的依赖标记
#   - "..."（省略号）: 必填
#   - File(default=None): 可选
#   - 和 Query()、Body()、Path() 一样，都是 FastAPI 的参数类型提示
# ================================================================
@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并自动索引

    TODO(用户): 手写文档上传 + 自动索引逻辑

    流程:
    1. 校验文件格式（仅允许 .pdf / .docx / .md / .txt）
    2. 保存到 data/documents/
    3. 调用 rag/loader.py 解析文档
    4. 调用 rag/splitter.py 切片
    5. 调用索引流水线写入 Qdrant 和 BM25
    6. 返回索引结果（切块数、耗时）

    Args:
        file: 通过 multipart/form-data 上传的文件

    curl 调用示例:
        curl -X POST http://localhost:8000/documents/upload \
          -F "file=@/path/to/行李运输规定.md"
    """
    # ================================================================
    # TODO(用户): 手写文档上传 + 索引逻辑
    # ================================================================
    # ================================================================
    # 学习要点:
    #
    # 步骤 1: os.path.splitext(file.filename)
    #   ("行李运输规定", ".md") → [1] 取后缀 → .lower() → ".md"
    #
    # 步骤 2: await file.read()
    #   async 读取文件内容到内存
    #   小文件没问题，大文件应改为流式写入（shutil.copyfileobj）
    #
    # 步骤 3: 解析 → 切片 → 索引
    #   这是 RAG 系统的核心流水线（TODO 用户实现）
    #   load_document(): 把文件解析为 ParsedDocument
    #   get_splitter("policy_clause"): 用条款感知切片器
    #   IndexingPipeline.index_incremental(): 写入 Qdrant + BM25
    # ================================================================

    # ---- 步骤 1: 格式校验 ----
    # os.path.splitext(file.filename):
    #   把文件名拆成 (名称, 后缀)
    #   例: "行李规定.md" → ("行李规定", ".md")
    #   [1] 取后缀, .lower() 转小写（防止 .PDF 不匹配 .pdf）
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""

    # 检查后缀是否在允许列表中
    if ext not in ALLOWED_EXTENSIONS:
        return APIResponse.fail(
            code=ErrorCode.PARAM_ERROR,
            detail=f"不支持的文件格式: {ext}，仅支持: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # ---- 步骤 2: 保存文件 ----
    # os.makedirs(..., exist_ok=True): 递归创建目录
    #   exist_ok=True = 目录已存在时不报错（幂等操作）
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # 拼接保存路径
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # await file.read(): 异步读取上传文件的全部内容 → bytes
    content = await file.read()

    # "wb" = 写二进制模式
    # f.write(content): 把 bytes 写入磁盘
    with open(file_path, "wb") as f:
        f.write(content)

    # ---- 步骤 3: 暂时返回"已保存"，索引功能待实现 ----
    # TODO(用户): 在这里串联解析 → 切片 → 索引
    return APIResponse.ok(data={
        "file_name": file.filename,
        "file_size": len(content),                   # len(bytes) = 字节数
        "status": "已保存。索引功能待实现(TODO用户)",
        "tip": "请在 document.py 的 upload_document 中实现 解析→切片→索引 流水线",
    })


# =============================================================================
# DELETE /documents/{doc_id} —— 删除文档
# =============================================================================

# ================================================================
# @router.delete("/documents/{doc_id}")
# {doc_id} 是路径参数（Path Parameter）
#
# FastAPI 路径参数工作机制:
#   1. URL 中 {doc_id} 声明这是一个路径参数
#   2. 函数参数 doc_id: str 同名的参数会自动接收 URL 中的值
#   3. 例: DELETE /documents/abc123 → doc_id = "abc123"
#
# 和查询参数的区别:
#   路径参数: /documents/abc123        → DELETE /documents/{doc_id}
#   查询参数: /documents?id=abc123     → DELETE /documents?doc_id=abc123
#   路径参数是 URL 的一部分（必填），查询参数在 ? 后面（可设默认值）
# ================================================================
@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除指定文档及对应的向量索引

    TODO(用户): 手写文档删除逻辑

    流程:
    1. 根据 doc_id 找到对应的文件
    2. 从 Qdrant 中删除该文档的所有向量
    3. 从 BM25 索引中移除（BM25 不支持单条删除，需重建索引）
    4. 删除磁盘上的文件
    """
    # ================================================================
    # TODO(用户): 手写文档删除逻辑
    # ================================================================
    # ================================================================
    # 学习要点:
    #
    # 1. doc_id 的匹配:
    #    doc_id 是文件名的 MD5 前 8 位
    #    需要遍历目录，对每个文件名算 MD5，找到匹配的
    #
    # 2. Qdrant 删除:
    #    VectorStore.delete_by_source(filename) 删除某文档的所有向量
    #    底层用 Qdrant 的 payload 过滤 + 批量删除
    #
    # 3. BM25 删除:
    #    BM25 没有"删除单条"的能力（倒排索引不支持增量删除）
    #    删除文档后需要重建整个 BM25 索引
    #    → 调用 bm25.index(all_remaining_chunks)
    # ================================================================
    return APIResponse.fail(
        code=ErrorCode.PARAM_ERROR,
        detail="删除功能待实现(TODO用户)。请参考注释中的实现逻辑。"
    )


# =============================================================================
# POST /documents/index —— 触发重新索引
# =============================================================================

@router.post("/documents/index")
async def reindex_all():
    """重新索引所有文档

    适用场景:
    - 切换了 Embedding 模型（向量维度变了，旧数据不兼容）
    - 调整了切片策略（chunk_size/overlap 变了，需要重新切）
    - Qdrant 数据丢失（容器被删重建）

    流程:
    1. 读取 data/documents/ 下所有文档
    2. 调用 loader → splitter → indexing_pipeline
    3. 返回索引结果统计
    """
    # TODO(用户): 实现全量重新索引逻辑
    return APIResponse.fail(
        code=ErrorCode.PARAM_ERROR,
        detail="索引功能待实现(TODO用户)"
    )


# =============================================================================
# GET /documents/status —— 索引状态
# =============================================================================

@router.get("/documents/status")
async def index_status():
    """查询当前索引状态"""
    # TODO(用户): 从 Qdrant 查询真实索引状态
    return APIResponse.ok(data={
        "indexed_count": 0,
        "total_chunks": 0,
        "last_indexed_at": None,
        "message": "请先在 document.py 中实现索引逻辑",
    })
