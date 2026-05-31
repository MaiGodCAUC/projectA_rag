"""
文档管理接口 —— 上传、列表、索引、删除

GET    /documents           → 文档列表 + 索引状态
POST   /documents/upload    → 上传文档 + 自动索引
DELETE /documents/{doc_id}  → 删除文档 + 清理索引
POST   /documents/index     → 触发重新索引

----------------------------------------------------------------------
## 你需要自己写的部分

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
from fastapi import APIRouter, UploadFile, File

from api.models import (
    APIResponse, DocumentInfo, DocumentListData,
    IndexStatusData, ErrorCode,
)

router = APIRouter(tags=["文档管理"])


# =============================================================================
# 配置
# =============================================================================

# 允许上传的文件格式
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

# 上传文件存储目录
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data", "documents")


# =============================================================================
# GET /documents —— 文档列表 + 索引状态
# =============================================================================

@router.get("/documents")
async def list_documents():
    """获取所有文档列表及索引状态

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
    # TODO(用户): 手写文档列表逻辑
    # ================================================================
    #
    # 实现参考:
    #
    # docs = []
    # for filename in os.listdir(UPLOAD_DIR):
    #     if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
    #         continue
    #     filepath = os.path.join(UPLOAD_DIR, filename)
    #     file_size = os.path.getsize(filepath)
    #     doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
    #
    #     # TODO: 检查是否已索引（需要查询 Qdrant / 索引状态文件）
    #     docs.append({
    #         "id": doc_id,
    #         "file_name": filename,
    #         "file_size": file_size,
    #         "indexed": False,    # 需要从索引状态文件读取
    #         "chunk_count": 0,
    #     })
    #
    # return APIResponse.ok(data={"total": len(docs), "documents": docs})
    #
    # ================================================================
    docs = []
    if os.path.isdir(UPLOAD_DIR):
        for filename in sorted(os.listdir(UPLOAD_DIR)):
            if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue
            filepath = os.path.join(UPLOAD_DIR, filename)
            file_size = os.path.getsize(filepath)
            doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]
            docs.append(DocumentInfo(
                id=doc_id, file_name=filename, file_size=file_size,
                indexed=False, chunk_count=0
            ).model_dump())

    return APIResponse.ok(data=DocumentListData(
        total=len(docs), documents=docs
    ).model_dump())


# =============================================================================
# POST /documents/upload —— 上传文档
# =============================================================================

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并自动索引

    TODO(用户): 手写文档上传 + 自动索引逻辑

    流程:
    1. 校验文件格式（仅允许 .pdf / .docx / .md / .txt）
    2. 保存到 data/documents/
    3. 调用 rag/loader.py 解析文档
    4. 调用 rag/splitter.py 切片
    5. 调用索引流水线写入 Qdrant
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
    #
    # 实现参考:
    #
    # # ---- 步骤 1: 格式校验 ----
    # ext = os.path.splitext(file.filename)[1].lower()
    # if ext not in ALLOWED_EXTENSIONS:
    #     return APIResponse.fail(
    #         code=ErrorCode.PARAM_ERROR,
    #         detail=f"不支持的文件格式: {ext}，仅支持: {', '.join(ALLOWED_EXTENSIONS)}"
    #     )
    #
    # # ---- 步骤 2: 保存文件 ----
    # os.makedirs(UPLOAD_DIR, exist_ok=True)
    # file_path = os.path.join(UPLOAD_DIR, file.filename)
    # content = await file.read()
    # with open(file_path, "wb") as f:
    #     f.write(content)
    #
    # # ---- 步骤 3: 解析 → 切片 → 索引 ----
    # from rag.loader import load_document
    # from rag.splitter import get_splitter
    # from rag.indexing_pipeline import IndexingPipeline
    #
    # doc = load_document(file_path)
    # splitter = get_splitter("policy_clause")
    # chunks = splitter.split(doc)
    #
    # pipeline = IndexingPipeline()
    # result = pipeline.index_incremental(chunks)
    #
    # return APIResponse.ok(data={
    #     "file_name": file.filename,
    #     "chunk_count": len(chunks),
    #     "indexed_chunks": result.get("indexed", 0),
    #     "status": "已完成索引",
    # })
    #
    # ================================================================
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return APIResponse.fail(
            code=ErrorCode.PARAM_ERROR,
            detail=f"不支持的文件格式: {ext}，仅支持: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return APIResponse.ok(data={
        "file_name": file.filename,
        "file_size": len(content),
        "status": "已保存。索引功能待实现(TODO用户)",
        "tip": "请在 document.py 的 upload_document 中实现 解析→切片→索引 流水线",
    })


# =============================================================================
# DELETE /documents/{doc_id} —— 删除文档
# =============================================================================

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除指定文档及对应的向量索引

    TODO(用户): 手写文档删除逻辑

    流程:
    1. 根据 doc_id 找到对应的文件
    2. 从 Qdrant 中删除该文档的所有向量
    3. 从 BM25 索引中移除
    4. 删除磁盘上的文件
    """
    # ================================================================
    # TODO(用户): 手写文档删除逻辑
    # ================================================================
    #
    # 实现参考:
    #
    # # 在 UPLOAD_DIR 中找到 doc_id 对应的文件
    # target_file = None
    # for filename in os.listdir(UPLOAD_DIR):
    #     if hashlib.md5(filename.encode()).hexdigest()[:8] == doc_id:
    #         target_file = filename
    #         break
    #
    # if not target_file:
    #     return APIResponse.fail(code=ErrorCode.PARAM_ERROR, detail="文档不存在")
    #
    # # 从 Qdrant 删除
    # from rag.vector_store import VectorStore
    # vs = VectorStore()
    # vs.delete_by_source(target_file)
    #
    # # 从 BM25 删除（需要重新索引——BM25 不支持单条删除）
    #
    # # 删除文件
    # file_path = os.path.join(UPLOAD_DIR, target_file)
    # os.remove(file_path)
    #
    # return APIResponse.ok(data={"deleted": target_file})
    #
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
    - Embedding 模型切换后
    - 切片策略调整后
    - Qdrant 数据丢失后
    """
    return APIResponse.fail(
        code=ErrorCode.PARAM_ERROR,
        detail="索引功能待实现(TODO用户)"
    )


# =============================================================================
# GET /documents/status —— 索引状态
# =============================================================================

@router.get("/documents/status")
async def index_status():
    """查询索引状态"""
    return APIResponse.ok(data={
        "indexed_count": 0,
        "total_chunks": 0,
        "last_indexed_at": None,
        "message": "请先在 document.py 中实现索引逻辑",
    })
