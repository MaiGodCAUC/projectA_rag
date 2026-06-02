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

    # ------------------------------------------------------------------
    # 步骤 3: 解析 → 切片 → 索引 流水线
    # ------------------------------------------------------------------
    # TODO(用户): 手写文档上传的索引流水线
    #
    # ┌──────────────────────────────────────────────────────────────────┐
    # │                       索引流水线全景                              │
    # │                                                                   │
    # │   file_path (.md/.pdf/.docx)                                     │
    # │        │                                                          │
    # │        ▼                                                          │
    # │   ┌──────────┐  load_document() 把文件解析为 ParsedDocument      │
    # │   │ ① 解析   │   内部根据后缀选择 PDFLoader/MDLoader/DocxLoader  │
    # │   └────┬─────┘   输出: ParsedDocument(元数据 + sections + tables) │
    # │        │                                                          │
    # │        ▼                                                          │
    # │   ┌──────────┐  get_splitter("policy_clause").split(doc)         │
    # │   │ ② 切片   │   条款感知切片，保证每条条款自成 chunk              │
    # │   └────┬─────┘   输出: List[TextChunk] (每个 chunk 含 content +    │
    # │        │           clause_id + section_title + source_file)       │
    # │        │                                                          │
    # │        ▼                                                          │
    # │   ┌──────────┐  vector_store.upsert_chunks(chunks)                │
    # │   │ ③ 向量化 │   每条 chunk → Embedding → 写入 Qdrant            │
    # │   │  + 索引  │   bm25.index(chunks) → 写入 BM25 倒排索引         │
    # │   └──────────┘   输出: 写入成功的 chunk 数量                       │
    # │                                                                   │
    # └──────────────────────────────────────────────────────────────────┘
    #
    # ================================================================
    # 实现方案 A（推荐，利用已有 IndexingPipeline）:
    #
    # 最简单的方式: 文件保存后，调 pipeline.index_incremental()。
    # IndexingPipeline 自动扫描目录、计算 SHA256 哈希、只索引新增/变更文档。
    # ================================================================
    #
    # from rag.indexing_pipeline import IndexingPipeline
    # from rag.vector_store import VectorStore
    #
    # # IndexingPipeline(向量存储, 切片策略)
    # pipeline = IndexingPipeline(
    #     vector_store=VectorStore(),
    #     splitter_strategy="policy_clause",
    # )
    #
    # # index_incremental() 扫描 UPLOAD_DIR，对比 SHA256 哈希
    # # 新文件/变更文件 → 解析 → 切片 → Embedding → Qdrant
    # # 未变更文件 → 跳过
    # result = pipeline.index_incremental(UPLOAD_DIR)
    #
    # return APIResponse.ok(data={
    #     "file_name": file.filename,
    #     "file_size": len(content),
    #     "status": "已完成索引",
    #     "total_chunks": result.get("total_chunks", 0),
    #     "indexed_count": result.get("indexed", 0),
    # })
    #
    # ================================================================
    # 实现方案 B（手动串联每一步，理解内部细节）:
    # ================================================================
    #
    import time
    from rag.loader import load_document
    from rag.splitter import get_splitter
    from rag.vector_store import VectorStore
    from rag.bm25 import BM25Retriever

    start_time = time.time()
    #
    # # load_document() 内部根据后缀选择 Loader:
    # #   .md/.txt → MDLoader     → 正则提取标题层级 + 段落内容
    # #   .pdf     → PDFLoader    → PyMuPDF 提取文本 + pdfplumber 提取表格
    # #   .docx    → DocxLoader   → python-docx 解析段落/表格
    # # 返回 ParsedDocument(sections=[...], tables=[...], metadata={...})
    # ---- ③-1: 解析文档 ----
    try:
        doc = load_document(file_path)
    except Exception as e:
        return APIResponse.fail(
            code = ErrorCode.DOC_PARSE_ERROR,
            detail=f"文档解析失败：{str(e)}"
        )
    # # ---- ③-2: 条款感知切片 ----
    # # get_splitter("policy_clause") → PolicyClauseSplitter 实例
    # # .split(doc) → List[TextChunk]
    # # 每个 TextChunk 自动附带:
    # #   clause_id="第3.2条"     ← 从文本中识别的条款编号
    # #   section_title="破损赔偿"  ← 从 doc.sections 中继承的章节标题
    # #   source_file="托运行李运输规定.md"
    # splitter = get_splitter("policy_clause")
    # chunks = splitter.split(doc)
    splitter = get_splitter("policy_clause")
    chunks = splitter.split(doc)

    if not chunks:
        return APIResponse.fail(
            code = ErrorCode.DOC_PARSE_ERROR,
            detail="文档切片后为空，请检查文档内容"
        )
    # # ---- ③-3: 写入向量索引 (Qdrant) ----
    # # upsert_chunks() 内部流程:
    # #   ① 取每个 chunk.content → Embedding 模型编码 → 1024 维向量
    # #   ② 构造 PointStruct(id=chunk_id, vector=向量, payload={元信息})
    # #   ③ 批量写入 Qdrant（已存在的 chunk_id → 更新向量）
    vs = VectorStore()
    try:
        vs.upsert_chunks(chunks)
    except Exception as e:
        return APIResponse.fail(
            code = ErrorCode.SEARCH_ERROR,
            detail=f"向量索引写入失败：{str(e)}"
        )
    # # ---- ③-4: 写入关键词索引 (BM25) ----
    # # BM25 用于精确匹配查询（"CA1234"、"第3.2条"、"Y舱"等）
    # # add_documents() 内部:
    # #   ① jieba 分词 → ["托运行李", "损坏", "赔偿", "标准"]
    # #   ② 增量构建倒排索引（词 → 文档列表）
    bm25 = BM25Retriever()
    try:
        bm25.add_documents(chunks)
    except Exception as e:
        # BM25 失败不致命——向量检索还能用
        print(f"[警告] BM25 索引写入失败: {e}")

    cost_ms = int((time.time() - start_time) * 1000)

    # ================================================================
    # 当前占位（替换为方案 A 或 B）
    # ================================================================
    return APIResponse.ok(data={
        "file_name": file.filename,
        "file_size": len(content),                   # len(bytes) = 字节数
        "chunk_count": len(chunks),
        "cost_ms": cost_ms,
        "status": "已完成索引",
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
    # 实现参考:
    #
    # 删除文档涉及三个存储层的清理:
    # ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    # │ Qdrant 向量  │    │ BM25 倒排   │    │ 磁盘文件     │
    # │ (按 source   │    │ (不支持单条  │    │ (os.remove) │
    # │  filter删除) │    │  删除，需重建)│    │             │
    # └─────────────┘    └─────────────┘    └─────────────┘
    # ================================================================
    #
    # ---- 步骤 1: 根据 doc_id 找到对应的文件名 ----
    # # 遍历 UPLOAD_DIR，计算每个文件名的 MD5 前 8 位
    # # 找到和请求中 doc_id 匹配的那个
    # target_file = None
    # for filename in os.listdir(UPLOAD_DIR):
    #     # .encode() 把字符串转 bytes（md5 只能处理 bytes）
    #     # hexdigest()[:8] 和 list_documents() 中的算法一致
    #     if hashlib.md5(filename.encode()).hexdigest()[:8] == doc_id:
    #         target_file = filename
    #         break  # 找到了，停止搜索
    #
    # if not target_file:
    #     return APIResponse.fail(
    #         code=ErrorCode.PARAM_ERROR,
    #         detail=f"文档不存在: {doc_id}"
    #     )
    #
    # # ---- 步骤 2: 从 Qdrant 删除该文档的所有向量 ----
    # # delete_by_source() 的工作原理:
    # #   1. 构建过滤条件: payload.source_file == target_file
    # #   2. Qdrant 找出所有匹配的 point
    # #   3. 批量删除
    # #   4. 返回删除状态（COMPLETED = 成功）
    # from rag.vector_store import VectorStore
    # vs = VectorStore()
    # deleted_count = vs.delete_by_source(target_file)
    # # deleted_count 是删除的 point 数量（可能不是精确值，Qdrant 异步删除）
    #
    # # ---- 步骤 3: 从 BM25 删除 ----
    # # ⚠️ BM25 倒排索引不支持"删除单条文档"
    # # 原因: 倒排索引一旦构建，词的统计信息（IDF）和文档的关联
    # # 已经嵌入索引结构中，无法局部修改。
    # #
    # # 解决方案: 删除后重建 BM25 索引
    # #   1. 收集所有剩余的文档
    # #   2. 用 loader → splitter 重新解析和切片
    # #   3. 调用 bm25.index(all_chunks) 重建整个索引
    # # 如果文档数量不大（< 100），这一步很快（< 1 秒）
    # from rag.bm25 import BM25Retriever
    # from rag.loader import load_document
    # from rag.splitter import get_splitter
    #
    # bm25 = BM25Retriever()
    # all_chunks = []
    # for fname in os.listdir(UPLOAD_DIR):
    #     if fname == target_file:
    #         continue  # 跳过被删除的文件
    #     if not any(fname.endswith(ext) for ext in ALLOWED_EXTENSIONS):
    #         continue
    #     fpath = os.path.join(UPLOAD_DIR, fname)
    #     try:
    #         doc = load_document(fpath)
    #         splitter = get_splitter("policy_clause")
    #         chunks = splitter.split(doc)
    #         all_chunks.extend(chunks)
    #     except Exception:
    #         pass  # 个别文件解析失败不影响整体
    #
    # # 重建 BM25 索引
    # bm25.index(all_chunks)
    #
    # # ---- 步骤 4: 删除磁盘文件 ----
    # file_path = os.path.join(UPLOAD_DIR, target_file)
    # os.remove(file_path)
    #
    # return APIResponse.ok(data={
    #     "deleted": target_file,
    #     "vector_points_removed": deleted_count,
    #     "bm25_rebuilt_with": len(all_chunks),
    #     "status": "已删除",
    # })
    # ================================================================
    return APIResponse.fail(
        code=ErrorCode.PARAM_ERROR,
        detail="删除功能待实现(TODO用户)。请参考上方注释中的实现逻辑。"
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
    # ================================================================
    # TODO(用户): 实现全量重新索引逻辑
    # ================================================================
    # ================================================================
    # 实现参考:
    #
    # 重新索引 = 清空旧数据 + 重新走完整流水线
    # 适用场景:
    #   - 换了 Embedding 模型（向量维度变了，Qdrant 里的旧向量不兼容）
    #   - 改了切片策略（chunk_size/overlap 变了，旧 chunk 不再有效）
    #   - Qdrant 容器被误删/重建
    #   - 新增了一批文档到 data/documents/ 目录
    #
    # 流程:
    # ┌─────────────┐
    # │ 扫描目录     │ → 收集所有 .md/.pdf/.docx/.txt
    # └──────┬──────┘
    #        ▼
    # ┌─────────────┐
    # │ 清空 Qdrant  │ → 删除旧 Collection，重建空 Collection
    # └──────┬──────┘   (或用 index_all 覆盖写入)
    #        ▼
    # ┌─────────────┐
    # │ 逐文件索引   │ → loader → splitter → Embedding → Qdrant
    # └──────┬──────┘   loader → splitter → jieba分词 → BM25
    #        ▼
    # ┌─────────────┐
    # │ 返回统计     │ → 总文件数、切块数、耗时
    # └─────────────┘
    # ================================================================
    #
    # import time
    # from rag.indexing_pipeline import IndexingPipeline
    # from rag.vector_store import VectorStore
    # from rag.bm25 import BM25Retriever
    # from rag.loader import load_document
    # from rag.splitter import get_splitter
    #
    # start_time = time.time()
    #
    # # ---- 步骤 1: 扫描所有支持的文档 ----
    # doc_files = []
    # if os.path.isdir(UPLOAD_DIR):
    #     for fname in sorted(os.listdir(UPLOAD_DIR)):
    #         if any(fname.endswith(ext) for ext in ALLOWED_EXTENSIONS):
    #             doc_files.append(os.path.join(UPLOAD_DIR, fname))
    #
    # if not doc_files:
    #     return APIResponse.fail(
    #         code=ErrorCode.PARAM_ERROR,
    #         detail="没有找到可索引的文档，请先上传文档"
    #     )
    #
    # # ---- 步骤 2: 重建 Qdrant 索引 ----
    # # 方式 A: 用 pipeline.index_all() 一键完成（推荐）
    # pipeline = IndexingPipeline(
    #     vector_store=VectorStore(),
    #     splitter_strategy="policy_clause",
    # )
    # result = pipeline.index_all(UPLOAD_DIR)
    #
    # # ---- 步骤 3: 重建 BM25 索引 ----
    # # 方式 B: 手动遍历文件，收集所有 chunk，统一建 BM25 索引
    # bm25 = BM25Retriever()
    # all_chunks = []
    # failed_files = []
    #
    # splitter = get_splitter("policy_clause")
    # for fpath in doc_files:
    #     try:
    #         doc = load_document(fpath)
    #         chunks = splitter.split(doc)
    #         all_chunks.extend(chunks)
    #     except Exception as e:
    #         failed_files.append(os.path.basename(fpath))
    #         print(f"[警告] {fpath} 解析失败: {e}")
    #
    # # 用所有chunk重建BM25索引（覆盖模式）
    # bm25.index(all_chunks)
    #
    # # ---- 步骤 4: 返回统计 ----
    # cost_ms = int((time.time() - start_time) * 1000)
    #
    # return APIResponse.ok(data={
    #     "total_files": len(doc_files),
    #     "indexed_files": len(doc_files) - len(failed_files),
    #     "failed_files": failed_files,
    #     "total_chunks": len(all_chunks),
    #     "cost_ms": cost_ms,
    #     "status": "已完成全量重新索引",
    # })
    # ================================================================
    return APIResponse.fail(
        code=ErrorCode.PARAM_ERROR,
        detail="索引功能待实现(TODO用户)。参考上方注释中的实现逻辑。"
    )


# =============================================================================
# GET /documents/status —— 索引状态
# =============================================================================

@router.get("/documents/status")
async def index_status():
    """查询当前索引状态"""
    # ================================================================
    # TODO(用户): 从 Qdrant 查询真实索引状态
    # ================================================================
    # ================================================================
    # 实现参考:
    #
    # 索引状态来自两个数据源:
    #   Qdrant → 向量数量、Collection 信息
    #   磁盘   → 文件列表、最近修改时间
    #
    # 注意: 这不是实时查询每个文档是否已索引（太慢），
    # 而是返回 Qdrant 中总的 points 数量 + 磁盘文件列表。
    # "indexed" 的判断依据是: 文件存在 + Qdrant 中有点 = 已索引
    # ================================================================
    #
    # import os
    # from rag.vector_store import VectorStore
    #
    # # ---- 查询 Qdrant ----
    # vs = VectorStore()
    # total_vectors = vs.count()           # Qdrant 中的向量总数
    # collection_info = vs.collection_info()  # Collection 元信息
    #
    # # ---- 查询磁盘 ----
    # doc_count = 0
    # last_modified = None
    # if os.path.isdir(UPLOAD_DIR):
    #     for fname in os.listdir(UPLOAD_DIR):
    #         if any(fname.endswith(ext) for ext in ALLOWED_EXTENSIONS):
    #             doc_count += 1
    #             fpath = os.path.join(UPLOAD_DIR, fname)
    #             mtime = os.path.getmtime(fpath)  # 文件最后修改时间（Unix 时间戳）
    #             if last_modified is None or mtime > last_modified:
    #                 last_modified = mtime
    #
    # # 把 Unix 时间戳转为可读的 ISO 格式
    # from datetime import datetime
    # last_indexed_iso = (
    #     datetime.fromtimestamp(last_modified).isoformat()
    #     if last_modified else None
    # )
    #
    # return APIResponse.ok(data={
    #     "total_documents": doc_count,
    #     "total_vectors": total_vectors,
    #     "collection_exists": collection_info.get("exists", False),
    #     "collection_name": collection_info.get("name", ""),
    #     "last_indexed_at": last_indexed_iso,
    #     "status": "已索引" if total_vectors > 0 else "未索引",
    # })
    # ================================================================
    return APIResponse.ok(data={
        "total_documents": 0,
        "total_vectors": 0,
        "collection_exists": False,
        "last_indexed_at": None,
        "status": "未索引",
        "message": "请参考上方注释实现索引状态查询逻辑",
    })
