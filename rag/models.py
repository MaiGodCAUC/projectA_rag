"""
RAG 数据模型

定义文档解析、切片、检索、生成的统一数据结构。
所有 RAG 模块共享这些 Pydantic 模型，确保类型安全。
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# 文档解析阶段
# =============================================================================

class TableData(BaseModel):
    """结构化表格数据

    与纯文本摊平不同，表格保留行列结构，后续检索时可做精确匹配。
    例如行李限额表中的"经济舱 / 23kg / 2件"不会和正文混在一起。
    """
    headers: list[str] = Field(description="表格列名列表，如 ['舱位', '免费行李额', '件数限制']")
    rows: list[list[str]] = Field(description="表格数据行，每行是一个 list，长度与 headers 一致")
    caption: Optional[str] = Field(default=None, description="表格标题/说明")
    page: Optional[int] = Field(default=None, description="所在页码（PDF 文档）")


class SectionMeta(BaseModel):
    """文档章节元数据

    记录每个章节的标题层级和位置信息，用于：
    - 切片时按章节边界切分
    - 检索时显示来源章节上下文
    - 引用溯源时定位到具体章节
    """
    title: str = Field(description="章节标题文本，如 '托运行李重量限制'")
    level: int = Field(description="标题层级：1=# 2=## 3=### 4=####")
    start_char: int = Field(default=0, description="章节在文档中的起始字符位置")
    end_char: int = Field(default=0, description="章节在文档中的结束字符位置")


class ParsedDocument(BaseModel):
    """文档解析后的统一数据结构

    无论原始文件是 PDF、DOCX 还是 Markdown，解析后都统一为此格式。
    后续的切片、索引、检索全部基于 ParsedDocument 操作，不关心原始格式。
    """
    file_name: str = Field(description="原始文件名，如 '行李运输规定.md'")
    file_type: str = Field(description="文件类型：pdf / docx / md")
    file_hash: str = Field(default="", description="文件 SHA256 哈希，用于增量索引去重")
    raw_text: str = Field(default="", description="完整纯文本内容（去除格式标记）")
    sections: list[SectionMeta] = Field(default_factory=list, description="章节结构列表")
    tables: list[TableData] = Field(default_factory=list, description="结构化表格列表")
    parsed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="解析时间戳",
    )
    char_count: int = Field(default=0, description="总字符数")
    table_count: int = Field(default=0, description="表格总数")
    section_count: int = Field(default=0, description="章节总数")

    def model_post_init(self, __context):
        """Pydantic v2 钩子：模型初始化后自动统计"""
        self.char_count = len(self.raw_text)
        self.table_count = len(self.tables)
        self.section_count = len(self.sections)


# =============================================================================
# 切片阶段
# =============================================================================

class TextChunk(BaseModel):
    """文档切片后的文本块

    每个 chunk 是检索的最小单元，embedding 按 chunk 粒度生成。
    """
    chunk_id: str = Field(description="唯一标识，如 '行李运输规定_第3条'")
    content: str = Field(description="切片文本内容")
    source_file: str = Field(description="来源文档名")
    clause_id: Optional[str] = Field(default=None, description="条款编号（PolicyClauseSplitter 识别）")
    section_title: Optional[str] = Field(default=None, description="所属章节标题")
    chunk_index: int = Field(default=0, description="在文档中的序号")
    metadata: dict = Field(default_factory=dict, description="附加元数据（含关联表格等）")


# =============================================================================
# 检索阶段
# =============================================================================

class RetrievalResult(BaseModel):
    """单条检索结果"""
    chunk: TextChunk = Field(description="命中的文本块")
    score: float = Field(description="相似度分数或 RRF 分数")
    source: str = Field(description="来源：vector / bm25 / hybrid")


# =============================================================================
# 生成阶段
# =============================================================================

class Citation(BaseModel):
    """引用溯源信息

    记录回答中每一条引用对应的原文出处，支持前端点击弹窗展示原文。
    """
    doc_name: str = Field(description="来源文档名，如 '行李运输规定'")
    clause_id: Optional[str] = Field(default=None, description="条款编号，如 '第3.2条'")
    section_title: Optional[str] = Field(default=None, description="所在章节标题")
    original_text: str = Field(description="引用的原文片段全文")


class CitedAnswer(BaseModel):
    """带引用溯源的 RAG 回答

    与普通文本回答不同，每个结论都关联到具体条款，可追溯到原文。
    """
    answer_text: str = Field(description="回答正文（含 [来源: xxx] 标记）")
    citations: list[Citation] = Field(default_factory=list, description="引用列表")
    trace_id: str = Field(default="", description="LangSmith Trace ID")
    cost_ms: int = Field(default=0, description="全链路耗时（毫秒）")
