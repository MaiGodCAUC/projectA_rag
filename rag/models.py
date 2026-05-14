"""
RAG 数据模型

定义文档解析、切片、检索、生成的统一数据结构。
所有 RAG 模块共享这些 Pydantic 模型，确保类型安全。

----------------------------------------------------------------------
## 你需要自己写的部分

本文件由工程辅助生成，你主要需要理解而非重写。

学习重点：
1. 理解 Pydantic BaseModel 的基本用法：类型声明 → 自动校验 → 序列化
2. 理解 Field() 的作用：添加描述、默认值、default_factory
3. 理解为什么要把文档解析结果统一成 ParsedDocument —— 解耦，后续模块不关心原始格式
4. 理解 model_post_init 钩子的时机：模型创建完成后自动执行，用于统计字段

如果你要手写：
- 可以先手写 ParsedDocument 类（核心），其余按需补充
- 重点是理解字段含义，不是抄代码

没有 TODO(用户) 标记 —— 本文件全部由工程辅助完成，无需要你补充的逻辑。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# datetime: 用于生成解析时间戳（如 "2026-05-09T10:30:00"）
from datetime import datetime

# Optional: 类型提示，表示字段可以为 None（如 caption: Optional[str] = None）
from typing import Optional

# BaseModel: Pydantic 的核心基类，继承它就能获得类型校验、序列化等能力
# Field: 给字段添加元数据——描述(description)、默认值(default)、默认工厂(default_factory)
from pydantic import BaseModel, Field


# =============================================================================
# 文档解析阶段 —— 这些模型描述「原始文档被解析后的数据结构」
# =============================================================================

class TableData(BaseModel):
    """结构化表格数据

    与纯文本摊平不同，表格保留行列结构，后续检索时可做精确匹配。
    例如行李限额表中的"经济舱 / 23kg / 2件"不会和正文混在一起。

    这是本项目的核心设计决策之一：表格不走「摊平为纯文本」的路线，
    而是保留为结构化对象，回答"经济舱能带几件行李"时可以直接命中表格数据。
    """

    # headers: 表格列名，如 ['舱位', '免费行李额', '件数限制']
    # list[str] 表示这是一个字符串列表，Pydantic 会自动校验每个元素是否为 str
    # Field(description=...) 是对字段的说明，也会出现在 JSON Schema 中
    headers: list[str] = Field(description="表格列名列表，如 ['舱位', '免费行李额', '件数限制']")

    # rows: 表格数据行，二维列表 —— 外层是行，内层是单元格
    # 例如 [['经济舱', '23kg', '2件'], ['商务舱', '32kg', '2件']]
    rows: list[list[str]] = Field(description="表格数据行，每行是一个 list，长度与 headers 一致")

    # caption: 表格标题/说明，可选字段（Optional）
    # default=None 表示不传这个字段时默认为 None
    caption: Optional[str] = Field(default=None, description="表格标题/说明")

    # page: 所在页码，用于 PDF 文档的表格溯源
    # Optional[int] 表示可以是 int 或 None
    page: Optional[int] = Field(default=None, description="所在页码（PDF 文档）")


class SectionMeta(BaseModel):
    """文档章节元数据

    记录每个章节的标题层级和位置信息，用于：
    - 切片时按章节边界切分
    - 检索时显示来源章节上下文
    - 引用溯源时定位到具体章节

    举例：文档中 ## 托运行李重量限制 → SectionMeta(title="托运行李重量限制", level=2, start_char=150, end_char=3200)
    """

    # title: 章节标题文本，去除 # 标记后的纯文字
    title: str = Field(description="章节标题文本，如 '托运行李重量限制'")

    # level: 标题层级，映射自 Markdown # 数量或 Word Heading 级别
    # 1=# 顶层标题, 2=## 二级标题, 3=### 三级标题, 4=#### 四级标题
    level: int = Field(description="标题层级：1=# 2=## 3=### 4=####")

    # start_char: 章节在文档全文中的起始字符位置（从 0 开始计数）
    # 用于定位"这个章节从文档的第几个字符开始"
    start_char: int = Field(default=0, description="章节在文档中的起始字符位置")

    # end_char: 章节在文档全文中的结束字符位置
    # start_char 到 end_char 之间的文本就是这个章节的内容
    end_char: int = Field(default=0, description="章节在文档中的结束字符位置")


class ParsedDocument(BaseModel):
    """文档解析后的统一数据结构

    无论原始文件是 PDF、DOCX 还是 Markdown，解析后都统一为此格式。
    后续的切片、索引、检索全部基于 ParsedDocument 操作，不关心原始格式。

    这是整个 RAG 流水线的「统一数据契约」：
    所有 Loader 的输出 → ParsedDocument → Splitter 的输入 → TextChunk

    设计要点：
    - raw_text: 纯文本，用于全文索引
    - sections: 章节结构，用于章节级切片
    - tables: 结构化表格，单独索引，支持精确匹配
    - file_hash: 增量索引去重的关键字段
    """

    # file_name: 原始文件名，如 '行李运输规定.md'
    # 用于检索结果中显示来源，以及引用溯源
    file_name: str = Field(description="原始文件名，如 '行李运输规定.md'")

    # file_type: 文件类型标识 —— pdf / docx / md
    # 虽然解析后格式统一，但保留原始类型便于调试和日志
    file_type: str = Field(description="文件类型：pdf / docx / md")

    # file_hash: 文件内容的 SHA256 哈希值
    # 核心用途：增量索引去重 —— 同一文件内容不变就不重新索引
    # default="" 表示解析器会在后续阶段填充这个值
    file_hash: str = Field(default="", description="文件 SHA256 哈希，用于增量索引去重")

    # raw_text: 去除所有格式标记后的纯文本内容
    # 所有格式（加粗、斜体、标题标记等）都被剥离，只保留可读字符
    raw_text: str = Field(default="", description="完整纯文本内容（去除格式标记）")

    # sections: 章节结构列表，每个元素是一个 SectionMeta 对象
    # default_factory=list 表示默认值是空列表（不能直接写 default=[]，这是 Pydantic 的规则）
    sections: list[SectionMeta] = Field(default_factory=list, description="章节结构列表")

    # tables: 结构化表格列表，每个元素是一个 TableData 对象
    # 表格不走纯文本路线，保留行列结构
    tables: list[TableData] = Field(default_factory=list, description="结构化表格列表")

    # parsed_at: 解析时间戳，ISO 8601 格式字符串
    # default_factory=lambda: datetime.now().isoformat() 表示每次创建模型时自动获取当前时间
    # 用 lambda 而不是直接调用，确保每次创建时时间都是最新（而不是导入模块时的时间）
    parsed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="解析时间戳",
    )

    # 以下三个是统计字段，在 model_post_init 中自动计算，不需要手动传入
    char_count: int = Field(default=0, description="总字符数")
    table_count: int = Field(default=0, description="表格总数")
    section_count: int = Field(default=0, description="章节总数")

    def model_post_init(self, __context):
        """Pydantic v2 钩子：模型初始化后自动统计

        Pydantic v2 中，模型创建完成后会自动调用 model_post_init 方法。
        这里我们利用它来自动计算统计字段，避免手动维护。

        __context 参数是 Pydantic 内部传递的上下文对象，
        我们不使用它，但函数签名必须保留以符合 Pydantic 规范。
        """
        # 统计 raw_text 的字符总数
        self.char_count = len(self.raw_text)
        # 统计表格数量
        self.table_count = len(self.tables)
        # 统计章节数量
        self.section_count = len(self.sections)


# =============================================================================
# 切片阶段 —— 这些模型描述「文档被切分后的文本块」
# =============================================================================

class TextChunk(BaseModel):
    """文档切片后的文本块

    每个 chunk 是检索的最小单元，embedding 按 chunk 粒度生成。

    设计要点：
    - chunk_id 必须全局唯一，后续检索结果靠它关联到原始文档位置
    - metadata 是一个自由字典，可存放关联表格、条款编号等扩展信息
    - clause_id 是亮点字段，仅在 PolicyClauseSplitter 切分时才填充
    """

    # chunk_id: 唯一标识，命名规范为 '{文档名}_{编号}'
    # 例如 '行李运输规定_第3条' —— 既能看到来源又能定位位置
    chunk_id: str = Field(description="唯一标识，如 '行李运输规定_第3条'")

    # content: 这个文本块的实际文本内容
    # embedding 就是对这个 content 做向量化
    content: str = Field(description="切片文本内容")

    # source_file: 来源文档名，用于检索结果中显示出处
    source_file: str = Field(description="来源文档名")

    # clause_id: 条款编号，如 '第3.2条'、'第5条'
    # 亮点功能：仅 PolicyClauseSplitter 识别条款时才填充
    # Optional[str] = None 表示不保证每个 chunk 都有条款编号
    clause_id: Optional[str] = Field(default=None, description="条款编号（PolicyClauseSplitter 识别）")

    # section_title: 所属章节标题，如 '托运行李重量限制'
    # 用于检索时显示"这个答案来自 XX 章节"，增强可解释性
    section_title: Optional[str] = Field(default=None, description="所属章节标题")

    # chunk_index: 在当前文档中的切片序号，从 0 开始
    # 用于保持检索结果的原始顺序
    chunk_index: int = Field(default=0, description="在文档中的序号")

    # metadata: 扩展字段字典，可存放任意附加信息
    # 例如关联的表格数据、原始页码、文档类别等
    # default_factory=dict 生成空字典作为默认值
    metadata: dict = Field(default_factory=dict, description="附加元数据（含关联表格等）")


# =============================================================================
# 检索阶段 —— 这些模型描述「检索系统返回的结果」
# =============================================================================

class RetrievalResult(BaseModel):
    """单条检索结果

    封装一次检索命中的完整信息，包括命中的文本块、分数、来源。
    source 字段标注结果来自哪种检索方式，方便调试和结果分析。
    """

    # chunk: 命中的文本块，包含完整内容和元数据
    chunk: TextChunk = Field(description="命中的文本块")

    # score: 相似度分数 —— 向量检索是余弦相似度，BM25 是 TF-IDF 分数
    # 混合检索（RRF 融合后）这个值就是 RRF 分数
    score: float = Field(description="相似度分数或 RRF 分数")

    # source: 标记检索来源
    # 'vector' = 向量语义检索, 'bm25' = 关键词精确检索, 'hybrid' = 混合检索
    # 这个字段对调试很有用 —— 你可以分析「哪些问题更依赖于哪种检索方式」
    source: str = Field(description="来源：vector / bm25 / hybrid")


# =============================================================================
# 生成阶段 —— 这些模型描述「LLM 生成回答的结果」
# =============================================================================

class Citation(BaseModel):
    """引用溯源信息

    记录回答中每一条引用对应的原文出处，支持前端点击弹窗展示原文。

    面试亮点：这是企业级 RAG 区别于 demo 的关键 ——
    不是"大模型说啥是啥"，而是每条结论都可以追溯到原文。
    """

    # doc_name: 来源文档名，如 '行李运输规定'
    doc_name: str = Field(description="来源文档名，如 '行李运输规定'")

    # clause_id: 条款编号，如 '第3.2条'、'第5条'
    # 可选，因为不是所有文档都有明确的条款编号
    clause_id: Optional[str] = Field(default=None, description="条款编号，如 '第3.2条'")

    # section_title: 所在章节标题
    # 当没有条款编号时，可以用章节标题来定位
    section_title: Optional[str] = Field(default=None, description="所在章节标题")

    # original_text: 从文档中检索到的原文片段全文
    # 和 LLM 生成的回答文字对应，用户可以对比"原文怎么说 ↔ LLM 怎么说"
    original_text: str = Field(description="引用的原文片段全文")


class CitedAnswer(BaseModel):
    """带引用溯源的 RAG 回答

    与普通文本回答不同，每个结论都关联到具体条款，可追溯到原文。

    这是本项目的差异化亮点：
    - answer_text: LLM 生成的回答正文，内嵌 [来源: xxx 第X条] 标记
    - citations: 结构化引用列表，前端可做「点击查看原文」弹窗
    - trace_id: LangSmith 追踪 ID，用于可观测性
    - cost_ms: 全链路耗时，用于性能监控和优化
    """

    # answer_text: LLM 生成的回答正文
    # 文中内嵌 [来源: 文档名 条款编号] 标记，让读者知道哪句话来自哪里
    answer_text: str = Field(description="回答正文（含 [来源: xxx] 标记）")

    # citations: 结构化引用列表
    # 和 answer_text 中的 [来源: ...] 标记一一对应
    citations: list[Citation] = Field(default_factory=list, description="引用列表")

    # trace_id: LangSmith 的一整条调用链追踪 ID
    # 用于在 LangSmith 后台查看完整的 LLM 调用过程（prompt、响应、耗时等）
    trace_id: str = Field(default="", description="LangSmith Trace ID")

    # cost_ms: 从用户提问到返回回答的总耗时（毫秒）
    # 这是面试中可以说的指标："我们的全链路延迟控制在 X ms 以内"
    cost_ms: int = Field(default=0, description="全链路耗时（毫秒）")
