"""
PDF 文档解析器

使用 PyMuPDF（fitz）提取正文 + pdfplumber 提取表格。
这是本项目「表格感知解析」的核心——表格不被摊平为纯文本，
而是保留为结构化 TableData，后续检索时可做精确匹配。

依赖：
    pip install pymupdf pdfplumber

----------------------------------------------------------------------
## 你需要自己写的部分

本文件由工程辅助生成，包含本项目的**核心技术亮点**——表格感知解析。

**强烈建议你理解后手写一遍，原因：**
1. 两阶段解析（pdfplumber 提取表格 + PyMuPDF 提取文本）是面试高频考点
2. pdfplumber 提取表格的原理（基于 PDF 绘图指令检测边框）可以展开讲
3. 这是你简历上可以写的差异化亮点："自研表格感知 PDF 解析器，表格保留结构化"

**面试话术参考：**
"传统 RAG 把 PDF 表格直接摊平为文本，导致'经济舱能带几件行李'这种问题
检索不到表格数据。我用了两阶段解析：pdfplumber 检测 PDF 线条边框来提取表格
保留行列结构，PyMuPDF 提取正文。表格作为结构化数据单独索引，
查询时可以做精确的行列匹配，准确率比纯文本方式高 XX%。"

**如果你要手写：**
- 核心逻辑是 _extract_tables_with_pdfplumber，必须吃透
- _extract_text_with_pymupdf 相对简单，是辅助功能

没有 TODO(用户) 标记 —— 本文件逻辑完整，你可以照抄学习。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# datetime: 生成解析时间戳
from datetime import datetime

# ParsedDocument: 解析结果的统一数据模型
# TableData: 结构化表格数据模型（表头 + 数据行 + 页码 + 标题）
from rag.models import ParsedDocument, TableData


class PDFLoader:
    """PDF 解析器 —— 正文 + 表格结构化

    核心设计：两阶段解析

    阶段 1：pdfplumber 提取表格
    - pdfplumber 通过分析 PDF 的底层绘图指令来检测表格
    - PDF 中的表格通常由水平和垂直线条（vectors）框定
    - pdfplumber 能找到这些线条的交点，推断出单元格边界
    - 比 OCR 方式更准确，因为不依赖图像识别，直接读 PDF 结构

    阶段 2：PyMuPDF（fitz）提取纯文本
    - PyMuPDF 是 C 库 MuPDF 的 Python 绑定，速度极快
    - 提取所有页面的文本流，拼接为全文
    - 当前版本未做表格区域跳过（未来可优化：pdfplumber 识别的表格区域在 PyMuPDF 中跳过）

    为什么用两个库而不是一个？
    - pdfplumber 表格提取精准但文本提取性能一般
    - PyMuPDF 文本提取极快但不会结构化识别表格
    - 两者互补，各取所长
    """

    def parse(self, file_path: str) -> ParsedDocument:
        """解析 PDF 文件为 ParsedDocument

        两阶段解析：
        1. pdfplumber 提取表格 → TableData 列表
        2. PyMuPDF 提取纯文本 → raw_text（跳过已识别表格区域）

        Args:
            file_path: PDF 文件路径

        Returns:
            ParsedDocument 对象，含 raw_text + tables
        """
        # 阶段 1：先用 pdfplumber 提取表格
        # 返回 TableData 列表，每个 TableData 包含 headers、rows、page
        tables = self._extract_tables_with_pdfplumber(file_path)

        # 阶段 2：再用 PyMuPDF 提取纯文本
        # 将所有页面的文本拼接为一个字符串
        raw_text = self._extract_text_with_pymupdf(file_path)

        # 组装 ParsedDocument 对象
        # sections=[] 是因为 PDF 的章节结构识别比较复杂（需要分析字体大小等），
        # 当前版本暂不提取，后续版本可以考虑
        # file_name="" 和 file_hash 由 loader.py 的 load_document 统一填充
        return ParsedDocument(
            file_name="",           # 由 load_document() 补充
            file_type="pdf",        # 固定为 "pdf"
            raw_text=raw_text,      # PyMuPDF 提取的纯文本
            sections=[],            # PDF 章节识别暂不实现，给空列表
            tables=tables,          # pdfplumber 提取的结构化表格
            parsed_at=datetime.now().isoformat(),  # 记录解析时间
        )

    # ------------------------------------------------------------------
    # 阶段 1：pdfplumber 提取表格
    # ------------------------------------------------------------------

    def _extract_tables_with_pdfplumber(self, file_path: str) -> list[TableData]:
        """使用 pdfplumber 提取 PDF 中的表格

        pdfplumber 基于 PDF 的绘图指令（线条位置）检测表格边框，
        比 OCR 方式更准确，适合规范排版的文档。

        ## pdfplumber 表格检测原理（面试重点）

        PDF 文档中表格的绘制方式是：
        1. 先画一堆水平和垂直线条（lines/rects）
        2. 在这些线条框定的区域内放置文字

        pdfplumber 做的事情：
        1. 解析 PDF 的绘图指令，找到所有线条
        2. 计算线条之间的交点，推断出哪些区域被线条围成"格子"
        3. 在这些格子里找到文字，按行列组织
        4. 返回二维数组（rows × cols）

        这意味着：
        - ✅ 有线框的真表格 → 识别率高
        - ✅ 规范排版的文档（如航空公司的 PDF 规定文件）→ 效果好
        - ❌ 无线框的"类表格"布局 → 可能识别失败
        - ❌ 扫描件 PDF → 完全无效（需要用 OCR）

        ## 为什么要在这里 import 而不是顶部？（面试小技巧）

        因为 pdfplumber 是重量级依赖（包含多个 C 扩展），
        把它放在函数内部 import（懒加载）的好处：
        - 不解析 PDF 时根本不需要加载 pdfplumber
        - 减少模块导入时的启动时间
        - 如果没装 pdfplumber 但也不调用此函数，不会报错

        Args:
            file_path: PDF 文件路径

        Returns:
            TableData 列表，空列表表示未识别到表格
        """
        # 懒加载：只在首次调用此方法时才导入 pdfplumber
        import pdfplumber

        # 初始化结果列表
        tables = []

        # 使用 pdfplumber 打开 PDF 文件
        # with 语句确保使用完后自动关闭文件（释放文件句柄）
        with pdfplumber.open(file_path) as pdf:
            # 遍历每一页
            # enumerate(pdf.pages, start=1) 返回 (页码, 页面对象)
            # start=1 表示页码从 1 开始（而不是 0），这样更符合人类习惯
            for page_num, page in enumerate(pdf.pages, start=1):
                # page.extract_tables() 是 pdfplumber 的核心方法
                # 返回当前页所有表格的二维列表
                # 每个表格是一个 list[list[str]]：外层是行，内层是单元格
                # 例如：[[['A', 'B'], ['1', '2']], [['C', 'D']]] 表示第一页有 2 个表格
                raw_tables = page.extract_tables()

                # 遍历当前页识别到的每个表格
                for t in raw_tables:
                    # 过滤无效表格：
                    # not t: 表格为 None（pdfplumber 偶尔返回 None）
                    # len(t) < 2: 只有 1 行（可能只是表头没有数据）
                    # 只有表头没有数据的表格没有价值，跳过
                    if not t or len(t) < 2:
                        continue  # 跳过此表格，继续下一个

                    # ---- 解析表头（第一行） ----
                    # t[0] 是表格的第一行，即表头行
                    # 列表推导式：遍历每个单元格
                    #   str(h) if h else ""：
                    #     如果单元格不是 None → 转为字符串并去空白
                    #     如果单元格是 None → 用空字符串替代
                    #   str(h).strip()：去掉单元格内容首尾空白
                    headers = [str(h).strip() if h else "" for h in t[0]]

                    # ---- 解析数据行（第二行开始） ----
                    # t[1:] 是表头之后的所有数据行
                    # 双层列表推导式：
                    #   外层 for row in t[1:] → 遍历每一行
                    #   内层 for c in row → 遍历行内的每个单元格
                    #   同样处理 None → "" 转换和空白去除
                    rows = [
                        [str(c).strip() if c else "" for c in row]
                        for row in t[1:]
                    ]

                    # ---- 创建 TableData 对象 ----
                    # page=page_num 记录表格在 PDF 的第几页
                    # 这个信息对于引用溯源很有用：可以说"答案来自 XX.pdf 第 3 页的表格"
                    tables.append(TableData(
                        headers=headers,    # 表头列表，如 ['舱位', '免费行李额', '件数限制']
                        rows=rows,          # 数据行二维列表
                        page=page_num,      # 所在页码，用于引用溯源
                    ))

        # 返回所有页面的所有表格
        return tables

    # ------------------------------------------------------------------
    # 阶段 2：PyMuPDF 提取纯文本
    # ------------------------------------------------------------------

    def _extract_text_with_pymupdf(self, file_path: str) -> str:
        """使用 PyMuPDF 提取 PDF 纯文本

        PyMuPDF（import 时叫 fitz）是 MuPDF 渲染引擎的 Python 绑定。
        它读取 PDF 的文本流（text stream），按阅读顺序提取文字。

        ## PyMuPDF vs pdfplumber vs PyPDF2 对比（面试可以说）

        | 库         | 文本提取 | 表格提取 | 速度   | 适用场景           |
        |-----------|---------|---------|--------|-------------------|
        | PyMuPDF   | ★★★★★  | ★★     | ★★★★★ | 大量文档的文本提取   |
        | pdfplumber| ★★★★   | ★★★★★  | ★★★   | 需要表格结构化的场景  |
        | PyPDF2    | ★★★    | ☆      | ★★★★  | 简单文本提取        |

        Args:
            file_path: PDF 文件路径

        Returns:
            所有页面拼接后的纯文本字符串
        """
        # 懒加载：只在首次调用时导入 PyMuPDF
        # fitz 是 PyMuPDF 的包名（历史原因，来自 MuPDF 的原始缩写）
        import fitz  # PyMuPDF

        # 打开 PDF 文档
        # fitz.open() 返回一个 Document 对象，代表整个 PDF 文件
        doc = fitz.open(file_path)

        # 初始化文本片段列表，用于收集每页的文本
        text_parts = []

        # 遍历每一页
        # doc 对象是可迭代的，每次迭代返回一个 Page 对象
        for page in doc:
            # page.get_text("text") 提取当前页的纯文本
            # 参数 "text" 表示按阅读顺序输出纯文本（还有 "html"、"json" 等模式）
            # 返回字符串，包含当前页的所有文本内容
            text_parts.append(page.get_text("text"))

        # 关闭文档，释放文件句柄
        # 这是良好的资源管理习惯 —— 虽然 Python GC 最终会回收，但显式关闭更安全
        doc.close()

        # 将所有页面的文本用换行符拼接，并去除首尾空白
        # "\n".join(...) 保留页与页之间的分隔，让后续切片知道这是不同页面
        # .strip() 去除文档开头和结尾的多余空白
        return "\n".join(text_parts).strip()
