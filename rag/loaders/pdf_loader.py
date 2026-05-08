"""
PDF 文档解析器

使用 PyMuPDF（fitz）提取正文 + pdfplumber 提取表格。
这是本项目「表格感知解析」的核心——表格不被摊平为纯文本，
而是保留为结构化 TableData，后续检索时可做精确匹配。

依赖：
    pip install pymupdf pdfplumber
"""

from datetime import datetime

from rag.models import ParsedDocument, TableData


class PDFLoader:
    """PDF 解析器 —— 正文 + 表格结构化"""

    def parse(self, file_path: str) -> ParsedDocument:
        """解析 PDF 文件为 ParsedDocument

        两阶段解析：
        1. pdfplumber 提取表格 → TableData 列表
        2. PyMuPDF 提取纯文本 → raw_text（跳过已识别表格区域）
        """
        tables = self._extract_tables_with_pdfplumber(file_path)
        raw_text = self._extract_text_with_pymupdf(file_path)

        return ParsedDocument(
            file_name="",
            file_type="pdf",
            raw_text=raw_text,
            sections=[],         # PDF 章节识别较复杂，暂不提取
            tables=tables,
            parsed_at=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # 阶段 1：pdfplumber 提取表格
    # ------------------------------------------------------------------

    def _extract_tables_with_pdfplumber(self, file_path: str) -> list[TableData]:
        """使用 pdfplumber 提取 PDF 中的表格

        pdfplumber 基于 PDF 的绘图指令（线条位置）检测表格边框，
        比 OCR 方式更准确，适合规范排版的文档。
        """
        import pdfplumber

        tables = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_tables = page.extract_tables()
                for t in raw_tables:
                    if not t or len(t) < 2:
                        continue  # 跳过空表或只有表头的表
                    headers = [str(h).strip() if h else "" for h in t[0]]
                    rows = [
                        [str(c).strip() if c else "" for c in row]
                        for row in t[1:]
                    ]
                    tables.append(TableData(
                        headers=headers,
                        rows=rows,
                        page=page_num,
                    ))
        return tables

    # ------------------------------------------------------------------
    # 阶段 2：PyMuPDF 提取纯文本
    # ------------------------------------------------------------------

    def _extract_text_with_pymupdf(self, file_path: str) -> str:
        """使用 PyMuPDF 提取 PDF 纯文本"""
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        doc.close()
        return "\n".join(text_parts).strip()
