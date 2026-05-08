"""
DOCX 文档解析器

使用 python-docx 提取段落和表格。
Word 文档中的表格保留为结构化 TableData。

依赖：
    pip install python-docx
"""

from datetime import datetime

from rag.models import ParsedDocument, SectionMeta, TableData


class DocxLoader:
    """DOCX（Word）解析器"""

    def parse(self, file_path: str) -> ParsedDocument:
        """解析 DOCX 文件为 ParsedDocument"""
        from docx import Document

        doc = Document(file_path)

        # 提取纯文本段落
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        raw_text = "\n\n".join(paragraphs)

        # 提取章节（基于 Word 内置标题样式）
        sections = self._extract_sections_from_docx(doc)

        # 提取表格
        tables = self._extract_tables_from_docx(doc)

        return ParsedDocument(
            file_name="",
            file_type="docx",
            raw_text=raw_text,
            sections=sections,
            tables=tables,
            parsed_at=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_sections_from_docx(self, doc) -> list[SectionMeta]:
        """从 Word 标题样式中提取章节结构

        Word 内置样式 Heading 1/2/3 对应 Markdown 的 #/##/###。
        python-docx 中 style.name 为 'Heading 1' / 'Heading 2' 等。
        """
        sections = []
        char_pos = 0
        for para in doc.paragraphs:
            text_l = len(para.text)
            style = para.style.name if para.style else ""

            if style.startswith("Heading "):
                try:
                    level = int(style.split()[-1])
                except ValueError:
                    level = 2
                sections.append(SectionMeta(
                    title=para.text.strip(),
                    level=level,
                    start_char=char_pos,
                    end_char=char_pos + text_l,
                ))
            char_pos += text_l + 2  # +2 模拟段落间距

        return sections

    def _extract_tables_from_docx(self, doc) -> list[TableData]:
        """提取 Word 文档中的表格"""
        tables = []
        for table in doc.tables:
            if len(table.rows) < 2:
                continue
            headers = [cell.text.strip() for cell in table.rows[0].cells]
            rows = [
                [cell.text.strip() for cell in row.cells]
                for row in table.rows[1:]
            ]
            tables.append(TableData(headers=headers, rows=rows))
        return tables
