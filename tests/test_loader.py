"""
文档加载器单元测试

运行方式：
    pytest tests/test_loader.py -v
"""

import pytest
from pathlib import Path

# 导入待测试模块
from rag.models import ParsedDocument, TableData, SectionMeta
from rag.loader import load_document


# 测试数据路径
DATA_DIR = Path(__file__).parent.parent / "data" / "documents"


class TestMDLoader:
    """Markdown 解析器测试"""

    def test_parse_markdown_document(self):
        """正常文档解析：应有文本、表格和章节"""
        doc = load_document(str(DATA_DIR / "04-托运行李运输规定.md"))

        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == "md"
        assert len(doc.raw_text) > 0, "应提取到文本内容"
        assert doc.table_count > 0, "应识别到至少1张表格"
        assert doc.section_count > 0, "应识别到至少1个章节"
        assert doc.char_count > 0
        assert doc.file_hash != ""

    def test_parse_tables(self):
        """表格解析：舱位代码对照表应有3张表格"""
        doc = load_document(str(DATA_DIR / "03-舱位代码对照表.md"))

        assert doc.table_count >= 3, f"应有3张表格，实际识别 {doc.table_count} 张"
        # 验证第一张表的结构
        first_table = doc.tables[0]
        assert len(first_table.headers) > 0, "表头不应为空"
        assert len(first_table.rows) > 0, "表格应有数据行"

    def test_parse_empty_file(self):
        """空文件不应报错，返回空 ParsedDocument"""
        empty_file = DATA_DIR / "_empty_test.md"
        empty_file.write_text("", encoding="utf-8")
        try:
            doc = load_document(str(empty_file))
            assert isinstance(doc, ParsedDocument)
        finally:
            empty_file.unlink()  # 清理测试文件


class TestPDFLoader:
    """PDF 解析器测试"""

    def test_parse_pdf(self):
        """解析 PDF 版本：对应 markdown 转 PDF 后的文件"""
        pdf_file = DATA_DIR / "04-托运行李运输规定.pdf"
        if not pdf_file.exists():
            pytest.skip("PDF 文件尚未生成，请先将 markdown 转为 PDF")

        doc = load_document(str(pdf_file))
        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == "pdf"
        assert len(doc.raw_text) > 0, "应提取到 PDF 文本内容"


class TestDocxLoader:
    """DOCX 解析器测试"""

    def test_parse_docx(self):
        """解析 DOCX 版本"""
        docx_file = DATA_DIR / "04-托运行李运输规定.docx"
        if not docx_file.exists():
            pytest.skip("DOCX 文件尚未生成")

        doc = load_document(str(docx_file))
        assert isinstance(doc, ParsedDocument)
        assert doc.file_type == "docx"


class TestLoaderErrors:
    """异常场景测试"""

    def test_file_not_found(self):
        """不存在的文件应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_document(str(DATA_DIR / "不存在的文件.md"))

    def test_unsupported_format(self):
        """不支持的格式应抛出 ValueError"""
        fake_file = DATA_DIR / "_test.xyz"
        fake_file.write_text("test", encoding="utf-8")
        try:
            with pytest.raises(ValueError, match="不支持的文档格式"):
                load_document(str(fake_file))
        finally:
            fake_file.unlink()
