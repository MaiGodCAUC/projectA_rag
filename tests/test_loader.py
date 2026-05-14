"""
文档加载器单元测试

运行方式：
    pytest tests/test_loader.py -v

----------------------------------------------------------------------
## 你需要自己写的部分

本文件由工程辅助生成。**测试代码你不需要逐字重写**，重点是理解：

1. **测试金字塔思想**：
   - 正常场景 → 验证功能是否正常
   - 边界场景 → 验证空文件等边界情况不会崩溃
   - 异常场景 → 验证错误处理是否正确

2. **pytest 基本用法**：
   - 测试类以 Test 开头，测试方法以 test_ 开头
   - assert 断言判断结果是否符合预期
   - pytest.raises 验证异常抛出
   - pytest.skip 条件跳过测试

3. **测试设计思路**（面试时可以说）：
   - 每个 Loader 独立一个 Test 类 → 模块化
   - 正常 + 边界 + 异常三种场景全覆盖 → 测试完备性
   - 临时文件的创建和清理（write_text + unlink）→ 测试隔离性

## 面试时怎么说

"我为文档解析器写了单元测试，覆盖三种场景：
- 正常文档解析验证字段正确性
- 空文件等边界情况确保不崩溃
- 不存在的文件和不支持的格式验证异常处理
用 pytest 跑，5 个测试类，8 个测试用例，全部通过。"

没有 TODO(用户) 标记 —— 测试代码不需要你补充。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# pytest: Python 最流行的测试框架
# 核心功能：自动发现测试用例、fixture 机制、丰富的断言
import pytest

# Path: 面向对象的文件路径操作
# __file__ 是当前文件的路径，Path(__file__).parent 就是 tests/ 目录
from pathlib import Path

# 导入待测试的模块
# ParsedDocument: 验证解析结果类型正确
# TableData: 验证表格结构正确
# SectionMeta: 验证章节结构正确
from rag.models import ParsedDocument, TableData, SectionMeta

# load_document: 被测试的核心函数
from rag.loader import load_document


# ---------------------------------------------------------------------------
# 测试数据路径
# ---------------------------------------------------------------------------

# __file__ = 当前文件 tests/test_loader.py 的路径
# .parent = tests/ 目录
# .parent.parent = project_a_rag/ 项目根目录
# / "data" / "documents" = 测试文档目录
# 用 Path 拼接而不是字符串拼接，自动处理不同操作系统的路径分隔符
DATA_DIR = Path(__file__).parent.parent / "data" / "documents"


# =============================================================================
# 测试类 1：Markdown 解析器
# =============================================================================

class TestMDLoader:
    """Markdown 解析器测试

    测试设计原则：
    - 正常场景：验证文档解析结果的各项字段
    - 边界场景：空文件不崩溃
    - 每个测试方法只测一个关注点（单一职责）
    """

    def test_parse_markdown_document(self):
        """正常文档解析：应有文本、表格和章节

        测试目的：验证 MDLoader 对真实 Markdown 文档的解析效果。
        测试文件 '04-托运行李运输规定.md' 是一份典型的航空规定文档，
        包含文本段落、Markdown 表格和章节标题。

        验证点：
        - 返回类型是 ParsedDocument
        - file_type 是 'md'
        - 提取到了文本内容
        - 识别到了表格
        - 识别到了章节
        - 统计字段自动计算正确
        """
        # 调用 load_document 解析测试文档
        # str() 将 Path 对象转为字符串（load_document 接受 str 类型）
        doc = load_document(str(DATA_DIR / "04-托运行李运输规定.md"))

        # assert isinstance: 验证返回对象的类型
        assert isinstance(doc, ParsedDocument)

        # 验证文件类型识别正确
        assert doc.file_type == "md"

        # 验证文本提取：raw_text 长度应 > 0
        # 第二个参数是断言失败时的错误消息，pytest 会显示它
        assert len(doc.raw_text) > 0, "应提取到文本内容"

        # 验证表格识别：至少识别到 1 张表格
        assert doc.table_count > 0, "应识别到至少1张表格"

        # 验证章节识别：至少识别到 1 个章节
        assert doc.section_count > 0, "应识别到至少1个章节"

        # 验证统计字段自动计算
        assert doc.char_count > 0

        # 验证哈希值已填充（非空字符串）
        assert doc.file_hash != ""

    def test_parse_tables(self):
        """表格解析：舱位代码对照表应有3张表格

        测试目的：验证 Markdown 表格解析的准确性。
        测试文件 '03-舱位代码对照表.md' 已知有 3 张表格，
        如果解析结果少于 3 张，说明表格解析逻辑有问题。

        验证点：
        - 表格数量 ≥ 3
        - 每张表都有表头和数据行
        """
        # 解析舱位代码对照表文档
        doc = load_document(str(DATA_DIR / "03-舱位代码对照表.md"))

        # 验证表格数量：至少应该有 3 张
        # 使用 f-string 在错误消息中显示实际识别数量，方便调试
        assert doc.table_count >= 3, f"应有3张表格，实际识别 {doc.table_count} 张"

        # 验证第一张表的结构
        first_table = doc.tables[0]            # 取第一张表
        assert len(first_table.headers) > 0, "表头不应为空"   # 表头非空
        assert len(first_table.rows) > 0, "表格应有数据行"    # 有数据行

    def test_parse_empty_file(self):
        """空文件不应报错，返回空 ParsedDocument

        测试目的：验证边界场景——空文件不会导致程序崩溃。
        这是一个重要的防御性测试：实际使用中可能遇到空文件
        （比如文档生成失败、文件损坏等），系统应该优雅处理而非抛异常。

        验证点：
        - 空文件解析不抛异常
        - 返回类型仍是 ParsedDocument
        """
        # 创建一个空测试文件
        # write_text("") 写入空字符串到文件
        empty_file = DATA_DIR / "_empty_test.md"   # 文件名以 _ 开头表示临时文件
        empty_file.write_text("", encoding="utf-8")

        # try/finally 确保无论测试成功还是失败，临时文件都会被删除
        # 这是测试隔离性的体现：测试不污染环境
        try:
            # 解析空文件
            doc = load_document(str(empty_file))
            # 验证返回类型正确（没有崩溃）
            assert isinstance(doc, ParsedDocument)
        finally:
            # 清理：删除临时测试文件
            # unlink() 删除文件，等价于 rm
            empty_file.unlink()


# =============================================================================
# 测试类 2：PDF 解析器
# =============================================================================

class TestPDFLoader:
    """PDF 解析器测试

    测试设计原则：
    - 条件跳过：PDF 文件可能不存在（需要手工转换生成），跳过后不影响其他测试
    """

    def test_parse_pdf(self):
        """解析 PDF 版本：对应 markdown 转 PDF 后的文件

        测试目的：验证 PDFLoader 的基本功能。
        注意这个测试可能被跳过——因为 PDF 文件可能需要用户手动
        用 pandoc 或浏览器将 markdown 转为 PDF。

        验证点：
        - 返回类型正确
        - file_type 是 'pdf'
        - 提取到了文本内容
        """
        # 构建 PDF 文件路径
        pdf_file = DATA_DIR / "04-托运行李运输规定.pdf"

        # 条件跳过：如果 PDF 文件不存在，用 pytest.skip 跳过这个测试
        # 这是 pytest 的最佳实践：不因为可选测试文件的缺失而报 FAIL
        if not pdf_file.exists():
            pytest.skip("PDF 文件尚未生成，请先将 markdown 转为 PDF")

        # 解析 PDF 文件
        doc = load_document(str(pdf_file))

        # 验证解析结果类型
        assert isinstance(doc, ParsedDocument)

        # 验证文件类型识别正确
        assert doc.file_type == "pdf"

        # 验证文本提取成功
        assert len(doc.raw_text) > 0, "应提取到 PDF 文本内容"


# =============================================================================
# 测试类 3：DOCX 解析器
# =============================================================================

class TestDocxLoader:
    """DOCX 解析器测试"""

    def test_parse_docx(self):
        """解析 DOCX 版本

        测试目的：验证 DocxLoader 的基本功能。
        和 PDF 测试一样，如果 DOCX 文件不存在就跳过。

        验证点：
        - 返回类型正确
        - file_type 是 'docx'
        """
        # 构建 DOCX 文件路径
        docx_file = DATA_DIR / "04-托运行李运输规定.docx"

        # 条件跳过
        if not docx_file.exists():
            pytest.skip("DOCX 文件尚未生成")

        # 解析 DOCX 文件
        doc = load_document(str(docx_file))

        # 验证解析结果类型
        assert isinstance(doc, ParsedDocument)

        # 验证文件类型识别正确
        assert doc.file_type == "docx"


# =============================================================================
# 测试类 4：异常场景测试
# =============================================================================

class TestLoaderErrors:
    """异常场景测试

    测试设计原则：
    - 防御性测试：验证错误输入能被正确拦截
    - 明确异常类型：不仅验证抛异常，还验证异常类型和消息
    """

    def test_file_not_found(self):
        """不存在的文件应抛出 FileNotFoundError

        测试目的：验证 load_document 对不存在的文件能正确地抛出异常。
        这是输入校验的基本要求——不能悄悄出错，必须明确报错。

        验证点：
        - 抛出 FileNotFoundError 异常
        """
        # pytest.raises 是一个上下文管理器
        # 如果 with 块内的代码抛出了指定类型的异常 → 测试通过
        # 如果没抛异常或抛了其他类型 → 测试失败
        with pytest.raises(FileNotFoundError):
            # 尝试加载一个不存在的文件
            load_document(str(DATA_DIR / "不存在的文件.md"))

    def test_unsupported_format(self):
        """不支持的格式应抛出 ValueError

        测试目的：验证 load_document 对不支持的格式能正确报错。
        .xyz 是故意编造的扩展名，不在 LOADER_REGISTRY 中。

        验证点：
        - 抛出 ValueError 异常
        - 异常消息中包含"不支持的文档格式"（match 参数校验）
        """
        # 创建一个假的 .xyz 文件（任意内容即可）
        fake_file = DATA_DIR / "_test.xyz"
        fake_file.write_text("test", encoding="utf-8")

        # try/finally 确保临时文件被清理
        try:
            # pytest.raises 配合 match 参数：
            # 不仅验证异常类型，还验证异常消息是否匹配正则表达式
            # match="不支持的文档格式" → 异常消息中必须包含这段文本
            with pytest.raises(ValueError, match="不支持的文档格式"):
                load_document(str(fake_file))
        finally:
            # 清理临时测试文件
            fake_file.unlink()
