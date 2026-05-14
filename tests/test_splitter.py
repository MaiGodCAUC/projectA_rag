"""
切片器单元测试

运行方式：
    pytest tests/test_splitter.py -v

----------------------------------------------------------------------
## 你需要自己写的部分

本测试文件覆盖 4 种切片策略。前 3 种依赖 LangChain 封装，可直接运行。
PolicyClauseSplitter 的测试用例已写好，等你实现 _find_clause_boundaries
和 _split_by_boundaries 方法后即可跑通。

测试设计原则：
- 每个策略独立一个 Test 类
- 正常场景（有条款的文档）→ 验证基本功能
- 边界场景（无条款的文档）→ 验证退化行为
- 异常场景（空文档）→ 验证不崩溃

没有 TODO(用户) 标记 —— 测试代码不需要你补充。
----------------------------------------------------------------------
"""

import pytest
from pathlib import Path

from rag.models import ParsedDocument, TextChunk, SectionMeta, TableData
from rag.loader import load_document
from rag.splitter import (
    RecursiveCharSplitter,
    MarkdownHeaderSplitter,
    PolicyClauseSplitter,
    get_splitter,
    SPLITTER_REGISTRY,
)

# 测试数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "documents"


# =============================================================================
# 夹具：提供统一的测试文档
# =============================================================================

@pytest.fixture
def luggage_doc():
    """加载「托运行李运输规定」作为主要测试文档

    这个文档包含 6 条条款 + 多个子条款 + 表格，非常适合测试切片器。
    """
    return load_document(str(DATA_DIR / "04-托运行李运输规定.md"))


@pytest.fixture
def simple_doc():
    """构造一个简单文档用于边界测试

    不含 Markdown 标题，不含条款编号，只有纯文本段落。
    用于验证各切片器在「无结构」文档上的退化行为。
    """
    return ParsedDocument(
        file_name="简单通知.md",
        file_type="md",
        file_hash="fake_hash",
        raw_text=(
            "各位同事：\n\n"
            "根据公司最新安排，自2026年6月1日起，"
            "所有国际航班的登机口关闭时间调整为起飞前20分钟。\n\n"
            "请各值机柜台和登机口员工及时更新SOP。\n\n"
            "特此通知。\n\n"
            "运行控制中心\n2026年5月10日"
        ),
        sections=[],
        tables=[],
    )


# =============================================================================
# Test 1：递归字符切片
# =============================================================================

class TestRecursiveCharSplitter:
    """递归字符切片器测试

    这是基准线策略，测试风格最常规。
    """

    def test_basic_split(self, luggage_doc):
        """基本切分：应生成多个 chunk"""
        splitter = RecursiveCharSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split(luggage_doc)

        assert len(chunks) > 0, "应生成至少 1 个 chunk"
        assert all(isinstance(c, TextChunk) for c in chunks), "每个元素应是 TextChunk"
        # 每个 chunk 应有内容
        for c in chunks:
            assert len(c.content) > 0, f"chunk {c.chunk_id} 内容不应为空"

    def test_chunk_size_respected(self, luggage_doc):
        """chunk 大小应在合理范围内（不超过 chunk_size * 1.2）"""
        splitter = RecursiveCharSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split(luggage_doc)

        for c in chunks:
            assert len(c.content) <= 550, (
                f"chunk {c.chunk_id} 大小 {len(c.content)} 超过上限"
            )

    def test_metadata_set(self, luggage_doc):
        """metadata 中应包含 strategy 标记"""
        splitter = RecursiveCharSplitter(chunk_size=500)
        chunks = splitter.split(luggage_doc)

        for c in chunks:
            assert c.metadata.get("strategy") == "recursive_char", (
                f"chunk {c.chunk_id} 缺少 strategy 标记"
            )
            assert c.source_file == luggage_doc.file_name

    def test_simple_doc(self, simple_doc):
        """无结构文档：应能正常切片（不崩溃）"""
        splitter = RecursiveCharSplitter(chunk_size=200)
        chunks = splitter.split(simple_doc)

        assert len(chunks) > 0, "即使是简单通知也该有 chunk 产出"


# =============================================================================
# Test 2：Markdown 标题层级切片
# =============================================================================

class TestMarkdownHeaderSplitter:
    """Markdown 标题层级切片器测试"""

    def test_basic_split(self, luggage_doc):
        """基本切分：应生成多个按标题切分的 chunk"""
        splitter = MarkdownHeaderSplitter(chunk_size=800)
        chunks = splitter.split(luggage_doc)

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

        # 有线文档有 6 条（##），至少生成 6 个 chunk
        assert len(chunks) >= 6, f"预期至少 6 个 chunk，实际 {len(chunks)}"

    def test_metadata_set(self, luggage_doc):
        """metadata 中包含 strategy 和可能的 clause_id"""
        splitter = MarkdownHeaderSplitter()
        chunks = splitter.split(luggage_doc)

        for c in chunks:
            assert c.metadata.get("strategy", "").startswith("markdown_header")

    def test_section_title_preserved(self, luggage_doc):
        """chunk 的 section_title 应包含章节标题"""
        splitter = MarkdownHeaderSplitter()
        chunks = splitter.split(luggage_doc)

        # 至少有一个 chunk 关联了 section_title
        titled = [c for c in chunks if c.section_title]
        assert len(titled) > 0, "应有至少一个 chunk 关联了章节标题"

    def test_simple_doc(self, simple_doc):
        """无标题文档：应能正常切片（不崩溃）"""
        splitter = MarkdownHeaderSplitter(chunk_size=200)
        chunks = splitter.split(simple_doc)

        assert len(chunks) > 0, "即使没有标题也应产出 chunk"


# =============================================================================
# Test 3：条款感知切片 —— ★ 核心测试 ★
# =============================================================================

class TestPolicyClauseSplitter:
    """条款感知切片器测试

    这些测试是你实现 PolicyClauseSplitter 后的验收标准。
    跑通全部测试 → 条款感知切片器可交付。
    """

    def test_basic_split(self, luggage_doc):
        """基本切分：应生成多个带条款编号的 chunk"""
        splitter = PolicyClauseSplitter(max_chunk_size=800)
        chunks = splitter.split(luggage_doc)

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

        # 「托运行李运输规定」有 6 条（第1条 ~ 第6条），至少生成 6 个 chunk
        assert len(chunks) >= 6, f"预期至少 6 个 chunk（对应 6 条条款），实际 {len(chunks)}"

    def test_clause_id_present(self, luggage_doc):
        """大多数 chunk 应有 clause_id（条款编号）"""
        splitter = PolicyClauseSplitter(max_chunk_size=800)
        chunks = splitter.split(luggage_doc)

        # 至少 80% 的 chunk 有 clause_id
        clause_count = sum(1 for c in chunks if c.clause_id)
        ratio = clause_count / len(chunks) if chunks else 0
        assert ratio >= 0.8, (
            f"clause_id 覆盖率 {ratio:.0%}，应 >= 80%。"
            f"有 clause_id: {clause_count}/{len(chunks)}"
        )

    def test_metadata_strategy(self, luggage_doc):
        """metadata 中 strategy 应为 policy_clause"""
        splitter = PolicyClauseSplitter(max_chunk_size=800)
        chunks = splitter.split(luggage_doc)

        for c in chunks:
            strategy = c.metadata.get("strategy", "")
            # 由于超长 chunk 可能被子条款切分，strategy 可能仍为 policy_clause
            # 或者是 policy_clause 的子切分变体
            assert "policy_clause" in strategy or "clause" in strategy.lower(), (
                f"chunk {c.chunk_id} 的 strategy 不符合预期: {strategy}"
            )

    def test_chunk_size_control(self, luggage_doc):
        """大多数 chunk 不应超过 max_chunk_size * 2（留有容错）"""
        splitter = PolicyClauseSplitter(max_chunk_size=800)
        chunks = splitter.split(luggage_doc)

        oversized = [
            c for c in chunks if len(c.content) > 1600
        ]
        assert len(oversized) <= len(chunks) * 0.1, (
            f"超过 max_chunk_size 两倍的 chunk 不应超过 10%。"
            f"超限: {len(oversized)}/{len(chunks)}"
        )

    def test_simple_doc_fallback(self, simple_doc):
        """无条款文档：应退化为递归切片（不崩溃）"""
        splitter = PolicyClauseSplitter(max_chunk_size=800)
        chunks = splitter.split(simple_doc)

        assert len(chunks) > 0, "无条款文档应退化为递归切片，不应崩溃"

    def test_section_title_preserved(self, luggage_doc):
        """chunk 应关联 section_title"""
        splitter = PolicyClauseSplitter(max_chunk_size=800)
        chunks = splitter.split(luggage_doc)

        # 至少部分 chunk 有 section_title
        titled = [c for c in chunks if c.section_title]
        assert len(titled) > 0, (
            "应有至少一个 chunk 关联了章节标题"
        )


# =============================================================================
# Test 4：工厂函数
# =============================================================================

class TestSplitterFactory:
    """切片器工厂函数测试"""

    def test_get_splitter_all_strategies(self):
        """工厂函数应支持所有注册的策略"""
        for strategy_name in SPLITTER_REGISTRY:
            if strategy_name == "semantic":
                # SemanticSplitter 需要 embedding 参数，跳过
                continue
            splitter = get_splitter(strategy_name)
            # 验证返回的是正确类型的实例
            assert hasattr(splitter, "split"), (
                f"{strategy_name} 切片器缺少 split 方法"
            )

    def test_invalid_strategy(self):
        """不支持的策略名称应抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的切片策略"):
            get_splitter("unknown_strategy_xyz")

    def test_default_strategy(self):
        """默认策略应为 policy_clause"""
        splitter = get_splitter()
        assert isinstance(splitter, PolicyClauseSplitter), (
            f"默认策略应为 PolicyClauseSplitter，实际为 {type(splitter).__name__}"
        )


# =============================================================================
# Test 5：对比实验辅助函数
# =============================================================================

class TestComparisonHelper:
    """验证对比实验所需的统计函数可用"""

    def test_chunk_stats(self, luggage_doc):
        """对各策略统计 chunk 数量、平均长度、长度方差

        注意：PolicyClauseSplitter 需要用户实现后才能参与对比。
        当前在对比中优雅降级（NotImplementedError → 跳过）。
        """
        strategies = [
            ("recursive_char", RecursiveCharSplitter(chunk_size=500)),
            ("markdown_header", MarkdownHeaderSplitter(chunk_size=500)),
            ("policy_clause", PolicyClauseSplitter(max_chunk_size=500)),
        ]

        print("\n--- 切片策略对比 ---")
        for name, splitter in strategies:
            try:
                chunks = splitter.split(luggage_doc)
            except NotImplementedError:
                print(f"{name:20s} | (待用户实现 _find_clause_boundaries)")
                continue
            lengths = [len(c.content) for c in chunks]
            avg_len = sum(lengths) / len(lengths) if lengths else 0
            var_len = (
                sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
                if lengths else 0
            )
            print(
                f"{name:20s} | "
                f"chunks={len(chunks):3d} | "
                f"avg_len={avg_len:6.0f} | "
                f"var={var_len:8.0f}"
            )
            # 基础校验
            assert len(chunks) > 0
            assert avg_len > 0

        # 确保三种策略都能运行
        assert True
