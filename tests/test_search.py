"""BM25 + 混合检索 + 重排序 单元测试"""
import pytest
from pathlib import Path

from rag.models import TextChunk, RetrievalResult
from rag.loader import load_document
from rag.splitter import PolicyClauseSplitter
from rag.bm25 import BM25Retriever

DATA_DIR = Path(__file__).parent.parent / "data" / "documents"


@pytest.fixture
def sample_chunks():
    """从测试文档中提取 chunks 供检索测试使用"""
    doc = load_document(str(DATA_DIR / "04-托运行李运输规定.md"))
    splitter = PolicyClauseSplitter(max_chunk_size=800)
    return splitter.split(doc)


class TestBM25Retriever:
    """BM25 检索器测试"""

    def test_index_and_search(self, sample_chunks):
        """基本索引 + 检索：有结果返回"""
        bm25 = BM25Retriever()
        bm25.index(sample_chunks)
        assert bm25.is_ready()
        assert bm25.doc_count > 0

        results = bm25.search("免费行李额", top_k=5)
        assert len(results) > 0
        assert isinstance(results[0][0], TextChunk)

    def test_no_results_for_nonsense(self, sample_chunks):
        """无意义查询应返回空或低分"""
        bm25 = BM25Retriever()
        bm25.index(sample_chunks)
        results = bm25.search("xyz123不存在的内容", top_k=5)
        # BM25 可能返回结果但分数很低
        if results:
            assert results[0][1] < 1.0

    def test_empty_index(self):
        """未索引时返回空列表"""
        bm25 = BM25Retriever()
        assert not bm25.is_ready()
        assert bm25.search("测试") == []

    def test_clause_query(self, sample_chunks):
        """条款编号精确查询：BM25 应能匹配"""
        bm25 = BM25Retriever()
        bm25.index(sample_chunks)
        results = bm25.search("第3条", top_k=3)
        assert len(results) > 0
