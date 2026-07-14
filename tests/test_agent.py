"""
Agent 路由图结构验证

测试 LangGraph 状态图的编译和节点连接，不涉及 LLM 调用。
"""

import pytest
import sys
import os

# 确保 agent 模块可以导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAgentGraphStructure:
    """Agent 图结构验证（不调 LLM）"""

    def test_graph_builds(self):
        """图编译成功"""
        from agent.router_graph import build_agent_graph
        graph = build_agent_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        """图中包含全部 6 个节点"""
        from agent.router_graph import build_agent_graph
        graph = build_agent_graph()

        # 通过 get_graph() 获取底层图结构，再取 nodes
        inner = graph.get_graph()
        node_names = list(inner.nodes.keys())

        expected_nodes = [
            "intent_classify",
            "retrieve",
            "evaluate_confidence",
            "rewrite_query",
            "generate",
            "generate_fallback",
        ]
        for node in expected_nodes:
            assert node in node_names, f"Node '{node}' not found in {node_names}"

    def test_graph_singleton(self):
        """get_agent_graph() 返回单例"""
        from agent.router_graph import get_agent_graph
        g1 = get_agent_graph()
        g2 = get_agent_graph()
        assert g1 is g2

    def test_get_graph_mermaid(self):
        """Mermaid 可视化输出有效字符串"""
        from agent.router_graph import get_graph_mermaid
        mermaid = get_graph_mermaid()
        assert isinstance(mermaid, str)
        assert "graph TD" in mermaid or "stateDiagram" in mermaid or "---" in mermaid
        # 至少有 100 个字符（正常的图描述不会很短）
        assert len(mermaid) > 50

    def test_agent_state_initialization(self):
        """AgentState 初始值正确"""
        from agent.router_graph import AgentState
        state: AgentState = {
            "query": "test",
            "top_k": 5,
            "intent": "",
            "intent_reason": "",
            "retrieval_results": [],
            "rewrite_count": 0,
            "rewritten_query": "",
            "confidence": 0.0,
            "confidence_reason": "",
            "answer": "",
            "citations": [],
            "trace_id": "",
            "error": "",
        }
        assert state["query"] == "test"
        assert state["top_k"] == 5
        assert state["confidence"] == 0.0
        assert state["rewrite_count"] == 0

    def test_constants(self):
        """常量值正确"""
        from agent.router_graph import MAX_REWRITES, CONFIDENCE_THRESHOLD
        assert MAX_REWRITES == 2
        assert 0 < CONFIDENCE_THRESHOLD < 1

    def test_route_by_intent_policy(self):
        """意图路由：policy_query → retrieve"""
        from agent.router_graph import route_by_intent
        state = {"intent": "policy_query"}
        assert route_by_intent(state) == "retrieve"

    def test_route_by_intent_emergency(self):
        """意图路由：emergency → retrieve"""
        from agent.router_graph import route_by_intent
        state = {"intent": "emergency"}
        assert route_by_intent(state) == "retrieve"

    def test_route_by_intent_uncertain(self):
        """意图路由：uncertain → generate_fallback"""
        from agent.router_graph import route_by_intent
        state = {"intent": "uncertain"}
        assert route_by_intent(state) == "generate_fallback"

    def test_route_by_confidence_high(self):
        """置信度路由：高置信度 → generate"""
        from agent.router_graph import route_by_confidence
        state = {"confidence": 0.85, "rewrite_count": 0, "retrieval_results": [{"score": 0.8}]}
        assert route_by_confidence(state) == "generate"

    def test_route_by_confidence_low_retry(self):
        """置信度路由：低置信度但可重试 → rewrite_query"""
        from agent.router_graph import route_by_confidence
        state = {"confidence": 0.3, "rewrite_count": 0, "retrieval_results": [{"score": 0.1}]}
        assert route_by_confidence(state) == "rewrite_query"

    def test_route_by_confidence_low_maxed(self):
        """置信度路由：低置信度且重试已满 → generate_fallback"""
        from agent.router_graph import route_by_confidence
        state = {"confidence": 0.3, "rewrite_count": 2, "retrieval_results": [{"score": 0.1}]}
        assert route_by_confidence(state) == "generate_fallback"

    def test_route_by_confidence_empty(self):
        """置信度路由：无检索结果 → generate_fallback"""
        from agent.router_graph import route_by_confidence
        state = {"confidence": 0.0, "rewrite_count": 0, "retrieval_results": []}
        assert route_by_confidence(state) == "generate_fallback"


class TestSerializationHelpers:
    """序列化/反序列化工具函数测试"""

    def test_serialize_roundtrip(self):
        """RetrievalResult → dict → RetrievalResult 往返一致"""
        from agent.router_graph import _serialize_results, _deserialize_results
        from rag.models import RetrievalResult, TextChunk

        original = [
            RetrievalResult(
                score=0.85,
                source="hybrid",
                chunk=TextChunk(
                    chunk_id="test_001",
                    content="托运行李损坏赔偿标准",
                    source_file="行李规定.md",
                    clause_id="第3.2条",
                    section_title="破损赔偿",
                ),
            )
        ]

        serialized = _serialize_results(original)
        deserialized = _deserialize_results(serialized, top_k=5)

        assert len(deserialized) == 1
        assert deserialized[0].score == 0.85
        assert deserialized[0].source == "hybrid"
        assert deserialized[0].chunk.content == "托运行李损坏赔偿标准"
        assert deserialized[0].chunk.clause_id == "第3.2条"
        assert deserialized[0].chunk.source_file == "行李规定.md"

    def test_serialize_empty_list(self):
        """空列表序列化"""
        from agent.router_graph import _serialize_results, _deserialize_results
        assert _serialize_results([]) == []
        assert _deserialize_results([], top_k=5) == []
