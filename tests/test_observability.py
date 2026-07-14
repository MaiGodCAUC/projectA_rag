"""
ObservabilityCollector 单元测试

验证指标收集器的核心功能：
- 基本计数（请求数/错误数/Token）
- 延迟分位数（P50/P95/P99）
- 节点耗时累计
- 查询频率统计
- 线程安全
- 重置功能
"""

import pytest
import threading
import time
from core.observability import ObservabilityCollector, get_collector


class TestObservabilityCollector:
    """指标收集器核心功能测试"""

    def test_initial_state(self):
        """初始状态：所有计数器为 0"""
        c = ObservabilityCollector()
        snap = c.get_snapshot()
        assert snap["request_count"] == 0
        assert snap["error_count"] == 0
        assert snap["error_rate_percent"] == 0.0
        assert snap["total_tokens"] == 0
        assert snap["avg_latency_ms"] == 0.0
        assert snap["p50_latency_ms"] == 0.0
        assert snap["uptime_seconds"] >= 0

    def test_record_basic_counts(self):
        """基本计数：请求数 + 错误数 + Token"""
        c = ObservabilityCollector()
        c.record_request({"total_ms": 100, "tokens": 200, "query": "q1", "success": True})
        c.record_request({"total_ms": 200, "tokens": 300, "query": "q2", "success": False})
        c.record_request({"total_ms": 300, "tokens": 100, "query": "q1", "success": True})

        snap = c.get_snapshot()
        assert snap["request_count"] == 3
        assert snap["error_count"] == 1
        assert snap["total_tokens"] == 600
        # error_rate = 1/3 * 100 = 33.33
        assert snap["error_rate_percent"] == 33.33

    def test_latency_percentiles(self):
        """延迟分位数计算"""
        c = ObservabilityCollector()
        # 插入 100 条延迟记录：1, 2, 3, ..., 100
        for i in range(1, 101):
            c.record_request({"total_ms": float(i), "tokens": 0, "query": "test", "success": True})

        snap = c.get_snapshot()
        # P50: ceil(0.50*100)=50 → index 49 → 50ms
        assert snap["p50_latency_ms"] == 50.0
        # P95: ceil(0.95*100)=95 → index 94 → 95ms
        assert snap["p95_latency_ms"] == 95.0
        # P99: ceil(0.99*100)=99 → index 98 → 99ms
        assert snap["p99_latency_ms"] == 99.0
        # avg = (1+100)*100/2/100 = 50.5
        assert snap["avg_latency_ms"] == 50.5

    def test_percentile_edge_cases(self):
        """分位数边界情况"""
        c = ObservabilityCollector()
        # 空列表 → 0
        assert c._percentile(0.50) == 0.0

        # 单条记录
        c.record_request({"total_ms": 42.0, "tokens": 0, "query": "t", "success": True})
        assert c._percentile(0.50) == 42.0
        assert c._percentile(0.95) == 42.0
        assert c._percentile(0.99) == 42.0

    def test_node_stats(self):
        """节点耗时累计"""
        c = ObservabilityCollector()
        c.record_request({
            "total_ms": 1000, "tokens": 100, "query": "q",
            "nodes": {"llm_generate": 800, "vector_search": 50},
            "success": True,
        })
        c.record_request({
            "total_ms": 800, "tokens": 80, "query": "q2",
            "nodes": {"llm_generate": 600, "vector_search": 40},
            "success": True,
        })

        snap = c.get_snapshot()
        nodes = snap["nodes"]
        assert "llm_generate" in nodes
        assert nodes["llm_generate"]["count"] == 2
        assert nodes["llm_generate"]["total_ms"] == 1400.0
        assert nodes["llm_generate"]["avg_ms"] == 700.0

    def test_query_counter(self):
        """查询频率统计"""
        c = ObservabilityCollector()
        for _ in range(5):
            c.record_request({"total_ms": 100, "tokens": 50, "query": "行李损坏", "success": True})
        for _ in range(3):
            c.record_request({"total_ms": 100, "tokens": 50, "query": "退票规定", "success": True})

        snap = c.get_snapshot()
        top = dict(snap["top_queries"])
        assert top["行李损坏"] == 5
        assert top["退票规定"] == 3

    def test_query_empty_skipped(self):
        """空查询不统计"""
        c = ObservabilityCollector()
        c.record_request({"total_ms": 100, "tokens": 50, "query": "", "success": True})
        c.record_request({"total_ms": 100, "tokens": 50, "query": "valid", "success": True})

        snap = c.get_snapshot()
        top = dict(snap["top_queries"])
        assert "valid" in top
        assert "" not in top

    def test_latency_sliding_window(self):
        """延迟记录滑动窗口（保留最近 1000 条）"""
        c = ObservabilityCollector()
        # 插入 1500 条
        for i in range(1500):
            c.record_request({"total_ms": float(i % 100), "tokens": 0, "query": "t", "success": True})

        assert len(c.latency_records) == 1000  # 只保留最近1000条

    def test_reset(self):
        """重置计数器"""
        c = ObservabilityCollector()
        c.record_request({"total_ms": 100, "tokens": 50, "query": "test", "success": True})

        c.reset()
        snap = c.get_snapshot()
        assert snap["request_count"] == 0
        assert snap["total_tokens"] == 0
        assert snap["p50_latency_ms"] == 0.0
        assert len(snap["top_queries"]) == 0

    def test_get_collector_singleton(self):
        """全局单例：两次调用返回同一个实例"""
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2


class TestObservabilityThreadSafety:
    """线程安全测试"""

    def test_concurrent_record(self):
        """多线程并发写入不丢失数据"""
        c = ObservabilityCollector()
        num_threads = 10
        records_per_thread = 100

        def worker():
            for i in range(records_per_thread):
                c.record_request({
                    "total_ms": float(i),
                    "tokens": i,
                    "query": f"q_{threading.get_ident()}",
                    "success": True,
                })

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = c.get_snapshot()
        assert snap["request_count"] == num_threads * records_per_thread
