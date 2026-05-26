"""
LangSmith 回调 —— RAG 全链路可观测

在检索、生成等关键节点埋入回调，记录耗时、token 消耗、检索命中数，
用于后续性能分析和问题排查。

----------------------------------------------------------------------
## 你需要自己写的部分

LangSmith 是 LangChain 生态的可观测性平台。回调机制让你能在
RAG 流程的关键节点「插桩」，收集运行时指标。

学习重点：
1. Callback 机制：每个节点执行前后触发 on_*_start / on_*_end
2. Trace 树：嵌套的 Run 形成一棵调用树，根节点 = 一次用户请求
3. 自定义 Tag：给 Run 打标签（如 "retrieval", "generation"），方便过滤分析

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# time: 记录各阶段耗时
import time

# uuid: 生成唯一 trace_id
import uuid

# typing: 类型提示
from typing import Optional, Dict, Any, List

# LangChain 回调基类
from langchain_core.callbacks.base import BaseCallbackHandler

# 配置
from core.config import get_settings


class RAGTraceCallback(BaseCallbackHandler):
    """RAG 全链路追踪回调

    在 RAG 流程的关键节点插桩，记录：
    - 每个阶段的耗时（检索、生成、总耗时）
    - LLM 调用的 token 消耗
    - 检索命中数

    面试话术:
    "我在 RAG 流程的关键节点埋了回调，每个节点记录耗时和状态。
    通过 LangSmith 可以看到完整的调用链——检索花了多少 ms、
    LLM 生成花了多少 ms、哪一步是瓶颈。这对生产环境排查问题非常关键。"
    """

    def __init__(self):
        """初始化回调"""
        self.trace_id = str(uuid.uuid4())[:8]  # 短 trace_id，方便日志展示
        self._start_time = time.time()
        self._node_times: Dict[str, float] = {}  # 每个节点的耗时
        self._current_node: Optional[str] = None  # 当前执行的节点名
        self._node_start: float = 0.0

        # 指标收集
        self.metrics: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retrieval_count": 0,
            "nodes": {},
        }

    # ------------------------------------------------------------------
    # 节点追踪
    # ------------------------------------------------------------------

    def start_node(self, name: str):
        """标记进入某个处理节点

        TODO(用户): 理解节点追踪机制

        在 RAG 流水线的每个阶段开始时调用：
        - "embed_query": Embedding 查询向量化
        - "vector_search": 向量检索
        - "bm25_search": BM25 检索
        - "rrf_fusion": RRF 融合
        - "rerank": 重排序
        - "llm_generate": LLM 生成回答
        - "citation_extract": 引用提取

        Args:
            name: 节点名称
        """
        self._current_node = name
        self._node_start = time.time()

    def end_node(self, extra: Optional[Dict[str, Any]] = None):
        """标记退出当前节点，记录耗时

        Args:
            extra: 额外的节点信息（如检索命中数）
        """
        if self._current_node:
            elapsed = (time.time() - self._node_start) * 1000  # 转为 ms
            self._node_times[self._current_node] = elapsed
            self.metrics["nodes"][self._current_node] = {
                "cost_ms": round(elapsed, 2),
                **(extra or {}),
            }
            self._current_node = None

    # ------------------------------------------------------------------
    # LangChain 回调钩子
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs,
    ):
        """LLM 调用开始时触发（LangChain 自动调用）"""
        self.start_node("llm_generate")
        # 估算 prompt token 数（中文约 1 字符 = 0.5 token）
        total_chars = sum(len(p) for p in prompts)
        self.metrics["prompt_tokens"] = int(total_chars * 0.5)

    def on_llm_end(self, response, **kwargs):
        """LLM 调用结束时触发（LangChain 自动调用）"""
        # 提取 token 用量（如果 LLM 返回了）
        try:
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
                self.metrics["total_tokens"] = usage.get("total_tokens", 0)
                self.metrics["completion_tokens"] = usage.get("completion_tokens", 0)
        except Exception:
            pass
        self.end_node()

    def on_llm_error(self, error, **kwargs):
        """LLM 调用出错时触发"""
        self.end_node({"error": str(error)})

    # ------------------------------------------------------------------
    # 报告生成
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        """生成全链路耗时报告

        Returns:
            {
                "trace_id": "a1b2c3d4",
                "total_cost_ms": 1234,
                "nodes": {
                    "embed_query": {"cost_ms": 15.2},
                    "vector_search": {"cost_ms": 45.8, "hit_count": 20},
                    "bm25_search": {"cost_ms": 3.1, "hit_count": 15},
                    "rrf_fusion": {"cost_ms": 1.2},
                    "llm_generate": {"cost_ms": 1050.5},
                },
                "total_tokens": 1520,
            }
        """
        total_ms = (time.time() - self._start_time) * 1000
        return {
            **self.metrics,
            "total_cost_ms": round(total_ms, 2),
        }

    def print_report(self):
        """打印耗时报告（开发调试用）"""
        report = self.get_report()
        print(f"\n{'='*50}")
        print(f"Trace ID: {report['trace_id']}")
        print(f"Total: {report['total_cost_ms']:.0f}ms")
        print(f"{'='*50}")
        for node, info in report.get("nodes", {}).items():
            extra = ""
            if info.get("hit_count"):
                extra = f" (hits: {info['hit_count']})"
            print(f"  {node:20s}: {info['cost_ms']:8.2f}ms{extra}")
        print(f"  {'Total tokens':20s}: {report.get('total_tokens', 0)}")
