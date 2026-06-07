"""
RAG 可观测性指标收集器 —— 运行时聚合统计

和 LangSmith 的区别:
- LangSmith: 单次请求级别的 Trace 追踪（调试用）
- ObservabilityCollector: 时间窗口级别的聚合统计（趋势/告警用）

----------------------------------------------------------------------
## 你需要自己写的部分

这个模块是「RAG 系统的监控后台」——不是看单次请求，而是看整体趋势。
P50/P95/P99 延迟是后台开发面试的常见考点。

学习重点:
1. 延迟分位数（P50/P95/P99）的含义和计算方法
2. 内存级指标收集（不依赖外部数据库）
3. 线程安全（多请求并发时的数据一致性）

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

import math
import time
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict


# =============================================================================
# ObservabilityCollector —— 全局单例指标收集器
# =============================================================================
# 设计思路:
# - 内存级: 所有数据存在 Python 对象中，不依赖 Redis/Prometheus 等外部系统
#   优点: 零依赖、启动即用、面试演示方便
#   缺点: 进程重启数据丢失（生产环境应改用 Prometheus + Grafana）
#
# - 线程安全: 用 threading.Lock 保护共享数据
#   FastAPI 是多线程处理请求的，多个请求同时调用 record_request()
#   会导致数据竞争（data race），所以需要加锁
#
# - 全局单例: 模块加载时创建唯一实例，所有请求共享同一个收集器
#   get_collector() 返回这个全局实例


class ObservabilityCollector:
    """RAG 运行时指标收集器（内存级、线程安全）

    ================================================================
    使用方式:
        from core.observability import get_collector
        collector = get_collector()

        # 在一次 RAG 请求完成后调用:
        collector.record_request({
            "trace_id": "a1b2c3d4",
            "total_ms": 1523,
            "tokens": 500,
            "nodes": {"llm_generate": 1200, "vector_search": 45, ...},
            "query": "旅客行李损坏怎么赔",
            "success": True,
        })

        # 查询统计:
        snapshot = collector.get_snapshot()
        print(snapshot["p95_latency_ms"])  # P95 延迟
    ================================================================
    """

    def __init__(self):
        """初始化指标收集器"""
        # ============================================================
        # _lock: 线程锁
        #   threading.Lock() 创建一个互斥锁
        #   同一时刻只有一个线程能拿到锁，其他线程等待
        #   with self._lock: → 获取锁 → 执行 → 释放锁
        #   保证 record_request() 在多线程并发时数据不被破坏
        # ============================================================
        self._lock = threading.Lock()

        # ---- 计数指标 ----
        self.request_count: int = 0        # 总请求数
        self.error_count: int = 0          # 错误请求数（LLM 失败 / 检索失败）
        self.total_tokens: int = 0         # 累计 Token 消耗（= 累计费用）

        # ---- 延迟指标 ----
        # 存储最近 N 次请求的耗时（用于计算分位数）
        # 为什么只存最近 1000 次？
        #   - 内存可控（1000 × 8 bytes ≈ 8KB）
        #   - 太旧的数据对趋势分析没意义
        #   - 生产环境应该用环形缓冲区（collections.deque(maxlen=1000)）
        self.latency_records: List[float] = []  # 每次请求的 total_ms

        # ---- 节点耗时 ----
        # 存储每个节点的累计耗时和调用次数
        # 结构: {"llm_generate": {"total_ms": 120000, "count": 100}, ...}
        self.node_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total_ms": 0.0, "count": 0}
        )

        # ---- 查询统计 ----
        # 统计最高频的查询（取前 20）
        # 结构: {"旅客行李损坏怎么赔": 5, "金卡能带人进休息室吗": 3, ...}
        self.query_counter: Dict[str, int] = defaultdict(int)

        # ---- 启动时间 ----
        self.start_time: float = time.time()

    # ------------------------------------------------------------------
    # record_request() —— 记录一次 RAG 请求
    # ------------------------------------------------------------------

    def record_request(self, metrics: Dict[str, Any]):
        """记录一次 RAG 请求的指标

        TODO(用户): 手写指标记录逻辑

        这个方法在每次 RAG 请求完成后调用（由 generator 或 API 路由触发）。
        把单次请求的指标累加到全局计数器中。

        metrics 字典结构:
        {
            "trace_id": str,         # 请求追踪 ID
            "total_ms": float,       # 全链路耗时（毫秒）
            "tokens": int,           # Token 消耗量
            "nodes": {               # 各节点耗时
                "context_manage": 1.2,
                "build_context": 0.5,
                "llm_generate": 1200.0,
                "citation_extract": 2.0,
            },
            "query": str,            # 用户问题（用于统计高频查询）
            "success": bool,         # 是否成功
        }
        """
        # ═══════════════════════════════════════════════════════════════
        # TODO(用户): 手写指标记录逻辑（5 个子任务）
        # ═══════════════════════════════════════════════════════════════
        # 把下面注释中的代码「翻译」成实际可运行的代码。
        # 写完后删掉 pass，保留你手写的实现。
        #
        # ── 前置知识 ───────────────────────────────────────────────
        # 1. with self._lock:
        #    threading.Lock() 上下文管理器，同一时刻只允许一个线程进入。
        #    FastAPI 多线程并发时，多个请求同时写 self.request_count
        #    会导致「丢失更新」（lost update），加锁解决这个问题。
        #
        # 2. metrics.get("key", default)
        #    dict 的安全取值方式，key 不存在时返回 default 而不会抛异常。
        #    例: metrics.get("tokens", 0) → tokens 字段存在就取值，
        #        不存在就返回 0。
        #
        # 3. self.latency_records.pop(0)
        #    list.pop(0) 删除并返回第一个元素（FIFO）。
        #    因为 list 底层是数组，pop(0) 需要把所有后续元素前移，
        #    时间复杂度 O(n)。生产环境建议用 collections.deque
        #    （双端队列，popleft O(1)），这里为易读性先用 list。
        #
        # 4. self.node_stats 的结构:
        #    defaultdict(lambda: {"total_ms": 0.0, "count": 0})
        #    访问不存在的 key 时自动创建 {"total_ms": 0.0, "count": 0}
        #    所以可以直接 stats["total_ms"] += node_ms 而不需要检查 key 是否存在。
        #
        # 5. self.query_counter 的结构:
        #    defaultdict(int)
        #    访问不存在的 key 时自动创建 int() = 0
        #    所以可以直接 self.query_counter[query] += 1
        # ─────────────────────────────────────────────────────────────

        # ---- 参考实现（取消注释并逐行理解后重写） ----

        # with self._lock:                                # 获取线程锁
        with self._lock:
            # (1)基本计数
            self.request_count += 1       # 每次请求+1
            if not metrics.get("success", True):
                self.error_count += 1     # 错误数+1

        # ② Token 累计
            self.total_tokens += metrics.get("tokens",0)  # 累加Token消耗
        # ③ 延迟记录（滑动窗口：只保留最近 1000 条）
            total_ms = metrics.get("total_ms", 0)
            self.latency_records.append(total_ms)
            if len(self.latency_records) > 1000:
                self.latency_records.pop(0)
        # ④ 节点耗时累计
            for node_name, node_ms in metrics.get("nodes",{}).items():
                stats = self.node_stats[node_name]          # defaultdict 自动创建
                stats["total_ms"] += node_ms                # 累加总耗时
                stats["count"] += 1                         # 调用次数 +1
        # ⑤ 查询频率统计
            query = metrics.get("query","")
            if query:
                self.query_counter[query] += 1
    # ------------------------------------------------------------------
    # _percentile() —— 计算延迟分位数
    # ------------------------------------------------------------------

    def _percentile(self, p: float) -> float:
        """计算延迟分位数

        TODO(用户): 理解分位数的计算原理

        P50（中位数）: 50% 的请求延迟低于此值
        P95:        95% 的请求延迟低于此值（5% 的请求比这慢）
        P99:        99% 的请求延迟低于此值（长尾最差情况）

        为什么不用平均值？
        平均值掩盖了长尾问题。比如 100 次请求：
          99 次 100ms + 1 次 10000ms → 平均值 = 199ms
          199ms 看起来还好，但实际上有 1% 的用户等了 10 秒！
          P99 = 10000ms → 一下暴露问题

        面试话术:
        "我不只看平均延迟，还看 P95 和 P99。平均值会被极端值拉偏，
        分位数才能反映真实的长尾体验。P99 高说明有少数用户遇到了
        严重延迟——通常是 LLM API 偶发超时。"
        """
        # ═══════════════════════════════════════════════════════════════
        # TODO(用户): 手写分位数计算逻辑
        # ═══════════════════════════════════════════════════════════════
        # 把下面注释中的代码「翻译」成实际可运行的代码。
        # 写完后删掉 pass，保留你手写的实现。
        #
        # ── 前置知识：分位数计算步骤 ──────────────────────────────
        # 1. 防御性检查：如果还没有任何延迟记录，返回 0.0
        #
        # 2. 排序：sorted(self.latency_records)
        #    - sorted() 返回一个新的升序列表，不修改原列表
        #    - 为什么不直接 self.latency_records.sort()？
        #      因为 sort() 是原地修改，如果此时另一个线程正在读
        #      latency_records 就会读到半排序的数据。
        #      sorted() 不会修改原列表，更安全。
        #
        # 3. 计算索引：math.ceil(p * n) - 1
        #    - p 是分位点（0.50 / 0.95 / 0.99）
        #    - n = len(sorted_latency)，记录总数
        #    - math.ceil() 向上取整（ceil  = 天花板）
        #    - 为什么用 ceil 而不是 int/floor？
        #      int(0.95 * 99) = int(94.05) = 94 → index = 93
        #      ceil(0.95 * 99) = ceil(94.05) = 95 → index = 94
        #      用 ceil 保证「至少 p% 的数据 ≤ 返回值」（保守估计）
        #    - 减 1：因为数组索引从 0 开始
        #      第 1 条 → index 0, 第 95 条 → index 94
        #
        # 4. 边界保护：max(0, min(index, n-1))
        #    - min(index, n-1) 确保不超过最后一个元素
        #    - max(0, ...) 确保不小于第一个元素
        #    - 防止 ceil 或浮点精度导致的越界
        #
        # 5. 返回：round(sorted_latency[index], 2)
        #    - 四舍五入到 2 位小数
        #    - 单位是毫秒（ms）
        #
        # ── 图解示例 ─────────────────────────────────────────────
        # 假设 latency_records = [100, 200, 500, 300, 150, 800, 250]
        # sorted → [100, 150, 200, 250, 300, 500, 800]  (n=7)
        #
        # P50: ceil(0.50*7)=4 → index=3 → sorted[3] = 250ms
        #      含义：50% 的请求延迟 ≤ 250ms（中位数）
        #
        # P95: ceil(0.95*7)=7 → index=6 → sorted[6] = 800ms
        #      含义：95% 的请求延迟 ≤ 800ms（只有 5% 超过 800ms）
        #
        # P99: ceil(0.99*7)=7 → index=6 → sorted[6] = 800ms
        #      数据太少时 P95 和 P99 可能相同，数据越多越精确
        # ─────────────────────────────────────────────────────────
        if not self.latency_records:
            return 0.0

        sorted_latency = sorted(self.latency_records)
        index = math.ceil(p * len(sorted_latency)) - 1
        index = max(0, min(index, len(sorted_latency) - 1))
        return round(sorted_latency[index], 2)

    # ------------------------------------------------------------------
    # get_snapshot() —— 获取当前所有指标快照
    # ------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, Any]:
        """获取当前指标快照（不重置计数器）

        返回给 /metrics API 的完整数据：
        - 请求量统计
        - 延迟分位数
        - Token 消耗
        - 各节点耗时分布
        - Top 高频查询
        - 运行时长
        """
        with self._lock:
            # 计算错误率（百分比）
            error_rate = 0.0
            if self.request_count > 0:
                error_rate = round(self.error_count / self.request_count * 100, 2)

            # 平均延迟
            avg_latency = 0.0
            if self.latency_records:
                avg_latency = round(sum(self.latency_records) / len(self.latency_records), 2)

            return {
                # ---- 运行状态 ----
                "uptime_seconds": round(time.time() - self.start_time, 0),
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate_percent": error_rate,

                # ---- 延迟 ----
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": self._percentile(0.50),
                "p95_latency_ms": self._percentile(0.95),
                "p99_latency_ms": self._percentile(0.99),

                # ---- Token ----
                "total_tokens": self.total_tokens,
                "avg_tokens_per_request": round(
                    self.total_tokens / max(self.request_count, 1), 1
                ),

                # ---- 节点耗时 ----
                "nodes": {
                    name: {
                        "avg_ms": round(stats["total_ms"] / max(stats["count"], 1), 2),
                        "total_ms": round(stats["total_ms"], 2),
                        "count": int(stats["count"]),
                    }
                    for name, stats in self.node_stats.items()
                },

                # ---- Top 查询 ----
                "top_queries": sorted(
                    self.query_counter.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20],
            }

    # ------------------------------------------------------------------
    # reset() —— 重置所有计数器
    # ------------------------------------------------------------------

    def reset(self):
        """重置所有计数器（用于测试或定时归零）"""
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.total_tokens = 0
            self.latency_records.clear()
            self.node_stats.clear()
            self.query_counter.clear()
            self.start_time = time.time()


# =============================================================================
# 全局单例
# =============================================================================
# Python 模块加载时创建唯一实例
# 所有 import 这个模块的地方共享同一个 collector
# 这就是「全局单例模式」——不需要设计模式书里那种复杂的 __new__ 实现
# Python 的模块本身就是天然单例

_collector: Optional[ObservabilityCollector] = None


def get_collector() -> ObservabilityCollector:
    """获取全局指标收集器单例

    所有模块通过这个函数拿到同一个 collector 实例：
        from core.observability import get_collector
        collector = get_collector()
        collector.record_request({...})

    首次调用时创建实例，后续返回同一实例。
    """
    global _collector
    if _collector is None:
        _collector = ObservabilityCollector()
    return _collector
