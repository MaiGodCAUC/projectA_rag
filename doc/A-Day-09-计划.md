# A-Day 9：可观测性 & LangSmith 集成

## 核心目标

建立 RAG 系统的可观测性体系：LangSmith 全链路追踪 + 自定义指标收集器 + 指标 API 端点。让系统从「黑盒」变为「白盒」，面试时能回答「你的系统延迟瓶颈在哪」「P95 是多少」。

---

## 学习内容

### 1. LLM 应用可观测性核心

| 维度 | 关注指标 | 面试问法 |
|------|---------|---------|
| **延迟** | P50/P95/P99 延迟、各环节耗时占比 | 「检索和生成各占多少时间？」 |
| **质量** | Faithfulness、Answer Relevancy 趋势 | 「怎么发现回答质量下降？」 |
| **成本** | Token 消耗趋势、单次请求成本 | 「日均 Token 消耗多少？」 |
| **稳定性** | 错误率、LLM 调用成功率 | 「怎么监控 LLM 挂了？」 |
| **使用模式** | Top 高频查询、低质量回答占比 | 「员工最常问什么？」 |

### 2. LangSmith Trace / Run / Feedback 机制

```
LangSmith 数据模型：

Project（项目）
  └── Trace（一次完整请求的全链路）
        ├── Run: embed_query（向量化）
        ├── Run: vector_search（Qdrant 检索）
        ├── Run: bm25_search（BM25 检索）
        ├── Run: rrf_fusion（RRF 融合）
        ├── Run: rerank（重排序）
        ├── Run: llm_generate（LLM 生成）
        │     ├── Run: ChatOpenAI（底层 API 调用）
        │     └── ...
        └── Feedback（人工/自动评分）
              ├── faithfullness: 0.85
              ├── answer_relevancy: 0.90
              └── ...
```

### 3. 自定义 Callback 实现业务级埋点

BaseCallbackHandler 钩子：
- `on_llm_start` → LLM 开始调用（记录开始时间）
- `on_llm_end` → LLM 调用结束 → 记录 Token 消耗 + 耗时
- `on_llm_error` → LLM 调用出错 → 记录错误信息
- `on_chain_start/end` → Chain 节点开始/结束

### 4. 内存级指标收集 vs 外部监控系统

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **内存级收集器** | 零依赖、启动即用 | 重启丢失、无法跨进程 | Demo/面试/小规模 |
| **Prometheus + Grafana** | 持久化、告警、可视化 | 需要额外部署 | 生产环境 |
| **LangSmith** | LLM 专用、开箱即用 | 付费、外部依赖 | 调试 + 追踪 |

---

## 代码任务

### 任务 1：实现 `core/observability.py` —— 指标收集器

**核心代码（你需要手写的部分）：**

#### (a) `record_request()` —— 记录一次请求

```python
def record_request(self, metrics: Dict[str, Any]):
    with self._lock:
        # 1. 基本计数
        self.request_count += 1
        if not metrics.get("success", True):
            self.error_count += 1

        # 2. Token 累计
        self.total_tokens += metrics.get("tokens", 0)

        # 3. 延迟记录（保留最近 1000 条）
        total_ms = metrics.get("total_ms", 0)
        self.latency_records.append(total_ms)
        if len(self.latency_records) > 1000:
            self.latency_records.pop(0)

        # 4. 节点耗时累计
        for node_name, node_ms in metrics.get("nodes", {}).items():
            stats = self.node_stats[node_name]
            stats["total_ms"] += node_ms
            stats["count"] += 1

        # 5. 查询频率
        query = metrics.get("query", "")
        if query:
            self.query_counter[query] += 1
```

#### (b) `_percentile()` —— 计算延迟分位数

```python
def _percentile(self, p: float) -> float:
    if not self.latency_records:
        return 0.0

    sorted_latency = sorted(self.latency_records)
    import math
    index = math.ceil(p * len(sorted_latency)) - 1
    index = max(0, min(index, len(sorted_latency) - 1))
    return round(sorted_latency[index], 2)
```

**关键问题——为什么用 `math.ceil`？**

- `p=0.95`，100 条记录 → `ceil(95) - 1 = 94` → 第 95 条（0-index: 94）
- 如果用 `floor`：`floor(95) - 1 = 94` → 一样（100 条时巧合）
- 如果用 `int`（截断）：`int(95) - 1 = 94` → 一样
- 当 `p * n` 不是整数时才有区别：`p=0.95, n=99` → `ceil(94.05)-1=94` vs `floor(94.05)-1=93`
- **ceil 保证「向上取」，确保至少 p% 的数据 ≤ 返回值**（保守估计）

### 任务 2：实现 `api/routes/observability.py` —— 指标 API

4 个端点：

| 端点 | 方法 | 用途 |
|------|------|------|
| `/metrics` | GET | 指标总览（请求量/延迟/Token/错误率） |
| `/metrics/nodes` | GET | 各节点耗时分布 |
| `/metrics/queries` | GET | Top 高频查询 |
| `/metrics/reset` | POST | 重置计数器 |

### 任务 3：集成到 chat.py

在 `/chat` 请求完成后调用 `get_collector().record_request({...})`，把 trace_id、total_ms、tokens、nodes、query、success 上报到收集器。

---

## 差异化亮点

1. **RAG 全链路耗时拆解**：能回答「检索 200ms + 重排序 150ms + 生成 800ms」→ 证明你懂性能瓶颈分析
2. **P50/P95/P99 延迟分位数**：面试高频考点，证明你不只看平均值
3. **内存级零依赖**：不依赖 Prometheus/Grafana，启动即用，面试演示方便
4. **线程安全**：`threading.Lock` 保护共享数据，FastAPI 多线程并发请求下保证数据一致性
5. **三级可观测体系**：
   - LangSmith：单次请求 Trace 追踪（调试用）
   - RAGTraceCallback：管道节点级计时（性能分析用）
   - ObservabilityCollector：时间窗口聚合统计（趋势/告警用）

---

## 验收标准

- [x] `core/observability.py` 指标收集器实现完整（含 P50/P95/P99）
- [x] `api/routes/observability.py` 4 个端点正常工作
- [x] `/chat` 请求完成后自动上报指标
- [x] `main.py` 注册 observability 路由
- [ ] 用户亲自实现 record_request() 和 _percentile() 中的 TODO 方法
- [ ] `GET /metrics` 返回真实数据
- [ ] LangSmith 中能看到完整 RAG 链路（7 个节点）

---

## 面试话术准备

**Q: 你的系统怎么做监控？**

> "我建立了三级可观测体系。最底层是 LangSmith 全链路 Trace——每次 RAG 请求的 embedding、检索、融合、重排序、LLM 生成都有独立 Run，能看到各环节耗时。中间层是自定义 RAGTraceCallback——通过 BaseCallbackHandler 钩子在 LLM 调用前后采集 token 消耗和耗时。最上层是 ObservabilityCollector——聚合级指标收集器，计算 P50/P95/P99 延迟、错误率、Top 查询等。这三层分别对应调试、性能分析、趋势监控三个不同需求。"

**Q: 为什么用 P95/P99 而不是平均值？**

> "平均值掩盖长尾问题。比如 99 次请求 100ms + 1 次 10s → 平均值 199ms，看起来还好，但实际有 1% 的用户等了 10 秒。P99 = 10s 能立刻暴露这个问题。生产环境通常告警阈值设在 P95 或 P99，而不是平均值。"

---

*Day 9 / 14 · 可观测性 & LangSmith 集成*
