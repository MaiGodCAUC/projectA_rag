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

## 通俗理解 P50 / P95 / P99

### 一句话总结

> **P50 = 一半用户的体验，P95 = 绝大多数用户的体验，P99 = 最倒霉用户的体验。**

### 生活类比 ——「食堂打饭」

你去公司食堂打饭，假设你记录了最近 100 天排队的时间：

| 天数 | 排队时间 | 你是什么感觉 |
|------|---------|------------|
| 前面 50 天 | ≤ 3 分钟 | 「还行，挺快的」 |
| 第 51~90 天 | 3~8 分钟 | 「有点慢，但能忍」 |
| 第 91~95 天 | 8~15 分钟 | 「今天怎么回事？」 |
| 第 96~99 天 | 15~25 分钟 | 「快疯了，要投诉」 |
| 最倒霉 1 天 | **40 分钟** | 「厨师长跑路了？」 |

从这些数据里提取三个数：

```
P50 = 3 分钟   ← 你一半的日子只需要等 3 分钟（正常情况）
P95 = 15 分钟  ← 20 天里只有 1 天比这个慢（偶发拥堵）
P99 = 25 分钟  ← 100 天里只有 1 天比这个慢（极端拥堵）
```

**如果你只看「平均等待时间」呢？**

```
平均 = (3×50 + 6×40 + 12×5 + 20×4 + 40×1) / 100
     = 650 / 100
     = 6.5 分钟
```

6.5 分钟——看起来「还行啊」。但它完全掩盖了你有 5 天等了超过 15 分钟、有 1 天等了 40 分钟的事实。**这就是平均值的问题：它把你最差的体验平均掉了。**

### 换成 RAG 系统的场景

把你换成「国航一线员工」，排队时间换成「知识库查询响应时间」：

```
P50 = 800ms   ← 一半的查询不到 1 秒就出结果（体验好）
P95 = 2500ms  ← 20 次里只有 1 次超过 2.5 秒（偶尔 LLM 慢）
P99 = 8000ms  ← 100 次里只有 1 次超过 8 秒（极少数情况）
```

面试官问你「系统延迟多少」，不要回答「平均 1.2 秒」——要回答「P50 是 800ms，P95 是 2.5s，P99 是 8s」。这样面试官知道你真的懂性能监控。

### 为什么这三个数就能描述整个系统？

想象你把所有请求的延迟从小到大排成一行：

```
[50ms] [80ms] [100ms] ... [500ms] ... [800ms] ... [2s] ... [5s] ... [10s]
   ↑                        ↑                   ↑              ↑
  最好                     中间                 95%的线       最差
```

- **P50（中位数）**：站在中间位置的那个值。一半请求比它快，一半比它慢。
- **P95**：站在「95% 位置」的那个值。只有 5% 的请求比它慢。**这是生产环境最常用的告警阈值。**
- **P99**：站在「99% 位置」的那个值。只关注最差的那 1%。用来发现「极端长尾」问题。

### 面试加分表达

> 「我不只看平均延迟，因为平均值会被极端值拉偏。我关注 P50/P95/P99 三个分位数——P50 反映常态体验，P95 作为告警阈值，P99 用来抓长尾异常。比如某次 P99 突然从 3s 飙到 15s，说明有少数请求遇到了严重问题，大概率是 LLM API 超时。」

---

## ObservabilityCollector 是什么？

### 一句话总结

> **ObservabilityCollector 就是 RAG 系统的「黑匣子」——飞行过程中不停记录数据，落地后可以回看整个飞行过程的状态。**

### 它做了什么？（通俗版）

想象你开了一家奶茶店：

| 奶茶店 | RAG 系统 | 对应指标 |
|--------|---------|---------|
| 今天卖了多少杯 | 处理了多少次请求 | `request_count` |
| 有几位客人投诉 | 多少次 LLM 调用失败 | `error_count` |
| 每杯制作花了多久 | 每次查询的响应时间 | `latency_records` |
| 用了多少毫升牛奶 | 消耗了多少 Token | `total_tokens` |
| 煮茶花了多久 vs 加料花了多久 | 检索花了多久 vs LLM 生成花了多久 | `node_stats` |
| 客人最常点什么 | 员工最常问什么问题 | `query_counter` |

一天结束后，你不需要翻看每一张小票——看一眼黑板上的汇总数字就知道今天的运营情况。

**ObservabilityCollector 就是这块黑板。** 每次 RAG 请求完成后，「哔」一声在黑板上更新数字。想看的时候，`get_snapshot()` 就把黑板拍照给你。

### 数据流图

```
用户发来一个问题 "行李摔坏了怎么赔？"
        │
        ▼
   POST /chat ──────────────────────────┐
        │                               │
        ▼                               │
   hybrid_searcher.search()  ← 检索     │  RAGTraceCallback
        │                     200ms     │  记录每个节点耗时
        ▼                               │
   generator.generate()      ← LLM生成  │
        │                     1200ms    │
        ▼                               │
   返回回答给用户 ───────────────────────┘
        │
        ▼
   get_collector().record_request({     ← 「哔」打卡
       "total_ms": 1452,                ┐
       "tokens": 380,                   │ 这些数据进入
       "query": "行李摔坏了怎么赔？",     │ ObservabilityCollector
       "success": True,                 │ 的 6 个计数器
   })                                   ┘
        │
        ▼
   下次 GET /metrics → get_snapshot() 就能看到更新后的统计
```

---

## ObservabilityCollector 设计思路

### 设计决策 1：为什么要存在内存里而不是数据库里？

```
方案 A：存在 Python 变量里（现在这样做）
  优点：零依赖、启动即用、面试演示不需要额外装任何东西
  缺点：进程重启数据丢失

方案 B：存在 Redis/PostgreSQL 里
  优点：持久化、多进程共享、支持 Grafana 可视化
  缺点：需要在 docker-compose 里多加一个服务

选择 A 的原因：
  - 这是 Demo/面试项目，不需要生产级持久化
  - 「一行代码不写就能看到指标」比「需要先启动 Redis」更好演示
  - 如果真的需要 Redis，把 record_request() 改成写 Redis 就行
    ——接口不变，换存储实现即可（策略模式的思想）
```

### 设计决策 2：为什么用 `threading.Lock`？

```
问题场景：FastAPI 是多线程的

线程1                    线程2                    结果
──────────────────────────────────────────────────────────
读取 count=0                                           count=0
                         读取 count=0                  count=0
count = 0+1 = 1                                        count=1
                          count = 0+1 = 1               count=1  ← 丢了 1 次！

两次请求，计数应该是 2，但结果是 1。这就是「丢失更新」（Lost Update）

加了 Lock 以后：

线程1                    线程2                    结果
──────────────────────────────────────────────────────────
🔒 获取锁                 等待...                    
读取 count=0              等待...                    
count = 0+1 = 1           等待...                    
写入 count=1              等待...                    
🔓 释放锁                 🔒 获取锁                  
                          读取 count=1               
                          count = 1+1 = 2            
                          写入 count=2               
                          🔓 释放锁                  

结果正确：count = 2 ✓
```

**一句话理解 Lock**：就像厕所门锁——进去的人锁上，外面的人等着，出来的人解锁，下一个才能进。

### 设计决策 3：为什么延迟记录只保留最近 1000 条？

```
不限制数量：           限制为 1000 条：
                       ┌─────────────────────┐
[1年前的请求][...]     │ [最近1000条]         │
[上个月的请求][...]     │ 内存 ≈ 8KB          │
[上周的请求][...]       │ P50/P95/P99 始终    │
[昨天的请求][...]       │ 反映「当前」系统状态 │
[今天的请求][...]       └─────────────────────┘
内存 = 无限增长
P99 = 被历史数据拉偏
       ↓
  内存泄漏            ✓ 内存可控
  指标失真            ✓ 指标有时效性
```

生产环境不这样做（会用 Prometheus + 时间窗口），但 Demo 项目这个方案足够了。

### 设计决策 4：为什么是「全局单例」？

```
错误做法（每次创建新的）：
  POST /chat 处理 → collector1 = ObservabilityCollector()  ← 新的！
  POST /chat 处理 → collector2 = ObservabilityCollector()  ← 又是新的！
  结果：每个 collector 只看到自己那一次的请求，看不到全局统计

正确做法（全局单例）：
  POST /chat 处理 → get_collector() → 返回同一个 collector  ← 共享的
  GET /metrics   → get_collector() → 返回同一个 collector  ← 同一个！
  结果：所有数据汇入同一个收集器，/metrics 能看到全部统计
```

Python 里实现单例最简单的方式：**模块级变量**。Python 的模块本身就是天然的「只有一个」。

```python
_collector = None   # 模块加载时创建，整个进程唯一

def get_collector():
    global _collector
    if _collector is None:       # 第一次调用才创建
        _collector = ObservabilityCollector()
    return _collector            # 之后永远返回同一个
```

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
