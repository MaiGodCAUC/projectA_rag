# A-Day 10：智能检索路由 Agent（LangGraph）

## 核心目标

基于 LangGraph 构建智能检索路由 Agent，在 RAG 基础上叠加 Agent 能力——意图分类、多路检索、置信度评估、Query 改写（Self-Reflection）。让系统从「被动检索」升级为「主动决策」。

---

## 学习内容

### 1. LangGraph 核心概念

```
LangGraph 三大核心：

StateGraph（状态图）
  ├── State（状态）: TypedDict 定义，在节点间流转
  ├── Node（节点）: 处理函数，接收 State → 返回 State 更新
  └── Edge（边）: 连接节点，决定执行顺序
       ├── add_edge("A", "B")       → 固定跳转
       └── add_conditional_edges    → 条件跳转（Router 模式的核心）
```

### 2. Agent 路由逻辑设计

```
员工 Query 进入
    │
    ▼
┌──────────────────────┐
│  意图分类 Node        │  LLM 判断：政策查询 / 操作指引 / 应急 / 不确定
└──────┬───────────────┘
       │
       ├── 政策查询 → RAG 知识库检索
       ├── 操作指引 → 全文搜索（更宽松匹配）
       ├── 应急场景 → 标记紧急 + 优先检索
       └── 不确定   → 提示查阅完整文档
       │
       ▼
┌──────────────────────┐
│  检索执行 Node        │  调用 hybrid_searcher.search()
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  置信度评估 Node      │  检索结果 Top-1 相似度是否 > 阈值？
└──────┬───────────────┘
       │
       ├── 高置信度 → 生成回答
       └── 低置信度 → Query 改写（换说法重试，最多 2 次）
              │
              ├── 改写后够好了 → 生成回答
              └── 2 次还不够   → 诚实告知 + 建议联系主管
       │
       ▼
┌──────────────────────┐
│  生成回答 Node        │  调用 RAGGenerator.generate()
└──────────────────────┘
```

### 3. 核心知识点

| 概念 | 作用 | 代码体现 |
|------|------|---------|
| **StateGraph** | Agent 的骨架 | `StateGraph(AgentState)` |
| **Node** | 处理逻辑 | `async def intent_classify(state)` |
| **ConditionalEdge** | 路由决策 | `add_conditional_edges("intent", router, {...})` |
| **Self-Reflection** | 自我纠错 | 置信度低 → 改写 Query → 重新检索 |
| **Checkpoint** | 状态持久化 | 每次节点执行后保存 State 快照 |

---

## 代码任务

### 任务 1：`agent/router_graph.py` —— LangGraph 状态图

**State 设计：**

```python
class AgentState(TypedDict):
    query: str              # 用户原始问题
    intent: str             # 意图分类结果
    top_k: int              # 检索数量
    retrieval_results: list # 检索结果
    confidence: float       # 检索置信度
    rewrite_count: int      # Query 改写次数
    rewritten_query: str    # 改写后的 Query
    answer: str             # 最终回答
    citations: list         # 引用列表
    trace_id: str           # 追踪 ID
    error: str              # 错误信息
```

**5 个 Node：**

| Node | 职责 | 输入 → 输出 |
|------|------|------------|
| `intent_classify` | LLM 判断查询类型 | query → intent |
| `retrieve` | 调用混合检索 | query + top_k → retrieval_results |
| `evaluate_confidence` | 评估检索质量 | retrieval_results → confidence |
| `rewrite_query` | LLM 改写 Query | query → rewritten_query |
| `generate` | 调用 RAG 生成器 | query + retrieval_results → answer + citations |

### 任务 2：API 集成

- `POST /agent/chat` —— Agent 路由对话（非流式）
- `POST /agent/chat/stream` —— Agent 路由对话（SSE 流式）

### 任务 3：状态图可视化

- 用 LangGraph 内置方法输出 Mermaid 图或 PNG

---

## 差异化亮点

1. **意图路由**：不是所有 Query 都走同一条检索链，而是根据意图选择最优检索策略
2. **Self-Reflection**：检索不足时自动改写 Query 重试，体现 Agent 自我纠错能力
3. **置信度评估**：不是「检索到了就生成」，而是先评估质量再决定下一步
4. **可控最大重试**：最多 2 次改写，避免死循环
5. **降级策略**：2 次重试仍不足 → 诚实告知用户而非强行编造

---

## 验收标准

- [ ] LangGraph 状态图可运行，5 个 Node 全部执行
- [ ] 意图分类准确率 > 80%（政策查询/操作指引/应急/不确定）
- [ ] Self-Reflection 改写机制生效（低置信度触发改写）
- [ ] /agent/chat 接口正常响应
- [ ] 状态图可视化输出
- [ ] 28 个已有测试不受影响

---

*Day 10 / 14 · 智能检索路由 Agent*
