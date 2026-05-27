# Day 6 附录：LangSmith 上手实操指南

## 前置步骤：注册 LangSmith

### 1. 注册账号（免费）

1. 打开浏览器访问 **https://smith.langchain.com**
2. 点击 **Sign Up**，用 GitHub / Google 账号注册
3. 注册后进入后台，点击左下角头像 → **Settings**
4. 在 **API Keys** 标签页点击 **Create API Key**
5. 复制生成的 Key（格式 `lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`）

### 2. 配置 .env

打开项目根目录的 `.env` 文件，取消注释并填入 Key：

```bash
# ---------- LangSmith 配置 ----------
LANGCHAIN_TRACING_V2=true                         # 开启追踪
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com # 上报地址
LANGCHAIN_API_KEY=lsv2_pt_你的key                 # 你的 API Key
LANGCHAIN_PROJECT=airchina-rag                    # 项目名
```

**重要：**
- Key 名是 `LANGCHAIN_API_KEY`（CHAIN 不是 SMITH），这是 LangChain SDK 直接读的
- `LANGCHAIN_TRACING_V2=true` 告诉 LangChain 开启自动追踪
- 这些变量是 LangChain SDK 在 `import langchain` 时自动检测的，不需要在 config.py 中定义

### 3. 验证配置

在终端运行：

```bash
cd project_a_rag
python -c "
import os
print('LANGCHAIN_TRACING_V2:', os.getenv('LANGCHAIN_TRACING_V2'))
print('LANGCHAIN_API_KEY:', os.getenv('LANGCHAIN_API_KEY', 'NOT SET')[:20] + '...')
print('LANGCHAIN_PROJECT:', os.getenv('LANGCHAIN_PROJECT'))
"
```

看到 `True` 就说明配置成功了。

---

## 操作指南：第一次看到你的 Trace

### 第一步：运行一个简单的 LLM 调用

```bash
cd project_a_rag
python -c "
from core.llm import get_llm
from core.config import get_settings
from langchain_core.messages import HumanMessage

llm = get_llm(get_settings())
response = llm.invoke([HumanMessage(content='你好，一句话介绍国航')])
print(response.content)
"
```

### 第二步：打开 LangSmith 后台

1. 打开 **https://smith.langchain.com**
2. 左侧边栏选择你的项目 **airchina-rag**
3. 你会看到一条新的 Trace 记录！

### 第三步：观察 Trace 详情

点击那条 Trace，你会看到一个树形结构：

```
ChatOpenAI (1.2s) ← 这是根 Run
  ├── metadata: {ls_provider: openai, ls_model_name: ...}
  ├── inputs: {messages: [[{content: "你好，一句话介绍国航"}]]}
  └── outputs: {generations: [[{text: "中国国际航空是..."}]]}
```

**每个节点都可以展开：**
- **inputs**：看 Prompt 是什么
- **outputs**：看 LLM 回答了什么
- **metadata**：看用了什么模型、温度等参数
- **latency**：看花了多少毫秒

---

## 操作指南：看一次完整的 RAG Trace

### 运行带 RAGTraceCallback 的完整流程

```bash
cd project_a_rag
python -c "
import asyncio
from rag.callbacks import RAGTraceCallback

# 模拟一次完整的 RAG 流程
cb = RAGTraceCallback()
print(f'Trace ID: {cb.trace_id}')

# ---- 模拟检索阶段 ----
cb.start_node('context_manage')
# ... 上下文窗口管理 ...
cb.end_node({'input_count': 5, 'output_count': 3})

cb.start_node('build_context')
# ... 构建上下文 ...
cb.end_node({'context_chars': 1200})

cb.start_node('assemble_messages')
# ... 组装消息 ...
cb.end_node()

# ---- 模拟 LLM 调用 ----
cb.on_llm_start({'name': 'ChatOpenAI'}, ['模拟 prompt 文本...' * 50])
# 这里实际调用 LLM
from core.llm import get_llm
from core.config import get_settings
from langchain_core.messages import HumanMessage
llm = get_llm(get_settings())
response = llm.invoke(
    [HumanMessage(content='你好，请一句话介绍国航的行李规定')],
    config={'callbacks': [cb]},      # ← 关键：把 callback 传给 LLM
)
print(f'LLM 回答: {response.content[:100]}...')
cb.on_llm_end(response)

# ---- 模拟引用提取 ----
cb.start_node('citation_extract')
# ... 提取引用 ...
cb.end_node({'citation_count': 2})

# ---- 打印报告 ----
cb.print_report()
"
```

---

## LangSmith 后台怎么看？

运行上面的代码后，打开 LangSmith 后台，你会看到：

### 视图 1：项目总览 (Project Dashboard)

```
airchina-rag 项目
┌────────────────────────────────────────────────────────────┐
│ 📊 今日统计                                                  │
│   Traces: 1         LLM Calls: 1        Total Tokens: 50   │
│   Avg Latency: 1.2s  Success Rate: 100%                     │
│                                                             │
│ 📋 最近 Traces                                               │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ 🟢 ChatOpenAI  ·  1.2s  ·  50 tokens   ·  2分钟前   │    │
│ └─────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

### 视图 2：单条 Trace 详情

点击一条 Trace 进入详情页：

```
Trace: 10baa057
┌────────────────────────────────────────────────────────────┐
│ 📊 概览                                                      │
│   Latency: 1.23s     Tokens: 50     Status: ✅ success      │
│                                                             │
│ 🌲 Trace 树                                                 │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ ▼ ChatOpenAI (1.23s)                                 │   │
│ │   ├─ input:  [{role: user, content: "你好..."}]      │   │
│ │   ├─ output: [{text: "中国国际航空..."}]              │   │
│ │   ├─ metadata: {ls_provider: deepseek,               │   │
│ │   │             ls_model_name: deepseek-chat}         │   │
│ │   └─ token_usage: {prompt: 30, completion: 20}       │   │
│ └──────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

### 视图 3：Feedback（反馈/评分）

在每个 Trace 详情页的右侧，你可以手动评分：

```
┌──────────────────────┐
│ 📝 Feedback          │
│                      │
│ Correctness: 👍 👎   │  ← 手动标注"是否正确"
│ Relevance:   👍 👎   │  ← 手动标注"是否相关"
│                      │
│ 💬 添加评语...        │
└──────────────────────┘
```

这些评分可以用于后续的评估分析——比如统计"有多少 trace 被标为不正确"。

---

## 关键概念速查表

| LangSmith 概念 | 对应到你项目中的什么 | 通俗理解 |
|---------------|-------------------|---------|
| **Project** | `airchina-rag` | 一个项目一个命名空间，所有 trace 按项目分组 |
| **Trace** | 一次 `RAGTraceCallback()` 实例 | 一次用户请求的完整记录（从收到问题到返回答案） |
| **Run** | `callback.start_node("build_context")` | Trace 树中的一个节点，代表一个步骤 |
| **Feedback** | 手动点赞/点踩 | 标注这次回答好不好，用于质量分析 |
| **Dataset** | `eval_dataset.json` | 测试用例集，用于批量评估 |
| **Experiment** | RAGAS 批量跑评估 | 用 Dataset 跑一轮评估，对比不同配置的效果 |

---

## 三种监控来源的对比

你的系统中，LangSmith 的 Trace 信息来自三个层面：

```
┌─────────────────────────────────────────────────────────────┐
│ 层面 1: LangChain 自动上报（设置 LANGCHAIN_TRACING_V2=true）  │
│   每次 self.llm.invoke() 自动生成一个 Run                     │
│   包含: inputs, outputs, token_usage, latency               │
│   不需要写任何代码                                            │
├─────────────────────────────────────────────────────────────┤
│ 层面 2: RAGTraceCallback 自定义埋点（我们的 callbacks.py）    │
│   每个 callback.start_node/end_node 记录一个环节的耗时        │
│   包含: build_context, assemble_messages, citation_extract   │
│   需要手动调用（但不需要改 LangChain 代码）                    │
├─────────────────────────────────────────────────────────────┤
│ 层面 3: LangSmith SDK 手动上报（高级用法，暂未使用）          │
│   可以直接用 langsmith SDK 发送任意事件                       │
│   适合上报「检索命中数」「用户反馈」等自定义指标              │
└─────────────────────────────────────────────────────────────┘
```

目前的实现中：
- **层面 1** 已经通过 `.env` 配置自动生效
- **层面 2** 已经通过 `config={"callbacks": [callback]}` 传递给了 LLM
- 但注意：`callback.start_node/end_node` 的信息只存在 Python 内存中（`callback.metrics`），不会自动上传到 LangSmith。要让这些节点也出现在 LangSmith Trace 树中，需要后续用 LangSmith SDK 手动上报。

---

## 下一步：Day 7 RAGAS 评估

LangSmith 还可以配合 RAGAS 做自动化评估：

1. 在 LangSmith 中创建 **Dataset**（上传 30 条标注好的 QA 对）
2. 用 RAGAS 跑一轮评估 → 生成 **Experiment** 结果
3. 在 LangSmith 后台对比不同配置的效果（如：有 reranker vs 无 reranker）

这就是 Day 7 要做的事情。
