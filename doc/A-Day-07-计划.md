# A-Day 7：RAGAS 评估体系

## 核心目标

建立 RAG 量化评估体系，用 RAGAS 框架的 4 项指标评估系统质量，
输出评估报告（含雷达图 + Bad Case 分析），形成「评估 → 优化 → 再评估」的闭环。

## 学习内容

### 1. RAGAS 四大指标

| 指标 | 英文 | 评估对象 | 通俗理解 | 计算方式 |
|------|------|---------|---------|---------|
| 忠实度 | Faithfulness | 生成回答 | 回答是否100%基于检索到的文档？有没有编造？ | 从回答中提取断言 → 逐一验证是否在上下文中 |
| 回答相关性 | Answer Relevancy | 生成回答 | 回答是否紧扣用户问题？有没有跑题？ | 用LLM生成反向问题 → 算余弦相似度 |
| 上下文精确度 | Context Precision | 检索结果 | 检索到的文档中，相关文档排在第几位？ | 检索结果中相关文档的平均排名 |
| 上下文召回率 | Context Recall | 检索结果 | 标注答案中的内容，检索结果覆盖了多少？ | 标注答案中的句子被检索结果覆盖的比例 |

### 2. 评估数据集构建

- **手工标注 30 条 QA 对**：覆盖 7 类文档，每类 4~5 条
- **LLM 辅助生成**：从文档中自动提取 QA 对扩充评估集
- **标注字段**：question / answer / context（期望命中的文档+条款）

### 3. RAGAS 工作原理

```
评估数据集 (QA pairs)
        │
        ▼
┌───────────────────┐
│ 你的 RAG 系统      │  ← 对每条 question 走完整流程
│ 检索 → 生成 → 回答 │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ RAGAS 评估器       │  ← 对比「系统回答」和「标注回答」
│ Faithfulness      │
│ Answer Relevancy  │
│ Context Precision │
│ Context Recall    │
└────────┬──────────┘
         │
         ▼
    评估报告（雷达图 + 指标表）
```

## 代码任务

### 任务 1：评估数据集 (`data/eval_dataset.json`)

30 条手工标注的民航员工 QA 对，按以下规范：
- `question`：一线员工的真实提问风格
- `answer`：标准答案（从文档中提取）
- `reference_context`：期望命中的文档 + 条款编号

### 任务 2：LLM 辅助 QA 生成器 (`eval/qa_generator.py`)

从已有文档自动生成 QA 对，用于扩充评估集：
- 输入：文档内容 + 章节标题
- 输出：多条 QA 对
- LLM Prompt：生成一线员工可能提出的问题

### 任务 3：RAGAS 评估流水线 (`eval/ragas_eval.py`)

核心逻辑（TODO 用户手写）：
- 加载评估数据集
- 对每条 question 执行完整 RAG 流程（检索 → 生成）
- 用 RAGAS 计算 4 项指标
- 输出评估报告 + 雷达图

### 任务 4：RAGAS 评估的运行方式

**重要差别**——RAGAS v0.3+ 有两种用法：

**方式 A（旧版，不推荐）**：用 RAGAS 内置的 `evaluate()` 函数
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
results = evaluate(dataset, metrics=[faithfulness, ...])
```

**方式 B（新版，推荐）**：用 RAGAS `SingleTurnSample` + 每个指标的 scorer
```python
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas import SingleTurnSample

sample = SingleTurnSample(
    user_input="...",
    response="...",
    retrieved_contexts=[...],
    reference="..."
)
score = faithfulness.single_turn_score(sample)
```

本项目用**方式 B**，因为：
- 更灵活——可以对每条样本单独处理
- 可以记录中间结果，方便 Bad Case 分析
- 和后续的 LangSmith Experiment 集成更容易

## 差异化亮点

1. **领域评估集**：30 条手工标注的民航员工 QA 对，面试时可以讲清楚每条的设计意图
2. **分文档类别的指标拆解**：不是只有一个总分，而是看「行李类文档检索精度高不高」「投诉类是否容易生成幻觉」
3. **Bad Case 分类**：检索失败 / 生成幻觉 / 引用错误 → 每类有改进方案
4. **RAGAS SingleTurnSample 模式**：代表你用新版 API，证明你关注技术演进

## 验收标准

- RAGAS 评估流程完整跑通（30 条 QA 对全部评估）
- Faithfulness > 0.70，Answer Relevancy > 0.75
- 评估报告含雷达图 + 按文档类别拆解 + Bad Case 分类
- 30 条手工 QA 对覆盖 7 类文档

## 文件清单

```
project_a_rag/
├── eval/
│   ├── qa_generator.py      # LLM 辅助 QA 生成（TODO 用户）
│   └── ragas_eval.py        # RAGAS 评估流水线（TODO 用户核心逻辑）
├── data/
│   └── eval_dataset.json    # 30 条手工标注 QA 对
└── doc/
    └── A-Day-07-计划.md      # 本文件
```
