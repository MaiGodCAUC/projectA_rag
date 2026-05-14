# A-Day 3：条款感知切片策略

## 核心目标

实现 4 种切片策略，核心亮点是 **PolicyClauseSplitter（条款感知切片器）**—— 保证每条政策条款作为完整的语义单元被切分，避免切断编号链导致法律效力丢失。

建立对比实验框架，用数据证明「为什么选这个策略」。

---

## 学习内容

| 知识点 | 说明 |
|--------|------|
| 切片策略原理 | RecursiveCharacter、MarkdownHeader、SemanticChunker 各自适用场景 |
| 民航政策文档特征 | 条款编号层级（第X条 → X.X → X.X.X），切断编号 = 丢失引用能力 |
| chunk_overlap 权衡 | overlap 太小丢失上下文，太大增加噪声 |
| LangChain splitter API | RecursiveCharacterTextSplitter、MarkdownHeaderTextSplitter、SemanticChunker |
| 中文分句 | 政策文档不是按句号分句，而是按条款编号分句 |
| 正则表达式工程化 | re.finditer 定位条款边界，逐段构建 chunk |

---

## 代码任务

### 1. `rag/splitter.py` —— 四种切片策略（300+ 行）

| 策略 | 类名 | 原理 | 适用场景 |
|------|------|------|---------|
| 递归字符切片 | `RecursiveCharSplitter` | 按段落 → 句子 → 字符逐级切 | 通用基准线 |
| 标题层级切片 | `MarkdownHeaderSplitter` | 按 # 标题层级切分 | 有明确章节结构的文档 |
| 语义切片 | `SemanticSplitter` | 按 embedding 相似度断点切 | 语义边界感强的文档 |
| **条款感知切片** | `PolicyClauseSplitter` | 识别「第X条」「X.X」编号 | **民航政策文档定制** |

### 2. `tests/test_splitter.py` —— 切片器单元测试（8+ 用例）

### 3. `doc/splitter_comparison.md` —— 对比实验报告

---

## 差异化亮点

- **PolicyClauseSplitter** 是真正的领域定制：识别 `第X条`、`X.X`、`（一）（二）` 等编号模式
- 不是简单按字符数切分，而是**语义感知 + 结构感知**双重保证
- 对比实验：同一批文档 × 4 种策略，量化输出（切片数、平均长度、长度方差、条款完整率）
- 面试时可以用一张表讲清楚「我为什么选这个方案」

---

## 验收标准

- [ ] 4 种策略都能正确切片，返回 `list[TextChunk]`
- [ ] PolicyClauseSplitter 条款完整率 > 90%（不切断编号链）
- [ ] 对比实验报告清晰呈现各策略差异
- [ ] 单测覆盖所有策略的正常/边界/异常场景

---

## PolicyClauseSplitter 核心算法

```
输入: ParsedDocument（raw_text + sections + tables）
  ↓
步骤1: 用正则找出所有条款边界
  - 「第X条」→ 一级条款边界
  - 「X.X」→ 二级子条款边界  
  - 「X.X.X」→ 三级子条款边界
  ↓
步骤2: 按边界切分 raw_text
  - 每个条款自成 chunk
  - 子条款保持父条款上下文（在 metadata 中记录编号链）
  ↓
步骤3: 关联 sections 和 tables
  - 判断每个 chunk 属于哪个 section
  - 判断每个 chunk 是否包含表格（将关联的表格放入 metadata）
  ↓
步骤4: 过长的 chunk 二次切分
  - 超过 max_chunk_size 的条款用 RecursiveCharacter 再切
  - 但保留条款编号链在 metadata 中
  ↓
输出: list[TextChunk]（含 clause_id + section_title + metadata）
```
