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

## 四种切分策略 —— 逻辑流程图

以下流程图与 `rag/splitter.py` 代码严格一一对应，可以边看代码边对照。

---

### 策略 1：RecursiveCharSplitter（递归字符切片）

入口：`RecursiveCharSplitter.split(doc)`  
核心：`_recursive_char_split(text, chunk_size, chunk_overlap, separators)`  
辅助：`_merge_splits()` / `_hard_split()` / `_add_overlap()`

```
┌─────────────────────────────────────────────────────────────────────┐
│                  RecursiveCharSplitter.split(doc)                   │
│                                                                     │
│  doc.raw_text ──→ _recursive_char_split(                            │
│                        text=raw_text,                               │
│                        chunk_size=500,   ← __init__ 参数              │
│                        chunk_overlap=50, ← __init__ 参数              │
│                        separators=None   ← 使用默认优先级列表          │
│                    )                                                 │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│            _recursive_char_split(text, size, overlap, seps)         │
│                                                                     │
│  seps 默认值: ["\n\n","\n","。","？","！","；","，"," ",""]          │
│                                                                     │
│  ┌──────────────────────┐                                           │
│  │ 终止条件              │                                           │
│  │ len(text) ≤ size ?   │──Yes──→ return [text]  (太短了不必切)      │
│  └────────┬─────────────┘                                           │
│           │ No                                                      │
│           ↓                                                         │
│  ┌──────────────────────┐                                           │
│  │ 取当前分隔符           │                                           │
│  │ sep = seps[0]        │                                           │
│  │ remaining = seps[1:] │                                           │
│  └────────┬─────────────┘                                           │
│           ↓                                                         │
│  ┌──────────────────────┐                                           │
│  │ sep == "" ?           │──Yes──→ _hard_split() → return           │
│  │ (最后一个兜底分隔符)    │          按 chunk_size 硬切字符            │
│  └────────┬─────────────┘                                           │
│           │ No                                                      │
│           ↓                                                         │
│  ┌──────────────────────┐                                           │
│  │ splits = text.split(sep)  ← 按分隔符拆成片段                      │
│  └────────┬─────────────┘                                           │
│           ↓                                                         │
│  ┌──────────────────────────────────────────────┐                   │
│  │ _merge_splits(splits, sep, size)              │                   │
│  │                                               │                   │
│  │ 贪心合并算法：                                  │                   │
│  │   merged = []                                 │                   │
│  │   current = ""                                │                   │
│  │                                               │                   │
│  │   for i, s in enumerate(splits):              │                   │
│  │     piece = s  或  sep + s（拼回分隔符）        │                   │
│  │     ┌─────────────────────┐                   │                   │
│  │     │ len(current)+piece  │                   │                   │
│  │     │     ≤ size ?        │                   │                   │
│  │     └───Yes──┬───No──────┘                   │                   │
│  │             ↓              ↓                   │                   │
│  │     current += piece   merged.append(current) │                   │
│  │                        current = piece片段     │                   │
│  │                                               │                   │
│  │   最后 current 有内容 → merged.append(current) │                   │
│  │                                               │                   │
│  │   例: [300字,100字,250字] size=500            │                   │
│  │   → chunk0=400字, chunk1=250字               │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         ↓                                           │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 遍历 merged 中每个 chunk_text                  │                   │
│  │                                               │                   │
│  │   ┌────────────────────┐                      │                   │
│  │   │ len(chunk_text)    │                      │                   │
│  │   │    ≤ size ?        │                      │                   │
│  │   └──Yes──┬──No───────┘                      │                   │
│  │          ↓            ↓                       │                   │
│  │   result.append   _recursive_char_split(       │                   │
│  │   (chunk_text)      chunk_text,               │   ← 递归！         │
│  │                     size, overlap,            │     用下一级分隔符   │
│  │                     remaining_seps)           │                   │
│  │                   → result.extend(子结果)      │   ← 打平，不嵌套   │
│  └──────────────────────┬───────────────────────┘                   │
│                         ↓                                           │
│  ┌──────────────────────┐                                           │
│  │ overlap > 0           │──Yes──→ _add_overlap(result, overlap)    │
│  │ 且 len(result) > 1?   │          前一个chunk结尾N字 → 拼到后一个   │
│  └────────┬─────────────┘                                           │
│           │ No                                                      │
│           ↓                                                         │
│       return result                                                 │
└─────────────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│           回到 RecursiveCharSplitter.split(doc)                      │
│                                                                     │
│  raw_splits = ["片段1","片段2",...]                                  │
│       ↓                                                             │
│  for i, content in enumerate(raw_splits):                           │
│     chunk_start = doc.raw_text.find(content)  ← 定位字符位置          │
│     section_title = _find_section(doc, chunk_start) ← 关联章节       │
│     → TextChunk(chunk_id, content, source_file, chunk_index,        │
│                 section_title, metadata={strategy:"recursive_char"}) │
│                                                                     │
│  return list[TextChunk]                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 策略 2：MarkdownHeaderSplitter（标题层级切片）

入口：`MarkdownHeaderSplitter.split(doc)`  
核心思路：复用 `doc.sections`（MDLoader 已解析好的章节结构），不重复解析 `raw_text`

```
┌─────────────────────────────────────────────────────────────────────┐
│               MarkdownHeaderSplitter.split(doc)                     │
│                                                                     │
│  输入: doc (ParsedDocument, 含 sections 列表)                        │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────┐                                   │
│  │ doc.sections 为空？           │──Yes──→ 退化为 RecursiveCharSplitter │
│  │ (文档没有章节结构)             │         .split(doc) → return       │
│  └────────────┬─────────────────┘                                   │
│               │ No                                                  │
│               ↓                                                     │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 步骤1: 在 raw_text 中定位每个 section title        │               │
│  │                                                  │               │
│  │ boundaries = []                                  │               │
│  │ for sec in doc.sections:                        │               │
│  │   pos = doc.raw_text.find(sec.title)             │               │
│  │   if pos >= 0:                                  │               │
│  │     boundaries.append({start:pos, level:sec.level,│               │
│  │                        title:sec.title})         │               │
│  │                                                  │               │
│  │ # sections 的 char 位置对应原始 Markdown，          │               │
│  │ # 用 title 在 raw_text 中搜索来重新定位（清洗后文本） │               │
│  └──────────────────────┬───────────────────────────┘               │
│                         ↓                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 去重 + 排序                                       │               │
│  │ - 同一 title 可能在 raw_text 中出现多次             │               │
│  │ - 按 start 升序排列                               │               │
│  └──────────────────────┬───────────────────────────┘               │
│                         ↓                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 步骤2: 按边界切分 raw_text                         │               │
│  │                                                  │               │
│  │ for i, b in enumerate(unique_bounds):            │               │
│  │   seg_start = b["start"]                        │               │
│  │   seg_end   = 下一个boundary的start 或 文末       │               │
│  │   content   = raw_text[seg_start : seg_end]      │               │
│  │                                                  │               │
│  │   section_title = b["title"]                    │               │
│  │   clause_id     = _extract_clause_id(title)      │               │
│  │   parent        = _find_parent_section(...)      │  ← 向上找父标题  │
│  └──────────────────────┬───────────────────────────┘               │
│                         ↓                                           │
│  ┌──────────────────────────────────────────────────┐               │
│  │ 步骤3: 处理过长章节                                │               │
│  │                                                  │               │
│  │   ┌────────────────────────┐                     │               │
│  │   │ len(content) ≤ size ?   │                     │               │
│  │   └──Yes───────┬──No──────┘                     │               │
│  │       ↓               ↓                          │               │
│  │   直接包装       _recursive_char_split(            │               │
│  │   TextChunk         content, size, overlap        │               │
│  │                  ) → 每个子结果包装为              │               │
│  │                      TextChunk                    │               │
│  │                      (metadata.strategy =         │               │
│  │                       "markdown_header_sub")      │               │
│  └──────────────────────┬───────────────────────────┘               │
│                         ↓                                           │
│                    return list[TextChunk]                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 策略 3：SemanticSplitter（语义切片）⏸️ 暂缓

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SemanticSplitter.split(doc)                     │
│                                                                     │
│  当前状态: NotImplementedError                                      │
│  原因: PyTorch 2.3.0 < 2.4, transformers 无法正常导入                │
│                                                                     │
│  预期流程（Day 4 升级 PyTorch 后实现）:                               │
│                                                                     │
│  1. 将 raw_text 按句号/换行切分为句子列表                             │
│  2. 逐句调用 embedding_function 生成向量                             │
│  3. 计算相邻句子的余弦相似度                                          │
│  4. 在相似度低于阈值（百分位数法）的位置标记为断点                      │
│  5. 在断点处切分 → 生成 chunks                                       │
│  6. 包装为 TextChunk(metadata={strategy:"semantic"})                │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 策略 4：PolicyClauseSplitter（条款感知切片）★ 核心

入口：`PolicyClauseSplitter.split(doc)`  
子方法：`_find_clause_boundaries` / `_split_by_boundaries` / `_associate_metadata` / `_handle_oversized` / `_merge_undersized`

```
┌─────────────────────────────────────────────────────────────────────┐
│                 PolicyClauseSplitter.split(doc)                     │
│                                                                     │
│  输入: doc (ParsedDocument, 含 raw_text + sections + tables)         │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1: _find_clause_boundaries(raw_text)                         │
│          ★ TODO(用户) 需要手写                                       │
│                                                                     │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 1. raw_text.split('\n') → lines               │                   │
│  │    char_pos = 0  (追踪累积字符位置)             │                   │
│  └────────────────────┬─────────────────────────┘                   │
│                       ↓                                             │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 2. for line in lines:                        │                   │
│  │      stripped = line.strip()                  │                   │
│  │                                              │                   │
│  │  ┌───────────────────────────────────────┐    │                   │
│  │  │ stripped 以 # 开头？                    │    │                   │
│  │  │ (是 Markdown 标题行)                    │    │                   │
│  │  └──Yes──┬──No──────────────────────────┘    │                   │
│  │         ↓            ↓                        │                   │
│  │  re.search(         re.match(                │                   │
│  │   CLAUSE_PATTERN,   SUBCLAUSE_PATTERN,       │                   │
│  │   stripped)         stripped)                │                   │
│  │         ↓            ↓                        │                   │
│  │  ┌──────────┐  ┌──────────┐                  │                   │
│  │  │匹配成功?  │  │匹配成功?  │                  │                   │
│  │  └Yes┐─No──┘  └Yes┐─No──┘                  │                   │
│  │     ↓               ↓     ↓                   │                   │
│  │  boundary:       boundary: re.match(         │                   │
│  │  level=1        level=     CN_NUM_PATTERN,   │                   │
│  │  label="第X条"   count('.')  stripped)        │                   │
│  │                  +1          ↓                │                   │
│  │                 label=    boundary:          │                   │
│  │                 "X.X"    level=2             │                   │
│  │                          label="（一）"       │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         ↓                                           │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 3. char_pos += len(line) + 1  (+1是\n)       │                   │
│  │    继续下一行                                  │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                     │
│  返回: [{start, level, label, line}, ...]                           │
│  例: [{"start":40,"level":1,"label":"第1条","line":"## 第1条 免费..."},│
│       {"start":54,"level":2,"label":"1.1","line":"### 1.1 国内航线"},  │
│       ...]                                                          │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ┌──────────────────────────────────────────────┐                   │
│  │ boundaries 为空？（文档无任何条款编号）          │                   │
│  └──Yes──→ 退化为 RecursiveCharSplitter.split(doc) → return         │
│     │ No                                                            │
└─────┼───────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2: _split_by_boundaries(raw_text, boundaries)                 │
│          ★ TODO(用户) 需要手写                                       │
│                                                                     │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 情况A: 前言（第一个boundary之前的内容）         │                   │
│  │   if boundaries[0]["start"] > 0:             │                   │
│  │     preface = raw_text[0 : boundaries[0].start]                   │
│  │     → segments.append({content:preface,       │                   │
│  │                        label:None, level:1})  │                   │
│  └────────────────────┬─────────────────────────┘                   │
│                       ↓                                             │
│  ┌──────────────────────────────────────────────┐                   │
│  │ 情况B: 按边界逐个切分                          │                   │
│  │                                              │                   │
│  │   for i, b in enumerate(boundaries):         │                   │
│  │     seg_start = b["start"]                   │                   │
│  │     ┌──────────────────────────┐             │                   │
│  │     │ i+1 < len(boundaries) ?   │             │                   │
│  │     └Yes──┬──No────────────────┘             │                   │
│  │          ↓            ↓                       │                   │
│  │   seg_end =         seg_end =                │                   │
│  │   boundaries[i+1]   len(raw_text)            │                   │
│  │   ["start"]          (文档末尾)               │                   │
│  │          ↓            ↓                       │                   │
│  │   content = raw_text[seg_start : seg_end]     │                   │
│  │   → segments.append({content, label, level,   │                   │
│  │                      start:seg_start})       │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         ↓                                           │
│  返回: [{content, label, level, start}, ...]                        │
│                                                                     │
│  例:                                                                 │
│   boundaries = [第1条@40, 第2条@220, 第3条@450]                      │
│   → segments = [                                                    │
│       {content:"前言文字...", label:None},                           │
│       {content:"## 第1条...全文", label:"第1条"},                     │
│       {content:"## 第2条...全文", label:"第2条"},                     │
│       {content:"## 第3条...全文", label:"第3条"},                     │
│     ]                                                               │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3: _associate_metadata(segments, doc)  [可抄写]               │
│                                                                     │
│  for i, seg in enumerate(segments):                                 │
│    ┌──────────────────────────────────────────────────────┐         │
│    │ ① _match_section(doc, seg.start)                     │         │
│    │    遍历 doc.sections, 找 start ≤ seg.start < end      │         │
│    │    → section_title (如 "第1条 免费行李额")             │         │
│    │                                                       │         │
│    │ ② _match_tables(seg.content, doc)                    │         │
│    │    检查表头关键词是否出现在 seg.content 中              │         │
│    │    → associated_tables (匹配到的 TableData 列表)      │         │
│    │                                                       │         │
│    │ ③ 构建 metadata:                                     │         │
│    │    {strategy:"policy_clause",                        │         │
│    │     clause_label: seg["label"],                      │         │
│    │     level: seg["level"],                             │         │
│    │     has_tables: True/False,                          │         │
│    │     tables: [...]  ← 关联表格摘要(前5列表头+行数)     │         │
│    │    }                                                 │         │
│    └──────────────────────────────────────────────────────┘         │
│    → TextChunk(chunk_id, content, source_file, chunk_index,         │
│                clause_id, section_title, metadata)                  │
│                                                                     │
│  返回: list[TextChunk] (初步)                                        │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4: _handle_oversized(chunks)  [可抄写]                        │
│                                                                     │
│  for chunk in chunks:                                               │
│    ┌─────────────────────────────────────┐                          │
│    │ len(content) ≤ max_chunk_size ?      │──Yes──→ result.append    │
│    └──No─────────────────────────────────┘         (保持不变)        │
│       │                                                              │
│       ↓                                                              │
│    ┌─────────────────────────────────────┐                          │
│    │ _split_on_subclauses(chunk)          │                          │
│    │ 在content中搜索子条款编号              │                          │
│    │ 正则: (?:^|\n)\s*(\d+\.\d+...)\s    │                          │
│    │                                     │                          │
│    │ ┌─────────────────────────┐         │                          │
│    │ │ 找到≥2个子条款边界？       │         │                          │
│    │ └Yes──┬──No───────────────┘         │                          │
│    │       ↓          ↓                   │                          │
│    │  result.extend  _split_by_paragraph  │  ← 兜底：按\n\n段落切    │
│    │  (子条款chunks)  (chunk)             │                          │
│    │                                      │                          │
│    │  例: "3.1 费率...\n3.2 计算..."      │                          │
│    │  → [chunk_3.1, chunk_3.2]           │                          │
│    └─────────────────────────────────────┘                          │
│                                                                     │
│  返回: list[TextChunk] (处理后的)                                    │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Step 5: _merge_undersized(chunks)  [可抄写]                        │
│                                                                     │
│  规则: len(content) < min_chunk_size 且 无 clause_id                │
│        → 合并到前一个 chunk (content 拼接 "\n\n")                    │
│                                                                     │
│  for chunk in chunks:                                               │
│    ┌─────────────────────────────────────────────┐                  │
│    │ len(content) < min 且 无 clause_id 且 有前驱?  │                  │
│    └Yes──→ merged[-1].content += "\n\n" + content │                  │
│     │ No                                           │                  │
│     └──→ merged.append(chunk)                     │                  │
│                                                                     │
│  返回: list[TextChunk] (最终输出)                                    │
└───────────────────────┬─────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     最终输出                                         │
│                                                                     │
│  list[TextChunk]  每个 chunk 包含:                                   │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ .chunk_id      = "行李运输规定_第1条_0"                   │        │
│  │ .content       = "## 第1条 免费行李额\n### 1.1 国内..."   │        │
│  │ .source_file   = "04-托运行李运输规定.md"                │        │
│  │ .chunk_index   = 0                                     │        │
│  │ .clause_id     = "第1条"          ← 条款编号             │        │
│  │ .section_title = "第1条 免费行李额" ← 章节标题            │        │
│  │ .metadata      = {                                     │        │
│  │     strategy: "policy_clause",                          │        │
│  │     clause_label: "第1条",                              │        │
│  │     level: 1,                                          │        │
│  │     has_tables: True,  ← 此 chunk 包含表格              │        │
│  │     tables: [...]      ← 关联表格摘要                   │        │
│  │   }                                                    │        │
│  └─────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 四种策略调用关系总览

```
                  get_splitter(strategy)
                         │
          ┌──────────────┼──────────────┬──────────────┐
          ↓              ↓              ↓              ↓
   "recursive_char" "markdown_header" "semantic"  "policy_clause"
          ↓              ↓              ↓              ↓
   RecursiveChar   MarkdownHeader  SemanticSplitter PolicyClause
   Splitter         Splitter         (暂缓)           Splitter
          │              │                              │
          │              │                              │
          ↓              ↓                              ↓
   _recursive_char  doc.sections            _find_clause_boundaries
   _split          → 定位 title                   ↓
   _merge_splits   → 按边界切              _split_by_boundaries
   _hard_split     → 过长二次切                    ↓
   _add_overlap                              _associate_metadata
                                                    ↓
                                            _handle_oversized
                                             → _split_on_subclauses
                                             → _split_by_paragraph
                                                    ↓
                                            _merge_undersized
```

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
