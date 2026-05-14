"""
文档切片策略 —— 4 种策略 × 统一接口

将 ParsedDocument 切分为 TextChunk 列表，每个 chunk 是检索的最小单元。

----------------------------------------------------------------------
## 你需要自己写的部分

本文件包含本项目的**核心技术亮点**——PolicyClauseSplitter（条款感知切片器）。

**强烈建议你手写 PolicyClauseSplitter 的核心方法，原因：**
1. 正则表达式在真实业务中的应用（条款编号识别）
2. 边界检测算法（如何在不切断编号链的前提下控制 chunk 大小）
3. 这是面试可以展开讲的差异化亮点："我手写了条款感知切片算法"

**学习路径：**
1. 先理解前 2 种策略（递归字符 / 标题层级）—— 可以照抄
2. 重点理解 PolicyClauseSplitter 的设计思路（见类 docstring）
3. 手写 _find_clause_boundaries（正则边界检测）和 _split_by_boundaries（边界切分）

标注了 TODO(用户) 的方法是你需要自己写的核心部分。
标注了 [可抄写] 的方法可以直接复制使用。

注意：本项目不依赖 langchain_text_splitters —— 所有切分逻辑都是自实现的。
这样做的好处：
  1. 你能真正理解切分算法，面试时可以讲清楚
  2. 避免 langchain_text_splitters → transformers → PyTorch 的依赖链问题
  3. 代码更轻量，没有多余抽象层
----------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# 导入依赖
# ---------------------------------------------------------------------------

# re: Python 正则表达式模块，用于 PolicyClauseSplitter 中检测条款编号边界
import re

# typing: 类型提示
from typing import Optional, List

# ParsedDocument: 文档解析后的统一数据结构，切片的输入
# TextChunk: 切片后文本块的数据模型，切片的输出
from rag.models import ParsedDocument, TextChunk


# =============================================================================
# 通用工具函数 —— 递归字符切分算法
# =============================================================================

def _recursive_char_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[str]:
    """递归字符切分算法 —— 自实现版，不依赖任何第三方库

    这是 RecursiveCharacterTextSplitter 的核心算法，面试高频考点。

    算法思想：
    按分隔符优先级逐级尝试切分——先找最优分隔符（段落），切不开再降级（句子→字符）。
    每一级尽量让每个 chunk ≤ chunk_size。

    算法步骤（面试可以画图讲）：
    ┌─────────────────────────────────────┐
    │ 输入: text, chunk_size, overlap,    │
    │       separators 优先级列表          │
    └──────────────┬──────────────────────┘
                   ↓
    ┌─────────────────────────────────────┐
    │ Step 1: 如果 text 已 ≤ chunk_size   │
    │   → 直接返回 [text]                 │
    └──────────────┬──────────────────────┘
                   ↓
    ┌─────────────────────────────────────┐
    │ Step 2: 取当前优先级最高的分隔符      │
    │   按此分隔符 split(text)            │
    └──────────────┬──────────────────────┘
                   ↓
    ┌─────────────────────────────────────┐
    │ Step 3: 合并（merge）小片段          │
    │   遍历 split 结果，尽量让每个 chunk  │
    │   接近 chunk_size 但不超过           │
    └──────────────┬──────────────────────┘
                   ↓
    ┌─────────────────────────────────────┐
    │ Step 4: 对仍然超大的片段，           │
    │   用下一个优先级分隔符递归切分        │
    └──────────────┬──────────────────────┘
                   ↓
    ┌─────────────────────────────────────┐
    │ Step 5: 最后一个兜底策略 ——          │
    │   按 chunk_size 硬切字符             │
    └─────────────────────────────────────┘

    面试话术：
    "递归字符切分的核心是 separator 优先级列表。
    我设置的是 ['\\n\\n', '\\n', '。', '？', '！', '；', '，', ' ', '']，
    优先在段落边界切，找不到段落边界再找句子边界、逗号边界，
    最后才硬切字符。这和人的阅读方式一致。"

    Args:
        text: 待切分的原始文本
        chunk_size: 目标 chunk 大小（字符数）
        chunk_overlap: 相邻 chunk 之间的重叠字符数
        separators: 分隔符优先级列表，默认按段落→句子→逗号→空格的顺序

    Returns:
        切分后的文本片段列表
    """
    # 默认分隔符优先级：段落 → 换行 → 句号 → 问号/感叹号 → 分号 → 逗号 → 空格 → 硬切
    if separators is None:
        separators = ["\n\n", "\n", "。", "？", "！", "；", "，", " ", ""]

    # ---- 终止条件：文本已足够短 ----
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # ---- 取当前优先级最高的分隔符 ----
    sep = separators[0]
    remaining_seps = separators[1:] if len(separators) > 1 else [""]

    # 按分隔符切分（保留分隔符内容）
    # 注意：不能简单 split(sep) 因为会丢失分隔符本身
    # 我们在合并时按顺序拼接，所以用 split 即可
    if sep == "":
        # 最后一个兜底：硬切字符
        # 不需要再尝试合并了，直接按 chunk_size 切
        return _hard_split(text, chunk_size, chunk_overlap)

    splits = text.split(sep)

    # ---- 合并小片段 ----
    merged = _merge_splits(splits, sep, chunk_size)

    # ---- 递归处理仍然超大的片段 ----
    result = []
    for chunk_text in merged:
        if len(chunk_text) <= chunk_size:
            result.append(chunk_text)
        else:
            # 用剩余分隔符递归切分
            sub_result = _recursive_char_split(
                chunk_text, chunk_size, chunk_overlap, remaining_seps
            )
            result.extend(sub_result)

    # ---- 添加 overlap（重叠） ----
    # overlap 的作用：相邻 chunk 之间共享一部分文本
    # 比如 chunk_size=500, chunk_overlap=50:
    #   chunk_0: text[0:500]
    #   chunk_1: text[450:950]  ← 和 chunk_0 重叠 50 字
    # 这样做的好处：检索时不会因为切分点恰好落在关键信息中间而漏掉答案
    if chunk_overlap > 0 and len(result) > 1:
        result = _add_overlap(result, chunk_overlap)

    return result


def _merge_splits(splits: list[str], separator: str, chunk_size: int) -> list[str]:
    """合并过小的片段，使每个 chunk 尽可能接近 chunk_size

    贪心策略：从头开始累加，累加到超过 chunk_size 就截断，开始新的 chunk。

    例如 chunk_size=500，splits = [300字, 100字, 250字, 200字]
    → chunk0 = 300+100 = 400（加第三个 250 会超 500，停）
    → chunk1 = 250
    → chunk2 = 200
    这是 [可抄写] 的辅助函数。

    Args:
        splits: 分隔后的文本片段列表
        separator: 分隔符（用于重新拼接）
        chunk_size: 目标大小

    Returns:
        合并后的片段列表
    """
    # merged = []
    # current = ""
    #
    # for i, s in enumerate(splits):
    #     # 拼上分隔符（除第一个片段外）
    #     piece = s if i == 0 else separator + s
    #
    #     # 判断：加上这个片段后是否超限？
    #     if len(current) + len(piece) <= chunk_size:
    #         # 不超 → 合并进来
    #         current += piece
    #     else:
    #         # 超了 → 保存当前 chunk，开始新的
    #         if current.strip():
    #             merged.append(current)
    #         # 当前片段成为新 chunk 的起点
    #         current = s if s.strip() else ""
    #
    # # 别忘了最后一个 chunk
    # if current.strip():
    #     merged.append(current)
    #
    # return merged

    merged = []
    current = ""

    for i, s in enumerate(splits):
        # 拼上分隔符（除第一个片段外）
        piece = s if i == 0 else separator + s

        # 判断：加上这个片段后是否超限？
        if len(current) + len(piece) <= chunk_size:
            # 不超 → 合并进来
            current += piece
        else:
            # 超了 → 添加当前 chunk，开始新的 chunk
            if current.strip():
                merged.append(current)
            current = s if s.strip() else ""

    # 最后一个chunk
    merged.append(current) if current.strip() else ""

    return  merged

def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """兜底策略：按 chunk_size 硬切字符

    这是最后的手段——当所有更优雅的分隔符都切不动时（比如整段文字
    没有任何标点符号），只能按固定长度硬切。

    面试时可以说：
    "我用硬切作为兜底策略。虽然理论上应该优先语义边界，
    但实际生产中总有极端情况需要退化处理。实际数据中硬切触发率 < 5%。"

    Args:
        text: 待切分文本
        chunk_size: 每块大小
        chunk_overlap: 重叠大小

    Returns:
        等长切分的文本片段列表
    """
    # chunks = []
    # start = 0
    # while start < len(text):
    #     end = start + chunk_size
    #     chunks.append(text[start:end])
    #     start += chunk_size - chunk_overlap  # 减去 overlap 确保连续性
    # return chunks

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap  # 减去 overlap 确保连续性
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """为相邻 chunk 添加重叠文本

    overlap 是 RAG 系统中平衡「上下文完整性」和「信噪比」的关键参数：
    - overlap 太小 → 切分点附近的信息可能丢失
    - overlap 太大 → 冗余信息多，检索信噪比降低
    - 经验值：chunk_size 的 10%（如 500 字的 chunk 用 50 字 overlap）

    Args:
        chunks: 无重叠的 chunk 列表
        overlap: 重叠字符数

    Returns:
        带重叠的 chunk 列表
    """
    # if overlap <= 0 or len(chunks) <= 1:
    #     return chunks
    #
    # result = [chunks[0]]  # 第一个 chunk 不变
    # for i in range(1, len(chunks)):
    #     # 从前一个 chunk 尾部取 overlap 字符 → 拼到当前 chunk 前面
    #     prev = chunks[i - 1]
    #     curr = chunks[i]
    #     # 取 prev 尾部 overlap 个字符（如果 prev 不够长就全取）
    #     overlap_text = prev[-overlap:] if len(prev) >= overlap else prev
    #     result.append(overlap_text + curr)
    #
    # return result

    if overlap <=0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1,len(chunks)):
        # 从前一个chunk 尾部取 overlap 字符 → 拼到当前 chunk 前面
        prev = chunks[i-1]
        curr = chunks[i]
        overlop_text = prev[-overlap:] if len(prev) >= overlap else prev
        result.append(overlop_text + curr)
    return  result


# =============================================================================
# 策略 1：递归字符切片 —— 通用基准线
# =============================================================================

class RecursiveCharSplitter:
    """递归字符切片器 —— 自实现版

    按「段落 → 句子 → 逗号 → 字符」的优先级逐级切分。
    这是最通用的切片方式，也是我们的 **基准线（baseline）**。

    面试时可以说：
    "我以递归字符切片作为基准线，通过对比实验证明
    PolicyClauseSplitter 在民航政策文档上的优势。"
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """初始化递归字符切片器

        Args:
            chunk_size: 每个文本块的最大字符数，默认 500
            chunk_overlap: 相邻文本块之间的重叠字符数，默认 50
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, doc: ParsedDocument) -> list[TextChunk]:
        """对 ParsedDocument 进行递归字符切片

        步骤：
        1. 调用 _recursive_char_split 将 raw_text 切分为多个片段
        2. 将每个片段包装为 TextChunk 对象
        3. 关联 section_title（根据字符位置判断片段属于哪个章节）

        Args:
            doc: 文档解析结果

        Returns:
            TextChunk 列表
        """
        # ---- 步骤 1：递归字符切分 ----
        # raw_splits = _recursive_char_split(
        #     text=doc.raw_text,
        #     chunk_size=self.chunk_size,
        #     chunk_overlap=self.chunk_overlap,
        # )
        raw_splits = _recursive_char_split(
            text = doc.raw_text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # ---- 步骤 2 + 3：包装为 TextChunk + 关联章节 ----
        chunks = []
        for i, content in enumerate(raw_splits):
            if not content.strip():
                continue

            # 根据 chunk 在 raw_text 中的位置找章节
            chunk_start = doc.raw_text.find(content) if content else 0

            chunks.append(TextChunk(
                chunk_id=f"{doc.file_name}_{i}",
                content=content,
                source_file=doc.file_name,
                chunk_index=i,
                section_title=self._find_section(doc, chunk_start),
                metadata={
                    "strategy": "recursive_char",
                    "char_start": chunk_start,
                    "char_end": chunk_start + len(content),
                },
            ))

        return chunks

    def _find_section(self, doc: ParsedDocument, char_pos: int) -> Optional[str]:
        """根据字符位置查找所属章节标题 [可抄写]

        Args:
            doc: 文档解析结果
            char_pos: 字符位置

        Returns:
            章节标题，找不到返回 None
        """
        for sec in doc.sections:
            if sec.start_char <= char_pos < sec.end_char:
                return sec.title
        return None


# =============================================================================
# 策略 2：Markdown 标题层级切片 —— 按章节结构切分
# =============================================================================

class MarkdownHeaderSplitter:
    """按 Markdown 标题层级切片 —— 自实现版

    识别 ## / ### 标题，按标题位置将文档切分为章节块。

    例如：
    ## 第1条 适用范围     ← 成为一个 chunk
    ### 1.1 基本原则      ← 包含在「第1条」的 chunk 中
    ### 1.2 定义          ← 包含在「第1条」的 chunk 中

    优点：章节边界精准，chunk 自然对应文档组织
    缺点：章节长短不一，可能过长/过短

    面试时可以说：
    "标题层级切片是最直觉的方案，但对民航政策文档有个问题——
    '第X条'下可能有 2000 字 + 多个表格，单个 chunk 太大，不适合 embedding。
    PolicyClauseSplitter 解决了这个问题。"
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 60):
        """初始化 Markdown 标题层级切片器

        Args:
            chunk_size: 二次切分的最大字符数（用于处理过长章节）
            chunk_overlap: 二次切分的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, doc: ParsedDocument) -> list[TextChunk]:
        """按章节结构切分文档

        关键设计：不自己解析 raw_text 中的标题（因为 raw_text 已被 MDLoader
        清洗掉了 # 标记），而是利用 doc.sections（MDLoader 已经解析好的章节结构）。

        doc.sections 是 SectionMeta 列表，每个包含 title、level、start_char、end_char。
        这些位置信息是基于原始 Markdown 文本的，但和 raw_text 大致对应。

        步骤：
        1. 如果 doc.sections 为空 → 退回递归切片
        2. 用 section 的 title 在 raw_text 中定位边界
        3. 按边界切分
        4. 过长章节用递归切分二次处理

        Args:
            doc: 文档解析结果

        Returns:
            TextChunk 列表
        """
        # ---- 如果没有章节信息 → 退回递归切片 ----
        if not doc.sections:
            fallback = RecursiveCharSplitter(self.chunk_size, self.chunk_overlap)
            return fallback.split(doc)

        # ---- 步骤 1：用 section title 在 raw_text 中找边界 ----
        # 找到每个 section 在 raw_text 中的起始位置
        boundaries = []
        for sec in doc.sections:
            # 在 raw_text 中搜索 section title 的位置
            # 因为 raw_text 是清洗后的文本，# 标记已被去除
            # 但 section title 的纯文字仍然保留
            pos = doc.raw_text.find(sec.title)
            if pos >= 0:
                boundaries.append({
                    "start": pos,
                    "level": sec.level,
                    "title": sec.title,
                })

        if not boundaries:
            fallback = RecursiveCharSplitter(self.chunk_size, self.chunk_overlap)
            return fallback.split(doc)

        # 去重并排序（同一 title 可能匹配多次）
        seen = set()
        unique_bounds = []
        for b in boundaries:
            if b["start"] not in seen:
                seen.add(b["start"])
                unique_bounds.append(b)
        unique_bounds.sort(key=lambda x: x["start"])

        # ---- 步骤 2：按边界切分 ----
        chunks = []
        chunk_idx = 0

        for i, b in enumerate(unique_bounds):
            seg_start = b["start"]
            seg_end = (
                unique_bounds[i + 1]["start"]
                if i + 1 < len(unique_bounds)
                else len(doc.raw_text)
            )
            content = doc.raw_text[seg_start:seg_end].strip()

            if not content or len(content) < self.chunk_overlap:
                continue

            # 提取 section_title 和 clause_id
            section_title = b["title"]
            clause_id = self._extract_clause_id(section_title)

            # 找父级标题（向上找到最近的更高级别 section）
            parent = self._find_parent_section(doc.sections, b["title"], b["level"])

            # ---- 步骤 3：处理过长章节 ----
            if len(content) <= self.chunk_size:
                chunks.append(TextChunk(
                    chunk_id=f"{doc.file_name}_{chunk_idx}",
                    content=content,
                    source_file=doc.file_name,
                    chunk_index=chunk_idx,
                    clause_id=clause_id,
                    section_title=parent or section_title,
                    metadata={
                        "strategy": "markdown_header",
                        "header_level": b["level"],
                    },
                ))
                chunk_idx += 1
            else:
                sub_splits = _recursive_char_split(
                    content, self.chunk_size, self.chunk_overlap
                )
                for sub_content in sub_splits:
                    if not sub_content.strip():
                        continue
                    chunks.append(TextChunk(
                        chunk_id=f"{doc.file_name}_{chunk_idx}",
                        content=sub_content,
                        source_file=doc.file_name,
                        chunk_index=chunk_idx,
                        clause_id=clause_id,
                        section_title=parent or section_title,
                        metadata={
                            "strategy": "markdown_header_sub",
                            "parent_clause": clause_id,
                            "header_level": b["level"],
                        },
                    ))
                    chunk_idx += 1

        return chunks

    def _extract_clause_id(self, text: str) -> Optional[str]:
        """从标题文字中提取条款编号 [可抄写]"""
        match = re.search(r'第\d+条', text)
        if match:
            return match.group(0)
        match = re.search(r'\d+\.\d+(?:\.\d+)?', text)
        if match:
            return match.group(0)
        return None

    def _find_parent_section(
        self,
        sections: list,
        title: str,
        level: int,
    ) -> Optional[str]:
        """在 sections 列表中向上查找父级标题 [可抄写]

        例如当前 section 是 level=3 的 '1.1 国内航线'，
        向上找到最近的 level=2 section → '第1条 免费行李额'。

        Args:
            sections: doc.sections 列表
            title: 当前 section 的标题
            level: 当前 section 的层级

        Returns:
            父标题文本，找不到返回 None
        """
        found_current = False
        for sec in reversed(sections):
            if sec.title == title:
                found_current = True
                continue
            if found_current and sec.level < level:
                return sec.title
        return None


# =============================================================================
# 策略 3：语义切片 —— 按语义边界切分（暂缓实现）
# =============================================================================

class SemanticSplitter:
    """语义切片器

    原理：计算相邻句子的 embedding 相似度，在相似度骤降处切分。

    当前状态：此策略依赖 embedding 模型（需要 PyTorch >= 2.4 + sentence_transformers），
    因本地 PyTorch 版本较低暂缓实现。Day 4（Embedding 选型 + Qdrant）完成后可回头补充。

    替代方案：
    如果面试时想强调「语义切片」这个点，可以用 LLM API 的 embedding 接口替代本地模型，
    但会增加调用成本。建议优先把 PolicyClauseSplitter 打磨好。

    面试时怎么说：
    "我设计了 4 种切片策略的对比实验框架，其中包括 SemanticChunker。
    当前因为本地 embedding 环境限制暂未跑通语义切片，但我的对比框架
    已经准备好，切换环境后即可补充数据。"
    """

    def __init__(self, embedding_function=None, chunk_size: int = 600):
        self.embedding_function = embedding_function
        self.chunk_size = chunk_size

    def split(self, doc: ParsedDocument) -> list[TextChunk]:
        """暂未实现 —— 需要先解决 PyTorch 版本问题"""
        raise NotImplementedError(
            "SemanticSplitter 需要 PyTorch >= 2.4 + sentence_transformers。"
            "请在 Day 4 完成 embedding 环境搭建后实现此方法。"
            "策略：将文本按句号分句 → 逐句计算 embedding → 相邻句相似度低处切分。"
        )


# =============================================================================
# 策略 4：条款感知切片 —— ★ 本项目的核心差异化亮点 ★
# =============================================================================

class PolicyClauseSplitter:
    """条款感知切片器 —— 专为民航政策文档设计

    为什么需要这个切片器？

    民航政策文档（客规、行李规定等）的特征：
    条款编号层级深，编号链承载法律引用效力。

    例如：
      第3条 逾重行李费
        3.1 国内航线费率 → 表格
        3.2 计算示例 → 示例文本

    传统切分的问题：
    - RecursiveChar: 可能在表格中间切断 → 费率表不完整
    - MarkdownHeader: 「第3条」整块太大（2000+字）

    PolicyClauseSplitter 的做法：
    1. 先识别「第X条」作为一级边界
    2. 再识别「X.X」作为二级边界
    3. 优先在二级边界切，保证每个子条款自成 chunk
    4. 保留编号链在 metadata 中
    5. 表格始终跟随所属条款

    算法流程：
    输入: ParsedDocument（raw_text + sections + tables）
      ↓
    步骤1: _find_clause_boundaries()
           用正则找出所有条款边界位置
           「第X条」→ 一级边界  「X.X」→ 二级边界
      ↓
    步骤2: _split_by_boundaries()
           按边界将 raw_text 切分为条款单元
      ↓
    步骤3: _associate_metadata()
           关联：编号链、所属章节、关联表格
      ↓
    步骤4: _handle_oversized()
           超长条款在子条款边界二次切分
      ↓
    输出: list[TextChunk]（含 clause_id + section_title）

    面试话术参考：
    "民航政策文档的引用方式是'根据《行李运输规定》第3.2条'，
    如果切片把 3.2 切断了，检索时就无法精确匹配到完整条款。
    我设计了 PolicyClauseSplitter，识别条款编号模式作为切分边界，
    保证每条条款作为完整的检索单元存在。"
    """

    # ================================================================
    # 条款编号正则模式
    # ================================================================

    # 一级条款：「第数字章」或「第数字条」
    # 例：第1条、第12条、第一章
    CLAUSE_PATTERN = r'第(\d+)(?:章|条)'

    # 二级/三级子条款：「数字.数字」或「数字.数字.数字」
    # 例：1.1、3.2.1
    # count('.') + 1 = 编号层级
    SUBCLAUSE_PATTERN = r'\d+\.\d+(?:\.\d+)?'

    # 中文序号：「（一）」「（二）」等
    CN_NUM_PATTERN = r'（[一二三四五六七八九十]+）'

    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 50):
        """初始化条款感知切片器

        Args:
            max_chunk_size: 单个 chunk 的最大字符数
            min_chunk_size: 最小 chunk 字符数，小于此值会与相邻合并
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def split(self, doc: ParsedDocument) -> list[TextChunk]:
        """对文档进行条款感知切片（主入口）

        四步走：
        1. _find_clause_boundaries → 找边界
        2. _split_by_boundaries     → 按边界切分
        3. _associate_metadata      → 关联元数据
        4. _handle_oversized         → 处理超长条款

        Args:
            doc: 文档解析结果

        Returns:
            TextChunk 列表
        """
        # 步骤 1：找到所有条款边界位置
        boundaries = self._find_clause_boundaries(doc.raw_text)

        # 如果没有找到任何条款边界 → 退化为递归切片
        if not boundaries:
            fallback = RecursiveCharSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=50,
            )
            return fallback.split(doc)

        # 步骤 2：按边界切分为条款单元
        clause_segments = self._split_by_boundaries(doc.raw_text, boundaries)

        # 步骤 3：为每个条款单元关联元数据
        chunks = self._associate_metadata(clause_segments, doc)

        # 步骤 4：处理超长条款
        final_chunks = self._handle_oversized(chunks)

        # 步骤 5：合并过短的 chunk
        final_chunks = self._merge_undersized(final_chunks)

        return final_chunks

    # ------------------------------------------------------------------
    # 步骤 1：边界检测 —— ★ TODO(用户)：核心方法，需要你手写 ★
    # ------------------------------------------------------------------

    def _find_clause_boundaries(self, raw_text: str) -> list[dict]:
        """找出 raw_text 中所有条款编号的边界位置

        这是整个 PolicyClauseSplitter 中最核心的方法。
        它决定了「在哪里切」。

        需要识别的模式（按优先级）：
        1. 「第X条」或「第X章」—— 一级条款边界
        2. 「X.X」或「X.X.X」—— 二级/三级子条款边界
        3. 「（一）（二）等中文序号」—— 备选编号边界

        每个边界记录：
        - start: 在 raw_text 中的起始字符位置
        - level: 边界层级（1=第X条, 2=X.X, 3=X.X.X）
        - label: 条款编号文本（如 '第3条'、'1.1'、'（二）'）
        - line: 所在行的完整文本

        TODO(用户): 参考下面的实现思路，自己手写这个方法

        实现提示：
        1. 将 raw_text 按 \\n 拆分为行列表
        2. 用 char_pos 追踪当前累积字符位置（包括换行符）
        3. 遍历每一行，对每行用 re.match 从行首匹配模式
        4. 匹配成功 → 记录边界信息
        5. char_pos += len(line) + 1（+1 是换行符\\n）
        6. 返回按 start 排序的边界列表

        关键判断逻辑：
        - 如果行以 # 开头且包含「第X条」→ 一级边界（优先级最高）
        - 如果行以数字开头且包含「.」→ 二级边界
        - 如果行以中文括号开头「（X）」→ 备选边界

        Args:
            raw_text: 文档纯文本

        Returns:
            边界信息字典列表，每个包含 {start, level, label, line}
        """
        # ================================================================
        # TODO(用户): 从这里开始手写 —— 条款边界检测
        # ================================================================
        #
        # 实现思路（写在你的文件中）：
        #
        # boundaries = []
        # lines = raw_text.split('\n')
        # char_pos = 0
        #
        # for line in lines:
        #     stripped = line.strip()
        #
        #     # 判断1：当前行是 Markdown 标题行（以 # 开头）且包含「第X条」？
        #     if stripped.startswith('#'):
        #         match = re.search(self.CLAUSE_PATTERN, stripped)
        #         if match:
        #             boundaries.append({
        #                 "start": char_pos,
        #                 "level": 1,
        #                 "label": match.group(0),
        #                 "line": stripped,
        #             })
        #     else:
        #         # 判断2：当前行以数字.数字开头？（子条款编号）
        #         match = re.match(self.SUBCLAUSE_PATTERN, stripped)
        #         if match:
        #             label = match.group(0)
        #             level = label.count('.') + 1
        #             boundaries.append({
        #                 "start": char_pos,
        #                 "level": level,
        #                 "label": label,
        #                 "line": stripped,
        #             })
        #         else:
        #             # 判断3：中文序号 «（一）（二）»？
        #             match = re.match(self.CN_NUM_PATTERN, stripped)
        #             if match:
        #                 boundaries.append({
        #                     "start": char_pos,
        #                     "level": 2,
        #                     "label": match.group(0),
        #                     "line": stripped,
        #                 })
        #
        #     char_pos += len(line) + 1  # +1 是换行符 \n
        #
        # return boundaries
        #
        # ================================================================
        # 以上是需要你手写的核心逻辑
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 请参考注释中的实现思路，手写 _find_clause_boundaries 方法"
        )

    # ------------------------------------------------------------------
    # 步骤 2：按边界切分 —— ★ TODO(用户)：核心方法，需要你手写 ★
    # ------------------------------------------------------------------

    def _split_by_boundaries(
        self, raw_text: str, boundaries: list[dict]
    ) -> list[dict]:
        """按边界位置将文本切分为条款单元

        核心思想：相邻两个 boundary 之间的文本 = 一个条款。

        TODO(用户): 参考下面的实现思路，自己手写这个方法

        实现提示：
        1. 处理「前言」：第一个 boundary 之前的内容
        2. 遍历 boundaries：相邻边界之间的文本 → 一个条款
        3. 最后一个条款：最后一个 boundary 到文末

        Args:
            raw_text: 文档纯文本
            boundaries: _find_clause_boundaries 返回的边界列表

        Returns:
            条款单元列表 [{content, label, level, start}, ...]
        """
        # ================================================================
        # TODO(用户): 从这里开始手写 —— 按边界切分文本
        # ================================================================
        #
        # 实现思路：
        #
        # segments = []
        #
        # # 情况1：第一个边界之前的内容（前言/无编号文本）
        # if boundaries[0]["start"] > 0:
        #     preface = raw_text[0 : boundaries[0]["start"]].strip()
        #     if preface:
        #         segments.append({
        #             "content": preface,
        #             "label": None,
        #             "level": 1,
        #             "start": 0,
        #         })
        #
        # # 情况2：遍历边界，相邻边界之间 = 一个条款
        # for i, b in enumerate(boundaries):
        #     seg_start = b["start"]
        #     # 找下一个边界：如果有 → 作为结束位置，如果没有 → 到文末
        #     if i + 1 < len(boundaries):
        #         seg_end = boundaries[i + 1]["start"]
        #     else:
        #         seg_end = len(raw_text)
        #
        #     content = raw_text[seg_start:seg_end].strip()
        #     if content:
        #         segments.append({
        #             "content": content,
        #             "label": b["label"],
        #             "level": b["level"],
        #             "start": seg_start,
        #         })
        #
        # return segments
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 请参考注释中的实现思路，手写 _split_by_boundaries 方法"
        )

    # ------------------------------------------------------------------
    # 步骤 3：关联元数据 —— [可抄写]
    # ------------------------------------------------------------------

    def _associate_metadata(
        self,
        segments: list[dict],
        doc: ParsedDocument,
    ) -> list[TextChunk]:
        """为每个条款单元关联元数据 [可抄写]

        包装 raw text segments 为 TextChunk，关联：
        - section_title: 根据位置查找
        - clause_id: 从 segment label 提取
        - metadata: 关联的表格数据

        Args:
            segments: _split_by_boundaries 返回的条款单元
            doc: 原始 ParsedDocument

        Returns:
            TextChunk 列表
        """
        chunks = []

        for i, seg in enumerate(segments):
            # 关联章节标题
            section_title = self._match_section(doc, seg.get("start", 0))

            # 关联表格（启发式匹配）
            associated_tables = self._match_tables(seg["content"], doc)

            # 构建 metadata
            meta = {
                "strategy": "policy_clause",
                "clause_label": seg["label"],
                "level": seg["level"],
                "has_tables": len(associated_tables) > 0,
            }
            if associated_tables:
                meta["tables"] = [
                    {
                        "headers": t.headers[:5],
                        "row_count": len(t.rows),
                    }
                    for t in associated_tables
                ]

            chunks.append(TextChunk(
                chunk_id=f"{doc.file_name}_{seg['label'] or 'preface'}_{i}",
                content=seg["content"],
                source_file=doc.file_name,
                chunk_index=i,
                clause_id=seg.get("label"),
                section_title=section_title,
                metadata=meta,
            ))

        return chunks

    # ------------------------------------------------------------------
    # 步骤 4：处理超长条款 —— [可抄写]
    # ------------------------------------------------------------------

    def _handle_oversized(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """对超过 max_chunk_size 的条款进行二次切分 [可抄写]

        优先级：
        1. 在子条款边界切（如 3.1、3.2）
        2. 在段落边界切（\\n\\n）
        3. 在句子边界切（。）

        Args:
            chunks: 初步 TextChunk 列表

        Returns:
            处理后的 TextChunk 列表
        """
        result = []

        for chunk in chunks:
            if len(chunk.content) <= self.max_chunk_size:
                result.append(chunk)
                continue

            # 尝试子条款边界切分
            sub = self._split_on_subclauses(chunk)
            if len(sub) > 1:
                result.extend(sub)
            else:
                # 段落级切分
                result.extend(self._split_by_paragraph(chunk))

        return result

    def _split_on_subclauses(self, chunk: TextChunk) -> list[TextChunk]:
        """在子条款边界处二次切分 [可抄写]

        在 chunk.content 中搜索形如「\\n3.1」「\\n3.2」的子条款编号，
        在匹配位置切分。

        Args:
            chunk: 需要切分的 TextChunk

        Returns:
            子条款 TextChunk 列表
        """
        content = chunk.content
        # 搜索行首（或换行后）的子条款编号
        matches = list(re.finditer(
            r'(?:^|\n)\s*(\d+\.\d+(?:\.\d+)?)\s',
            content
        ))

        if len(matches) < 2:
            return [chunk]

        sub_chunks = []
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            sub_content = content[start:end].strip()

            if sub_content:
                sub_chunks.append(TextChunk(
                    chunk_id=f"{chunk.chunk_id}_sub{i}",
                    content=sub_content,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    clause_id=m.group(1),
                    section_title=chunk.section_title,
                    metadata={
                        **chunk.metadata,
                        "parent_clause": chunk.clause_id,
                    },
                ))

        return sub_chunks if sub_chunks else [chunk]

    def _split_by_paragraph(self, chunk: TextChunk) -> list[TextChunk]:
        """按段落边界切分 —— 兜底方案 [可抄写]

        Args:
            chunk: TextChunk

        Returns:
            段落级 TextChunk 列表
        """
        paragraphs = chunk.content.split('\n\n')
        if len(paragraphs) <= 1:
            return [chunk]

        result = []
        for j, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            result.append(TextChunk(
                chunk_id=f"{chunk.chunk_id}_p{j}",
                content=para,
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                clause_id=chunk.clause_id,
                section_title=chunk.section_title,
                metadata={
                    **chunk.metadata,
                    "is_sub_chunk": True,
                },
            ))
        return result if result else [chunk]

    # ------------------------------------------------------------------
    # 步骤 5：合并过短 chunk —— [可抄写]
    # ------------------------------------------------------------------

    def _merge_undersized(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """合并过短 chunk [可抄写]

        规则：如果 chunk 内容 < min_chunk_size 且没有 clause_id，
        则合并到前一个 chunk。

        Args:
            chunks: TextChunk 列表

        Returns:
            合并后的列表
        """
        if not chunks:
            return chunks

        merged = []
        for chunk in chunks:
            if (
                merged
                and len(chunk.content) < self.min_chunk_size
                and not chunk.clause_id
            ):
                merged[-1].content += "\n\n" + chunk.content
            else:
                merged.append(chunk)

        return merged

    # ------------------------------------------------------------------
    # 辅助方法 —— [可抄写]
    # ------------------------------------------------------------------

    def _match_section(self, doc: ParsedDocument, char_pos: int) -> Optional[str]:
        """根据字符位置查找所属章节 [可抄写]"""
        for sec in doc.sections:
            if sec.start_char <= char_pos < sec.end_char:
                return sec.title
        return None

    def _match_tables(self, content: str, doc: ParsedDocument) -> list:
        """找出 content 中包含的表格（启发式匹配）[可抄写]

        通过检查表格表头的第一个关键词是否在 content 中出现。

        Args:
            content: 条款文本
            doc: 原始文档

        Returns:
            匹配的 TableData 列表
        """
        matched = []
        for table in doc.tables:
            if table.headers and len(table.headers[0]) >= 2:
                if table.headers[0][:4] in content:
                    matched.append(table)
        return matched


# =============================================================================
# 工厂函数：根据策略名称获取切片器
# =============================================================================

# 切片器注册表
SPLITTER_REGISTRY = {
    "recursive_char": RecursiveCharSplitter,
    "markdown_header": MarkdownHeaderSplitter,
    "semantic": SemanticSplitter,
    "policy_clause": PolicyClauseSplitter,
}


def get_splitter(strategy: str = "policy_clause", **kwargs):
    """工厂函数：根据策略名称获取切片器实例 [可抄写]

    Usage:
        splitter = get_splitter("policy_clause", max_chunk_size=800)
        splitter = get_splitter("recursive_char", chunk_size=500)

    Args:
        strategy: 策略名称
        **kwargs: 传递给切片器 __init__ 的额外参数

    Returns:
        切片器实例

    Raises:
        ValueError: 不支持的切片策略
    """
    if strategy not in SPLITTER_REGISTRY:
        raise ValueError(
            f"不支持的切片策略: {strategy}。"
            f"支持: {', '.join(SPLITTER_REGISTRY.keys())}"
        )
    splitter_class = SPLITTER_REGISTRY[strategy]
    return splitter_class(**kwargs)
