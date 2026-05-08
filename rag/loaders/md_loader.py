"""
Markdown 文档解析器

解析 .md 和 .txt 文件，提取：
- 纯文本内容（去除格式标记）
- 章节层级结构（## → ### → ####）
- 结构化表格（| col1 | col2 |）

同时作为所有 Loader 的抽象基类参考。
"""

import re
from datetime import datetime

from rag.models import ParsedDocument, SectionMeta, TableData


class MDLoader:
    """Markdown / 纯文本解析器"""

    def parse(self, file_path: str) -> ParsedDocument:
        """解析 Markdown 文件为 ParsedDocument"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return ParsedDocument(
            file_name="",
            file_type="md",
            raw_text=self._extract_text(content),
            sections=self._extract_sections(content),
            tables=self._extract_tables(content),
            parsed_at=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _extract_text(self, content: str) -> str:
        """提取纯文本：去除 Markdown 格式标记，保留可读文本"""
        # 去除表格（表格单独解析，不计入纯文本）
        text = re.sub(r'\|.*\|[\s\S]*?\n\n', '\n', content)
        # 去除格式标记但保留文字
        text = re.sub(r'[#*>`\-]', '', text)
        # 合并多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _extract_sections(self, content: str) -> list[SectionMeta]:
        """提取章节层级结构

        识别 ## / ### / #### 标题，记录每个章节的起止位置。
        level 计算规则：## → 2, ### → 3, #### → 4
        """
        sections = []
        # 匹配 Markdown 标题行：## 标题 / ### 标题 / #### 标题
        pattern = r'^(#{2,4})\s+(.+)$'
        matches = list(re.finditer(pattern, content, re.MULTILINE))

        for i, match in enumerate(matches):
            level = len(match.group(1))  # ## = 2, ### = 3
            title = match.group(2).strip()
            start = match.start()
            # 结束位置：下一个同级或更高级标题的开始，或文档末尾
            end = self._find_section_end(content, start, i, matches)

            sections.append(SectionMeta(
                title=title,
                level=level,
                start_char=start,
                end_char=end,
            ))

        return sections

    def _find_section_end(self, content: str, current_start: int,
                          current_index: int, all_matches: list) -> int:
        """找到当前章节的结束位置"""
        if current_index + 1 < len(all_matches):
            return all_matches[current_index + 1].start()
        return len(content)

    def _extract_tables(self, content: str) -> list[TableData]:
        """提取 Markdown 表格为 TableData

        识别 | col1 | col2 | 格式的表格，解析表头和数据行。
        分隔行（|---|---|）自动跳过。
        """
        tables = []
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            # 检测表格开始：以 | 开头且包含表头
            if line.startswith('|') and line.endswith('|'):
                # 找表格前的标题作为 caption
                caption = self._find_table_caption(content, i)

                # 解析表头
                headers = [h.strip() for h in line.split('|')[1:-1]]

                # 跳过分隔行（|---|---|）
                i += 1
                if i < len(lines) and re.match(r'^[\|\s\-:]+$', lines[i]):
                    i += 1

                # 解析数据行
                rows = []
                while i < len(lines):
                    row_line = lines[i].strip()
                    if row_line.startswith('|') and row_line.endswith('|'):
                        cells = [c.strip() for c in row_line.split('|')[1:-1]]
                        rows.append(cells)
                        i += 1
                    else:
                        break

                tables.append(TableData(
                    headers=headers,
                    rows=rows,
                    caption=caption,
                ))
            else:
                i += 1

        return tables

    def _find_table_caption(self, content: str, table_line_index: int) -> str:
        """尝试在表格上方查找标题/说明作为 caption"""
        lines = content.split('\n')
        # 向前查找最近的非空行
        for j in range(table_line_index - 1, max(table_line_index - 5, -1), -1):
            line = lines[j].strip()
            if line and not line.startswith('|'):
                # 如果是标题格式（## 开头）或普通文本，作为 caption
                if line.startswith('#') or len(line) < 80:
                    return line.lstrip('#').strip()
        return ""
