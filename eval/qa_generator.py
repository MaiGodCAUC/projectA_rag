"""
LLM 辅助 QA 对生成器 —— 从民航文档中自动生成评估用 QA 对

用来扩充手工标注的 30 条评估集。输入一份文档（如"托运行李运输规定.md"），
输出多条模拟一线员工提问 + 标准答案的 QA 对。

----------------------------------------------------------------------
## 你需要自己写的部分

这个工具的核心理念是「用 LLM 造测试数据来评估用 LLM 的 RAG 系统」。
听起来套娃但很实用——LLM 擅长根据给定材料生成问题，而手工标注 30 条
已经确立了质量标准，LLM 可以参照这 30 条的风格批量生成。

学习重点:
1. 如何设计 QA 生成的 Prompt（让 LLM 产出符合员工风格的 query）
2. 生成内容的多样性控制（避免全是同一种问法）
3. 生成结果的质量筛选（手工审核 + 自动去重）

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

import os, sys, json
from typing import List, Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from core.llm import get_llm
from core.config import get_settings
from langchain_core.messages import SystemMessage, HumanMessage


# =============================================================================
# QA 生成 Prompt
# =============================================================================

QA_GENERATION_PROMPT = """你是国航内部业务培训师。请根据以下政策/规定文档，
生成一线员工在实际工作中可能提出的问题，以及对应的标准答案。

## 要求

1. 问题风格：模拟客服坐席、值机柜台、登机口员工的口吻
   - 直接问"怎么处理"、"要什么材料"、"多少钱"、"什么流程"
   - 不要学术化的提问方式

2. 答案要求：
   - 必须严格从给定文档中提取，不得编造
   - 简明扼要，直接给出可执行的指引
   - 引用文档中的具体条款编号

3. 覆盖多种难度：
   - 简单：单条款查询（如"XX是什么"）
   - 中等：条件判断查询（如"如果XX情况，怎么处理"）
   - 困难：跨条款对比（如"散客和团队客退票规则有什么区别"）

4. 输出格式（JSON 数组，不要其他文字）：
```json
[
  {
    "question": "员工的实际问题",
    "answer": "从文档中提取的标准答案（引用条款）",
    "reference": "文档名 第X条"
  }
]
```

请生成 3-5 条 QA 对。"""


# =============================================================================
# QA 生成器
# =============================================================================

class QAGenerator:
    """从文档自动生成评估用 QA 对

    使用方式:
        gen = QAGenerator()
        qa_pairs = gen.generate_from_doc("data/documents/04-托运行李运输规定.md")
    """

    def __init__(self):
        self.llm = get_llm(get_settings())
        self.llm.temperature = 0.3  # 稍高温度增加问题多样性

    def generate_from_doc(
        self,
        doc_path: str,
        num_pairs: int = 5,
    ) -> List[Dict[str, str]]:
        """从文档生成 QA 对

        TODO(用户): 手写 QA 生成逻辑

        流程:
        1. 读取文档内容
        2. 组装 Prompt（文档内容 + QA_GENERATION_PROMPT）
        3. 调用 LLM 生成 JSON
        4. 解析 JSON → QA 对列表
        5. 去重、筛选

        Args:
            doc_path: 文档路径（相对于项目根目录）
            num_pairs: 期望生成的 QA 对数量

        Returns:
            QA 对列表 [{"question": "...", "answer": "...", "reference": "..."}]
        """
        # ================================================================
        # TODO(用户): 手写 QA 生成逻辑
        # ================================================================
        #
        # 实现参考:
        #

        # ---- 步骤 1: 读取文档 ----
        # full_path = os.path.join(_PROJECT_ROOT, doc_path)
        # with open(full_path, "r", encoding="utf-8") as f:
        #     doc_content = f.read()
        #
        # # 如果文档太长，截取前 4000 字（LLM 上下文限制）
        # if len(doc_content) > 4000:
        #     doc_content = doc_content[:4000] + "\n...(文档过长，已截断)"
        #
        # ---- 步骤 2: 组装 Prompt ----
        # user_prompt = (
        #     f"请根据以下文档内容生成 {num_pairs} 条员工问答对：\n\n"
        #     f"## 文档内容\n\n{doc_content}"
        # )
        #
        # messages = [
        #     SystemMessage(content=QA_GENERATION_PROMPT),
        #     HumanMessage(content=user_prompt),
        # ]
        #
        # ---- 步骤 3: 调用 LLM ----
        # response = self.llm.invoke(messages)
        # raw_output = response.content
        #
        # ---- 步骤 4: 解析 JSON ----
        # # LLM 可能在 JSON 前后加 markdown 代码块标记，需要去掉
        # import re
        # json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_output)
        # if json_match:
        #     raw_output = json_match.group(1)
        #
        # qa_pairs = json.loads(raw_output)
        #
        # # 确保是 list
        # if isinstance(qa_pairs, dict):
        #     qa_pairs = [qa_pairs]
        #
        # ---- 步骤 5: 质量筛选 ----
        # valid_pairs = []
        # for pair in qa_pairs:
        #     # 必须有 question 和 answer
        #     if not pair.get("question") or not pair.get("answer"):
        #         continue
        #     # question 至少 5 个字
        #     if len(pair["question"]) < 5:
        #         continue
        #     valid_pairs.append({
        #         "question": pair["question"],
        #         "answer": pair["answer"],
        #         "reference": pair.get("reference", ""),
        #     })
        #
        # return valid_pairs

        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 参考上面的注释实现 QA 自动生成逻辑。\n"
            "步骤: 读文档 → 拼Prompt → 调LLM → 解析JSON → 质量筛选"
        )

    def generate_all(self, doc_dir: str = "data/documents", output_path: str = None) -> List[Dict]:
        """批量从所有文档生成 QA 对，保存到文件

        Args:
            doc_dir: 文档目录
            output_path: 输出 JSON 路径（可选）
        """
        all_pairs = []
        doc_dir_full = os.path.join(_PROJECT_ROOT, doc_dir)

        for filename in sorted(os.listdir(doc_dir_full)):
            if not filename.endswith(".md"):
                continue
            doc_path = os.path.join(doc_dir, filename)
            print(f"从 {filename} 生成 QA 对...")
            try:
                pairs = self.generate_from_doc(doc_path)
                # 标注来源文档
                for p in pairs:
                    p["source_doc"] = filename
                all_pairs.extend(pairs)
                print(f"  生成 {len(pairs)} 条")
            except NotImplementedError:
                print("  [跳过] QA 生成逻辑待实现(TODO用户)")
            except Exception as e:
                print(f"  失败: {e}")

        if output_path and all_pairs:
            output_full = os.path.join(_PROJECT_ROOT, output_path)
            with open(output_full, "w", encoding="utf-8") as f:
                json.dump(all_pairs, f, ensure_ascii=False, indent=2)
            print(f"\n共生成 {len(all_pairs)} 条 QA 对，已保存至 {output_path}")

        return all_pairs


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    gen = QAGenerator()
    gen.generate_all(output_path="data/generated_qa_pairs.json")
