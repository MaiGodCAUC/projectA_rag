"""
RAGAS 评估流水线 —— 量化评估 RAG 系统质量

使用 RAGAS v0.4+ 的 SingleTurnSample API，对 30 条手工标注的
民航员工 QA 对进行 4 项指标评估，输出报告 + 雷达图 + Bad Case 分析。

----------------------------------------------------------------------
## 你需要自己写的部分

RAGAS 评估是检验 RAG 系统质量的标准方法。面试中面试官大概率会问
"你怎么评估你的系统质量"，这个文件就是你的答案。

学习重点：
1. RAGAS 四大指标的含义和计算原理
2. SingleTurnSample 的用法（新版 API）
3. 评估报告的解读（哪个指标低 = 哪个环节需要优化）
4. Bad Case 分类和改进方案

TODO(用户) 标记的部分是你需要手写的核心逻辑。
----------------------------------------------------------------------
"""

# ===========================================================================
# 1. 导入依赖
# ===========================================================================

# json: 读取 eval_dataset.json
import json

# os, sys: 路径处理
import os, sys

# time: 记录评估耗时
import time

# typing: 类型提示
from typing import List, Dict, Any, Optional, Tuple

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ---- RAGAS 核心 ----
# SingleTurnSample: RAGAS v0.3+ 的评估样本类
#   每个样本包含一次完整 RAG 问答的所有信息：
#   - user_input: 用户问题
#   - response: 系统生成的回答
#   - retrieved_contexts: 检索到的文档片段列表
#   - reference: 标注的标准答案
from ragas import SingleTurnSample

# RAGAS 四大指标（每个都是独立的 scorer 类）
# Faithfulness:        忠实度——回答是否100%基于检索到的文档？有没有幻觉？
# AnswerRelevancy:     回答相关性——回答是否紧扣用户问题？有没有跑题？
# ContextPrecision:    上下文精确度——检索到的文档中，相关的排在第几位？
# ContextRecall:       上下文召回率——标注答案中的要点，检索结果覆盖了多少？
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# ---- 本项目内部模块 ----
# 混合检索（向量 + BM25 + RRF）
from rag.hybrid_search import HybridSearcher
# RAG 生成器
from rag.generator import RAGGenerator
# 向量存储
from rag.vector_store import VectorStore
# BM25 检索
from rag.bm25 import BM25Retriever
# Embedding 工厂
from core.embedding import get_embeddings
# 配置
from core.config import get_settings

# ---- 雷达图 ----
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，不需要 GUI
import matplotlib.pyplot as plt
import numpy as np


# ===========================================================================
# 2. 配置常量
# ===========================================================================

# 评估数据集路径
EVAL_DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "eval_dataset.json")

# 评估报告输出路径
EVAL_REPORT_DIR = os.path.join(_PROJECT_ROOT, "eval", "reports")
os.makedirs(EVAL_REPORT_DIR, exist_ok=True)

# 文档类别（用于分维度拆解）
CATEGORIES = ["客规运价", "行李规定", "会员手册", "特殊服务", "航班不正常", "证件签证", "投诉处理"]

# Bad Case 分类阈值
LOW_FAITHFULNESS_THRESHOLD = 0.6   # 忠实度低于此值 → 生成幻觉
LOW_PRECISION_THRESHOLD = 0.5      # 精确度低于此值 → 检索质量差
LOW_RECALL_THRESHOLD = 0.5         # 召回率低于此值 → 文档覆盖不足


# ===========================================================================
# 3. 评估核心类
# ===========================================================================

class RAGASEvaluator:
    """RAGAS 评估器 —— 一键评估 RAG 系统质量

    使用流程:
        evaluator = RAGASEvaluator()
        evaluator.load_dataset("data/eval_dataset.json")
        report = evaluator.run()        # ← TODO(用户): 核心评估逻辑
        evaluator.print_report()        # 终端打印报告
        evaluator.save_charts()         # 保存雷达图
    """

    def __init__(self):
        """初始化评估器 —— 加载 RAG 管线和 RAGAS 指标"""
        print("初始化 RAGAS 评估器...")

        # ---- 加载 RAG 管线组件 ----
        settings = get_settings()

        # 向量存储（需要 Qdrant 已启动并索引）
        self.vector_store = VectorStore()

        # BM25 检索器（需要先建立索引）
        self.bm25 = BM25Retriever()

        # 混合检索器
        self.hybrid_searcher = HybridSearcher(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25,
        )

        # RAG 生成器
        self.generator = RAGGenerator()

        # ---- 初始化 RAGAS 四大指标 scorer ----
        # 每个指标都是一个类实例，调用 .single_turn_score(sample) 来打分
        self.metrics = {
            "faithfulness": Faithfulness(),
            "answer_relevancy": AnswerRelevancy(),
            "context_precision": ContextPrecision(),
            "context_recall": ContextRecall(),
        }

        # 评估数据
        self.dataset: List[Dict[str, Any]] = []

        # 评估结果（每条样本的详细分数）
        self.results: List[Dict[str, Any]] = []

        # 汇总统计
        self.summary: Dict[str, Any] = {}

        print("  评估器初始化完成。")

    # ------------------------------------------------------------------
    # 加载评估数据集
    # ------------------------------------------------------------------

    def load_dataset(self, path: str = EVAL_DATASET_PATH):
        """加载手工标注的 QA 评估数据集

        Args:
            path: eval_dataset.json 文件路径
        """
        with open(path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        print(f"已加载 {len(self.dataset)} 条评估样本")

        # 按类别统计
        cat_counts = {}
        for item in self.dataset:
            cat = item.get("category", "未分类")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        for cat, count in cat_counts.items():
            print(f"  {cat}: {count} 条")

    # ==================================================================
    # run() —— 核心评估流程（TODO 用户手写）
    # ==================================================================

    def run(self) -> Dict[str, Any]:
        """执行完整 RAGAS 评估

        TODO(用户): 手写核心评估逻辑

        流程:
        ┌─────────────────────────────────────────────────────────────────┐
        │ for 每条 QA 样本 in dataset:                                     │
        │                                                                  │
        │   ① 检索: hybrid_searcher.search(question)                      │
        │      → 拿到 Top-5 检索结果                                       │
        │                                                                  │
        │   ② 生成: generator.generate(question, retrieval_results)       │
        │      → 拿到 LLM 回答                                             │
        │                                                                  │
        │   ③ 构建 SingleTurnSample:                                       │
        │      sample = SingleTurnSample(                                  │
        │          user_input=question,          # 用户问题                │
        │          response=answer_text,          # 系统回答                │
        │          retrieved_contexts=[...],      # 检索到的文档            │
        │          reference=reference_answer,    # 标注标准答案            │
        │      )                                                           │
        │                                                                  │
        │   ④ 打分: 对每个指标调用 .single_turn_score(sample)              │
        │      → faith_score = faithfulness.single_turn_score(sample)     │
        │      → relev_score = answer_relevancy.single_turn_score(sample)  │
        │      → prec_score  = context_precision.single_turn_score(sample) │
        │      → recall_score = context_recall.single_turn_score(sample)  │
        │                                                                  │
        │   ⑤ 记录结果: 保存到 self.results 列表                           │
        │                                                                  │
        │ 汇总: 计算各指标均值 → self.summary                               │
        └─────────────────────────────────────────────────────────────────┘

        面试话术:
        "我用 RAGAS 的 4 项指标量化评估系统质量。Faithfulness 衡量
        回答是否基于检索文档而非 LLM 幻觉，Answer Relevancy 衡量
        回答是否紧扣问题。Context Precision 和 Recall 分别衡量
        检索的精确度和覆盖面。30 条手工标注的领域 QA 对保证了
        评估的有效性。"

        Returns:
            summary 字典: {metric_name: avg_score, ...}
        """
        print("\n" + "=" * 60)
        print("开始 RAGAS 评估（共 {} 条样本）".format(len(self.dataset)))
        print("=" * 60)

        # ================================================================
        # TODO(用户): 从这里开始手写 —— RAGAS 评估核心逻辑
        # ================================================================
        #
        # 实现参考:
        #

        # ---- 预备: 初始化结果容器 ----
        self.results = []
        total_start = time.time()

        # ---- 逐条评估 ----
        for idx, item in enumerate(self.dataset):
            qid = item["id"]
            question = item["question"]
            reference_answer = item["answer"]
            category = item.get("category", "未分类")

            print(f"\n[{idx+1}/{len(self.dataset)}] {qid}: {question[:40]}...")
            sample_start = time.time()
            # ---- 步骤 1: 检索 ----
            # 用混合检索拿 Top-5 最相关文档片段
            retrieval_results = self.hybrid_searcher.search(
                query=question,
                top_k=5
            )
            # 提取检索到的文档内容列表（RAGAS 需要 List[str]）
            retrieval_contexts = [
                r.chunk.content for r in retrieval_results
            ]

            # ---- 步骤 2: 生成 ----
            # 用 RAGGenerator 生成带引用的回答
            cited_answer = self.generator.generate(
                query=question,
                retrieval_results=retrieval_results
            )
            answer_text = cited_answer.answer_text

            # ---- 步骤 3: 构建 RAGAS 评估样本 ----
            # SingleTurnSample 是 RAGAS v0.3+ 的标准样本格式
            # user_input:          用户问题
            # response:            RAG 系统生成的回答
            # retrieved_contexts:  检索到的文档内容列表（只取文本）
            # reference:           手工标注的标准答案
            sample = SingleTurnSample(
                user_input=question,
                response=answer_text,
                retrieval_contexts=retrieval_contexts,
                reference=reference_answer
            )

        #     # ---- 步骤 4: RAGAS 打分 ----
        #     # 每个指标调用 .single_turn_score(sample) 返回 0~1 的分数
        #     # 指标内部用 LLM 做评估判断（如 Faithfulness 用 LLM 检查
        #     # 回答中的每个陈述是否能在上下文中找到依据）
        #     try:
        #         faith_score = self.metrics["faithfulness"].single_turn_score(sample)
        #     except Exception as e:
        #         print(f"    ⚠ Faithfulness 打分失败: {e}")
        #         faith_score = 0.0
        #
        #     try:
        #         relev_score = self.metrics["answer_relevancy"].single_turn_score(sample)
        #     except Exception as e:
        #         print(f"    ⚠ AnswerRelevancy 打分失败: {e}")
        #         relev_score = 0.0
        #
        #     try:
        #         prec_score = self.metrics["context_precision"].single_turn_score(sample)
        #     except Exception as e:
        #         print(f"    ⚠ ContextPrecision 打分失败: {e}")
        #         prec_score = 0.0
        #
        #     try:
        #         recall_score = self.metrics["context_recall"].single_turn_score(sample)
        #     except Exception as e:
        #         print(f"    ⚠ ContextRecall 打分失败: {e}")
        #         recall_score = 0.0
        #
        #     # ---- 步骤 5: 记录结果 ----
        #     cost_ms = int((time.time() - sample_start) * 1000)
        #     self.results.append({
        #         "id": qid,
        #         "category": category,
        #         "question": question[:80],
        #         "faithfulness": round(faith_score, 4),
        #         "answer_relevancy": round(relev_score, 4),
        #         "context_precision": round(prec_score, 4),
        #         "context_recall": round(recall_score, 4),
        #         "cost_ms": cost_ms,
        #         "retrieval_count": len(retrieval_results),
        #         "reference_answer": reference_answer[:80],
        #         "generated_answer": answer_text[:80],
        #     })
        #
        #     print(f"    Faith: {faith_score:.3f} | Relev: {relev_score:.3f} | "
        #           f"Prec: {prec_score:.3f} | Recall: {recall_score:.3f}")

        # ---- 汇总统计 ----
        # 计算各指标的全集均值
        # total_cost = int((time.time() - total_start) * 1000)
        # self.summary = {
        #     "total_samples": len(self.results),
        #     "total_cost_ms": total_cost,
        #     "avg_faithfulness": self._avg("faithfulness"),
        #     "avg_answer_relevancy": self._avg("answer_relevancy"),
        #     "avg_context_precision": self._avg("context_precision"),
        #     "avg_context_recall": self._avg("context_recall"),
        #     "by_category": self._category_breakdown(),
        #     "bad_cases": self._bad_case_analysis(),
        # }
        #
        # return self.summary
        #
        # ================================================================
        raise NotImplementedError(
            "TODO(用户): 参考上面的注释实现 RAGAS 评估核心逻辑。\n"
            "步骤: 遍历 dataset → 检索 → 生成 → 构建SingleTurnSample → 打分 → 记录"
        )

    # ------------------------------------------------------------------
    # 辅助函数: 计算某指标的平均值
    # ------------------------------------------------------------------

    def _avg(self, metric_name: str) -> float:
        """计算某指标的均值（跳过失败样本）"""
        values = [r[metric_name] for r in self.results if r.get(metric_name, 0) > 0]
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)

    # ------------------------------------------------------------------
    # 按类别拆解
    # ------------------------------------------------------------------

    def _category_breakdown(self) -> Dict[str, Dict[str, float]]:
        """按文档类别拆解各指标均值

        面试价值: 能回答"行李类问答检索效果好不好？投诉类是否容易幻觉？"

        Returns:
            {category: {metric: avg_score, ...}, ...}
        """
        breakdown = {}
        for cat in CATEGORIES:
            cat_results = [r for r in self.results if r.get("category") == cat]
            if not cat_results:
                continue
            breakdown[cat] = {
                "count": len(cat_results),
                "faithfulness": round(sum(r["faithfulness"] for r in cat_results) / len(cat_results), 4),
                "answer_relevancy": round(sum(r["answer_relevancy"] for r in cat_results) / len(cat_results), 4),
                "context_precision": round(sum(r["context_precision"] for r in cat_results) / len(cat_results), 4),
                "context_recall": round(sum(r["context_recall"] for r in cat_results) / len(cat_results), 4),
            }
        return breakdown

    # ------------------------------------------------------------------
    # Bad Case 分析
    # ------------------------------------------------------------------

    def _bad_case_analysis(self) -> Dict[str, List[Dict]]:
        """Bad Case 自动分类

        三类 Bad Case:
        1. 生成幻觉（Hallucination）: Faithfulness < 阈值
           → 回答内容在检索文档中找不到依据 → 优化方向: 改进 Prompt / 换模型
        2. 检索失败（Retrieval Failure）: ContextPrecision < 阈值
           → 检索到的文档不相关 → 优化方向: 调整切片策略 / 混合检索权重
        3. 覆盖不足（Coverage Gap）: ContextRecall < 阈值
           → 检索结果覆盖不了标注答案 → 优化方向: 补充知识库文档

        Returns:
            {bad_case_type: [case, ...], ...}
        """
        bad_cases = {
            "hallucination": [],
            "retrieval_failure": [],
            "coverage_gap": [],
        }

        for r in self.results:
            faith = r.get("faithfulness", 1.0)
            prec = r.get("context_precision", 1.0)
            recall = r.get("context_recall", 1.0)

            if faith < LOW_FAITHFULNESS_THRESHOLD:
                bad_cases["hallucination"].append({
                    "id": r["id"],
                    "question": r["question"],
                    "faithfulness": faith,
                    "generated_answer": r.get("generated_answer", "")[:100],
                })

            if prec < LOW_PRECISION_THRESHOLD:
                bad_cases["retrieval_failure"].append({
                    "id": r["id"],
                    "question": r["question"],
                    "context_precision": prec,
                    "retrieval_count": r.get("retrieval_count", 0),
                })

            if recall < LOW_RECALL_THRESHOLD:
                bad_cases["coverage_gap"].append({
                    "id": r["id"],
                    "question": r["question"],
                    "context_recall": recall,
                })

        return bad_cases


# ===========================================================================
# 4. 雷达图绘制（TODO 用户手写核心逻辑）
# ===========================================================================

def plot_radar_chart(summary: Dict[str, Any], save_path: str):
    """绘制 4 项 RAGAS 指标的雷达图

    TODO(用户): 理解并手写雷达图绘制逻辑

    雷达图的每个轴代表一项指标（0-1 范围），面积越大 = 系统质量越好。

    面试话术:
    "我用雷达图把 4 项 RAGAS 指标可视化。面试官一眼就能看到
    系统的强项和短板——比如 Faithfulness 高说明防幻觉做得好，
    Context Recall 低说明知识库覆盖可能不够。"

    Args:
        summary: evaluator.summary，含 avg_faithfulness 等字段
        save_path: 图片保存路径
    """
    # ================================================================
    # TODO(用户): 手写雷达图绘制逻辑
    # ================================================================
    #
    # 实现参考:
    #

    # ---- 数据准备 ----
    # 四个指标的标签和数值
    # labels = ["Faithfulness\n忠实度", "Answer Relevancy\n回答相关性",
    #           "Context Precision\n上下文精确度", "Context Recall\n上下文召回率"]
    # values = [
    #     summary.get("avg_faithfulness", 0),
    #     summary.get("avg_answer_relevancy", 0),
    #     summary.get("avg_context_precision", 0),
    #     summary.get("avg_context_recall", 0),
    # ]
    #
    # # 闭合雷达图（首尾相连）
    # values += values[:1]
    #
    # # ---- 角度计算 ----
    # num_vars = 4
    # angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    # angles += angles[:1]  # 闭合
    #
    # # ---- 绘图 ----
    # fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    #
    # # 绘制数据区域
    # ax.fill(angles, values, alpha=0.25, color="steelblue")
    # ax.plot(angles, values, linewidth=2, color="steelblue", marker="o", markersize=8)
    #
    # # 设置刻度标签
    # ax.set_xticks(angles[:-1])
    # ax.set_xticklabels(labels, fontsize=11)
    #
    # # 设置 Y 轴范围
    # ax.set_ylim(0, 1)
    # ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    # ax.set_rlabel_position(30)
    #
    # # 标题
    # ax.set_title("RAGAS Evaluation - Air China RAG System",
    #              fontsize=14, fontweight="bold", pad=30)
    #
    # # 在每个数据点上标注数值
    # for angle, value in zip(angles[:-1], values[:-1]):
    #     ax.annotate(f"{value:.2f}",
    #                 xy=(angle, value),
    #                 xytext=(5, 5), textcoords="offset points",
    #                 fontsize=10, fontweight="bold", color="steelblue")
    #
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=150, bbox_inches="tight")
    # plt.close()
    # print(f"雷达图已保存至: {save_path}")
    #
    # ================================================================
    raise NotImplementedError(
        "TODO(用户): 参考上面的注释实现雷达图绘制逻辑。\n"
        "步骤: 准备数据 → 计算角度 → polar图 → 填充 → 标注 → 保存"
    )


# ===========================================================================
# 5. 报告输出
# ===========================================================================

def print_report(results: List[Dict], summary: Dict, bad_cases: Dict):
    """打印终端评估报告"""
    print("\n" + "=" * 70)
    print("  RAGAS 评估报告 —— 国航内部员工智能知识助手")
    print("=" * 70)

    # ---- 总览 ----
    print(f"\n  评估样本数: {summary.get('total_samples', 0)}")
    print(f"  总耗时: {summary.get('total_cost_ms', 0)}ms")
    print()

    # ---- 四项指标 ----
    print("  " + "-" * 50)
    print(f"  {'指标':25s} {'分数':>8s}  {'评级':>8s}")
    print("  " + "-" * 50)

    def _rate(score):
        if score >= 0.85:
            return "优秀"
        elif score >= 0.75:
            return "良好"
        elif score >= 0.60:
            return "一般"
        else:
            return "待优化"

    metrics = [
        ("Faithfulness (忠实度)", summary.get("avg_faithfulness", 0)),
        ("Answer Relevancy (相关性)", summary.get("avg_answer_relevancy", 0)),
        ("Context Precision (精确度)", summary.get("avg_context_precision", 0)),
        ("Context Recall (召回率)", summary.get("avg_context_recall", 0)),
    ]
    for name, score in metrics:
        bar = "█" * int(score * 20)
        print(f"  {name:25s} {score:.4f}  {_rate(score)}")
        print(f"  {'':25s} {bar}")

    print("  " + "-" * 50)

    # ---- 按类别拆解 ----
    if summary.get("by_category"):
        print("\n  --- 按文档类别拆解 ---")
        print(f"  {'类别':12s} {'样本':>4s} {'Faith':>7s} {'Relev':>7s} {'Prec':>7s} {'Recall':>7s}")
        print("  " + "-" * 55)
        for cat, stats in summary["by_category"].items():
            print(f"  {cat:12s} {stats['count']:>4d} "
                  f"{stats['faithfulness']:7.3f} {stats['answer_relevancy']:7.3f} "
                  f"{stats['context_precision']:7.3f} {stats['context_recall']:7.3f}")

    # ---- Bad Case ----
    if bad_cases:
        total_bad = sum(len(v) for v in bad_cases.values())
        print(f"\n  --- Bad Case 分析（共 {total_bad} 条）---")

        if bad_cases["hallucination"]:
            print(f"\n  🔴 生成幻觉 ({len(bad_cases['hallucination'])} 条):")
            print(f"     优化方向: 改进 System Prompt 约束、降低 temperature、换用更强模型")
            for case in bad_cases["hallucination"][:3]:
                print(f"     - {case['id']}: {case['question'][:50]}... (Faith: {case['faithfulness']:.3f})")

        if bad_cases["retrieval_failure"]:
            print(f"\n  🟡 检索失败 ({len(bad_cases['retrieval_failure'])} 条):")
            print(f"     优化方向: 调整切片策略(chunk_size/overlap)、调整混合检索权重(RRF k值)")
            for case in bad_cases["retrieval_failure"][:3]:
                print(f"     - {case['id']}: {case['question'][:50]}... (Prec: {case['context_precision']:.3f})")

        if bad_cases["coverage_gap"]:
            print(f"\n  🟠 覆盖不足 ({len(bad_cases['coverage_gap'])} 条):")
            print(f"     优化方向: 补充知识库文档、改进文档切片的条款完整性")
            for case in bad_cases["coverage_gap"][:3]:
                print(f"     - {case['id']}: {case['question'][:50]}... (Recall: {case['context_recall']:.3f})")

    print("\n" + "=" * 70)


# ===========================================================================
# 6. 入口
# ===========================================================================

def main():
    """RAGAS 评估入口 —— 一键运行完整评估流程"""
    print("=" * 60)
    print("  RAGAS 评估流水线 v1.0")
    print("  国航内部员工智能知识助手")
    print("=" * 60)

    # 1. 初始化评估器
    evaluator = RAGASEvaluator()

    # 2. 加载评估数据集
    evaluator.load_dataset()

    # 3. 执行评估
    summary = evaluator.run()

    # 4. 输出报告
    bad_cases = summary.get("bad_cases", {})
    print_report(evaluator.results, summary, bad_cases)

    # 5. 绘制雷达图
    try:
        chart_path = os.path.join(EVAL_REPORT_DIR, "ragas_radar.png")
        plot_radar_chart(summary, chart_path)
    except NotImplementedError:
        print("\n  [跳过] 雷达图绘制逻辑待实现(TODO用户)")

    # 6. 保存评估结果为 JSON
    report_path = os.path.join(EVAL_REPORT_DIR, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {k: v for k, v in summary.items() if k != "bad_cases"},
            "results": evaluator.results,
            "bad_cases": {
                k: [{"id": c["id"], "question": c["question"]} for c in v]
                for k, v in bad_cases.items()
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\n评估报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
