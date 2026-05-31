"""
评估接口 —— 触发 RAGAS 评估 + 查看报告

POST /eval/run    → 触发 RAGAS 全量评估
GET  /eval/report → 查看最新评估报告
"""

import json
import os
from fastapi import APIRouter

from api.models import APIResponse, EvalData

router = APIRouter(tags=["评估"])

_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
EVAL_REPORT_DIR = os.path.join(_PROJECT_ROOT, "eval", "reports")


@router.post("/eval/run")
async def run_evaluation():
    """触发 RAGAS 评估

    后台异步执行全量评估（30 条 QA 对），结果保存到 eval/reports/。

    注意: 评估需要大量 LLM 调用（约 60-90 次），预计耗时 2-5 分钟。
    生产环境应改为 Celery / BackgroundTasks 异步执行。
    """
    try:
        from eval.ragas_eval import RAGASEvaluator, print_report, plot_radar_chart

        evaluator = RAGASEvaluator()
        evaluator.load_dataset()
        summary = evaluator.run()

        bad_cases = summary.get("bad_cases", {})
        print_report(evaluator.results, summary, bad_cases)

        # 绘制雷达图
        try:
            chart_path = os.path.join(EVAL_REPORT_DIR, "ragas_radar.png")
            plot_radar_chart(summary, chart_path)
        except Exception:
            pass

        return APIResponse.ok(data={
            "total_samples": summary.get("total_samples", 0),
            "avg_faithfulness": summary.get("avg_faithfulness", 0),
            "avg_answer_relevancy": summary.get("avg_answer_relevancy", 0),
            "avg_context_precision": summary.get("avg_context_precision", 0),
            "avg_context_recall": summary.get("avg_context_recall", 0),
            "total_cost_ms": summary.get("total_cost_ms", 0),
            "bad_case_count": sum(len(v) for v in bad_cases.values()),
        })
    except Exception as e:
        return APIResponse.fail(code=9999, detail=f"评估执行失败: {str(e)}")


@router.get("/eval/report")
async def get_eval_report():
    """获取最新的评估报告（JSON）"""
    report_path = os.path.join(EVAL_REPORT_DIR, "eval_report.json")
    if not os.path.exists(report_path):
        return APIResponse.ok(data={"message": "暂未执行过评估，请先 POST /eval/run"})

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    return APIResponse.ok(data=report)
