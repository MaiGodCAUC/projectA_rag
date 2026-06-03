"""
评估接口 —— 触发 RAGAS 评估 + 查看报告

POST /eval/run    → 触发 RAGAS 全量评估
GET  /eval/report → 查看最新评估报告

----------------------------------------------------------------------
FastAPI 核心用法在本文件中的体现:

1. 同步函数做异步路由: 评估是计算密集型操作（大量 LLM 调用），
   async def + 同步调用 = 会阻塞事件循环。生产环境应改为
   BackgroundTasks 或 Celery 异步任务。

2. 延迟导入 (lazy import): 不在模块加载时导入 RAGASEvaluator，
   而是在路由函数内部导入——避免启动时就加载评估模块的所有依赖。

3. 文件读取 + 异常处理: 读取本地 JSON 文件返回给前端。
----------------------------------------------------------------------
"""

# json: 读取 eval_report.json
import json

# os: 路径拼接
import os

# APIRouter: 路由分组器
from fastapi import APIRouter

from api.models import APIResponse

router = APIRouter(tags=["评估"])

# _PROJECT_ROOT: 项目根目录的绝对路径
# __file__ = api/routes/eval.py
# dirname 三次: routes/ → api/ → 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
EVAL_REPORT_DIR = os.path.join(_PROJECT_ROOT, "eval", "reports")


@router.post("/eval/run")
async def run_evaluation():
    """触发 RAGAS 评估

    后台同步执行全量评估（30 条 QA 对），结果保存到 eval/reports/。

    ================================================================
    注意: 评估需要大量 LLM 调用（每条 QA 需要 2~3 次 LLM 调用：
    生成回答 + RAGAS Faithfulness 打分 + Answer Relevancy 打分），
    30 条约 60-90 次 LLM 调用，预计耗时 2-5 分钟。

    当前是同步执行——客户端会一直等到评估完成才收到响应。
    生产环境应改为 BackgroundTasks（FastAPI 内置）或 Celery（分布式任务队列）：

    from fastapi import BackgroundTasks

    @router.post("/eval/run")
    async def run_evaluation(background_tasks: BackgroundTasks):
        background_tasks.add_task(_do_evaluation)  # 后台执行
        return APIResponse.ok(data={"status": "评估已提交，请稍后查看结果"})
    ================================================================
    """
    try:
        # ============================================================
        # 延迟导入 (lazy import):
        # 不在模块顶部导入，而在函数内部导入
        # 原因: RAGASEvaluator 依赖 Qdrant、LLM 等重资源
        #   如果模块加载时就导入，启动 main.py 就会尝试连接 Qdrant
        #   延迟导入只在用户真正请求 /eval/run 时才加载
        #   这是 Python 中常见的优化手段
        # ============================================================
        from eval.ragas_eval import RAGASEvaluator, print_report, plot_radar_chart

        # 初始化评估器（加载 RAG 管线 + RAGAS 指标）
        evaluator = RAGASEvaluator()

        # 加载 30 条手工标注的评估数据集
        evaluator.load_dataset()

        # 执行全量评估（核心逻辑在 eval/ragas_eval.py 的 run() 方法中）
        summary = evaluator.run()
        # summary 结构:
        # {
        #   "total_samples": 30,
        #   "total_cost_ms": 120000,
        #   "avg_faithfulness": 0.78,
        #   "avg_answer_relevancy": 0.82,
        #   ...
        #   "bad_cases": {...}
        # }

        bad_cases = summary.get("bad_cases", {})

        # 打印终端报告（开发时可以看到，生产环境应删除或改为日志）
        print_report(evaluator.results, summary, bad_cases)

        # 绘制雷达图（保存到 eval/reports/ragas_radar.png）
        try:
            chart_path = os.path.join(EVAL_REPORT_DIR, "ragas_radar.png")
            plot_radar_chart(summary, chart_path)
        except Exception:
            pass   # 雷达图失败不影响主流程

        # ---- 保存评估报告 JSON ----
        # 这是 GET /eval/report 的数据来源，必须保存
        # 结构: {summary: {指标均值...}, results: [每条样本详情...], bad_cases: {...}}
        os.makedirs(EVAL_REPORT_DIR, exist_ok=True)
        report_path = os.path.join(EVAL_REPORT_DIR, "eval_report.json")
        # bad_cases 中的 Citation 等对象可能不可序列化，降级为 id+question 摘要
        serializable_bad_cases = {
            k: [{"id": c["id"], "question": c.get("question", "")} for c in v]
            for k, v in bad_cases.items()
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "evaluated_at": __import__("datetime").datetime.now().isoformat(),
                "summary": {
                    k: v for k, v in summary.items()
                    if k != "bad_cases"  # bad_cases 单独存（方便前端分开渲染）
                },
                "results": evaluator.results,
                "bad_cases": serializable_bad_cases,
            }, f, ensure_ascii=False, indent=2)

        # 返回评估摘要给前端
        return APIResponse.ok(data={
            "total_samples": summary.get("total_samples", 0),
            "avg_faithfulness": summary.get("avg_faithfulness", 0),
            "avg_answer_relevancy": summary.get("avg_answer_relevancy", 0),
            "avg_context_precision": summary.get("avg_context_precision", 0),
            "avg_context_recall": summary.get("avg_context_recall", 0),
            "total_cost_ms": summary.get("total_cost_ms", 0),
            "bad_case_count": sum(len(v) for v in bad_cases.values()),
            # sum(len(v) for v in bad_cases.values())
            #   bad_cases = {"hallucination": [case1, case2], "retrieval_failure": [case3]}
            #   → len = 2, 1, 0
            #   → sum = 3
        })
    except Exception as e:
        # 评估过程中的任何异常都会传到这里
        return APIResponse.fail(code=9999, detail=f"评估执行失败: {str(e)}")


@router.get("/eval/report")
async def get_eval_report():
    """获取最新的评估报告完整 JSON

    返回 eval/reports/eval_report.json 的全部内容。
    这个文件由 run_evaluation() 在评估完成后生成。

    如果还没有执行过评估 → 返回提示信息。
    """
    # 拼接报告文件路径
    report_path = os.path.join(EVAL_REPORT_DIR, "eval_report.json")

    # 检查文件是否存在
    if not os.path.exists(report_path):
        # 没有报告 → 返回提示
        return APIResponse.ok(data={
            "message": "暂未执行过评估，请先 POST /eval/run"
        })

    # 读取 JSON 文件内容
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # 把整个报告 JSON 返回给前端
    # 前端可以据此绘制图表、展示各指标的详细数据
    return APIResponse.ok(data=report)
