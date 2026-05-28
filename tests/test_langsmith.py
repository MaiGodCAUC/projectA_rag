"""
LangSmith 快速体验脚本 —— 运行一次即可在 LangSmith 后台看到 Trace

使用方式：
    python tests/test_langsmith.py

前提条件：
    1. 已在 https://smith.langchain.com 注册并获取 API Key
    2. .env 文件中已配置 LANGCHAIN_API_KEY=你的key
    3. .env 文件中已配置 LANGCHAIN_TRACING_V2=true
"""

import sys
import os

# ---- Windows 终端 UTF-8 编码修复 ----
# 解决 GBK 终端输出中文乱码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---- 确保项目根目录在 Python 路径中 ----
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ---- 加载 .env 文件 ----
# pydantic-settings 会自动读取 .env，但 os.getenv 不会
# 需要先手动加载，让 os.getenv 能读到里面的变量
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from core.llm import get_llm
from core.config import get_settings
from langchain_core.messages import HumanMessage, SystemMessage


def check_langsmith_config():
    """检查 LangSmith 是否已正确配置"""
    print("=" * 60)
    print("[Config] LangSmith 配置检查")
    print("=" * 60)

    config_items = [
        ("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2"), "'true' or 'false'"),
        ("LANGCHAIN_ENDPOINT",   os.getenv("LANGCHAIN_ENDPOINT"),
         "https://api.smith.langchain.com"),
        ("LANGCHAIN_API_KEY",    os.getenv("LANGCHAIN_API_KEY", ""),
         "lsv2_pt_..."),
        ("LANGCHAIN_PROJECT",    os.getenv("LANGCHAIN_PROJECT"),
         "airchina-rag"),
    ]

    all_ok = True
    for key, value, expected in config_items:
        # 决定每个配置项的显示值和状态
        if value is None or value == "":
            display = "(not set)"
            status = "[FAIL]"
        elif key == "LANGCHAIN_API_KEY":
            # API Key 用脱敏显示
            display = (value[:15] + "..." + value[-4:]) if len(value) > 20 else value
            is_valid = len(value) > 20 and value != "lsv2_pt_your_key_here"
            status = "[OK]" if is_valid else "[FAIL] need real key"
        elif key == "LANGCHAIN_TRACING_V2":
            display = value
            is_valid = value.lower() == "true"
            status = "[OK]" if is_valid else "[FAIL] should be 'true'"
        else:
            display = value
            status = "[OK]"

        if status != "[OK]":
            all_ok = False
        print(f"  {status}  {key:25s} = {display}")

    print()
    return all_ok


def demo_simple_llm_call():
    """演示 1：最简单的 LLM 调用（自动上报到 LangSmith）"""
    print("=" * 60)
    print(" 演示 1：基础 LLM 调用")
    print("=" * 60)

    llm = get_llm(get_settings())
    response = llm.invoke([
        HumanMessage(content="你好，请用一句话介绍中国国际航空（国航）。")
    ])
    print(f"  LLM 回答: {response.content}")
    print(f"   这条调用已自动上报到 LangSmith")
    print(f"   打开 https://smith.langchain.com → 项目 airchina-rag 查看")
    print()


def demo_rag_with_callback():
    """演示 2：带 RAGTraceCallback 的完整流程"""
    print("=" * 60)
    print(" 演示 2：完整 RAG 流程（带自定义埋点）")
    print("=" * 60)

    from rag.callbacks import RAGTraceCallback

    cb = RAGTraceCallback()
    print(f"  Trace ID: {cb.trace_id}")

    # 模拟 RAG 流程各节点
    cb.start_node("context_manage")
    # （实际项目中这里调用 _manage_context_window）
    cb.end_node({"input_count": 5, "output_count": 5})

    cb.start_node("build_context")
    cb.end_node({"context_chars": 1200})

    cb.start_node("assemble_messages")
    cb.end_node()

    # LLM 调用（带 callback，LangChain 自动触发 on_llm_start/end）
    llm = get_llm(get_settings())
    response = llm.invoke(
        [
            SystemMessage(content="你是国航内部业务支持助手。"),
            HumanMessage(content="请一句话说明：旅客托运行李损坏怎么赔偿？")
        ],
        config={"callbacks": [cb]},  # ← 关键：传递 callback
    )
    print(f"  LLM 回答: {response.content[:80]}...")

    cb.start_node("citation_extract")
    cb.end_node({"citation_count": 0})

    # 打印完整报告
    cb.print_report()
    print(f"   Trace ID {cb.trace_id} 已上报到 LangSmith")
    print()


def main():
    print()
    print("+======================================================+")
    print("|       LangSmith 快速体验 —— 国航 RAG 项目            |")
    print("+======================================================+")
    print()

    if not check_langsmith_config():
        print("️  请先完成 LangSmith 配置：")
        print("   1. 访问 https://smith.langchain.com 注册")
        print("   2. 获取 API Key")
        print("   3. 在 .env 中将 LANGCHAIN_API_KEY 替换为真实 Key")
        print("   4. 重新运行本脚本")
        sys.exit(1)

    # 演示 1：基础 LLM 调用
    demo_simple_llm_call()

    # 演示 2：完整 RAG 流程（含回调）
    demo_rag_with_callback()

    print("=" * 60)
    print(" 全部完成！现在打开浏览器访问：")
    print("   https://smith.langchain.com")
    print("   左侧选择项目：airchina-rag")
    print("   你会看到 2 条新的 Trace 记录！")
    print("=" * 60)


if __name__ == "__main__":
    main()
