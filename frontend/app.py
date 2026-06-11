"""
国航内部员工智能知识助手 —— Streamlit 工作台

启动方式:
    streamlit run frontend/app.py

访问: http://localhost:8501

======================================================================
布局设计:

┌──────────────────────────────────────────────────────────┐
│                     国航内部员工智能知识助手                │
├────────────────────────┬─────────────────────────────────┤
│     💬 对话区          │  📁 文档管理                      │
│                        │  🤖 Agent 决策面板                │
│  流式 Markdown         │  📋 预设 Query                    │
│  引用弹窗              │                                  │
└────────────────────────┴─────────────────────────────────┘

依赖: 后端 FastAPI 需先在 http://localhost:8000 启动
======================================================================
"""

import streamlit as st
import requests
import json
import os
import time
import re
from typing import Optional, Dict, Any, List

# =============================================================================
# 配置
# =============================================================================

# 后端 API 地址（Streamlit 和 FastAPI 通常在同一台机器上）
# 后端 API 地址（环境变量 > 默认值）
# 本地开发: http://127.0.0.1:8000
# Docker 内部: http://api:8000
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")

# 页面设置
st.set_page_config(
    page_title="国航内部员工智能知识助手",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS 样式
# =============================================================================

st.markdown("""
<style>
    /* 引用标记高亮 */
    .citation-link {
        color: #1a73e8;
        text-decoration: underline;
        cursor: pointer;
        font-weight: 500;
    }
    /* 状态栏 */
    .status-bar {
        padding: 8px 16px;
        background: #f0f2f6;
        border-radius: 8px;
        font-size: 13px;
        display: flex;
        gap: 24px;
        margin-top: 12px;
    }
    .status-ok { color: #0d904f; font-weight: 600; }
    .status-error { color: #d93025; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State 初始化
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []  # 对话历史

if "agent_info" not in st.session_state:
    st.session_state.agent_info = None  # Agent 决策信息

if "last_trace_id" not in st.session_state:
    st.session_state.last_trace_id = ""

if "last_cost_ms" not in st.session_state:
    st.session_state.last_cost_ms = 0

# =============================================================================
# API 调用工具函数
# =============================================================================


def check_health() -> tuple[bool, str]:
    """检查后端服务状态"""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200, f"{r.elapsed.total_seconds()*1000:.0f}ms"
    except Exception:
        return False, "不可达"


def upload_document(file_bytes: bytes, file_name: str) -> dict:
    """上传文档到后端"""
    try:
        files = {"file": (file_name, file_bytes)}
        r = requests.post(f"{API_BASE}/documents/upload", files=files, timeout=30)
        return r.json()
    except Exception as e:
        return {"code": -1, "data": {"detail": str(e)}}


def list_documents() -> dict:
    """获取文档列表"""
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        return r.json()
    except Exception as e:
        return {"code": -1, "data": {"detail": str(e)}}


def delete_document(doc_id: str) -> dict:
    """删除文档"""
    try:
        r = requests.delete(f"{API_BASE}/documents/{doc_id}", timeout=10)
        return r.json()
    except Exception as e:
        return {"code": -1, "data": {"detail": str(e)}}


def chat_stream(query: str, top_k: int = 5) -> tuple[str, List[Dict]]:
    """SSE 流式对话 —— 逐 token 返回

    Returns:
        (完整回答文本, 引用列表)
    """
    try:
        r = requests.post(
            f"{API_BASE}/chat/stream",
            json={"message": query, "top_k": top_k, "stream": True},
            stream=True,
            timeout=60,
        )
        full_text = ""
        citations = []

        for line in r.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                token = line_str[6:]  # 去掉 "data: " 前缀
                # 检查是否为引用标记
                if token.startswith("<!--CITATIONS:"):
                    # 解析引用 JSON
                    try:
                        json_str = token.replace("<!--CITATIONS:", "").replace("-->", "")
                        citations = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                    continue
                full_text += token
                yield token, citations

    except requests.exceptions.ConnectionError:
        yield "❌ 无法连接后端服务，请确认 FastAPI 已启动 (python main.py)", []
    except Exception as e:
        yield f"❌ 请求失败: {e}", []


def agent_chat(query: str, top_k: int = 5) -> dict:
    """Agent 路由对话（非流式）"""
    try:
        r = requests.post(
            f"{API_BASE}/agent/chat",
            json={"message": query, "top_k": top_k, "stream": False},
            timeout=120,
        )
        return r.json()
    except Exception as e:
        return {"code": -1, "data": {"detail": str(e)}}


# =============================================================================
# 工具函数
# =============================================================================


def render_answer_with_citations(text: str):
    """渲染回答文本，把 [来源: ...] 标记转为可点击的引用链接"""
    # 匹配 [来源: 文档名 条款编号] 格式
    pattern = r'\[来源:\s*([^\]]+)\]'

    parts = re.split(pattern, text)
    if len(parts) == 1:
        # 没有引用标记，直接显示
        st.markdown(text)
        return

    # parts: [before, citation1, after1, citation2, after2, ...]
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # 偶数索引 = 普通文本
            result.append(part)
        else:
            # 奇数索引 = 引用内容
            result.append(f"[来源: {part}]")  # Markdown 保留原始格式

    st.markdown("".join(result))


def show_citations_expander(citations: List[Dict]):
    """显示引用溯源弹窗"""
    if not citations:
        return

    with st.expander(f"📎 引用溯源（共 {len(citations)} 条）", expanded=False):
        for i, c in enumerate(citations):
            doc = c.get("doc_name", "未知文档")
            clause = c.get("clause_id", "")
            section = c.get("section_title", "")
            text = c.get("original_text", "")

            header = f"**{i+1}. {doc}**"
            if clause:
                header += f" —— {clause}"
            if section:
                header += f"（{section}）"

            st.markdown(header)
            if text:
                st.text_area(
                    label=f"原文片段 {i+1}",
                    value=text,
                    height=80,
                    disabled=True,
                    key=f"cite_{i}",
                    label_visibility="collapsed",
                )
            st.divider()


# =============================================================================
# 预设演示 Query
# =============================================================================

PRESET_QUERIES = {
    "客规运价": [
        "旅客买的经济舱Y舱全价票，起飞前1小时退票，手续费多少？",
        "公务舱C舱改签经济舱Y舱，差价怎么算？",
    ],
    "行李规定": [
        "旅客行李箱被摔坏了，赔偿标准是什么？需要什么材料？",
        "旅客带了一把小提琴要带上飞机，超尺寸怎么办？",
    ],
    "会员手册": [
        "金卡会员的同行人能一起优先登机吗？最多带几个人？",
        "凤凰知音白金卡升级标准是什么？",
    ],
    "特殊服务": [
        "无陪儿童的年龄范围是多少？国际航班能申请吗？",
        "轮椅旅客需要提前多久申请？",
    ],
    "航班不正常": [
        "CA888航班因天气原因取消，旅客要求签转到东航，能签转吗？",
        "因公司原因延误超过4小时，赔偿标准是什么？",
    ],
    "证件签证": [
        "旅客持美国绿卡经北京转机去日本，需要中国签证吗？",
        "国际航班婴儿旅客需要什么证件？",
    ],
    "投诉处理": [
        "旅客投诉行李延误，经济补偿审批权限是多少？",
    ],
}


# =============================================================================
# 侧边栏 —— 文档管理
# =============================================================================

def render_document_panel():
    """渲染文档管理面板"""
    st.subheader("📁 文档管理")

    # ---- 上传 ----
    # 用动态 key 解决 st.rerun() 后 file_uploader 不清空的死循环问题
    # 每次上传成功后 _upload_key 自增 → 新 key → 旧文件自动清空
    if "_upload_key" not in st.session_state:
        st.session_state._upload_key = 0

    uploaded_file = st.file_uploader(
        "上传民航文档",
        type=["pdf", "md", "txt", "docx"],
        key=f"doc_uploader_{st.session_state._upload_key}",
    )
    if uploaded_file is not None:
        with st.spinner(f"正在上传并索引 {uploaded_file.name}..."):
            result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
            if result.get("code") == 0:
                st.success(f"✅ {uploaded_file.name} 上传并索引成功")
                st.session_state._upload_key += 1  # 自增 key，下次渲染清空上传器
                st.rerun()
            else:
                detail = result.get("data", {}).get("detail", "未知错误")
                st.error(f"❌ 上传失败: {detail}")

    # ---- 文档列表 ----
    if st.button("🔄 刷新文档列表", use_container_width=True):
        st.rerun()

    result = list_documents()
    if result.get("code") == 0:
        data = result.get("data", {})
        docs = data.get("documents", [])
        total = data.get("total", 0)

        st.caption(f"共 {total} 份文档")

        for doc in docs:
            doc_id = doc.get("id", "")
            name = doc.get("file_name", "未知")
            indexed = doc.get("indexed", False)
            chunks = doc.get("chunk_count", 0)

            col1, col2 = st.columns([3, 1])
            with col1:
                status_icon = "✅" if indexed else "⬜"
                st.caption(f"{status_icon} {name} ({chunks} 块)")
            with col2:
                if st.button("🗑", key=f"del_{doc_id}", help=f"删除 {name}"):
                    del_result = delete_document(doc_id)
                    if del_result.get("code") == 0:
                        st.success("已删除")
                    else:
                        st.error("删除失败")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.caption("⚠️ 无法加载文档列表")


# =============================================================================
# 侧边栏 —— Agent 决策面板
# =============================================================================

def render_agent_panel():
    """渲染 Agent 决策可视化面板"""
    st.subheader("🤖 Agent 决策路径")

    agent_info = st.session_state.get("agent_info")

    if agent_info is None:
        st.caption("发送消息后将显示 Agent 决策过程")
        # 预留占位
        st.markdown("""
        ```
        ┌──────────┐
        │  START   │
        └────┬─────┘
             │
        ┌────▼─────┐
        │  intent  │  ⏳ 等待输入...
        └──────────┘
        ```
        """)
        return

    intent = agent_info.get("intent", "?")
    confidence = agent_info.get("confidence", 0)
    rewrites = agent_info.get("rewrites", 0)
    retrieval_count = agent_info.get("retrieval_count", 0)

    # 意图翻译
    intent_map = {
        "policy_query": "政策查询",
        "operation_guide": "操作指引",
        "emergency": "应急场景",
        "uncertain": "不确定",
    }
    intent_cn = intent_map.get(intent, intent)

    # 显示决策流程（简化版）
    lines = [
        "```",
        "          ┌──────────┐",
        "          │  START   │",
        "          └────┬─────┘",
        "               │",
        f"          ┌────▼──────┐",
        f"          │ 意图分类  │ → {intent_cn}",
        f"          └────┬──────┘",
        "               │",
    ]

    if intent == "uncertain":
        lines += [
            f"          ┌────▼──────────┐",
            f"          │ 降级兜底      │ 不属于业务范围",
            f"          └───────────────┘",
        ]
    else:
        lines += [
            f"          ┌────▼──────┐",
            f"          │   检索    │ → 命中 {retrieval_count} 条",
            f"          └────┬──────┘",
            "               │",
            f"          ┌────▼──────┐",
            f"          │ 置信度评估│ → {confidence:.0%}",
            f"          └────┬──────┘",
        ]

        if confidence >= 0.5:
            lines += [
                "               │",
                f"          ┌────▼──────┐",
                f"          │ ✅ 生成回答│",
                f"          └───────────┘",
            ]
        elif rewrites > 0:
            lines += [
                "               │",
                f"          ┌────▼──────┐",
                f"          │ 🔄 改写{rewrites}次 │",
                f"          └────┬──────┘",
                "               │",
                f"          ┌────▼──────┐",
                f"          │ 降级兜底  │",
                f"          └───────────┘",
            ]
        else:
            lines += [
                "               │",
                f"          ┌────▼──────┐",
                f"          │ 🔄 改写中  │",
                f"          └───────────┘",
            ]

    lines.append("```")
    st.markdown("\n".join(lines))

    # 指标卡片
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("置信度", f"{confidence:.0%}")
    with col2:
        st.metric("改写次数", rewrites)
    with col3:
        st.metric("检索命中", retrieval_count)


# =============================================================================
# 侧边栏 —— 预设 Query
# =============================================================================

def render_preset_queries():
    """渲染预设 Query 快捷按钮"""
    st.subheader("📋 预设演示 Query")

    for category, queries in PRESET_QUERIES.items():
        with st.expander(f"{category}（{len(queries)}条）"):
            for q in queries:
                if st.button(q, key=f"preset_{q[:20]}", use_container_width=True):
                    # 把预设问题填入输入并触发
                    st.session_state.preset_query = q
                    st.rerun()


# =============================================================================
# 主界面
# =============================================================================

def main():
    # ---- 标题 ----
    st.title("🛫 国航内部员工智能知识助手")
    st.caption("基于 RAG + Agent 的企业级民航知识库 · 面向一线员工")

    # ---- 侧边栏 ----
    with st.sidebar:
        render_document_panel()
        st.divider()
        render_agent_panel()
        st.divider()
        render_preset_queries()

    # ---- 对话区 ----
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer_with_citations(msg.get("content", ""))
                # 显示引用按钮
                citations = msg.get("citations", [])
                if citations:
                    show_citations_expander(citations)
            else:
                st.markdown(msg.get("content", ""))

    # ---- 模式切换 ----
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        use_agent = st.checkbox("🤖 Agent 路由", value=False, help="启用 LangGraph Agent 智能路由")
    with col2:
        top_k = st.number_input("Top-K", min_value=1, max_value=10, value=5, label_visibility="collapsed")

    # ---- 处理预设 Query ----
    preset = st.session_state.get("preset_query", "")
    if preset:
        prompt = preset
        st.session_state.preset_query = ""  # 清空，避免重复
    else:
        prompt = st.chat_input("请输入您的问题（如：旅客行李摔坏了怎么赔？）")

    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # 调用后端
        with st.chat_message("assistant"):
            if use_agent:
                # ---- Agent 路由模式 ----
                with st.spinner("Agent 决策中..."):
                    result = agent_chat(prompt, top_k)

                if result.get("code") == 0:
                    data = result.get("data", {})
                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    retrieval_count = data.get("retrieval_count", 0)
                    intent = data.get("agent_intent", "")
                    rewrites = data.get("agent_rewrites", 0)
                    confidence = data.get("agent_confidence", 0)

                    # 更新 Agent 信息
                    st.session_state.agent_info = {
                        "intent": intent,
                        "confidence": confidence,
                        "rewrites": rewrites,
                        "retrieval_count": retrieval_count,
                    }

                    render_answer_with_citations(answer)
                    show_citations_expander(citations)

                    # 保存消息
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    })
                else:
                    err = result.get("data", {}).get("detail", "未知错误")
                    st.error(f"Agent 请求失败: {err}")

            else:
                # ---- 普通 RAG 模式（流式） ----
                placeholder = st.empty()
                full_text = ""
                citations = []

                for token, cites in chat_stream(prompt, top_k):
                    full_text += token
                    if cites:
                        citations = cites
                    placeholder.markdown(full_text)

                # 流式结束后渲染引用
                show_citations_expander(citations)

                # 保存消息
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_text,
                    "citations": citations,
                })

    # ---- 底部状态栏 ----
    st.divider()
    healthy, latency = check_health()
    status_class = "status-ok" if healthy else "status-error"
    status_text = "🟢 服务正常" if healthy else "🔴 服务异常"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<span class="{status_class}">{status_text}</span> ({latency})', unsafe_allow_html=True)
    with col2:
        st.caption(f"模式: {'Agent 路由' if use_agent else 'RAG 直连'}")
    with col3:
        st.caption(f"Top-K: {top_k}")
    with col4:
        if st.button("🗑 清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.agent_info = None
            st.rerun()


# =============================================================================
# 启动入口
# =============================================================================

if __name__ == "__main__":
    main()
