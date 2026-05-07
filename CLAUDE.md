# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# 项目概述

国航内部员工智能知识助手 —— 面向国航一线员工（客服坐席、值机柜台、登机口、行李查询、特殊服务协调员）的企业级 RAG 智能知识库。

# 项目背景与个人目标
- 求职目标岗位：国内大模型应用开发工程师 / RAG开发工程师
- 技术基础：熟悉 Python、大模型 API 调用、LangChain 基础
- 目标：打造企业级 RAG 项目写入简历，要求有明确业务价值、工程深度和个人专属工作量
- **大模型相关核心代码由我亲自编写，Claude 辅助完成工程化部分**

# 常用技术栈
- 核心语言：Python 3.11+
- RAG 框架：LangChain >= 0.3
- Agent 框架：LangGraph >= 0.2（Day 13 加分项）
- API 框架：FastAPI >= 0.115
- 向量数据库：Qdrant（Docker 部署）
- 文档解析：PyMuPDF、pdfplumber、python-docx
- 评估框架：RAGAS >= 0.2
- 可观测性：LangSmith
- 重排序：FlagEmbedding (bge-reranker-v2-m3)
- 中文分词：jieba
- 前端演示：Streamlit
- 部署：Docker + Docker Compose + Nginx

# 项目结构

```
project_a_rag/
├── api/                    # FastAPI 接口层
│   ├── routes/             # 路由（chat, document, eval, health）
│   └── middleware.py       # 异常处理 + 请求日志中间件
├── core/                   # 核心配置与基础设施
│   ├── config.py           # pydantic-settings 配置管理
│   ├── llm.py              # LLM 工厂（OpenAI/通义千问/DeepSeek 热切换）
│   ├── embedding.py        # Embedding 工厂（多模型切换）
│   └── observability.py    # 指标收集器
├── rag/                    # RAG 核心引擎
│   ├── loader.py           # 文档加载器统一接口
│   ├── loaders/            # 各格式解析器（pdf, docx, md）
│   ├── models.py           # Pydantic 数据模型
│   ├── splitter.py         # 切片策略（含 PolicyClauseSplitter）
│   ├── vector_store.py     # Qdrant 操作层
│   ├── indexing_pipeline.py # 索引流水线
│   ├── bm25.py             # BM25 关键词检索
│   ├── hybrid_search.py    # 混合检索 + RRF 融合
│   ├── reranker.py         # 重排序模块
│   ├── generator.py        # RAG 生成器 + 引用溯源
│   └── callbacks.py        # LangSmith Tracing Callback
├── eval/                   # 评估模块
│   └── ragas_eval.py       # RAGAS 评估流水线
├── agent/                  # Agent 路由层（Day 13 加分项）
│   └── router_graph.py     # LangGraph 智能路由 Agent
├── frontend/               # 前端
│   └── app.py              # Streamlit 工作台
├── data/
│   ├── documents/          # 10 份民航模拟文档（md + pdf）
│   ├── test_queries.json   # 30 条检索测试集
│   └── eval_dataset.json   # 30 条 RAGAS 评估集
├── tests/                  # 测试
├── doc/                    # 每日详细计划文档
├── main.py                 # 入口
├── Dockerfile
├── docker-compose.yml
├── docker-compose.dev.yml
├── nginx.conf
├── .env.example
└── requirements.txt
```

# 开发命令

```bash
# 虚拟环境
python -m venv .venv && source .venv/Scripts/activate  # Windows
python -m venv .venv && source .venv/bin/activate      # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
python main.py

# 运行测试
pytest tests/ -v

# 测试覆盖率
pytest tests/ -v --cov=rag --cov=core --cov-report=term-missing

# Docker 部署
docker compose up -d              # 生产模式
docker compose -f docker-compose.dev.yml up -d  # 开发模式
```

# 核心架构决策

1. **PolicyClauseSplitter（自研核心）**：条款感知切片，保证民航政策条款语义完整
2. **混合检索**：BM25(精确匹配) + 向量检索(语义搜索) + RRF 融合 + bge-reranker
3. **条款级引用溯源**：LLM 回答中标注 `[来源: 文档名 第X条]`，可追溯到原文片段
4. **RAGAS 评估闭环**：30 条手工标注评估集，4 项指标量化系统质量
5. **E 端定位**：内部员工工具，面试演示不需要 C 端 UI 精致度

# 每日任务规范
- 每日开始前先输出详细计划文档到 `doc/A-Day-XX-计划.md`
- 每日计划格式：核心目标 → 学习内容 → 代码任务 → 差异化亮点 → 验收标准
- 遵循「环境准备 → 项目设计 → 核心开发 → 亮点打造 → 工程化上线 → 简历包装」顺序
- **涉及 LLM Prompt 设计、LLM 调用链路、LLM 生成逻辑的部分由我亲自编写**
- Claude 辅助：文档解析、数据模型、API 路由、测试、Docker 配置等工程化部分
