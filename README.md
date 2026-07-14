# 🛫 国航内部员工智能知识助手

基于 **RAG + LangGraph Agent** 的企业级民航知识库，面向航空公司一线员工（客服坐席、值机柜台、登机口、行李查询、特殊服务协调员）。

> "将海量民航政策文档构建为可检索知识库，帮助员工在面对旅客复杂问题时快速获取精准答案，降低培训成本，提升服务质量一致性。"

---

## 业务背景

航空业是典型的高合规、高文件密度行业。一线员工需要掌握客规运价、行李政策、会员体系、特殊旅客服务、航班不正常处置、证件签证、投诉处理等数十类文档。新员工上手需 3~6 个月，老员工也难以记住所有政策的细节条款。

本系统将 10+ 份民航政策文档构建为 **RAG 知识库**，结合 **LangGraph Agent 智能路由**，实现：
- 📄 **文档解析**：支持 PDF/Markdown/DOCX，表格感知提取
- 🔍 **混合检索**：向量语义 + BM25 关键词 + RRF 融合 + 重排序
- 🧠 **Agent 决策**：意图分类 → 置信度评估 → Self-Reflection 改写
- 📎 **条款溯源**：每条回答可追溯到原文条款，点击查看原文
- 📊 **质量评估**：RAGAS 4 项指标量化，30 条手工标注评估集
- 🔭 **可观测性**：LangSmith 全链路追踪 + P50/P95/P99 延迟监控

---

## 技术架构

```
┌─────────────────────────────────────────────────┐
│              Streamlit 工作台 (:8501)             │
│  对话区 · 文档管理 · Agent 决策面板 · 引用弹窗    │
└──────────────────┬──────────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────────┐
│              FastAPI (:8000)                     │
│  /chat · /agent/chat · /documents · /eval · /metrics │
└──────┬──────────────────────────┬───────────────┘
       │                          │
┌──────▼──────┐  ┌────────┐  ┌───▼──────────┐
│  LangGraph   │  │  RagAS  │  │ Observability│
│  Agent 路由  │  │  评估   │  │  指标收集    │
└──────┬──────┘  └────────┘  └──────────────┘
       │
┌──────▼──────────────────────────────┐
│           RAG 引擎                   │
│  文档解析 → 条款切片 → 混合检索 → 生成 │
│  (LangChain + BGE + BM25 + RRF)      │
└──────┬──────────────────────────────┘
       │
┌──────▼──────┐  ┌──────────┐
│   Qdrant    │  │ DeepSeek │
│  向量数据库  │  │   LLM    │
└─────────────┘  └──────────┘
```

---

## 快速启动

### 前置条件
- Python 3.11+
- LLM API Key（DeepSeek 或通义千问）

### 1. 克隆项目
```bash
git clone https://github.com/MaiGodCAUC/projectA_rag.git
cd projectA_rag
```

### 2. 配置环境
```bash
cp .env.example .env
# 编辑 .env，填写 LLM_API_KEY
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 启动服务
```bash
# 终端 1：后端 API
python main.py
# → http://localhost:8000/docs

# 终端 2：前端工作台
streamlit run frontend/app.py
# → http://localhost:8501
```

### 5. Docker 部署（可选）
```bash
docker compose up -d
# → http://localhost
```

---

## 项目结构

```
project_a_rag/
├── api/                        # FastAPI 接口层
│   ├── routes/                 # chat / agent / document / eval / health / observability
│   ├── models.py               # 统一请求/响应模型 + 错误码
│   └── middleware.py            # 异常处理 + 请求日志
├── core/                       # 核心基础设施
│   ├── config.py               # pydantic-settings 配置
│   ├── llm.py                  # LLM 工厂（DeepSeek/通义千问 热切换）
│   ├── embedding.py            # Embedding 工厂（BGE/m3e/千问）
│   └── observability.py        # 指标收集器（P50/P95/P99）
├── rag/                        # RAG 核心引擎
│   ├── loader.py               # 文档加载器统一接口
│   ├── splitter.py             # 切片策略（含 PolicyClauseSplitter）
│   ├── vector_store.py         # Qdrant 操作层
│   ├── hybrid_search.py        # 混合检索（向量 + BM25 + RRF）
│   ├── reranker.py             # bge-reranker 重排序
│   ├── generator.py            # RAG 生成器 + 引用溯源
│   └── callbacks.py            # LangSmith Tracing
├── agent/                      # Agent 路由层
│   └── router_graph.py         # LangGraph 智能路由 Agent
├── eval/                       # 评估模块
│   └── ragas_eval.py           # RAGAS 评估流水线
├── frontend/                   # 前端
│   └── app.py                  # Streamlit 工作台
├── data/
│   ├── documents/              # 10 份民航模拟文档
│   └── eval_dataset.json       # 30 条 RAGAS 评估集
├── tests/                      # 单元测试（29 条）
├── doc/                        # 每日计划 + 使用指南
├── main.py                     # FastAPI 入口
├── Dockerfile                  # 多阶段构建
├── docker-compose.yml          # 生产部署
└── DEPLOY.md                   # 部署文档
```

---

## 核心亮点

| 亮点 | 说明 | 技术 |
|------|------|------|
| **条款感知切片** | PolicyClauseSplitter 保证每条政策条款自成 chunk，不切断编号链 | 自研 |
| **混合检索** | BM25+向量+RRF 融合+Reranker，精确匹配与语义搜索互补 | LangChain + FlagEmbedding |
| **条款溯源** | 回答中标注 `[来源: 文档名 第X条]`，点击弹窗查看原文 | 自研 Prompt 设计 |
| **Agent 自我纠错** | 检索不足 → LLM 改写 Query → 重新检索（最多 2 轮） | LangGraph |
| **RAGAS 评估** | 4 项指标量化 + 30 条手工标注评估集 | RAGAS |
| **三级可观测** | LangSmith Trace + Callback 计时 + 聚合指标（P50/P95/P99） | LangSmith |

---

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat` | POST | RAG 对话（非流式） |
| `/chat/stream` | POST | RAG 对话（SSE 流式） |
| `/agent/chat` | POST | Agent 路由对话 |
| `/agent/chat/stream` | POST | Agent 路由对话（SSE） |
| `/documents` | GET | 文档列表（含索引状态） |
| `/documents/upload` | POST | 上传文档 + 自动索引 |
| `/documents/{doc_id}` | DELETE | 删除文档 |
| `/eval/run` | POST | 运行 RAGAS 评估 |
| `/eval/report` | GET | 查看评估报告 |
| `/metrics` | GET | 运行时指标（P50/P95/P99 等） |

---

## 运行测试

```bash
pytest tests/ -v                                          # 全部测试
pytest tests/ -v --cov=rag --cov=core --cov-report=term-missing  # 覆盖率
```

---

## License

MIT
