# 国航内部员工智能知识助手 —— 部署指南

## 架构图

```
                    ┌──────────────────────────────┐
                    │       Nginx (:80)             │
                    │   反向代理 + 路由分发          │
                    └──────┬───────────┬───────────┘
           /api/* /docs    │           │    /*
                          │           │
              ┌───────────▼──┐   ┌────▼──────────┐
              │  API (:8000) │   │ Streamlit      │
              │  FastAPI     │   │ (:8501)        │
              │  后端服务     │   │ 前端工作台     │
              └───────┬──────┘   └───────────────┘
                      │
                      │ HTTP (Qdrant 兼容 API)
                      │
              ┌───────▼──────┐
              │ Qdrant       │
              │ (:6333)      │
              │ 向量数据库    │
              └──────────────┘
```

## 快速启动

### 前置条件

- Docker Desktop（Windows/Mac）或 Docker Engine（Linux）
- LLM API Key（DeepSeek 或通义千问）
- 可选：LangSmith API Key（可观测性）

### 1. 配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑 .env，填写真实 API Key
# LLM_API_KEY=sk-your-real-key
```

### 2. 启动全部服务

```bash
# 生产模式（后台运行）
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

### 3. 访问

| 地址 | 服务 |
|------|------|
| http://localhost | 前端工作台（Streamlit） |
| http://localhost:8000/docs | API 文档（Swagger） |
| http://localhost:8000/health | 健康检查 |
| http://localhost:8501 | Streamlit 直连 |

## 开发模式

开发时只需要 Docker 运行 Qdrant，API 和前端在本地跑（支持热重载）：

```bash
# 1. 启动 Qdrant
docker compose -f docker-compose.dev.yml up -d

# 2. 本地启动 API（热重载）
python main.py

# 3. 本地启动前端（热重载）
streamlit run frontend/app.py
```

## 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 80 | Nginx | 统一入口 |
| 8000 | FastAPI | API 服务 |
| 8501 | Streamlit | 前端 |
| 6333 | Qdrant HTTP | 向量数据库 API |
| 6334 | Qdrant gRPC | 向量数据库 gRPC |

## 环境变量清单

| 变量 | 说明 | 示例 |
|------|------|------|
| `LLM_PROVIDER` | LLM 提供商 | `deepseek` / `qwen` |
| `LLM_API_KEY` | LLM API Key | `sk-xxx` |
| `LLM_MODEL` | LLM 模型名 | `deepseek-chat` |
| `EMBEDDING_PROVIDER` | 向量化模型 | `bge` / `m3e` / `qwen` |
| `EMBEDDING_MODEL` | 向量化模型名 | `BAAI/bge-large-zh-v1.5` |
| `EMBEDDING_DEVICE` | 推理设备 | `cpu` / `cuda` |
| `QDRANT_HOST` | Qdrant 地址 | `localhost` (开发) / `qdrant` (Docker) |
| `QDRANT_PORT` | Qdrant 端口 | `6333` |
| `QDRANT_PATH` | 本地模式路径 | `./qdrant_data`（本地开发用） |
| `QDRANT_COLLECTION` | Collection 名称 | `airchina_knowledge_base` |
| `LANGCHAIN_TRACING_V2` | 启用 LangSmith | `true` |
| `LANGCHAIN_ENDPOINT` | LangSmith 端点 | `https://api.smith.langchain.com` |
| `LANGCHAIN_API_KEY` | LangSmith Key | `lsv2_pt_xxx` |
| `LANGCHAIN_PROJECT` | LangSmith 项目名 | `airchina-rag` |

## 常见问题

### Q: Qdrant 连接失败？
确保 Docker 中 Qdrant 容器在运行：`docker compose ps`

### Q: LLM 调用超时？
检查 `.env` 中 `LLM_API_KEY` 是否正确，网络能否访问 DeepSeek API。

### Q: 流式输出卡住？
Nginx 已配置 `proxy_buffering off`，如果还卡住，检查是否经过了额外的代理层。

### Q: 容器重启后数据丢失？
Qdrant 数据存储在 named volume `airchina_qdrant_data` 中，`docker compose down` 不会删除。如需完全清空：`docker compose down -v`

### Q: 本地开发时不想用 Docker？
设置 `QDRANT_PATH=./qdrant_data`，Qdrant 以嵌入模式运行在 Python 进程内，无需 Docker。
