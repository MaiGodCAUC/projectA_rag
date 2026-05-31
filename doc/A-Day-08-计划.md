# A-Day 8：FastAPI 工程化

## 核心目标

将 RAG 能力封装为生产级 RESTful API，实现 Swagger 自动文档、
统一响应格式、统一错误码、请求日志中间件、SSE 流式输出。

## 学习内容

### 1. FastAPI 项目最佳实践

| 概念 | 说明 | 本项目对应 |
|------|------|-----------|
| **APIRouter** | 路由分组，替代把所有路由写在 main.py 里的做法 | `api/routes/` 下按业务模块拆分 |
| **依赖注入** | 用 `Depends()` 注入公共依赖（如配置、LLM 实例） | 后续 Day 9 加入 |
| **中间件链** | HTTP 请求 → CORS → 异常处理 → 日志 → 路由 → 响应 | `api/middleware.py` |
| **Lifespan** | 启动/关闭时执行的操作 | 已有，保持不变 |

### 2. 统一响应格式

所有接口返回同一 JSON 结构：

```json
{
  "code": 0,
  "data": {...},
  "trace_id": "a1b2c3d4",
  "cost_ms": 123
}
```

### 3. 统一错误码

| 错误码 | 含义 | 返回示例 |
|--------|------|---------|
| 0 | 成功 | `{"code": 0, "data": {...}}` |
| 1001 | 参数校验失败 | `{"code": 1001, "data": {"detail": "message 不能为空"}}` |
| 1002 | 文档解析失败 | `{"code": 1002, "data": {"detail": "PDF 解析失败: ..."}}` |
| 1003 | 检索超时/失败 | `{"code": 1003, "data": {"detail": "Qdrant 连接超时"}}` |
| 1004 | LLM 生成失败 | `{"code": 1004, "data": {"detail": "API Key 无效"}}` |
| 1005 | 文档未索引 | `{"code": 1005, "data": {"detail": "请先上传并索引文档"}}` |
| 9999 | 未知错误 | `{"code": 9999, "data": {"detail": "服务器内部错误"}}` |

## 代码任务

### 任务 1：重构项目结构

```
project_a_rag/
├── main.py                    # 入口，挂载路由
├── api/
│   ├── __init__.py
│   ├── middleware.py          # 异常处理 + 请求日志中间件（TODO 用户）
│   ├── models.py              # 统一请求/响应 Pydantic 模型
│   └── routes/
│       ├── __init__.py
│       ├── chat.py            # RAG 对话接口（普通/SSE流式）（TODO 用户核心逻辑）
│       ├── document.py        # 文档管理（上传/列表/解析/删除/索引）（TODO 用户核心逻辑）
│       ├── eval.py            # 评估接口（触发评估/查看报告）
│       └── health.py          # 健康检查
```

### 任务 2：API 路由设计

| 路由 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/chat` | POST | RAG 对话（非流式） |
| `/chat/stream` | POST | RAG 对话（SSE 流式） |
| `/documents` | GET | 文档列表 + 索引状态 |
| `/documents/upload` | POST | 上传文档 + 自动索引 |
| `/documents/{doc_id}` | DELETE | 删除文档 + 清理索引 |
| `/documents/index` | POST | 触发重新索引 |
| `/eval/run` | POST | 触发 RAGAS 评估 |
| `/eval/report` | GET | 查看最新评估报告 |

## 差异化亮点

1. **Swagger 中文文档**：所有接口描述、参数说明用中文，面试官打开 `http://localhost:8000/docs` 即可理解全部功能
2. **响应携带 cost_ms + trace_id**：生产级可观测性，每个请求的耗时和追踪 ID 都在响应中
3. **中间件自动记录请求日志**：请求路径、耗时、状态码自动打印
4. **SSE 流式输出**：非流式和流式两种模式可选

## 验收标准

- `python main.py` 启动后，`http://localhost:8000/docs` 可访问
- Swagger 文档完整展示所有接口，中文描述清晰
- `/chat` 返回 RAG 增强回答（含 [来源: ...] 引用标记）
- `/chat/stream` 在浏览器中逐字显示
- `/documents/upload` 上传文档后可查询索引状态
- 所有响应统一格式 `{"code": ..., "data": ..., "trace_id": ..., "cost_ms": ...}`
