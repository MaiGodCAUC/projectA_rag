# A-Day 12：Docker 容器化部署

## 核心目标

`docker compose up` 一行命令启动全部服务，实现「开发环境 = 生产环境」。

---

## 学习内容

### Docker 多阶段构建

```
阶段 1: builder（构建阶段）
  pip install 所有依赖 → 生成 site-packages

阶段 2: runtime（运行阶段）
  复制 site-packages + 源码 → 最终镜像
  不包含 gcc/make 等构建工具 → 镜像体积小
```

### Docker Compose 多服务编排

```
docker-compose.yml:

┌──────────────────────────────────────────────┐
│                  Nginx (:80)                  │
│              反向代理 + 静态文件               │
│           /api/* → api:8000                  │
│           /*     → streamlit:8501             │
└──────────────┬───────────────┬───────────────┘
               │               │
    ┌──────────▼──────┐  ┌─────▼──────────┐
    │  api (:8000)    │  │ streamlit (:8501)│
    │  FastAPI 后端   │  │ 前端工作台       │
    └────────┬────────┘  └─────────────────┘
             │
    ┌────────▼────────┐
    │  qdrant (:6333) │
    │  向量数据库      │
    └─────────────────┘
```

### Volume 持久化 vs 源码挂载

| 模式 | Volume 类型 | 代码修改 | 适用 |
|------|------------|---------|------|
| 生产 | named volume | 需重建镜像 | 部署 |
| 开发 | bind mount | 即时生效 | 开发 |

---

## 代码任务

### 1. `Dockerfile` —— 多阶段构建
- builder 阶段：安装 Python 依赖
- runtime 阶段：复制依赖 + 源码，最终镜像 < 1GB

### 2. `docker-compose.yml` —— 生产部署
- 4 个 service：qdrant + api + streamlit + nginx
- Volume 持久化 Qdrant 数据
- 健康检查 + depends_on

### 3. `docker-compose.dev.yml` —— 开发模式
- 只启动 Qdrant（API 和 Streamlit 本地跑）
- 热重载，无需重建镜像

### 4. `nginx.conf` —— 反向代理
- `/api/*` → FastAPI:8000
- `/*` → Streamlit:8501

### 5. `DEPLOY.md` —— 部署文档
- 架构图、端口说明、环境变量清单、常见问题

---

## 差异化亮点

1. **多阶段构建**：最终镜像不含编译工具，体积控制在合理范围
2. **dev/prod 双配置**：开发用源码挂载，生产用镜像部署
3. **SSE 流式支持**：Nginx 关闭缓冲，保证流式输出正常
4. **健康检查**：所有服务有 healthcheck，依赖链保证启动顺序

---

## 验收标准

- [ ] Dockerfile 语法正确
- [ ] `docker compose up -d` 所有服务启动
- [ ] 浏览器访问 http://localhost 完整可用
- [ ] 28 测试不受影响

---

*Day 12 / 14 · Docker 容器化部署*
