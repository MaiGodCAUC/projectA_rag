# ============================================================
# 国航内部员工智能知识助手 - Dockerfile（多阶段构建）
#
# 构建: docker build -t airchina-rag .
# 运行: docker run -p 8000:8000 --env-file .env airchina-rag
# ============================================================

# =============================================================================
# 阶段 1: builder —— 安装 Python 依赖
# =============================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# 安装系统依赖（pymupdf、FlagEmbedding 等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 先复制 requirements.txt，利用 Docker 缓存层
# 只要 requirements.txt 不变，pip install 就不会重新执行
COPY requirements.txt .

# 安装所有依赖到 /build/deps 目录
# --target: 把所有包安装到指定目录，方便下一阶段复制
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# =============================================================================
# 阶段 2: runtime —— 运行时镜像（精简）
# =============================================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# 只安装运行时必需的系统库（pymupdf 需要 libstdc++）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 阶段复制已安装的依赖
COPY --from=builder /build/deps /usr/local/lib/python3.12/site-packages

# 复制项目源码
COPY . .

# 创建数据目录
RUN mkdir -p data/documents qdrant_data

# 暴露 FastAPI 端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 启动 FastAPI（生产模式，无热重载）
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
