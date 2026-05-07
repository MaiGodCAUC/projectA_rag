# A-Day 1：环境准备 & 项目初始化

---

## 核心目标

搭建开发环境，初始化项目结构，跑通第一个 LLM API 调用。

---

## 学习内容

1. Python 虚拟环境管理（venv）与依赖管理（pip + requirements.txt）
2. LLM API 调用方式对比：OpenAI SDK vs LangChain ChatModel
3. 环境变量与配置管理模式：pydantic-settings（类型校验、热切换）
4. FastAPI 最小应用结构：路由、应用工厂模式

---

## 代码任务清单

| # | 任务 | 文件 | 负责 |
|---|------|------|------|
| 1 | 编写 `requirements.txt`，锁定核心依赖 | `requirements.txt` | Claude |
| 2 | 实现 `core/config.py` 配置管理 | `core/config.py` | Claude |
| 3 | 实现 `core/llm.py` LLM 工厂 | `core/llm.py` | **用户**（LLM 调用核心） |
| 4 | 实现 `main.py` 入口 + `/health` + `/chat` | `main.py` | Claude |
| 5 | 编写 `docker-compose.yml` | `docker-compose.yml` | Claude |
| 6 | 编写 `.env.example` | `.env.example` | Claude |
| 7 | 创建 `.gitignore` | `.gitignore` | Claude |

---

## 差异化亮点

- 多模型热切换：在 `.env` 中改 `LLM_PROVIDER=openai|qwen|deepseek` 一键切换，体现工程抽象能力
- pydantic-settings 配置校验：启动时自动检查必填环境变量，缺少 API Key 时报错退出而非运行时崩溃

---

## 验收标准

| # | 验收项 | 量化条件 |
|---|--------|---------|
| 1 | 服务启动 | `python main.py` → `GET /health` 返回 `{"status": "ok"}`，状态码 200 |
| 2 | LLM 连通 | `POST /chat` 发送 `{"message": "你好"}` → 状态码 200，response body 含非空 `content` 字段 |
| 3 | 配置热切换 | 修改 `.env` 中 `LLM_PROVIDER` 从 `openai` 改为 `qwen` 后重启 → `/chat` 仍返回 200 |
| 4 | 配置校验 | `.env` 中删除 `LLM_API_KEY` 后启动 → 进程报错退出，错误信息包含 `LLM_API_KEY` |
| 5 | Docker 语法 | `docker compose config` 无语法错误 |

---

## 用户需自行完成的核心部分

> ⚠️ 以下模块涉及大模型 API 调用核心逻辑，由你亲自编写：

### `core/llm.py` —— LLM 工厂

```python
# TODO: 你需要实现的内容：
# 1. 根据 config 中的 LLM_PROVIDER 选择对应的 ChatModel
# 2. 支持 OpenAI / 通义千问 / DeepSeek 三种 provider
# 3. 统一返回 LangChain BaseChatModel 实例
# 4. 处理各 provider 的 base_url 和 api_key 配置差异

from langchain_openai import ChatOpenAI
# from langchain_qwen import ...   # 按需选用
# from langchain_deepseek import ...  # 按需选用

def get_llm(config: Settings) -> BaseChatModel:
    """
    根据配置返回对应的 LLM 实例
    
    Args:
        config: Settings 对象，含 LLM_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    
    Returns:
        LangChain BaseChatModel 实例
    """
    # 你在这里实现多 provider 切换逻辑
    pass
```

> Claude 已准备好 `core/config.py` 中的 Settings 类，提供 `config.llm_provider`, `config.llm_api_key`, `config.llm_base_url`, `config.llm_model` 等字段供你使用。
