# A-Day 4：向量存储与 Embedding 选型

## 核心目标

对接 Qdrant 向量数据库，实现全量 + 增量索引流水线，完成 Embedding 模型对比实验，用数据支撑选型决策。

---

## 学习内容

| 知识点 | 说明 |
|--------|------|
| Embedding 原理 | 文本 → 高维向量，语义相近的文本向量距离近 |
| Embedding 模型对比 | bge-large-zh-v1.5（本地）、m3e-base（本地）、qwen text-embedding-v3（API） |
| 中文 Embedding 评测 | 民航术语"经停""值机""逾重行李"在各模型上的召回效果 |
| Qdrant 核心操作 | 创建 Collection、批量 upsert、条件查询、payload 过滤 |
| 增量索引 | SHA256 哈希对比，已存在跳过，变更覆盖 |
| 索引流水线 | 文档上传 → 解析 → 切片 → Embedding → 写入 Qdrant |

---

## 代码任务

### 1. `core/embedding.py` —— Embedding 工厂

| 模型 | 类型 | 维度 | 特点 |
|------|------|------|------|
| BAAI/bge-large-zh-v1.5 | 本地 | 1024 | 中文效果最好，需 GPU/CPU |
| moka-ai/m3e-base | 本地 | 768 | 轻量，CPU 可运行 |
| qwen text-embedding-v3 | API | 1536 | 免部署，有调用成本 |

### 2. `rag/vector_store.py` —— Qdrant 操作层

CRUD 操作：create_collection / upsert / search / delete / collection_info

### 3. `rag/indexing_pipeline.py` —— 索引流水线

全量索引 + 增量索引（hash 去重）

### 4. Embedding 选型实验

15 条民航员工典型 query × 3 种 Embedding → Top-5 命中率对比

---

## 差异化亮点

- **Embedding 选型实验报告**：用数据而非直觉选模型
- **民航术语覆盖分析**：验证各 Embedding 对专业术语的表示质量
- **增量索引**：基于 SHA256 哈希，不重复索引已存在文档

---

## 验收标准

- [ ] 3 种 Embedding 模型可热切换
- [ ] Qdrant Collection 创建 + 数据写入成功
- [ ] 文档 → 切片 → Embedding → 写入 全流程跑通
- [ ] 增量索引去重正确
- [ ] Embedding 对比实验有结论输出
