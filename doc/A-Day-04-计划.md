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

---

## 代码审查结果（2026-05-22）

### 审查通过项

| 文件 | 状态 | 备注 |
|------|------|------|
| `core/embedding.py` | ✅ | BGE/M3E/Qwen 三个 Backend 逻辑正确，工厂模式结构清晰 |
| `rag/vector_store.py` | ✅ | Qdrant CRUD 操作完整，upsert/search/delete 实现正确 |
| `rag/indexing_pipeline.py` | ✅ | index_all 逻辑正确，增量索引流程清晰 |

### 修复的 Bug

| Bug | 位置 | 修复 |
|-----|------|------|
| `files.append(glob())` 生成器未展开 | indexing_pipeline.py L135/L234 | `append` → `extend` |
| `index_incremental` 在 for 内 return | indexing_pipeline.py L273-297 | 移到循环外，补充 total_chunks/details 统计 |
| 多余 import `from importlib.metadata import metadata` | vector_store.py L27 | 删除 |

---

## Embedding 对比实验 —— 框架验证报告

### 实验条件

由于本地 PyTorch 2.3.0 < 2.4（BGE/M3E 不可用）且 Qwen API Key 未配置，
本次实验使用**哈希伪向量**验证对比实验框架的完整性和正确性。
真实 Embedding 数据需 PyTorch 升级后补充。

### 实验配置

| 参数 | 值 |
|------|-----|
| 文档数 | 8 份（data/documents/*.md） |
| Query 数 | 11 条（提取自各文档的 clause_id） |
| Reference 数 | 23 条（各文档的 chunk 内容片段） |
| 伪向量维度 | 128 维（SHA256 派生） |
| 切片策略 | PolicyClauseSplitter (max_chunk_size=800) |

### Query 示例（来自真实文档条款）

```
"第1条: 适用范围"
"1.1: 自愿退票"
"第1条: 客票有效期"
"1.2: 国内航线免费行李额"
...
```

### 框架验证结果

```
余弦相似度矩阵: (11 queries × 23 references)
Top-5 检索: 每条 query 返回 5 个最相似 reference
框架流程: 加载 → 切片 → 向量化 → 余弦相似度 → Top-5 → 指标统计 ✅ 全部跑通
```

### 预期真实 Embedding 对比结果（待环境就绪后补充）

| 指标 | BGE (1024维) | M3E (768维) | Qwen (1536维) |
|------|-------------|------------|---------------|
| Precision@5 | 待测 | 待测 | 待测 |
| MRR | 待测 | 待测 | 待测 |
| 平均延迟/条 | 待测 | 待测 | 待测 |
| 民航术语召回 | 待测 | 待测 | 待测 |

### 下一步

1. `pip install torch>=2.4` 升级 PyTorch
2. 重新运行 `python -c "from core.embedding import run_embedding_comparison; ..."`
3. 填入上表真实数据
4. 面试时用真实数据说明"为什么我选 BGE 作为默认 Embedding"
