# A-Day 5：混合检索 + RRF 融合 + 重排序

## 核心目标

实现向量检索 + BM25 关键词检索 + RRF 融合 + 重排序的完整检索链路，用对比实验证明混合检索优于纯向量检索。

---

## 学习内容

| 知识点 | 说明 |
|--------|------|
| BM25 算法 | TF-IDF 的改进版，基于词频和文档长度归一化的关键词匹配 |
| 中文分词（jieba） | BM25 需要分词才能建立倒排索引 |
| 为什么民航需要 BM25 | 员工常问精确匹配型 query（"Y 舱""CA1234""第12条"） |
| RRF 融合 | Reciprocal Rank Fusion，将多个排序列表合并为一个 |
| RRF 参数 k | 控制融合平滑度，k 越大越平滑 |
| 重排序 | bge-reranker-v2-m3（本地 Cross-Encoder）vs LLM-based rerank |

---

## 代码任务

### 1. `rag/bm25.py` —— BM25 关键词检索引擎

- 基于 rank_bm25 库（BM25Okapi 实现）
- jieba 中文分词
- 增量索引（add/remove documents）
- 统一接口：search(query, top_k) → list[(doc_id, score)]

### 2. `rag/hybrid_search.py` —— 混合检索 + RRF 融合

- 同时调用向量检索 + BM25 检索
- RRF 融合算法：`score = sum(1/(k + rank_i))`
- k 值可配置，支持网格搜索调优
- 返回融合后的结果列表

### 3. `rag/reranker.py` —— 重排序模块

- Cross-Encoder 重排序（bge-reranker-v2-m3）
- 对 Top-N 候选结果精排
- 输出精排后结果

---

## 差异化亮点

- **混合检索**：向量语义 + BM25 精确匹配互补
- **RRF 融合**：不需要归一化分数的排序融合方法
- **检索对比实验**：4 种策略 × 30 条 query 的 Precision@5 / MRR
- **民航关键词优势**：BM25 对"CA1234""第3.2条"等精确匹配查询天然优于向量

---

## 验收标准

- [ ] BM25 检索引擎可独立使用
- [ ] 混合检索（向量 + BM25 + RRF）跑通
- [ ] Reranker 可正常重排序
- [ ] 4 种检索策略对比实验有量化结果
