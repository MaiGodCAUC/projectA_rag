"""
可观测性接口 —— 查看 RAG 运行时指标

GET /metrics          → 指标总览（请求量/延迟分位数/Token/错误率）
GET /metrics/nodes    → 各节点耗时分布
GET /metrics/queries  → Top 高频查询
POST /metrics/reset   → 重置计数器

----------------------------------------------------------------------
数据来源: core/observability.py 的 ObservabilityCollector 全局单例
每次 /chat 请求完成后自动上报指标到 collector
----------------------------------------------------------------------
"""

import os
import json
from fastapi import APIRouter

from api.models import APIResponse
from core.observability import get_collector

router = APIRouter(tags=["可观测性"])


@router.get("/metrics")
async def get_metrics():
    """获取 RAG 服务运行时指标总览

    返回内容:
    - request_count: 总请求数
    - error_rate: 错误率（%）
    - avg_latency_ms: 平均延迟
    - p50_latency_ms / p95_latency_ms / p99_latency_ms: 延迟分位数
    - total_tokens: 累计 Token
    - avg_tokens_per_request: 每次请求平均 Token
    - uptime_seconds: 服务运行时长
    """
    collector = get_collector()
    snapshot = collector.get_snapshot()

    return APIResponse.ok(data={
        "uptime_seconds": snapshot["uptime_seconds"],
        "request_count": snapshot["request_count"],
        "error_count": snapshot["error_count"],
        "error_rate_percent": snapshot["error_rate_percent"],
        "avg_latency_ms": snapshot["avg_latency_ms"],
        "p50_latency_ms": snapshot["p50_latency_ms"],
        "p95_latency_ms": snapshot["p95_latency_ms"],
        "p99_latency_ms": snapshot["p99_latency_ms"],
        "total_tokens": snapshot["total_tokens"],
        "avg_tokens_per_request": snapshot["avg_tokens_per_request"],
    })


@router.get("/metrics/nodes")
async def get_node_metrics():
    """获取各 RAG 节点的耗时分布

    节点列表:
    - context_manage: 上下文窗口管理
    - build_context: 构建 Prompt 上下文
    - assemble_messages: 组装消息
    - llm_generate: LLM 生成（通常最耗时）
    - citation_extract: 引用提取

    返回每个节点的: 平均耗时、累计耗时、调用次数
    """
    collector = get_collector()
    snapshot = collector.get_snapshot()

    return APIResponse.ok(data={
        "nodes": snapshot.get("nodes", {}),
    })


@router.get("/metrics/queries")
async def get_top_queries():
    """获取高频查询 Top 20"""
    collector = get_collector()
    snapshot = collector.get_snapshot()

    return APIResponse.ok(data={
        "top_queries": [
            {"query": q, "count": c}
            for q, c in snapshot.get("top_queries", [])
        ],
    })


@router.post("/metrics/reset")
async def reset_metrics():
    """重置所有指标计数器（用于测试或演示前清理数据）"""
    collector = get_collector()
    collector.reset()
    return APIResponse.ok(data={"status": "已重置"})
