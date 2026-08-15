"""운영 지표 — 색인 크기, 임베딩 캐시 적중, 적용 중인 임계값.

임계값을 응답에 싣는 이유는 기권·거부가 났을 때 "무슨 기준으로 걸렀나"를
호출자가 되짚을 수 있어야 하기 때문이다.
"""
from __future__ import annotations

from fastapi import APIRouter

from api import config
from api.schemas import MetricsResponse
from api.state import runtime

router = APIRouter(tags=["monitoring"])


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(
        indexed=runtime.indexed,
        properties=len(runtime.indexer.properties),
        chunks=len(runtime.indexer.index),
        backend=runtime.backend,
        embedding_cache=runtime.indexer.cache_stats,
        thresholds={
            "min_score": config.MIN_SCORE,
            "min_margin": config.MIN_MARGIN,
            "max_regenerate": config.MAX_REGENERATE,
        },
    )
