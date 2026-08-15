"""FastAPI 앱 — 숙소 콘텐츠 생성 파이프라인의 HTTP 표면.

    POST /index            데모 숙소 색인 (증분)
    POST /search           메타 필터 → dense+BM25 → RRF, 기권 판정 포함
    POST /generate         세그먼트×형식 생성 + 사실 검증. 실패하면 422
    POST /generate/stream  단계별 SSE
    GET  /metrics          색인·캐시·임계값
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routers import generate, metrics, search
from api.state import runtime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")

app = FastAPI(
    title="Marketplace Content API",
    description="숙소 레코드에서 세그먼트별 마케팅 문구를 만들고, 내보내기 전에 원본과 대조한다.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    logger.info(
        "%s %s → %s (%.0fms)",
        request.method, request.url.path, response.status_code, (time.time() - t0) * 1000,
    )
    return response


app.include_router(search.router)
app.include_router(generate.router)
app.include_router(metrics.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "indexed": runtime.indexed}
