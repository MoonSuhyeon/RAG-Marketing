"""색인과 검색.

검색 응답에 기권 판정을 같이 싣는다. 호출자가 ``hits`` 만 보고 쓰면
빈약한 근거 위에 그럴듯한 문장을 만들게 되는데, 그게 v25에서 배운 실패 방식이다.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.property import SearchFilter
from retrieval import assess

from api import config
from api.schemas import (
    HitOut, IndexRequest, IndexResponse, SearchRequest, SearchResponse,
)
from api.state import runtime

router = APIRouter(tags=["retrieval"])


@router.post("/index", response_model=IndexResponse)
def build_index(body: IndexRequest | None = None) -> IndexResponse:
    """데모 숙소를 생성해 색인한다.

    인덱스 구조 자체는 매번 다시 만든다. 비싼 것은 그쪽이 아니라 임베딩이고,
    콘텐츠 해시가 같은 청크는 디스크 캐시에서 나오므로 ``embedded`` 가 0에 가깝다.
    숙소 한 곳만 바뀌었을 때는 ``PropertyIndexer.upsert`` 가 그 청크만 건드린다.
    """
    body = body or IndexRequest()
    props, report = runtime.load_demo(body.count, body.seed)
    return IndexResponse(
        properties=len(props),
        added=report.added,
        updated=report.updated,
        removed=report.removed,
        unchanged=report.unchanged,
        embedded=report.embedded,
    )


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest) -> SearchResponse:
    if not runtime.indexed:
        raise HTTPException(409, "색인이 비어 있습니다. 먼저 POST /index 를 호출하세요")

    flt = SearchFilter(
        region=body.region,
        min_capacity=body.min_capacity,
        max_price=body.max_price,
        property_type=body.property_type,
        required_amenities=body.required_amenities,
    )
    hits, stats = runtime.indexer.search(body.query, flt=flt, top_k=body.top_k)

    # 점수 자체가 낮거나 1·2위 격차가 없으면 근거로 쓰지 않는다.
    # assess 는 점수만 보므로 엔진 쪽 Hit 을 그대로 넘긴다.
    grounding = assess(hits, min_score=config.MIN_SCORE, min_margin=config.MIN_MARGIN)

    return SearchResponse(
        hits=[
            HitOut(
                chunk_id=h.chunk.chunk_id,
                property_id=h.chunk.property_id,
                doc_type=h.chunk.document_type.value,
                score=h.score,
                text=h.chunk.text,
            )
            for h in hits
        ],
        grounded=bool(grounding),
        reason=getattr(grounding, "reason", None),
        candidates_before_filter=stats.total_chunks,
        candidates_after_filter=stats.after_filter,
        filter_reduction=stats.filter_reduction,
    )
