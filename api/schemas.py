"""요청·응답 모델.

검색 응답에 ``filter_reduction`` 을 넣는 이유는 이 파이프라인의 핵심 주장이
"비싼 단계에 덜 보낸다" 이기 때문이다. 주장하는 수치는 응답에서 확인될 수 있어야 한다.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.engine.segment import ContentFormat, Segment
from app.schemas.property import PropertyType


class IndexRequest(BaseModel):
    count: int | None = Field(default=None, ge=1, le=1000)
    seed: int | None = None


class IndexResponse(BaseModel):
    properties: int
    added: int
    updated: int
    removed: int
    unchanged: int
    embedded: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    region: str | None = None
    min_capacity: int | None = Field(default=None, ge=1)
    max_price: int | None = Field(default=None, ge=0)
    property_type: PropertyType | None = None
    required_amenities: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)


class HitOut(BaseModel):
    chunk_id: str
    property_id: str
    doc_type: str
    score: float
    text: str


class SearchResponse(BaseModel):
    hits: list[HitOut]
    grounded: bool
    reason: str | None = None
    candidates_before_filter: int
    candidates_after_filter: int
    filter_reduction: float


class GenerateRequest(BaseModel):
    property_id: str = Field(min_length=1)
    segment: Segment
    format: ContentFormat


class ViolationOut(BaseModel):
    type: str
    detail: str


class GenerateResponse(BaseModel):
    property_id: str
    segment: Segment
    format: ContentFormat
    backend: str
    content: str
    valid: bool
    attempts: int
    violations: list[ViolationOut] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    indexed: bool
    properties: int
    chunks: int
    backend: str
    embedding_cache: dict
    thresholds: dict


__all__ = [
    "GenerateRequest", "GenerateResponse", "HitOut", "IndexRequest",
    "IndexResponse", "MetricsResponse", "SearchRequest", "SearchResponse",
    "ViolationOut",
]
