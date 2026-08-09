"""검색 순서와 증분 인덱싱 검증.

- 메타데이터 필터가 벡터 검색 **앞에** 적용되는가
- 가격 하나 바뀌었을 때 해당 청크만 다시 임베딩되는가
"""
from __future__ import annotations

import pytest

from app.data.generator import generate
from app.engine.chunker import chunk_property
from app.engine.embedder import Embedder, LocalEmbedder
from app.engine.indexer import PropertyIndexer
from app.schemas.property import DocumentType, PropertyStatus, SearchFilter


@pytest.fixture(scope="module")
def properties():
    return generate(n=60, seed=7)


@pytest.fixture(scope="module")
def indexer(properties, tmp_path_factory):
    from app.engine.embedder import EmbeddingCache

    cache = EmbeddingCache(tmp_path_factory.mktemp("emb"))
    idx = PropertyIndexer(Embedder(backend=LocalEmbedder(), cache=cache))
    idx.index_all(properties)
    return idx


# ------------------------------------------------------------------ 청킹
def test_chunk_types_follow_field_boundaries(properties):
    p = properties[0]
    chunks = chunk_property(p)
    types = {c.document_type for c in chunks}
    assert DocumentType.BASIC in types
    assert DocumentType.ROOM in types
    assert DocumentType.POLICY in types
    assert DocumentType.LOCATION in types
    # 객실 수만큼 ROOM 청크가 나온다
    assert sum(c.document_type is DocumentType.ROOM for c in chunks) == len(p.rooms)
    # 편의시설은 항목마다 하나씩
    assert sum(c.document_type is DocumentType.AMENITY for c in chunks) == len(p.amenities)


def test_every_chunk_carries_filter_metadata(properties):
    for c in chunk_property(properties[0]):
        assert c.region
        assert c.property_status
        assert c.content_hash


# ------------------------------------------------------------------ 검색
def test_metadata_filter_narrows_before_search(indexer):
    """필터가 후보를 실제로 줄여야 한다."""
    hits, stats = indexer.search("바다 근처 숙소", flt=SearchFilter(region="Jeju"))
    assert stats.after_filter < stats.total_chunks
    assert stats.filter_reduction > 0
    assert all(h.chunk.region == "Jeju" for h in hits)


def test_capacity_filter_excludes_small_properties(indexer):
    flt = SearchFilter(min_capacity=6)
    hits, _ = indexer.search("가족 여행", flt=flt)
    for h in hits:
        assert indexer.properties[h.chunk.property_id].max_capacity >= 6


def test_amenity_filter_is_exact(indexer):
    flt = SearchFilter(required_amenities=["수영장"])
    hits, _ = indexer.search("수영장 있는 숙소", flt=flt)
    assert hits, "수영장을 가진 숙소가 하나도 없다 — 생성 데이터를 확인할 것"
    for h in hits:
        assert "수영장" in indexer.properties[h.chunk.property_id].all_amenities


def test_inactive_properties_are_excluded_by_default(indexer):
    hits, _ = indexer.search("숙소")
    for h in hits:
        assert h.chunk.property_status is PropertyStatus.ACTIVE


def test_hybrid_merges_both_channels(indexer):
    hits, stats = indexer.search("제주 수영장", flt=SearchFilter(region="Jeju"))
    assert stats.dense_candidates > 0
    assert stats.bm25_candidates > 0
    # RRF 병합 결과는 점수 내림차순이어야 한다
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


# ------------------------------------------------------- 증분 인덱싱
def test_price_change_reindexes_only_that_chunk(indexer, properties):
    """가격 1건 변경 → 해당 ROOM 청크만 갱신."""
    p = indexer.properties[properties[1].property_id].model_copy(deep=True)
    p.rooms[0].price += 10_000

    report = indexer.upsert(p)
    assert report.updated == 1, str(report)
    assert report.added == 0
    assert report.unchanged > 0


def test_unchanged_property_costs_no_embedding(indexer, properties):
    """내용이 그대로면 임베딩 호출이 발생하지 않는다."""
    p = indexer.properties[properties[2].property_id]
    report = indexer.upsert(p.model_copy(deep=True))
    assert report.updated == 0
    assert report.added == 0
    assert report.embedded == 0, "내용이 같은데 임베딩이 다시 계산됐다"


def test_removed_room_drops_its_chunk(indexer, properties):
    pid = properties[3].property_id
    p = indexer.properties[pid].model_copy(deep=True)
    if len(p.rooms) < 2:
        pytest.skip("객실이 하나뿐인 숙소")
    p.rooms = p.rooms[:-1]
    report = indexer.upsert(p)
    assert report.removed >= 1
