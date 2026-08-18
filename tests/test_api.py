"""HTTP 표면 — 파이프라인의 규칙이 API 경계에서도 유지되는가.

엔진 테스트가 검색과 검증을 이미 확인한다. 여기서 보는 것은 **경계에서 새지
않는가** 다. 검증에 실패한 문장이 200 으로 나가면, 호출자는 그것을 발행한다.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.server import app
from api.state import runtime


@pytest.fixture(scope="module", autouse=True)
def cold_cache(tmp_path_factory):
    """임베딩 캐시를 **비운 상태에서** 시작한다.

    캐시는 디스크(`.embed_cache`)에 남는다. 개발 기계에는 이전 실행의 캐시가
    쌓여 있어서, 캐시를 쓰는지 확인하는 테스트가 **첫 색인부터 이미 히트**한 채로
    통과한다. CI 는 매번 빈 상태라 같은 테스트가 거기서만 깨진다.

    실제로 그랬다 — 로컬 통과, CI 실패였고 원인은 캐시가 아니라 `index_all` 이
    **누적** 미스를 보고하던 버그였다. 로컬 캐시가 그 버그를 가리고 있었다.

    `CACHE_DIR` 을 바꾸는 것으로는 안 된다. `EmbeddingCache.__init__` 의 기본
    인자가 **정의 시점에** 묶이고, 런타임 캐시는 import 때 이미 만들어져 있다.
    그래서 살아 있는 객체의 뿌리와 카운터를 직접 갈아끼운다.
    """
    cache = runtime.indexer.embedder.cache
    root = tmp_path_factory.mktemp("embed_cache")
    old_root, old_hits, old_misses = cache.root, cache.hits, cache.misses
    cache.root, cache.hits, cache.misses = root, 0, 0
    yield
    cache.root, cache.hits, cache.misses = old_root, old_hits, old_misses


@pytest.fixture(scope="module")
def client(cold_cache) -> TestClient:
    c = TestClient(app)
    c.post("/index", json={"count": 60, "seed": 7})
    return c


def test_reindex_reuses_the_embedding_cache(client):
    """다시 색인해도 임베딩은 다시 만들지 않는다.

    인덱스 구조는 매번 새로 만들지만 비싼 것은 임베딩이다. 콘텐츠 해시가 같으면
    캐시에서 나오므로, 두 번째 호출의 embedded 는 청크 수보다 훨씬 작아야 한다.
    """
    first = client.post("/index", json={"count": 60, "seed": 7}).json()
    second = client.post("/index", json={"count": 60, "seed": 7}).json()
    assert first["properties"] == second["properties"] == 60
    assert second["added"] > 0                      # 인덱스는 다시 만들어졌고
    assert second["embedded"] < second["added"]     # 임베딩은 다시 만들지 않았다


def test_search_without_index_is_refused():
    """색인 전에는 검색이 성립하지 않는다 — 빈 결과를 돌려주면 근거처럼 쓰인다."""
    fresh = TestClient(app)
    runtime.indexed = False
    try:
        r = fresh.post("/search", json={"query": "수영장"})
        assert r.status_code == 409
    finally:
        runtime.indexed = True


def test_search_reports_how_much_the_filter_removed(client):
    """이 파이프라인의 주장은 '비싼 단계에 덜 보낸다' 이므로 수치가 응답에 있어야 한다."""
    r = client.post("/search", json={"query": "수영장 있는 숙소", "region": "Jeju", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["candidates_after_filter"] <= body["candidates_before_filter"]
    assert 0.0 <= body["filter_reduction"] <= 1.0
    assert len(body["hits"]) <= 5


def test_search_returns_a_grounding_verdict(client):
    """검색은 언제나 뭔가를 돌려준다. 쓸 만한지는 별도 판정으로 온다."""
    body = client.post("/search", json={"query": "수영장", "top_k": 5}).json()
    assert isinstance(body["grounded"], bool)
    if not body["grounded"]:
        assert body["reason"]


def test_generate_returns_content_that_passed_validation(client):
    pid = next(iter(runtime.indexer.properties))
    r = client.post("/generate", json={"property_id": pid, "segment": "COUPLE", "format": "SNS"})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["violations"] == []
    assert body["content"].strip()
    assert body["backend"]


def test_generate_rejects_an_unknown_property(client):
    r = client.post("/generate", json={"property_id": "NOPE", "segment": "FAMILY", "format": "CRM"})
    assert r.status_code == 404


def test_failed_validation_never_returns_200(client, monkeypatch):
    """검증을 통과 못 한 문장은 200 으로 나가지 않는다 — 이 API 의 핵심 규칙."""
    pid = next(iter(runtime.indexer.properties))

    class Liar:
        name = "liar"

        def generate(self, prop, segment, fmt):
            # 실제로 없는 편의시설을 주장하게 만든다
            return f"{prop.name} 에는 최고급 수영장과 온수 욕조가 있습니다."

    monkeypatch.setattr(runtime, "generator", Liar())
    r = client.post("/generate", json={"property_id": pid, "segment": "COUPLE", "format": "AD_COPY"})

    if r.status_code == 200:
        # 그 숙소가 실제로 두 시설을 다 가진 경우 — 거짓이 아니므로 통과가 맞다
        assert r.json()["valid"] is True
    else:
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["valid"] is False
        assert detail["violations"]


def test_stream_emits_stages_then_done(client):
    pid = next(iter(runtime.indexer.properties))
    with client.stream("POST", "/generate/stream",
                       json={"property_id": pid, "segment": "BUSINESS", "format": "SNS"}) as r:
        assert r.status_code == 200
        events = [
            json.loads(line[len("data: "):])
            for line in r.iter_lines() if line.startswith("data: ")
        ]
    assert [e["type"] for e in events][-1] == "done"
    assert any(e.get("stage") == "validate" for e in events)
    assert events[-1]["content"].strip()


def test_metrics_expose_the_thresholds_used(client):
    body = client.get("/metrics").json()
    assert body["indexed"] is True
    assert body["properties"] > 0
    assert body["chunks"] > 0
    assert "min_score" in body["thresholds"]
    assert "min_margin" in body["thresholds"]
