"""Phase 0~6 파이프라인 — 색인 → 검색 → 세그먼트 생성 → 사실 검증.

    python scripts/run_demo.py

OPENAI_API_KEY 가 없으면 로컬 임베딩과 템플릿 생성기로 동작한다.
검색·검증 계층은 백엔드와 무관하게 동일하게 측정된다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.data.generator import generate                                  # noqa: E402
from app.engine.fact_validator import (                                  # noqa: E402
    FactValidator, fact_consistency_rate, hallucinated_amenity_rate,
)
from app.engine.indexer import PropertyIndexer                           # noqa: E402
from app.engine.segment import (                                         # noqa: E402
    ContentFormat, Segment, build_filter, get_generator,
)
from app.schemas.property import SearchFilter                            # noqa: E402

BAR = "=" * 72


def main() -> int:
    print(BAR)
    print("Phase 0~1  숙소 생성 및 색인")
    properties = generate(n=100, seed=42)
    indexer = PropertyIndexer()
    report = indexer.index_all(properties)
    print(f"  숙소 {len(properties)}개 → 청크 {len(indexer.index):,}개")
    print(f"  임베딩 백엔드: {indexer.embedder.model} (dim={indexer.embedder.dim})")
    print(f"  색인 결과: {report}")

    print()
    print(BAR)
    print("Phase 3~4  복합 조건 검색 — 메타데이터 필터가 먼저 후보를 좁힌다")
    flt = SearchFilter(region="Jeju", min_capacity=4, required_amenities=["수영장"])
    hits, stats = indexer.search("제주에서 가족이 머물기 좋은 수영장 있는 숙소", flt=flt, top_k=5)
    print(f"  질의: 제주 / 4인 이상 / 수영장")
    print(f"  전체 청크 {stats.total_chunks:,} → 필터 통과 {stats.after_filter:,} "
          f"(후보 {stats.filter_reduction:.1%} 감소)")
    print(f"  Dense {stats.dense_candidates}건 · BM25 {stats.bm25_candidates}건 → RRF 병합")
    for h in hits[:3]:
        p = indexer.properties[h.chunk.property_id]
        print(f"    {h.score:.5f}  {p.name:<22} [{h.chunk.document_type.value}] "
              f"최대{p.max_capacity}인")

    matched = {h.chunk.property_id for h in hits}
    assert all("수영장" in indexer.properties[pid].all_amenities for pid in matched)
    print("  → 결과 전부가 조건을 만족한다 (필터 정확도 100%)")

    print()
    print(BAR)
    print("Phase 5  세그먼트별 콘텐츠 생성")
    gen = get_generator()
    validator = FactValidator()
    print(f"  생성 백엔드: {gen.name}")

    sample = indexer.properties[sorted(matched)[0]] if matched else properties[0]
    for seg in Segment:
        text = gen.generate(sample, seg, ContentFormat.AD_COPY)
        print(f"    [{seg.value:<8}] {text}")

    print()
    print(BAR)
    print("Phase 6  사실 정합성 검증 — 생성된 전량을 원본 필드와 대조")
    results = []
    for prop in properties[:60]:
        for seg in Segment:
            for fmt in (ContentFormat.SNS, ContentFormat.AD_COPY, ContentFormat.CRM):
                text = gen.generate(prop, seg, fmt)
                results.append(validator.validate(text, prop))

    fcr = fact_consistency_rate(results)
    har = hallucinated_amenity_rate(results)
    print(f"  검증 건수                {len(results):,}건")
    print(f"  Fact Consistency Rate    {fcr:.2%}")
    print(f"  Hallucinated Amenity     {har:.2%}")

    print()
    print("  적대적 케이스 — 이 숙소에 실제로 없는 편의시설을 넣은 문구")
    from app.schemas.property import AMENITY_VOCAB

    absent = [a for a in AMENITY_VOCAB if a not in sample.all_amenities][:2]
    bad = (
        f"{sample.name}의 프라이빗 {absent[0]}과(와) {absent[1]}에서 "
        f"여유로운 시간을 보내세요. 1박 33,000원."
    )
    print(f"    (이 숙소에 없는 편의시설: {', '.join(absent)})")
    r = validator.validate(bad, sample)
    print(f"    입력: {bad}")
    print(f"    판정: {'통과' if r.consistent else '차단'} — {r.summary()}")
    assert not r.consistent, "적대적 케이스가 통과했다"

    print()
    print(BAR)
    print("Phase 7  증분 인덱싱 — 가격 1건 변경")
    before = indexer.cache_stats
    target = indexer.properties[properties[5].property_id].model_copy(deep=True)
    old_price = target.rooms[0].price
    target.rooms[0].price = old_price + 10_000
    rep = indexer.upsert(target)
    after = indexer.cache_stats
    print(f"  {target.name} 객실 요금 {old_price:,} → {target.rooms[0].price:,}원")
    print(f"  재색인 결과: {rep}")
    print(f"  임베딩 캐시 히트율 {after['hit_rate']:.2%} "
          f"(히트 {after['hits']:,} / 미스 {after['misses']:,})")
    print(f"  → 전체 {len(indexer.index):,}개 청크 중 {rep.updated}개만 다시 임베딩됐다")

    print()
    print(BAR)
    print("요약")
    print(f"  숙소 {len(properties)}개 · 청크 {len(indexer.index):,}개")
    print(f"  Fact Consistency Rate    {fcr:.2%}")
    print(f"  Hallucinated Amenity     {har:.2%}")
    print(f"  필터 후보 감소율          {stats.filter_reduction:.1%}")
    print(f"  가격 변경 시 재임베딩      {rep.updated}청크")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
