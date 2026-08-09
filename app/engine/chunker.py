"""필드 경계 청킹.

비정형 문서에서는 "청크 경계에서 의미가 잘리는 문제" 때문에 Overlap·Semantic 청킹이
필요했다. 숙소 데이터는 **필드 경계가 곧 의미 경계**라 그 고민이 사라진다.

대신 새 과제가 생긴다 — 필드를 가로지르는 질의("수영장 있는 4인 숙소")를 어떻게 잇는가.
그건 메타데이터 필터와 Multi-doc 조합이 담당한다.
"""
from __future__ import annotations

import hashlib

from app.schemas.property import Chunk, DocumentType, Property


def _hash(text: str) -> str:
    """청크 내용 해시. 증분 인덱싱에서 변경 감지에 쓴다."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _mk(prop: Property, doc_type: DocumentType, suffix: str, text: str, **extra) -> Chunk:
    return Chunk(
        chunk_id=f"{prop.property_id}:{doc_type.value}:{suffix}",
        property_id=prop.property_id,
        document_type=doc_type,
        text=text.strip(),
        region=prop.region,
        property_type=prop.property_type,
        property_status=prop.status,
        price_range=prop.price_range(),
        updated_at=prop.updated_at,
        content_hash=_hash(text.strip()),
        **extra,
    )


def chunk_property(prop: Property) -> list[Chunk]:
    """숙소 하나를 검색 단위로 쪼갠다.

    BASIC / ROOM(객실마다) / AMENITY / POLICY / LOCATION.
    """
    chunks: list[Chunk] = []

    # --- 기본 정보 -------------------------------------------------------
    basic = (
        f"{prop.name}은(는) {prop.region}에 위치한 {prop.property_type.value} 숙소입니다. "
        f"{prop.description} "
        f"평점 {prop.rating}점, 리뷰 {prop.review_count}건."
    )
    chunks.append(
        _mk(prop, DocumentType.BASIC, "0", basic,
            capacity=prop.max_capacity, price=prop.min_price)
    )

    # --- 객실: 객실마다 하나씩 -------------------------------------------
    for room in prop.rooms:
        text = (
            f"{prop.name}의 {room.room_type} 객실은 최대 {room.capacity}명까지 이용 가능하며 "
            f"1박 요금은 {room.price:,}원입니다."
        )
        if room.amenities:
            text += f" 객실 편의시설: {', '.join(room.amenities)}."
        chunks.append(
            _mk(prop, DocumentType.ROOM, room.room_id, text,
                capacity=room.capacity, price=room.price, room_id=room.room_id)
        )

    # --- 편의시설: 항목마다 하나씩 ---------------------------------------
    # 항목 단위로 쪼개야 "수영장 있는 숙소" 질의에서 정확히 걸린다.
    for amenity in prop.amenities:
        text = f"{prop.name}에서는 {amenity}을(를) 이용하실 수 있습니다."
        chunks.append(
            _mk(prop, DocumentType.AMENITY, amenity, text,
                capacity=prop.max_capacity, price=prop.min_price, amenity_key=amenity)
        )

    # --- 이용 규칙 -------------------------------------------------------
    policy = (
        f"{prop.name}의 체크인은 {prop.policy.check_in}, 체크아웃은 {prop.policy.check_out}입니다. "
        f"취소 규정: {prop.policy.cancellation}."
    )
    if prop.policy.house_rules:
        policy += f" 이용 규칙: {', '.join(prop.policy.house_rules)}."
    chunks.append(
        _mk(prop, DocumentType.POLICY, "0", policy,
            capacity=prop.max_capacity, price=prop.min_price)
    )

    # --- 주변 정보 -------------------------------------------------------
    loc = f"{prop.name}은(는) {prop.location.address}에 있습니다."
    if prop.location.attractions:
        loc += f" 주변 관광지: {', '.join(prop.location.attractions)}."
    if prop.location.transportation:
        loc += f" 교통: {prop.location.transportation}."
    if prop.location.neighborhood:
        loc += f" {prop.location.neighborhood}"
    chunks.append(
        _mk(prop, DocumentType.LOCATION, "0", loc,
            capacity=prop.max_capacity, price=prop.min_price)
    )

    return chunks


def chunk_all(properties: list[Property]) -> list[Chunk]:
    out: list[Chunk] = []
    for p in properties:
        out.extend(chunk_property(p))
    return out


__all__ = ["chunk_property", "chunk_all"]
