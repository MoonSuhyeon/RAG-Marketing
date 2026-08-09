"""FactValidator 검증 — 없는 편의시설 광고를 실제로 막는지.

README 의 DoD 는 "FactValidator 가 없는 편의시설 언급을 차단한다 — 테스트 케이스로 증명"이다.
이 파일이 그 증명이다.
"""
from __future__ import annotations

import pytest

from app.engine.fact_validator import (
    FactValidator, ViolationType, fact_consistency_rate, hallucinated_amenity_rate,
)
from app.schemas.property import (
    Location, Policy, Property, PropertyType, Room,
)


@pytest.fixture
def prop() -> Property:
    """수영장이 **없고** 주차·와이파이만 있는 숙소."""
    return Property(
        property_id="P0001",
        name="제주 스테이 001",
        region="Jeju",
        property_type=PropertyType.PENSION,
        description="조용한 숙소입니다.",
        rooms=[
            Room(room_id="R1", room_type="디럭스", capacity=4, price=120_000,
                 amenities=["와이파이"]),
            Room(room_id="R2", room_type="스탠다드", capacity=2, price=80_000,
                 amenities=["와이파이"]),
        ],
        amenities=["주차", "와이파이"],
        policy=Policy(check_in="15:00", check_out="11:00",
                      cancellation="체크인 7일 전까지 전액 환불"),
        location=Location(address="Jeju 12로 3"),
        rating=4.5,
        review_count=100,
    )


@pytest.fixture
def validator() -> FactValidator:
    return FactValidator()


def test_truthful_content_passes(validator, prop):
    text = "제주 스테이 001, 주차와 와이파이를 갖춘 숙소입니다. 디럭스 1박 120,000원."
    r = validator.validate(text, prop)
    assert r.consistent, r.summary()


def test_hallucinated_amenity_is_blocked(validator, prop):
    """수영장이 없는데 수영장을 광고하면 반드시 걸려야 한다."""
    text = "제주 스테이 001의 프라이빗 수영장에서 여유로운 시간을 보내세요."
    r = validator.validate(text, prop)
    assert not r.consistent
    assert "수영장" in r.hallucinated_amenities
    assert any(v.type is ViolationType.HALLUCINATED_AMENITY for v in r.violations)


def test_amenity_alias_is_caught(validator, prop):
    """표현이 달라도 잡아야 한다 — '풀장'은 '수영장'의 다른 말이다."""
    text = "제주 스테이 001 풀장이 매력적입니다."
    r = validator.validate(text, prop)
    assert "수영장" in r.hallucinated_amenities


def test_price_mismatch_is_caught(validator, prop):
    text = "제주 스테이 001 특가 1박 55,000원."
    r = validator.validate(text, prop)
    assert any(v.type is ViolationType.PRICE_MISMATCH for v in r.violations)


def test_actual_price_passes(validator, prop):
    text = "제주 스테이 001 스탠다드 1박 80,000원."
    r = validator.validate(text, prop)
    assert not any(v.type is ViolationType.PRICE_MISMATCH for v in r.violations)


def test_capacity_exceeded_is_caught(validator, prop):
    text = "제주 스테이 001 최대 10인까지 이용 가능합니다."
    r = validator.validate(text, prop)
    assert any(v.type is ViolationType.CAPACITY_EXCEEDED for v in r.violations)


def test_wrong_region_is_caught(validator, prop):
    text = "제주 스테이 001은 부산 해운대 근처입니다."
    r = validator.validate(text, prop)
    assert any(v.type is ViolationType.REGION_MISMATCH for v in r.violations)


def test_missing_name_is_caught(validator, prop):
    text = "조용하고 아늑한 숙소입니다."
    r = validator.validate(text, prop)
    assert any(v.type is ViolationType.NAME_MISMATCH for v in r.violations)


def test_rates_are_computed(validator, prop):
    good = validator.validate("제주 스테이 001, 주차 가능합니다.", prop)
    bad = validator.validate("제주 스테이 001 수영장 완비.", prop)
    assert fact_consistency_rate([good, bad]) == 0.5
    assert hallucinated_amenity_rate([good, bad]) == 0.5
    assert hallucinated_amenity_rate([good, good]) == 0.0
