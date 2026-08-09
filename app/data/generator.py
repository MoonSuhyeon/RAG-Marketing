"""더미 숙소 데이터 생성기.

편의시설 조합을 일부러 불균등하게 만든다. 모든 숙소가 수영장을 갖고 있으면
"없는 편의시설을 언급했는지" 검증이 아무것도 걸러내지 못하기 때문이다.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from app.schemas.property import (
    AMENITY_VOCAB, Location, Policy, Property, PropertyStatus, PropertyType, Room,
)

REGIONS = {
    "Seoul": (["경복궁", "남산타워", "홍대", "명동"], "지하철 2호선 도보 5분"),
    "Busan": (["해운대", "광안리", "감천문화마을"], "부산역에서 택시 15분"),
    "Jeju": (["성산일출봉", "협재해변", "우도"], "제주공항에서 차량 40분"),
    "Gangneung": (["경포대", "안목해변", "오죽헌"], "강릉역에서 차량 10분"),
    "Gyeongju": (["불국사", "첨성대", "동궁과 월지"], "경주역에서 차량 20분"),
}

ROOM_TYPES = ["스탠다드", "디럭스", "스위트", "온돌", "패밀리"]
ALL_AMENITIES = list(AMENITY_VOCAB.keys())

CANCELLATION = [
    "체크인 7일 전까지 전액 환불",
    "체크인 3일 전까지 50% 환불",
    "체크인 14일 전까지 전액 환불, 이후 환불 불가",
]
HOUSE_RULES = ["실내 금연", "파티 금지", "22시 이후 정숙", "반려동물 동반 불가"]


def make_property(i: int, rng: random.Random) -> Property:
    region = rng.choice(list(REGIONS))
    attractions, transport = REGIONS[region]
    ptype = rng.choice(list(PropertyType))

    # 편의시설을 3~7개만 준다. 없는 것이 있어야 검증이 의미를 갖는다.
    amenities = rng.sample(ALL_AMENITIES, k=rng.randint(3, 7))

    n_rooms = rng.randint(1, 3)
    rooms = [
        Room(
            room_id=f"R{j + 1}",
            room_type=rng.choice(ROOM_TYPES),
            capacity=rng.randint(2, 8),
            price=rng.randint(6, 30) * 10_000,
            amenities=rng.sample(amenities, k=min(2, len(amenities))),
        )
        for j in range(n_rooms)
    ]

    return Property(
        property_id=f"P{i:04d}",
        name=f"{region} {rng.choice(['스테이', '하우스', '레지던스', '빌라', '호텔'])} {i:03d}",
        region=region,
        property_type=ptype,
        description=rng.choice(
            [
                "조용한 골목 안쪽에 자리한 아늑한 공간입니다.",
                "넓은 창으로 햇살이 가득 들어옵니다.",
                "여행의 피로를 풀기 좋은 편안한 숙소입니다.",
                "가족 단위 여행객에게 인기가 많은 곳입니다.",
            ]
        ),
        rooms=rooms,
        amenities=amenities,
        policy=Policy(
            check_in=f"{rng.randint(14, 16)}:00",
            check_out=f"{rng.randint(10, 12)}:00",
            cancellation=rng.choice(CANCELLATION),
            house_rules=rng.sample(HOUSE_RULES, k=rng.randint(1, 3)),
        ),
        location=Location(
            address=f"{region} {rng.randint(1, 200)}로 {rng.randint(1, 99)}",
            attractions=rng.sample(attractions, k=min(2, len(attractions))),
            transportation=transport,
            neighborhood=rng.choice(["주변에 카페와 식당이 많습니다.", "조용한 주거 지역입니다."]),
        ),
        rating=round(rng.uniform(3.5, 5.0), 1),
        review_count=rng.randint(0, 900),
        status=PropertyStatus.ACTIVE if rng.random() > 0.05 else PropertyStatus.INACTIVE,
        updated_at=datetime(2025, 1, 1) + timedelta(days=rng.randint(0, 300)),
    )


def generate(n: int = 100, seed: int = 42) -> list[Property]:
    rng = random.Random(seed)
    return [make_property(i + 1, rng) for i in range(n)]


__all__ = ["generate", "make_property", "REGIONS"]
