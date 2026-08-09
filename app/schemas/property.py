"""숙소 도메인 스키마.

문서 RAG와 달리 원본이 **정형 필드**다. 그래서 두 가지가 가능해진다.

1. 필드 경계로 청킹할 수 있다 (의미 경계를 따로 찾을 필요가 없다)
2. 생성물을 원본 필드와 대조하는 **정량 검증**이 가능하다
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PropertyType(str, Enum):
    APARTMENT = "APARTMENT"
    HOTEL = "HOTEL"
    GUESTHOUSE = "GUESTHOUSE"
    PENSION = "PENSION"
    HOUSE = "HOUSE"


class PropertyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class DocumentType(str, Enum):
    """청크 종류. 필드 경계와 1:1로 대응한다."""

    BASIC = "BASIC"
    ROOM = "ROOM"
    AMENITY = "AMENITY"
    POLICY = "POLICY"
    LOCATION = "LOCATION"


# 편의시설 통제 어휘.
# FactValidator 가 생성물에서 이 목록의 단어를 찾아 원본과 대조한다.
AMENITY_VOCAB: dict[str, tuple[str, ...]] = {
    "수영장": ("수영장", "풀장", "인피니티풀"),
    "주차": ("주차", "주차장"),
    "와이파이": ("와이파이", "wifi", "wi-fi", "무선인터넷"),
    "조식": ("조식", "아침식사", "브렉퍼스트"),
    "반려동물": ("반려동물", "애견", "펫"),
    "바비큐": ("바비큐", "바베큐", "bbq"),
    "스파": ("스파", "사우나", "찜질"),
    "주방": ("주방", "취사", "키친"),
    "세탁기": ("세탁기", "세탁"),
    "에어컨": ("에어컨", "냉방"),
    "오션뷰": ("오션뷰", "바다뷰", "바다 전망"),
    "발코니": ("발코니", "테라스"),
    "피트니스": ("피트니스", "헬스장", "짐"),
    "금연": ("금연",),
}


class Room(BaseModel):
    room_id: str
    room_type: str
    capacity: int = Field(ge=1)
    price: int = Field(ge=0, description="1박 요금(원)")
    amenities: list[str] = Field(default_factory=list)


class Policy(BaseModel):
    check_in: str
    check_out: str
    cancellation: str
    house_rules: list[str] = Field(default_factory=list)


class Location(BaseModel):
    address: str
    attractions: list[str] = Field(default_factory=list)
    transportation: str = ""
    neighborhood: str = ""


class Property(BaseModel):
    """숙소 하나. 인덱싱과 검증 모두 이 모델을 원천으로 삼는다."""

    property_id: str
    name: str
    region: str
    property_type: PropertyType
    description: str
    rooms: list[Room]
    amenities: list[str] = Field(default_factory=list)
    policy: Policy
    location: Location
    rating: float = Field(ge=0, le=5)
    review_count: int = Field(ge=0)
    status: PropertyStatus = PropertyStatus.ACTIVE
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # ---------------------------------------------------------- 파생 속성
    @property
    def max_capacity(self) -> int:
        return max((r.capacity for r in self.rooms), default=0)

    @property
    def min_price(self) -> int:
        return min((r.price for r in self.rooms), default=0)

    @property
    def all_amenities(self) -> set[str]:
        """숙소 공용 + 객실 편의시설 전체."""
        s = set(self.amenities)
        for r in self.rooms:
            s.update(r.amenities)
        return s

    def price_range(self) -> str:
        prices = [r.price for r in self.rooms]
        if not prices:
            return "UNKNOWN"
        p = min(prices)
        if p < 80_000:
            return "BUDGET"
        if p < 150_000:
            return "MID"
        return "PREMIUM"


class Chunk(BaseModel):
    """검색 단위. 메타데이터를 함께 들고 다닌다."""

    chunk_id: str
    property_id: str
    document_type: DocumentType
    text: str
    # --- 필터에 쓰는 메타데이터 ---
    region: str
    property_type: PropertyType
    property_status: PropertyStatus
    capacity: int = 0
    price: int = 0
    price_range: str = "UNKNOWN"
    room_id: str | None = None
    amenity_key: str | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    content_hash: str = ""


class SearchFilter(BaseModel):
    """벡터 검색 **이전에** 후보를 좁히는 조건."""

    region: str | None = None
    min_capacity: int | None = None
    max_price: int | None = None
    property_type: PropertyType | None = None
    required_amenities: list[str] = Field(default_factory=list)
    status: PropertyStatus | None = PropertyStatus.ACTIVE
    document_types: list[DocumentType] = Field(default_factory=list)


__all__ = [
    "AMENITY_VOCAB", "Chunk", "DocumentType", "Location", "Policy",
    "Property", "PropertyStatus", "PropertyType", "Room", "SearchFilter",
]
