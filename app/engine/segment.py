"""세그먼트별 콘텐츠 생성.

같은 숙소라도 세그먼트에 따라 **검색 필터와 강조 필드**가 달라진다.
커플에게 수용 인원을 말하고 가족에게 감성을 말하면 둘 다 실패한다.

생성 백엔드는 교체 가능하다. API 키가 없으면 템플릿 기반 결정적 생성기를 쓴다.
검증 계층(FactValidator)을 시크릿 없이 테스트하기 위해서다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from app.schemas.property import DocumentType, Property, SearchFilter


class Segment(str, Enum):
    COUPLE = "COUPLE"
    FAMILY = "FAMILY"
    BUSINESS = "BUSINESS"


class ContentFormat(str, Enum):
    SNS = "SNS"
    AD_COPY = "AD_COPY"
    CRM = "CRM"


@dataclass(frozen=True)
class SegmentProfile:
    """세그먼트마다 무엇을 먼저 찾고 무엇을 강조할지."""

    segment: Segment
    focus_amenities: tuple[str, ...]
    focus_documents: tuple[DocumentType, ...]
    tone: str
    min_capacity: int | None = None


PROFILES: dict[Segment, SegmentProfile] = {
    Segment.COUPLE: SegmentProfile(
        segment=Segment.COUPLE,
        focus_amenities=("오션뷰", "스파", "발코니"),
        focus_documents=(DocumentType.BASIC, DocumentType.LOCATION, DocumentType.AMENITY),
        tone="감성적이고 분위기를 강조하는",
    ),
    Segment.FAMILY: SegmentProfile(
        segment=Segment.FAMILY,
        focus_amenities=("주방", "주차", "수영장", "세탁기"),
        focus_documents=(DocumentType.ROOM, DocumentType.AMENITY, DocumentType.POLICY),
        tone="실용적이고 편의성을 강조하는",
        min_capacity=4,
    ),
    Segment.BUSINESS: SegmentProfile(
        segment=Segment.BUSINESS,
        focus_amenities=("와이파이", "피트니스", "조식"),
        focus_documents=(DocumentType.BASIC, DocumentType.LOCATION, DocumentType.POLICY),
        tone="간결하고 효율을 강조하는",
    ),
}


def build_filter(segment: Segment, region: str | None = None,
                 max_price: int | None = None) -> SearchFilter:
    """세그먼트를 검색 조건으로 옮긴다."""
    p = PROFILES[segment]
    return SearchFilter(
        region=region,
        max_price=max_price,
        min_capacity=p.min_capacity,
        document_types=list(p.focus_documents),
    )


@dataclass
class GeneratedContent:
    property_id: str
    segment: Segment
    fmt: ContentFormat
    text: str
    backend: str
    sources: list[str] = field(default_factory=list)


class TemplateGenerator:
    """검색된 필드만 조합하는 결정적 생성기.

    **없는 정보를 만들어내지 않는다는 점이 이 백엔드의 특징이다.**
    원본 필드만 문장으로 엮으므로 구조적으로 환각이 생길 수 없다.
    """

    name = "template"

    def generate(self, prop: Property, segment: Segment, fmt: ContentFormat) -> str:
        profile = PROFILES[segment]
        have = prop.all_amenities
        # 세그먼트가 중시하는 것 중 **실제로 있는 것만** 고른다
        highlights = [a for a in profile.focus_amenities if a in have]
        if not highlights:
            highlights = sorted(have)[:2]

        cheapest = min(prop.rooms, key=lambda r: r.price) if prop.rooms else None
        parts = [f"{prop.name}"]

        if fmt is ContentFormat.SNS:
            parts.append(f"{prop.region}에서 보내는 하루.")
            if highlights:
                parts.append(f"{' · '.join(highlights)}까지 준비되어 있습니다.")
            if prop.location.attractions:
                parts.append(f"{prop.location.attractions[0]}도 가까워요.")
        elif fmt is ContentFormat.AD_COPY:
            if cheapest:
                parts.append(
                    f"{cheapest.room_type} 1박 {cheapest.price:,}원, "
                    f"최대 {cheapest.capacity}인 이용 가능."
                )
            if highlights:
                parts.append(f"{', '.join(highlights)} 완비.")
            parts.append(f"평점 {prop.rating}점.")
        else:  # CRM
            parts.append(f"체크인 {prop.policy.check_in}, 체크아웃 {prop.policy.check_out}.")
            if highlights:
                parts.append(f"{', '.join(highlights)}을(를) 이용하실 수 있습니다.")
            parts.append(f"취소 규정: {prop.policy.cancellation}")

        return " ".join(parts)


class LLMGenerator:
    """LLM 생성. 검색된 컨텍스트만 근거로 쓰도록 프롬프트를 고정한다."""

    name = "llm"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def generate(self, prop: Property, segment: Segment, fmt: ContentFormat,
                 context: str = "") -> str:
        profile = PROFILES[segment]
        system = (
            "너는 숙박 플랫폼의 마케팅 카피라이터다. "
            "아래 '숙소 정보'에 있는 사실만 사용해라. "
            "정보에 없는 편의시설·가격·인원을 절대 지어내지 마라."
        )
        user = (
            f"[숙소 정보]\n{context or self._context(prop)}\n\n"
            f"[요청] {profile.tone} 톤으로 {fmt.value} 문구를 2~3문장으로 작성해라. "
            f"숙소명 '{prop.name}'을 반드시 포함해라."
        )
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.7,
        )
        return res.choices[0].message.content.strip()

    @staticmethod
    def _context(prop: Property) -> str:
        rooms = "; ".join(
            f"{r.room_type} 최대{r.capacity}인 {r.price:,}원" for r in prop.rooms
        )
        return (
            f"숙소명: {prop.name}\n지역: {prop.region}\n"
            f"편의시설: {', '.join(sorted(prop.all_amenities))}\n"
            f"객실: {rooms}\n평점: {prop.rating}"
        )


def get_generator():
    """API 키가 있으면 LLM, 없으면 템플릿."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            return LLMGenerator()
        except Exception:
            pass
    return TemplateGenerator()


__all__ = [
    "ContentFormat", "GeneratedContent", "LLMGenerator", "PROFILES",
    "Segment", "SegmentProfile", "TemplateGenerator", "build_filter", "get_generator",
]
