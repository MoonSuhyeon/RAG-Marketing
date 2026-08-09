"""사실 정합성 검증 — 이 프로젝트에서 가장 중요한 안전장치.

문서 RAG는 답이 틀리는 데 그치지만, 없는 편의시설을 광고하면 **실제 클레임**이 된다.

원본이 정형 필드라서 생성물에서 속성을 추출해 필드와 대조하는 **정량 검증**이 가능하다.
LLM 자기평가에 기대지 않고 규칙으로 판정하므로 결정적이고, 테스트로 고정할 수 있다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from app.schemas.property import AMENITY_VOCAB, Property


class ViolationType(str, Enum):
    HALLUCINATED_AMENITY = "hallucinated_amenity"
    PRICE_MISMATCH = "price_mismatch"
    CAPACITY_EXCEEDED = "capacity_exceeded"
    REGION_MISMATCH = "region_mismatch"
    NAME_MISMATCH = "name_mismatch"


@dataclass
class Violation:
    type: ViolationType
    detail: str
    evidence: str = ""

    def __str__(self) -> str:
        return f"[{self.type.value}] {self.detail}"


@dataclass
class ValidationResult:
    consistent: bool
    violations: list[Violation] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def hallucinated_amenities(self) -> list[str]:
        return [
            v.evidence for v in self.violations
            if v.type is ViolationType.HALLUCINATED_AMENITY
        ]

    def summary(self) -> str:
        if self.consistent:
            return f"통과 ({len(self.checked)}개 항목 검사)"
        return f"위반 {len(self.violations)}건: " + "; ".join(str(v) for v in self.violations)


# 지역 표기 흔들림 흡수 (제주 / 제주도 등)
_REGION_ALIASES = {
    "Seoul": ("서울",),
    "Busan": ("부산",),
    "Jeju": ("제주", "제주도"),
    "Gangneung": ("강릉",),
    "Gyeongju": ("경주",),
}

_PRICE_RE = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})\s*원")
_CAPACITY_RE = re.compile(r"(?:최대\s*)?([0-9]{1,2})\s*인")


def _mentioned_amenities(text: str) -> dict[str, str]:
    """생성물에서 언급된 편의시설 → (표준명: 발견된 표현)."""
    low = text.lower()
    found: dict[str, str] = {}
    for canonical, aliases in AMENITY_VOCAB.items():
        for alias in aliases:
            if alias.lower() in low:
                found[canonical] = alias
                break
    return found


class FactValidator:
    """생성된 마케팅 콘텐츠를 원본 숙소 필드와 대조한다."""

    def __init__(self, price_tolerance: float = 0.0):
        # 가격은 정확해야 한다. 기본 허용 오차 0.
        self.price_tolerance = price_tolerance

    def validate(self, content: str, prop: Property) -> ValidationResult:
        violations: list[Violation] = []
        checked: list[str] = []

        # --- 편의시설: 없는 것을 말하면 안 된다 -------------------------
        checked.append("amenity")
        have = {a for a in prop.all_amenities}
        for canonical, evidence in _mentioned_amenities(content).items():
            if canonical not in have:
                violations.append(
                    Violation(
                        ViolationType.HALLUCINATED_AMENITY,
                        f"'{canonical}'을(를) 언급했으나 이 숙소에 없다",
                        evidence=canonical,
                    )
                )

        # --- 가격: 실제 객실 요금 중 하나여야 한다 ----------------------
        checked.append("price")
        room_prices = {r.price for r in prop.rooms}
        for m in _PRICE_RE.finditer(content):
            value = int(m.group(1).replace(",", ""))
            if value < 10_000:  # 금액이 아닌 숫자는 건너뛴다
                continue
            ok = any(
                abs(value - p) <= max(1, p * self.price_tolerance) for p in room_prices
            )
            if not ok:
                violations.append(
                    Violation(
                        ViolationType.PRICE_MISMATCH,
                        f"{value:,}원은 실제 요금 {sorted(room_prices)} 중 어느 것과도 다르다",
                        evidence=m.group(0),
                    )
                )

        # --- 수용 인원: 최대 정원을 넘을 수 없다 ------------------------
        checked.append("capacity")
        cap = prop.max_capacity
        for m in _CAPACITY_RE.finditer(content):
            n = int(m.group(1))
            if n > cap:
                violations.append(
                    Violation(
                        ViolationType.CAPACITY_EXCEEDED,
                        f"{n}인을 언급했으나 최대 수용 인원은 {cap}명이다",
                        evidence=m.group(0),
                    )
                )

        # --- 지역: 다른 지역을 말하면 안 된다 ---------------------------
        checked.append("region")
        own = _REGION_ALIASES.get(prop.region, ())
        for region, aliases in _REGION_ALIASES.items():
            if region == prop.region:
                continue
            for alias in aliases:
                if alias in content and not any(o in content for o in own if alias in o):
                    violations.append(
                        Violation(
                            ViolationType.REGION_MISMATCH,
                            f"'{alias}'을(를) 언급했으나 이 숙소는 {prop.region}에 있다",
                            evidence=alias,
                        )
                    )
                    break

        # --- 숙소명 ------------------------------------------------------
        checked.append("name")
        if prop.name not in content:
            violations.append(
                Violation(
                    ViolationType.NAME_MISMATCH,
                    f"숙소명 '{prop.name}'이 콘텐츠에 없다",
                )
            )

        return ValidationResult(
            consistent=not violations, violations=violations, checked=checked
        )


def hallucinated_amenity_rate(results: list[ValidationResult]) -> float:
    """생성 건수 대비 없는 편의시설을 언급한 비율. 목표는 0."""
    if not results:
        return 0.0
    bad = sum(1 for r in results if r.hallucinated_amenities)
    return round(bad / len(results), 4)


def fact_consistency_rate(results: list[ValidationResult]) -> float:
    """전체 검증 통과율."""
    if not results:
        return 0.0
    return round(sum(1 for r in results if r.consistent) / len(results), 4)


__all__ = [
    "FactValidator", "ValidationResult", "Violation", "ViolationType",
    "fact_consistency_rate", "hallucinated_amenity_rate",
]
