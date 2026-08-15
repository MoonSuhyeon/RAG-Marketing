"""공유 상태 — 인덱서 하나, 검증기 하나, 생성기 하나.

라우터가 각자 인덱서를 만들면 색인한 라우터와 검색하는 라우터가 서로 다른
인덱스를 보게 된다. 그래서 여기서 한 벌만 만들고 모두 이것을 쓴다.
"""
from __future__ import annotations

from app.data.generator import generate as generate_properties
from app.engine.fact_validator import FactValidator
from app.engine.indexer import PropertyIndexer
from app.engine.segment import get_generator
from app.schemas.property import Property

from api import config


class Runtime:
    """프로세스 하나가 들고 있는 파이프라인."""

    def __init__(self) -> None:
        self.indexer = PropertyIndexer()
        self.validator = FactValidator()
        self.generator = get_generator()
        self.indexed = False

    # ------------------------------------------------------------ 색인
    def load_demo(self, n: int | None = None, seed: int | None = None):
        """데모 숙소를 만들어 색인한다. API 키 없이도 전부 동작한다."""
        props = generate_properties(
            n=n if n is not None else config.DEMO_PROPERTY_COUNT,
            seed=seed if seed is not None else config.DEMO_SEED,
        )
        report = self.indexer.index_all(props)
        self.indexed = True
        return props, report

    def get_property(self, property_id: str) -> Property | None:
        return self.indexer.properties.get(property_id)

    @property
    def backend(self) -> str:
        """어느 생성 백엔드가 붙어 있는지 — 응답에 그대로 실어 보낸다."""
        return getattr(self.generator, "name", type(self.generator).__name__)


runtime = Runtime()

__all__ = ["Runtime", "runtime"]
