"""생성 — 그리고 내보내기 전에 대조.

이 라우터의 핵심은 생성이 아니라 **거부**다. 생성물이 원본 필드와 어긋나면
한 번 다시 만들고, 그래도 어긋나면 422 로 거절한다. 통과하지 못한 문장은
나가지 않는다.
"""
from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.engine.fact_validator import ValidationResult

from api import config
from api.schemas import GenerateRequest, GenerateResponse, ViolationOut
from api.state import runtime

router = APIRouter(tags=["generation"])


def _violations(result: ValidationResult) -> list[ViolationOut]:
    return [ViolationOut(type=v.type.value, detail=v.detail) for v in result.violations]


def _attempt(body: GenerateRequest):
    """생성 → 검증을 최대 ``MAX_REGENERATE + 1`` 회. 마지막 결과를 돌려준다."""
    prop = runtime.get_property(body.property_id)
    if prop is None:
        raise HTTPException(404, f"숙소를 찾을 수 없습니다: {body.property_id}")

    text = ""
    result: ValidationResult | None = None
    for attempt in range(1, config.MAX_REGENERATE + 2):
        text = runtime.generator.generate(prop, body.segment, body.format)
        result = runtime.validator.validate(text, prop)
        if result.consistent:
            return prop, text, result, attempt
    return prop, text, result, config.MAX_REGENERATE + 1


@router.post("/generate", response_model=GenerateResponse)
def generate(body: GenerateRequest) -> GenerateResponse:
    if not runtime.indexed:
        raise HTTPException(409, "색인이 비어 있습니다. 먼저 POST /index 를 호출하세요")

    prop, text, result, attempts = _attempt(body)
    payload = GenerateResponse(
        property_id=prop.property_id,
        segment=body.segment,
        format=body.format,
        backend=runtime.backend,
        content=text,
        valid=result.consistent,
        attempts=attempts,
        violations=_violations(result),
    )
    if not result.consistent:
        # 검증에 실패한 문장은 200 으로 내보내지 않는다. 호출자가 실수로 쓰게 된다.
        raise HTTPException(422, detail=payload.model_dump(mode="json"))
    return payload


@router.post("/generate/stream")
def generate_stream(body: GenerateRequest) -> StreamingResponse:
    """단계별 진행을 SSE 로 흘린다.

    토큰 단위 스트리밍이 아니라 **파이프라인 단계** 스트리밍이다. 이 파이프라인에서
    기다리게 만드는 것은 토큰 생성이 아니라 검색과 검증이고, 진행 중에 알고 싶은
    것도 "지금 어느 단계인가" 이기 때문이다.
    """
    if not runtime.indexed:
        raise HTTPException(409, "색인이 비어 있습니다. 먼저 POST /index 를 호출하세요")

    def sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def stream() -> Iterator[str]:
        yield sse({"type": "stage", "stage": "generate", "backend": runtime.backend})
        try:
            prop, text, result, attempts = _attempt(body)
        except HTTPException as e:
            yield sse({"type": "error", "detail": e.detail})
            return

        yield sse({"type": "stage", "stage": "validate", "attempts": attempts})
        yield sse({
            "type": "done",
            "property_id": prop.property_id,
            "content": text,
            "valid": result.consistent,
            "attempts": attempts,
            "violations": [v.model_dump() for v in _violations(result)],
        })

    return StreamingResponse(stream(), media_type="text/event-stream")
