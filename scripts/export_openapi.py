"""OpenAPI 스키마를 파일로 떨어뜨린다.

이 파일을 커밋해 두고 CI 가 변경을 감지한다. **스키마를 바꾼 사람이 자기 PR 에서
알게 하려는 것** — 소비자(운영 콘솔)가 나중에 조용히 깨지는 대신.

키를 정렬해 쓴다. 정렬하지 않으면 내용이 같아도 순서만 달라져 diff 가 뜬다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.server import app  # noqa: E402

out = ROOT / "openapi.json"
out.write_text(
    json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"작성 {out.name} · 경로 {len(app.openapi()['paths'])}개")
