"""설정 — 상수·임계값·모델명을 한 곳에서.

v24에서 배운 것을 그대로 가져왔다. 임계값이 코드 여기저기에 흩어져 있으면
"기권 기준을 얼마로 뒀더라"를 확인하려고 파일을 뒤지게 된다.
"""
from __future__ import annotations

import os

# 검색
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
MAX_TOP_K = int(os.getenv("RAG_MAX_TOP_K", "50"))

# 기권 임계값 — 검색은 언제나 뭔가를 돌려주므로, 돌려준 것이 쓸 만한지 따로 판단한다.
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.016"))
MIN_MARGIN = float(os.getenv("RAG_MIN_MARGIN", "0.0005"))

# 생성물이 검증에 걸렸을 때 다시 시도하는 횟수.
# 1회로 둔 이유는 v17 의 교훈이다 — 재시도를 늘려도 통과율은 거의 안 오르고
# 지연과 비용만 늘었다. 두 번째도 실패하면 고칠 대상은 생성기가 아니라 원본이다.
MAX_REGENERATE = int(os.getenv("RAG_MAX_REGENERATE", "1"))

# 데모 데이터
DEMO_PROPERTY_COUNT = int(os.getenv("RAG_DEMO_PROPERTIES", "100"))
DEMO_SEED = int(os.getenv("RAG_DEMO_SEED", "42"))

__all__ = [
    "DEFAULT_TOP_K", "MAX_TOP_K", "MIN_SCORE", "MIN_MARGIN",
    "MAX_REGENERATE", "DEMO_PROPERTY_COUNT", "DEMO_SEED",
]
