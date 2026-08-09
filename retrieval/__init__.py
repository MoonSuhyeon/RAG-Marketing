"""재사용 가능한 검색 코어.

숙소 도메인과 분리되어 있어 다른 저장소에서 그대로 가져다 쓸 수 있다.
Agent-Customer-Support 가 취소·환불 정책 문서 검색에 이 패키지를 사용한다.
"""
from retrieval.core import (
    Doc, Grounding, Hit, HybridIndex, SearchStats, assess, tokenize,
)
from retrieval.embedder import Embedder, EmbeddingCache, LocalEmbedder, OpenAIEmbedder

__all__ = [
    "Doc", "Embedder", "EmbeddingCache", "Grounding", "Hit", "HybridIndex",
    "LocalEmbedder", "OpenAIEmbedder", "SearchStats", "assess", "tokenize",
]
