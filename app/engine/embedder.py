"""임베딩 — 재사용 코어(`retrieval`)로 옮겼다. 기존 임포트 경로 호환용 재수출."""
from retrieval.embedder import (  # noqa: F401
    CACHE_DIR, DIM_LOCAL, Embedder, EmbeddingCache, LocalEmbedder, OpenAIEmbedder,
)

__all__ = ["Embedder", "EmbeddingCache", "LocalEmbedder", "OpenAIEmbedder",
           "CACHE_DIR", "DIM_LOCAL"]
