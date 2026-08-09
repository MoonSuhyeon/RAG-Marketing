"""임베딩 — 백엔드 교체 가능 + 해시 캐시.

API 키가 있으면 OpenAI, 없으면 결정적 로컬 임베딩을 쓴다.
로컬 백엔드가 있어야 **시크릿 없이 CI에서 검색 테스트를 돌릴 수 있다.**

캐시는 청크 내용의 MD5 로 건다. 숙소 1건의 가격만 바뀌었을 때
전체를 다시 임베딩하는 비용을 없애는 장치다.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

DIM_LOCAL = 384
CACHE_DIR = Path(os.getenv("EMBED_CACHE_DIR", ".embed_cache"))


def _key(text: str, model: str) -> str:
    return hashlib.md5(f"{model}::{text}".encode("utf-8")).hexdigest()


class EmbeddingCache:
    """MD5 기반 디스크 캐시. 히트 수를 세어 효율을 지표로 노출한다."""

    def __init__(self, root: Path = CACHE_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        f = self.root / f"{key}.npy"
        if f.exists():
            self.hits += 1
            return np.load(f)
        self.misses += 1
        return None

    def put(self, key: str, vec: np.ndarray) -> None:
        np.save(self.root / f"{key}.npy", vec)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class LocalEmbedder:
    """결정적 문자 n-gram 해싱 임베딩.

    한글은 형태소 경계가 모호해 문자 2·3-gram 이 잘 동작한다.
    학습이 없고 외부 호출이 없어 테스트에서 재현 가능하다.
    """

    model = "local-hash-ngram"
    dim = DIM_LOCAL

    @staticmethod
    def _grams(text: str) -> list[str]:
        t = "".join(ch for ch in text.lower() if not ch.isspace())
        out = [t[i : i + 2] for i in range(len(t) - 1)]
        out += [t[i : i + 3] for i in range(len(t) - 2)]
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for g in self._grams(text):
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                # 부호를 섞어 충돌이 한쪽으로 쏠리지 않게 한다
                vecs[i, h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)


class OpenAIEmbedder:
    """`text-embedding-3-small`. API 키가 있을 때만 쓴다."""

    model = "text-embedding-3-small"
    dim = 1536

    def __init__(self):
        from openai import OpenAI  # 지연 임포트

        self.client = OpenAI()

    def embed(self, texts: list[str]) -> np.ndarray:
        res = self.client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in res.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)


class Embedder:
    """캐시를 앞에 둔 임베딩 진입점."""

    def __init__(self, backend=None, cache: EmbeddingCache | None = None):
        self.backend = backend or self._auto_backend()
        self.cache = cache or EmbeddingCache()

    @staticmethod
    def _auto_backend():
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIEmbedder()
            except Exception:  # 키가 있어도 초기화 실패하면 로컬로
                pass
        return LocalEmbedder()

    @property
    def dim(self) -> int:
        return self.backend.dim

    @property
    def model(self) -> str:
        return self.backend.model

    def embed(self, texts: list[str]) -> np.ndarray:
        """캐시에 없는 것만 백엔드로 보낸다."""
        out: list[np.ndarray | None] = [None] * len(texts)
        todo_idx, todo_txt = [], []

        for i, t in enumerate(texts):
            hit = self.cache.get(_key(t, self.model))
            if hit is not None:
                out[i] = hit
            else:
                todo_idx.append(i)
                todo_txt.append(t)

        if todo_txt:
            fresh = self.backend.embed(todo_txt)
            for j, i in enumerate(todo_idx):
                out[i] = fresh[j]
                self.cache.put(_key(texts[i], self.model), fresh[j])

        return np.vstack(out).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


__all__ = ["Embedder", "EmbeddingCache", "LocalEmbedder", "OpenAIEmbedder"]
