"""증분 인덱싱.

가격·재고·편의시설은 수시로 바뀐다. 하나 바뀔 때마다 전체를 재색인하면
임베딩 비용이 숙소 수에 비례해 늘어난다.

청크 내용 해시를 비교해 **바뀐 청크만** 다시 만든다. 임베딩 캐시가 앞단에 있어
내용이 같으면 API 호출 자체가 발생하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.engine.chunker import chunk_property
from app.engine.embedder import Embedder
from app.engine.retriever import ChunkIndex
from app.schemas.property import Chunk, Property


@dataclass
class IndexReport:
    """무엇이 실제로 다시 만들어졌는지 남긴다."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    embedded: int = 0
    properties_touched: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"added={self.added} updated={self.updated} removed={self.removed} "
            f"unchanged={self.unchanged} embedded={self.embedded}"
        )


class PropertyIndexer:
    """숙소를 청킹·임베딩해 인덱스에 반영한다."""

    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self.index = ChunkIndex(dim=self.embedder.dim)
        self.properties: dict[str, Property] = {}
        self._hashes: dict[str, str] = {}   # chunk_id -> content_hash
        self._chunks: dict[str, Chunk] = {}  # chunk_id -> chunk

    # ------------------------------------------------------------ 최초 색인
    def index_all(self, properties: list[Property]) -> IndexReport:
        report = IndexReport()
        chunks: list[Chunk] = []
        for p in properties:
            self.properties[p.property_id] = p
            chunks.extend(chunk_property(p))

        vectors = self.embedder.embed([c.text for c in chunks])
        self.index.rebuild(chunks, vectors)

        for c in chunks:
            self._hashes[c.chunk_id] = c.content_hash
            self._chunks[c.chunk_id] = c

        report.added = len(chunks)
        report.embedded = self.embedder.cache.misses
        report.properties_touched = [p.property_id for p in properties]
        return report

    # ------------------------------------------------------------ 증분 반영
    def upsert(self, prop: Property) -> IndexReport:
        """숙소 하나를 갱신한다. 바뀐 청크만 다시 임베딩한다."""
        report = IndexReport(properties_touched=[prop.property_id])
        before = self.embedder.cache.misses

        new_chunks = chunk_property(prop)
        new_ids = {c.chunk_id for c in new_chunks}
        old_ids = {cid for cid in self._chunks if cid.startswith(f"{prop.property_id}:")}

        changed: list[Chunk] = []
        for c in new_chunks:
            prev = self._hashes.get(c.chunk_id)
            if prev is None:
                report.added += 1
                changed.append(c)
            elif prev != c.content_hash:
                report.updated += 1
                changed.append(c)
            else:
                report.unchanged += 1

        for cid in old_ids - new_ids:          # 객실 삭제 등
            self._chunks.pop(cid, None)
            self._hashes.pop(cid, None)
            report.removed += 1

        for c in changed:
            self._chunks[c.chunk_id] = c
            self._hashes[c.chunk_id] = c.content_hash

        self.properties[prop.property_id] = prop

        # 인덱스는 전체 재구축하되, **임베딩은 변경분만** 새로 계산된다.
        # (내용이 같은 청크는 캐시 히트로 API 호출이 발생하지 않는다)
        self._rebuild()
        report.embedded = self.embedder.cache.misses - before
        return report

    def remove(self, property_id: str) -> IndexReport:
        report = IndexReport(properties_touched=[property_id])
        for cid in [c for c in self._chunks if c.startswith(f"{property_id}:")]:
            self._chunks.pop(cid, None)
            self._hashes.pop(cid, None)
            report.removed += 1
        self.properties.pop(property_id, None)
        self._rebuild()
        return report

    def _rebuild(self) -> None:
        chunks = list(self._chunks.values())
        if not chunks:
            self.index = ChunkIndex(dim=self.embedder.dim)
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.index.rebuild(chunks, vectors)

    # ------------------------------------------------------------ 조회
    def search(self, query: str, flt=None, top_k: int = 10):
        qv = self.embedder.embed_one(query)
        return self.index.search(query, qv, self.properties, flt=flt, top_k=top_k)

    @property
    def cache_stats(self) -> dict:
        return self.embedder.cache.stats()


__all__ = ["PropertyIndexer", "IndexReport"]
