"""검색 — Metadata First → Dense + BM25 → RRF.

핵심은 순서다. **벡터 공간을 훑기 전에 업무 조건으로 후보를 좁힌다.**

전체를 검색하고 나중에 거르면 "제주 4인 수영장" 질의에서 서울 2인 숙소가 상위를
차지한 뒤 필터에 걸려 사라진다. 결과적으로 뽑을 게 없어진다.
업무 조건은 애초에 정답이 정해져 있으므로 먼저 적용하는 것이 맞다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas.property import Chunk, Property, SearchFilter

RRF_K = 60


def tokenize(text: str) -> list[str]:
    """BM25 용 토크나이저. 한글은 문자 2-gram, 영숫자는 단어 단위."""
    import re

    tokens: list[str] = []
    for word in re.findall(r"[0-9A-Za-z]+|[가-힣]+", text.lower()):
        if word[0].isalnum() and word.isascii():
            tokens.append(word)
        else:
            tokens += [word[i : i + 2] for i in range(max(1, len(word) - 1))]
    return tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class RetrievalStats:
    """관측용. 필터가 실제로 후보를 얼마나 줄였는지 남긴다."""

    total_chunks: int = 0
    after_filter: int = 0
    dense_candidates: int = 0
    bm25_candidates: int = 0

    @property
    def filter_reduction(self) -> float:
        if not self.total_chunks:
            return 0.0
        return round(1 - self.after_filter / self.total_chunks, 4)


class ChunkIndex:
    """FAISS IndexFlatIP + 메타데이터 스토어 + BM25."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._corpus: list[list[str]] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("청크 수와 벡터 수가 다르다")
        if not chunks:
            return
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self.chunks.extend(chunks)
        self._corpus.extend(tokenize(c.text) for c in chunks)
        self._bm25 = BM25Okapi(self._corpus)

    def rebuild(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index = faiss.IndexFlatIP(self.dim)
        self.chunks, self._corpus, self._bm25 = [], [], None
        self.add(chunks, vectors)

    # ------------------------------------------------------------ 필터
    def matching_ids(self, flt: SearchFilter, properties: dict[str, Property]) -> list[int]:
        """메타데이터 조건을 만족하는 청크 인덱스."""
        ids: list[int] = []
        for i, c in enumerate(self.chunks):
            if flt.status and c.property_status != flt.status:
                continue
            if flt.region and c.region != flt.region:
                continue
            if flt.property_type and c.property_type != flt.property_type:
                continue
            if flt.document_types and c.document_type not in flt.document_types:
                continue
            prop = properties.get(c.property_id)
            if prop is None:
                continue
            if flt.min_capacity and prop.max_capacity < flt.min_capacity:
                continue
            if flt.max_price and prop.min_price > flt.max_price:
                continue
            if flt.required_amenities:
                have = prop.all_amenities
                if not all(a in have for a in flt.required_amenities):
                    continue
            ids.append(i)
        return ids

    # ------------------------------------------------------------ 검색
    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        properties: dict[str, Property],
        flt: SearchFilter | None = None,
        top_k: int = 10,
        pool: int = 50,
    ) -> tuple[list[Hit], RetrievalStats]:
        """메타 필터 → Dense·BM25 병렬 → RRF 병합."""
        stats = RetrievalStats(total_chunks=len(self.chunks))
        if not self.chunks:
            return [], stats

        flt = flt or SearchFilter()
        allowed = self.matching_ids(flt, properties)
        stats.after_filter = len(allowed)
        if not allowed:
            return [], stats

        allowed_set = set(allowed)

        # --- Dense: 후보군이 좁으므로 넉넉히 뽑고 필터로 거른다 ---------
        n_probe = min(len(self.chunks), max(pool * 4, 200))
        scores, idxs = self.index.search(
            np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32), n_probe
        )
        dense = [i for i in idxs[0] if i in allowed_set][:pool]
        stats.dense_candidates = len(dense)

        # --- BM25: 허용된 청크만 대상으로 ------------------------------
        bm25_ranked: list[int] = []
        if self._bm25 is not None:
            bm_scores = self._bm25.get_scores(tokenize(query))
            order = np.argsort(bm_scores)[::-1]
            bm25_ranked = [int(i) for i in order if int(i) in allowed_set][:pool]
        stats.bm25_candidates = len(bm25_ranked)

        # --- RRF 병합 ---------------------------------------------------
        rr: dict[int, float] = {}
        dr: dict[int, int] = {}
        br: dict[int, int] = {}
        for rank, i in enumerate(dense):
            rr[i] = rr.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            dr[i] = rank + 1
        for rank, i in enumerate(bm25_ranked):
            rr[int(i)] = rr.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
            br[int(i)] = rank + 1

        merged = sorted(rr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        hits = [
            Hit(chunk=self.chunks[i], score=round(s, 6),
                dense_rank=dr.get(i), bm25_rank=br.get(i))
            for i, s in merged
        ]
        return hits, stats


__all__ = ["ChunkIndex", "Hit", "RetrievalStats", "tokenize", "RRF_K"]
