"""도메인에 묶이지 않은 검색 코어.

숙소 스키마에 의존하던 부분을 걷어내 **다른 저장소에서 재사용**할 수 있게 분리했다.
문서는 `Doc`(본문 + 메타데이터 dict) 하나로 표현하고, 필터는 술어 함수로 받는다.

Agent-Customer-Support 가 이 코어를 그대로 가져다 취소·환불 **정책 문서** 검색에 쓴다.
정책 검색에는 한 가지가 더 필요했다 — **기권(abstain)**. 검색은 언제나 무언가를 돌려주지만,
근거가 약하면 "모른다"고 말할 수 있어야 에이전트가 추측하지 않는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60


def tokenize(text: str) -> list[str]:
    """BM25 용 토크나이저. 한글은 문자 2-gram, 영숫자는 단어 단위."""
    tokens: list[str] = []
    for word in re.findall(r"[0-9A-Za-z]+|[가-힣]+", text.lower()):
        if word.isascii():
            tokens.append(word)
        else:
            tokens += [word[i : i + 2] for i in range(max(1, len(word) - 1))]
    return tokens


@dataclass
class Doc:
    """검색 단위. 도메인 타입을 모른다."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    doc: Doc
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


@dataclass
class SearchStats:
    total: int = 0
    after_filter: int = 0
    dense: int = 0
    bm25: int = 0

    @property
    def filter_reduction(self) -> float:
        return round(1 - self.after_filter / self.total, 4) if self.total else 0.0


Predicate = Callable[[Doc], bool]


class HybridIndex:
    """FAISS(IndexFlatIP) + BM25 + RRF.

    검색 **이전에** 술어로 후보를 좁힌다. 전체를 훑고 나중에 거르면
    상위 결과가 조건에 맞지 않는 문서로 채워진다.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.docs: list[Doc] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self.docs)

    def build(self, docs: Iterable[Doc], vectors: np.ndarray) -> None:
        docs = list(docs)
        if len(docs) != len(vectors):
            raise ValueError("문서 수와 벡터 수가 다르다")
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs = docs
        self._corpus = [tokenize(d.text) for d in docs]
        self._bm25 = BM25Okapi(self._corpus) if docs else None
        if docs:
            self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(
        self,
        query: str,
        query_vec: np.ndarray,
        where: Predicate | None = None,
        top_k: int = 10,
        pool: int = 50,
    ) -> tuple[list[Hit], SearchStats]:
        stats = SearchStats(total=len(self.docs))
        if not self.docs:
            return [], stats

        allowed = [i for i, d in enumerate(self.docs) if where is None or where(d)]
        stats.after_filter = len(allowed)
        if not allowed:
            return [], stats
        allowed_set = set(allowed)

        n_probe = min(len(self.docs), max(pool * 4, 200))
        _, idxs = self.index.search(
            np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32), n_probe
        )
        dense = [int(i) for i in idxs[0] if int(i) in allowed_set][:pool]
        stats.dense = len(dense)

        bm25_ranked: list[int] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query))
            order = np.argsort(scores)[::-1]
            bm25_ranked = [int(i) for i in order if int(i) in allowed_set][:pool]
        stats.bm25 = len(bm25_ranked)

        rr: dict[int, float] = {}
        dr: dict[int, int] = {}
        br: dict[int, int] = {}
        for rank, i in enumerate(dense):
            rr[i] = rr.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            dr[i] = rank + 1
        for rank, i in enumerate(bm25_ranked):
            rr[i] = rr.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
            br[i] = rank + 1

        merged = sorted(rr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        hits = [
            Hit(doc=self.docs[i], score=round(s, 6),
                dense_rank=dr.get(i), bm25_rank=br.get(i))
            for i, s in merged
        ]
        return hits, stats


# --------------------------------------------------------------- 기권 판정
@dataclass
class Grounding:
    """검색 결과를 근거로 써도 되는지에 대한 판정.

    Agent 처럼 **틀리면 안 되는** 소비자를 위해 존재한다.
    검색은 언제나 무언가를 돌려주므로, 돌려준 것이 쓸 만한지는 따로 판정해야 한다.
    """

    grounded: bool
    reason: str
    top_score: float = 0.0
    margin: float = 0.0

    def __bool__(self) -> bool:
        return self.grounded


def assess(hits: list[Hit], min_score: float, min_margin: float = 0.0) -> Grounding:
    """상위 점수와 1·2위 격차로 근거 충분성을 판정한다.

    격차를 보는 이유: 여러 문서가 비슷하게 걸리면 **어느 것이 답인지 모른다**는 뜻이다.
    점수만 높고 격차가 없으면 기권하는 편이 낫다.
    """
    if not hits:
        return Grounding(False, "검색 결과가 없다")
    top = hits[0].score
    if top < min_score:
        return Grounding(False, f"최고 점수 {top:.4f} 가 임계 {min_score:.4f} 미만이다", top)
    margin = top - (hits[1].score if len(hits) > 1 else 0.0)
    if margin < min_margin:
        return Grounding(
            False, f"1·2위 격차 {margin:.4f} 가 임계 {min_margin:.4f} 미만이다", top, margin
        )
    return Grounding(True, "충분", top, margin)


__all__ = [
    "Doc", "Grounding", "Hit", "HybridIndex", "Predicate", "RRF_K",
    "SearchStats", "assess", "tokenize",
]
