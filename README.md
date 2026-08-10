# Hybrid RAG with Fact Validation for Content Generation

*Personal project*

![Retrieval Engineering](https://img.shields.io/badge/Retrieval%20Engineering-0E1725?style=flat-square)
![hybrid retrieval](https://img.shields.io/badge/hybrid%20retrieval-41506A?style=flat-square) ![metadata filtering](https://img.shields.io/badge/metadata%20filtering-41506A?style=flat-square) ![reranking](https://img.shields.io/badge/reranking-41506A?style=flat-square) ![grounding](https://img.shields.io/badge/grounding-41506A?style=flat-square) ![evaluation](https://img.shields.io/badge/evaluation-41506A?style=flat-square)

A listing's selling points are scattered across fields — rooms, amenities,
price, house rules, location. **Marketing has to turn those fields into copy,
per customer segment, for every listing.** Doing it by hand does not scale;
handing the whole record to a model does not either.

The usual RAG problem is answering questions from documents. This is different
in a way that matters: **the source is structured records, and being wrong has a
price.** An assistant that misreads a PDF gives a bad answer. An ad that
promises a pool the property does not have produces a guest who arrives, finds
no pool, and files a complaint.

So the question is not "can the model write good copy" but **"can I prove that
every claim in the copy came from the record?"**

I built a retrieval and generation pipeline where **business filters narrow
candidates before the vector search runs, generated copy is checked field by
field against the source record, and anything that fails validation is refused
rather than shipped.** The retrieval core is packaged separately and reused by
[Agent-Customer-Support](https://github.com/MoonSuhyeon/Agent-Customer-Support)
for policy lookup.

Built without LangChain or LlamaIndex — retrieval, reranking, compression and
generation are implemented directly, so a bad result can be traced to the stage
that produced it.

---

## Architecture

```
              ┌────────────────────────────────────────────────┐
              │                 PROPERTY DB                    │
              │                                                │
              │  Property · Room · Amenity · Policy · Location  │
              │  updated_at drives incremental reindex         │
              └───────────────────────┬────────────────────────┘
                                      │
                                      ▼
        ┌──────────────────────────────────────────────────────────┐
        │                       INDEXER                            │
        │                                                          │
        │  ┌──────────────────────┐   ┌─────────────────────────┐  │
        │  │ FIELD-BOUNDARY CHUNK │   │ METADATA ENRICHMENT     │  │
        │  │                      │   │                         │  │
        │  │ BASIC   · one        │   │ property_id · region    │  │
        │  │ ROOM    · per room   │   │ capacity · price_range  │  │
        │  │ AMENITY · per item   │   │ property_type · status  │  │
        │  │ POLICY  · one        │   │ amenity_key · room_id   │  │
        │  │ LOCATION· one        │   │ content_hash            │  │
        │  └──────────────────────┘   └─────────────────────────┘  │
        │                                                          │
        │  incremental: hash unchanged → chunk skipped entirely    │
        │  properties without a policy are NOT indexed             │
        └───────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │                  retrieval  (shared package)             │
        │                                                          │
        │  ┌──────────────────┐        ┌────────────────────────┐  │
        │  │ EMBEDDER         │        │ INDEX                  │  │
        │  │ OpenAI  |  local │───────▶│ FAISS IndexFlatIP      │  │
        │  │ MD5 disk cache   │        │ BM25Okapi              │  │
        │  │ hit → no API call│        │ metadata store         │  │
        │  └──────────────────┘        └────────────────────────┘  │
        │                                                          │
        │  ┌────────────────────────────────────────────────────┐  │
        │  │ SEARCH   metadata filter FIRST, then vectors       │  │
        │  │                                                    │  │
        │  │   query ─▶ predicate ─┬─▶ dense  ─┐                │  │
        │  │                       └─▶ BM25   ─┴─▶ RRF ─▶ hits  │  │
        │  └────────────────────────────────────────────────────┘  │
        │                                                          │
        │  ┌────────────────────────────────────────────────────┐  │
        │  │ assess(hits, min_score, min_margin) → Grounding    │  │
        │  │ lets a consumer abstain instead of guessing        │  │
        │  └────────────────────────────────────────────────────┘  │
        └───────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │                      GENERATION                          │
        │                                                          │
        │  ┌────────────────────┐      ┌────────────────────────┐  │
        │  │ SEGMENT PROFILE    │      │ OUTPUT FORMAT          │  │
        │  │ COUPLE   view·spa  │  ×   │ SNS · AD_COPY · CRM    │  │
        │  │ FAMILY   kitchen   │      │                        │  │
        │  │ BUSINESS wifi·gym  │      │ backend: LLM | template│  │
        │  └────────────────────┘      └────────────────────────┘  │
        │                                                          │
        │  highlights = segment priorities ∩ amenities the         │
        │  property actually has                                   │
        └───────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────┐
        │                    FACT VALIDATOR                        │
        │            generated text  ⟷  source fields              │
        │                                                          │
        │  amenity   controlled vocabulary + aliases (풀장→수영장)  │
        │  price     must match an actual room rate                │
        │  capacity  must not exceed max occupancy                 │
        │  region    no other region may be named                  │
        │  name      the property must be named                    │
        │                                                          │
        │        ┌──────────────┬───────────────────────┐          │
        │        │ consistent   │ violation             │          │
        │        │ ship         │ regenerate → refuse   │          │
        │        └──────────────┴───────────┬───────────┘          │
        └────────────────────────────────────┼─────────────────────┘
                                             ▼
              ┌──────────────────────────────────────────────┐
              │                 EVALUATION                   │
              │  RAGAS  faithfulness · answer relevancy ·    │
              │         context precision                    │
              │  own    fact consistency · hallucinated      │
              │         amenity rate                         │
              │  ops    P50/P95/P99 · tokens · cache hit     │
              │  FailureDataset → JSONL by failure type      │
              └──────────────────────────────────────────────┘

              ┌──────────────────────────────────────────────┐
              │                  SERVING                     │
              │  FastAPI · SSE   |   Streamlit client        │
              └──────────────────────────────────────────────┘
```

---

## Results

100 properties → 986 chunks. Query: *Jeju · 4+ guests · pool*.

| | |
|---|---|
| Metadata filter | 986 → **31 chunks** (96.9% narrowed) |
| Filter precision | 100% — every returned property met all three conditions |
| Fact consistency | **100%** across 540 generations (60 properties × 3 segments × 3 formats) |
| Hallucinated amenity | **0%** |
| Price change reindex | **1 chunk** of 986 |

Adversarial case — copy naming two amenities the property does not have:

```
blocked · 3 violations
  hallucinated_amenity  '와이파이' is not at this property
  hallucinated_amenity  '조식' is not at this property
  price_mismatch        33,000원 matches neither 210,000 nor 230,000
```

Validation is rule-based, not model self-assessment. Because the source is
structured, the check is deterministic and pinned by tests — including alias
normalization, wrong price, over-capacity and wrong region.

The 100% figure is measured with the template backend, which composes only
retrieved fields and therefore cannot hallucinate by construction. Swapping in
an LLM will move that number, and that is the point at which the validator
starts doing real work.

## Stack

| | |
|---|---|
| Language | Python 3.11 |
| Serving | FastAPI · Pydantic v2 · SSE · Streamlit |
| Retrieval | FAISS (IndexFlatIP) · rank_bm25 · RRF |
| Embedding | `text-embedding-3-small` \| local hash n-gram · MD5 disk cache |
| Generation | OpenAI SDK (`gpt-4o-mini`) \| deterministic template |
| Evaluation | RAGAS · custom fact metrics |
| Testing | pytest — 19 tests |

No LangChain, no LlamaIndex. Runs without an API key; the local backends keep
retrieval and validation testable in CI without secrets.

## Run locally

```bash
pip install -r requirements.txt
pytest                       # 19 tests
python scripts/run_demo.py   # index → search → generate → validate
```

## Docs

| | |
|---|---|
| `retrieval/` | Shared retrieval core, installable as `marketplace-retrieval` |
| `app/engine/fact_validator.py` | Field-by-field grounding check |
| `app/engine/indexer.py` | Incremental reindex on content hash |
| `change_logs/` | v1–v26 evolution of the pipeline |
