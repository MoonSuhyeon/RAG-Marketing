# Hybrid RAG with Fact Validation for Content Generation

*Personal project*

![Retrieval Engineering](https://img.shields.io/badge/Retrieval%20Engineering-0B1220?style=for-the-badge)

![hybrid retrieval](https://img.shields.io/badge/hybrid%20retrieval-0F766E?style=for-the-badge) ![metadata filtering](https://img.shields.io/badge/metadata%20filtering-0F766E?style=for-the-badge) ![reranking](https://img.shields.io/badge/reranking-0F766E?style=for-the-badge) ![grounding](https://img.shields.io/badge/grounding-BE123C?style=for-the-badge) ![evaluation](https://img.shields.io/badge/evaluation-B45309?style=for-the-badge)

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

**100% fact consistency** and **0% hallucinated amenities** across 540 generations · **96.9%** candidate reduction before search · **28 tests**

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
              │  fact consistency rate                       │
              │  hallucinated amenity rate                   │
              │  filter reduction before the expensive step  │
              │  embedding cache hit rate                    │
              │                                              │
              │  all four are computed from the record, not  │
              │  scored by a model — so they are pinned by   │
              │  tests and need no API key                   │
              └──────────────────────────────────────────────┘

              ┌──────────────────────────────────────────────┐
              │                  SERVING                     │
              │  POST /index            incremental reindex  │
              │  POST /search           filter → RRF → assess│
              │  POST /generate         422 if validation    │
              │                         fails after a retry  │
              │  POST /generate/stream  stage events (SSE)   │
              │  GET  /metrics          cache · thresholds   │
              │                                              │
              │  Streamlit client shows content and its      │
              │  validation verdict side by side             │
              └──────────────────────────────────────────────┘
```

---

## Stack

| | |
|---|---|
| Language | Python 3.11 |
| Serving | FastAPI · Pydantic v2 · SSE · Streamlit |
| Retrieval | FAISS (IndexFlatIP) · rank_bm25 · RRF |
| Embedding | `text-embedding-3-small` \| local hash n-gram · MD5 disk cache |
| Generation | OpenAI SDK (`gpt-4o-mini`) \| deterministic template |
| Evaluation | Deterministic fact metrics — computed against the record, no judge model |
| Testing | pytest — 28 tests |

No LangChain, no LlamaIndex. Runs without an API key; the local backends keep
retrieval and validation testable in CI without secrets.

---

## Trade-offs

Measured on 100 properties → 986 chunks.

### Metadata filter before the vector search

Business conditions have a known answer. Running them first is cheaper than
running them last.

**Buys** — for *Jeju · 4+ guests · pool*, the candidate set drops from 986 chunks
to **31**, a 96.9% reduction, and every returned property satisfied all three
conditions. Reranking then runs over 3% of the corpus instead of all of it, so
the expensive stage is the one that shrinks.
**Costs** — the query has to be parsed into filters correctly. A wrong filter
returns nothing rather than something imperfect, which is a louder failure but
still a failure.

### Rule-based fact validation instead of LLM self-assessment

Claiming a pool that does not exist is not a wrong answer; it is a guest arriving
to a complaint.

**Buys** — deterministic verdicts that tests can pin, and no extra model call per
generation. Across 540 generations (60 properties × 3 segments × 3 formats):
**fact consistency 100%**, **hallucinated amenity 0%**. An adversarial case
naming two absent amenities was blocked with three violations, including a price
that matched no room rate.
**Costs** — it only catches what the controlled vocabulary covers. A claim
phrased outside that vocabulary passes.

The 100% is measured with the template backend, which composes retrieved fields
and therefore cannot invent by construction. An LLM backend will move that
number — and that is the point where the validator starts doing real work.

### Incremental indexing on a content hash

Prices and availability change constantly. Full reindex cost grows with catalogue
size, not with what changed.

**Buys** — a single price edit re-embeds **1 chunk of 986**. Unchanged chunks hit
the embedding cache, so no API call is made at all.
**Costs** — change detection has to know which fields map to which chunks. That
mapping is now something to maintain.

### FAISS in memory instead of a managed vector database

**Buys** — no network hop on the search path and no per-query bill, at a scale of
thousands of chunks where a managed service is overhead rather than help.
**Costs** — the index rebuilds on restart and cannot scale horizontally. At
hundreds of thousands of listings this decision reverses.

### Implemented directly, without a framework

**Buys** — when a result is wrong, the stage that produced it is identifiable.
Retrieval, reranking, compression and generation are each replaceable in
isolation, which is what made the retrieval core extractable into a shared
package for [Agent-Customer-Support](https://github.com/MoonSuhyeon/Agent-Customer-Support).
**Costs** — connectors and glue had to be written rather than imported.

## Run locally

```bash
pip install -r requirements.txt
pytest                          # 28 tests
python scripts/run_demo.py      # index → search → generate → validate

uvicorn api.server:app --reload  # API at /docs
streamlit run api/client.py      # demo client
```

The API refuses rather than ships: `POST /generate` returns **422** with the
violation list when generated text disagrees with the record after one retry.
Runs without an API key — retrieval and validation use the local backends.

## Docs

| | |
|---|---|
| `retrieval/` | Shared retrieval core, installable as `marketplace-retrieval` |
| `app/engine/fact_validator.py` | Field-by-field grounding check |
| `app/engine/indexer.py` | Incremental reindex on content hash |
| `docs/rag-evolution.md` | How the pipeline got here — v1 to v26, and what was dropped |
| `api/server.py` | HTTP surface over the pipeline |
| `tests/test_api.py` | The refusal rule holds at the API boundary too |
