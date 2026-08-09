# RAG-Marketing

> Airbnb형 숙박 플랫폼의 **숙소 상세정보를 검색 가능한 지식으로 구조화하고, 고객 세그먼트별 마케팅 콘텐츠를 생성하는 RAG 시스템**

숙소명·객실·편의시설·가격·이용규칙·지역 정보를 구조화하여 인덱싱하고,
검색된 **실제 숙소 데이터에 근거해서만** 마케팅 콘텐츠를 생성합니다.

LangChain / LlamaIndex 없이 검색·리랭킹·생성 파이프라인을 **직접 구현**했습니다.

---

## 1. Architecture

```text
[Client] Streamlit (client_app.py)                    ✅
    ↓ HTTP / SSE
[Server] FastAPI (server_api.py)                      ✅
    ├── routers/auth.py        JWT 인증               ✅
    ├── routers/metrics.py     지표·실패 데이터셋      ✅
    ├── routers/admin.py       사용자 관리            ✅
    ├── routers/properties.py  숙소 등록·색인         🆕
    ├── routers/search.py      숙소 검색              🆕
    └── routers/marketing.py   콘텐츠 생성            🆕
    ↓
[Engine] rag_engine.py                                ✅
    ├── MultiHopPlanner      질문 분해·hop별 검색     ✅
    ├── SelfRAGChecker       검색 충분성 자동 판단     ✅
    ├── AsyncRAGEngine       asyncio 비동기 파이프라인 ✅
    ├── MetricsCollector     P50/P95/P99 지연         ✅
    ├── FailureDataset       실패 케이스 JSONL        ✅
    ├── PropertyIndexer      숙소 필드 단위 색인       🆕
    ├── SegmentPrompter      세그먼트별 프롬프트       🆕
    └── FactValidator        생성물 ↔ 원본 대조       🆕
    ↓
[Index] FAISS IndexFlatIP (in-memory)                 ✅
    ├── chunk_index  / sent_index / kw_index          ✅
    └── metadata store (property_id, region, ...)     🔨
[Eval] evaluate_ragas.py                              ✅
    └── + 사실 정합성 지표                             🆕
```

---

## 2. Implementation

검색과 생성을 분리하고, 생성물은 반드시 원본과 대조한다.

| 단계 | 구현 | 핵심 |
|------|------|------|
| **Index** | `engine/indexer.py` | 필드 경계로 청킹하고 메타데이터를 붙인다. 변경 감지 시 해당 청크만 재생성 |
| **Retrieve** | `engine/retriever.py` | 메타데이터 필터로 후보를 좁힌 뒤 Dense + BM25를 RRF로 병합 |
| **Rerank** | `engine/reranker.py` | LLM이 0~10으로 채점하고, 컨텍스트를 압축해 프롬프트를 줄인다 |
| **Generate** | `engine/segment_prompter.py` | 세그먼트별로 강조 필드를 바꿔 문구를 만든다 |
| **Validate** | `engine/fact_validator.py` | 생성물에서 속성을 추출해 **원본 필드와 대조** |

### Failure handling

없는 편의시설을 광고하면 실제 클레임이 된다. 실패는 임의 생성이 아니라 명시적 처리로 흡수한다.

```text
LLM / 임베딩 API 실패
    ↓ retry — 지수 백오프, 임베딩은 해시 캐시로 재호출 회피
검색 품질 미달 (score 임계 이하)
    ↓ Query Rewrite 후 재검색
Fact validation 실패 (없는 편의시설·가격 불일치)
    ↓ validation — 재생성 1회
재생성도 실패
    ↓ fallback — 콘텐츠 반환 거부, 실패 상태 반환
    ↓ FailureDataset(JSONL)에 유형별 적재
반복 실패 유형
    ↓ human review — 청킹·프롬프트 전략 재검토
```

**Silent Fallback을 금지한다.** 검색이 부족하면 그럴듯한 문구를 만들지 않고 실패를 반환한다.

---

## 3. Evaluation

> 아직 구현 전이므로 아래 수치는 **측정값이 아니라 목표치**다. 구현과 함께 계측을 붙이고 측정되는 대로 갱신한다.

| 지표 | 대상 | 목표 |
|------|------|------|
| **Fact Consistency Rate** | 생성물 ↔ 원본 필드 일치율 | 100% |
| **Hallucinated Amenity Rate** | 없는 편의시설 언급 비율 | 0% |
| Faithfulness | 검색 근거 충실도 (RAGAS) | 버전별 비교 |
| Answer Relevancy | 요청 부합도 (RAGAS) | 버전별 비교 |
| Context Precision | 검색 정밀도 (RAGAS) | 버전별 비교 |
| Retrieval P50 / P95 / P99 | 검색 지연 | `MetricsCollector`로 계측 중 |
| Token cost / request | 생성 비용 | 압축 전후 비교 |
| 증분 인덱싱 효율 | 가격 1건 변경 시 재색인 청크 수 | 해당 청크만 |
| 캐시 히트 시 임베딩 호출 | API 호출 수 | 0회 |

**버전별로 비교한다.** 청킹·검색·프롬프트를 바꿀 때마다 같은 평가셋으로 돌려 개선 여부를 수치로 확인한다.

현재 측정값: 지연·토큰 계측기는 동작하나 숙소 도메인 기준 수치는 없음

---

## 4. Engineering Decisions

| 결정 | 채택 | 이유와 대안 |
|------|------|-------------|
| 파이프라인 | **프레임워크 없이 직접 구현** | 검색이 나빠졌을 때 어느 단계에서 깨졌는지 추적할 수 있어야 한다. 추상화 계층은 디버깅 비용을 올린다 |
| 검색 순서 | **메타데이터 필터 → 벡터·BM25** | 전체 벡터 공간을 훑고 나중에 거르면 무관한 숙소가 상위를 차지한다. 업무 조건은 애초에 정답이 정해져 있다 |
| 벡터 저장소 | **FAISS in-memory** | 숙소 수천 건 규모에서 관리형 벡터 DB의 운영 비용이 정당화되지 않는다 |
| 리랭킹 | **LLM-as-Reranker** | 전용 리랭커 모델은 별도 서빙 인프라가 필요하다. 이미 쓰는 LLM으로 품질을 확보한다 |
| 청킹 | **필드 경계 기준** | 고정 길이·semantic 청킹은 비정형 문서용이다. 숙소 데이터는 필드 경계가 곧 의미 경계다 |
| 색인 갱신 | **증분 + 임베딩 해시 캐시** | 가격 하나 바뀔 때마다 전체 재색인하면 비용이 선형으로 는다 |
| 검증 | **생성 후 필드 대조** | 원본이 정형 필드라 정량 검증이 가능하다. LLM 자기평가만으로는 근거가 약하다 |

### Trade-offs

| 얻은 것 | 포기한 것 |
|---------|-----------|
| 전 구간을 추적·수정할 수 있다 | 커넥터·유틸을 **직접 만들어야 한다** |
| 상위 결과 품질이 올라간다 | 요청마다 **LLM 호출이 추가되어 지연·비용이 는다** |
| 인프라가 단순하고 검색이 빠르다 | **재시작 시 재색인**이 필요하고 수평 확장이 어렵다 |
| 필드 단위로 정확히 검색된다 | 필드를 가로지르는 질의에 약해 **Multi-doc chain으로 보완**해야 한다 |
| 잘못된 광고가 나가지 않는다 | 검증만큼 **응답이 늦어지고** 재생성 비용이 든다 |

---

## 문서 성격

이 문서는 **구현 명세(Spec)** 입니다. 완성된 결과물이 아니라 만들어 가는 목표 상태를 기술합니다.
각 항목의 진행 상태는 다음 표기로 구분합니다.

| 표기 | 의미 |
|------|------|
| ✅ | 구현 완료 |
| 🔨 | 진행 중 |
| 🆕 | 예정 |

검색 파이프라인은 v1~v26에 걸쳐 단계적으로 구축했으며, 각 버전은 실제로 관찰된 문제에서 출발했습니다. 버전별 변경 내역은 [`change_logs/`](./change_logs)를 참고하세요.

---

## 목적

```text
숙소 상세정보
      ↓
구조화 · 인덱싱
      ↓
검색 (Hybrid + Metadata Filter)
      ↓
고객 세그먼트 / 마케팅 목적
      ↓
콘텐츠 생성
      ↓
사실 정합성 검증
      ↓
Marketing Asset
```

마케팅팀이 숙소 정보를 일일이 확인하며 문구를 쓰는 반복 작업을, **실제 데이터에 근거한 생성**으로 대체하는 것이 목표입니다.

---

## 문제 정의

### 문서 RAG와 상품 RAG는 다르다 🆕

일반적인 RAG는 **비정형 문서**를 다룹니다. 숙소 정보는 **DB 레코드**입니다. 이 차이가 설계 전반을 바꿉니다.

| | 일반 문서 RAG | 상품 RAG (본 프로젝트) |
|---|---|---|
| 입력 | 비정형 텍스트 | **정형 필드 + 일부 텍스트** |
| 청킹 기준 | 의미 경계 탐색 필요 | **필드 경계가 이미 존재** |
| 메타데이터 | 추출해야 함 | **처음부터 구조화되어 있음** |
| 갱신 | 문서 교체 (드묾) | **가격·재고는 수시로 변경** |
| 오류의 비용 | 답변이 틀림 | **없는 편의시설을 광고 → 실제 클레임** |

→ 이 차이 때문에 **메타데이터 필터를 검색보다 먼저** 적용하고, **증분 인덱싱**과 **사실 정합성 검증**이 필수가 됩니다.

### 하나의 숙소에 흩어진 정보

```text
Property
├── Basic      name / description / location
├── Room       room_type / capacity / price / amenities
├── Policy     check_in / check_out / cancellation / house_rules
└── Local      attractions / transportation / neighborhood
```

LLM에 전체를 통째로 넣으면 서로 다른 숙소의 정보가 섞이거나 핵심 필드가 누락됩니다.

---

## 기술 범위

### 숙소 청킹 전략 🔨

문서 RAG의 semantic chunking 대신 **필드 경계 청킹**을 사용합니다.

```text
숙소 설명   →  Basic Chunk
객실 정보   →  Room Chunk (객실마다 1개)
편의시설    →  Amenity Chunk
이용규칙    →  Policy Chunk
주변 정보   →  Location Chunk
```

각 청크에 붙는 메타데이터:

```text
property_id · property_type · room_id · region
amenity_type · price_range · capacity
document_type · property_status · updated_at
```

> 비정형 문서에서는 "청크 경계에서 의미가 잘리는 문제" 때문에 Overlap·Semantic 청킹이 필요했습니다. 숙소 데이터는 **필드 경계가 곧 의미 경계**라 그 고민이 사라지는 대신, **필드 간 관계(객실 ↔ 편의시설)를 어떻게 이을지**가 새 과제가 됩니다.

### Metadata First 검색 🔨

전체 벡터 공간을 훑기 전에 **후보군을 업무 조건으로 먼저 좁힙니다.**

```text
Query: "제주도 가족 여행에 좋은 숙소"
    ↓
Intent / Metadata Extraction
    ↓
Filter:  region = "Jeju"
         capacity >= 4
         property_status = "active"
    ↓
Vector Search  +  BM25          ✅ (v4 하이브리드)
    ↓
RRF 병합                        ✅
    ↓
LLM Reranking (gpt-4o-mini)     ✅ (v4)
    ↓
Context Compression             ✅ (v19/v21/v23)
```

> Query Routing(v8)은 질의를 **숙소 질의 유형(지역 추천 / 편의시설 조건 / 가격대 / 복합 마케팅 요청)** 으로 분류해 검색 전략을 자동 선택합니다. 모든 질의에 같은 검색 파라미터를 쓰면 지역 추천과 정확 매칭 중 한쪽이 반드시 나빠집니다.

### 증분 인덱싱 🆕

가격·재고·편의시설은 수시로 바뀝니다. 전체 재색인은 비용이 큽니다.

```text
숙소 정보 변경
      ↓
변경 감지 (updated_at 비교)
      ↓
해당 Property의 영향받은 Chunk만 재생성
      ↓
Embedding 갱신 (캐시 미스분만)
      ↓
Vector DB Update
```

처리 대상: 신규 등록 / 정보 수정 / 객실 변경 / 가격 변경 / 편의시설 변경 / 삭제·비활성화

> **MD5 해시 기반 임베딩 디스크 캐시**(v19)가 여기서 결정적입니다. 청크 내용이 그대로면 해시가 같으므로 캐시 히트로 임베딩 API 호출을 건너뜁니다. 숙소 1건의 가격만 바뀌었을 때 전체를 다시 임베딩하는 비용을 없앱니다.

### 세그먼트별 콘텐츠 생성 🆕

동일한 숙소라도 세그먼트에 따라 **검색 필터와 강조 필드**가 달라집니다.

| 세그먼트 | 우선 검색 필드 | 강조 |
|---------|---------------|------|
| 커플 | 뷰, 프라이빗 공간, 인테리어 | 감성 |
| 가족 | 수용 인원, 객실 크기, 주방, 주차 | 실용 |
| 비즈니스 | 교통 접근성, Wi-Fi, 업무 공간, 체크인 편의 | 효율 |

출력 형식도 목적별로 분기합니다.

```text
SNS Content   |  Ad Copy      |  CRM Message
Instagram     |  Meta Ad      |  Email
Blog          |  Search Ad    |  Push
```

### 사실 정합성 검증 (FactValidator) 🆕

**이 프로젝트에서 가장 중요한 안전장치입니다.** 수영장이 없는 숙소에 "프라이빗 수영장에서의 여유"라는 문구가 나가면 실제 클레임이 됩니다.

```text
Retrieved Context
       ↓
Content Generation
       ↓
FactValidator
  ├── 숙소명 일치
  ├── 객실 정보 일치
  ├── 가격 정보 일치
  ├── 편의시설 존재 여부   ← 원본 필드와 직접 대조
  ├── 이용규칙 일치
  └── 지역 정보 일치
       ↓
┌───────────────┬────────────────┐
│ Consistent    │ Inconsistent   │
│ 콘텐츠 반환    │ 재생성 / Reject │
└───────────────┴────────────────┘
```

문서 RAG와 달리 **원본이 정형 필드**이므로, 생성물에서 속성을 추출해 필드와 대조하는 **정량 검증이 가능**합니다. 이것이 상품 RAG의 이점입니다.

### 복합 요청 처리 🔨

> "제주도에서 4인 가족이 머물기 좋고 수영장이 있는 숙소를 찾아서 인스타그램 광고 문구를 만들어줘."

```text
User Request
      ↓
Task Decomposition                    ✅ (v25 Multi-Hop)
      ↓
 1. region = 제주
 2. capacity >= 4
 3. amenity contains 수영장
 4. 숙소 정보 검색
 5. 마케팅 콘텐츠 생성
      ↓
Self-RAG: 검색 충분성 판단             ✅ (v25)
      ↓
Content Generation → Fact Validation
```

### 평가 🔨

| 지표 | 출처 | 대상 |
|------|------|------|
| Faithfulness | RAGAS ✅ | 검색 근거 충실도 |
| Answer Relevancy | RAGAS ✅ | 요청 부합도 |
| Context Precision | RAGAS ✅ | 검색 정밀도 |
| **Fact Consistency Rate** | 자체 🆕 | 생성물 ↔ 원본 필드 일치율 |
| **Hallucinated Amenity Rate** | 자체 🆕 | 없는 편의시설 언급 비율 |

---

## 구현 로드맵

| Phase | 산출물 | 착수 조건 |
|-------|--------|-----------|
| **0** | 숙소 스키마 정의 + 더미 숙소 데이터 (50~100개) | — |
| **1** | `PropertyIndexer` — 필드 단위 청킹 + 메타데이터 부착 | Phase 0 |
| **2** | `routers/properties.py` — 숙소 등록·색인 API | Phase 1 |
| **3** | 메타데이터 필터 + 기존 하이브리드 검색 연결 | Phase 2 |
| **4** | `routers/search.py` — 숙소 검색 API | Phase 3 |
| **5** | `SegmentPrompter` + `routers/marketing.py` — 콘텐츠 생성 | Phase 4 |
| **6** | `FactValidator` — 생성물 ↔ 원본 대조 | Phase 5 |
| **7** | 증분 인덱싱 (변경 감지 → 부분 재색인) | Phase 2 |
| **8** | 평가 확장 (RAGAS + 자체 지표) | Phase 6 |

**Phase 5까지가 최소 완성선**입니다. 검색된 실제 숙소 데이터로 세그먼트별 문구가 나오면 프로젝트로서 성립합니다. Phase 6은 그 다음으로 중요합니다.

---

## 완료 정의 (DoD)

- [ ] 숙소 100개가 필드 단위로 색인되고 메타데이터 필터가 동작한다
- [ ] "제주 / 4인 이상 / 수영장" 같은 복합 조건 검색이 정확한 후보를 반환한다
- [ ] 세그먼트 3종(커플·가족·비즈니스)에 대해 서로 다른 문구가 생성된다
- [ ] `FactValidator`가 없는 편의시설 언급을 차단한다 — 테스트 케이스로 증명
- [ ] 가격 변경 시 해당 숙소 청크만 재색인된다 (전체 재색인 아님)
- [ ] RAGAS 3개 지표 + 사실 정합성 지표가 수치로 기록된다

---

## 기술 스택

### 핵심 (Python / FastAPI)

| 영역 | 기술 |
|------|------|
| 언어 | Python 3.11 |
| API | **FastAPI**, Pydantic v2, Uvicorn, SSE |
| LLM | OpenAI SDK (`gpt-4o-mini`) |
| 임베딩 | `text-embedding-3-small` + 디스크 캐시 |
| 벡터 검색 | FAISS (`IndexFlatIP`) |
| 키워드 검색 | `rank_bm25` (BM25Okapi) |
| 비동기 | asyncio |
| 인증 | python-jose (JWT) |
| DB | PostgreSQL, SQLAlchemy 2.0 (async) |
| 평가 | RAGAS |
| 테스트 | pytest |
| 클라이언트 | Streamlit |

프레임워크(LangChain / LlamaIndex) 없이 파이프라인 전 구성요소를 직접 구현합니다.

---

## 프로젝트 구조 (목표)

```text
RAG-Marketing/
├── app/
│   ├── main.py                 # FastAPI 진입점
│   ├── config.py
│   ├── deps.py
│   ├── routers/
│   │   ├── auth.py             ✅
│   │   ├── properties.py       🆕
│   │   ├── search.py           🆕
│   │   ├── marketing.py        🆕
│   │   ├── metrics.py          ✅
│   │   └── admin.py            ✅
│   ├── engine/
│   │   ├── rag_engine.py       ✅
│   │   ├── indexer.py          🆕
│   │   ├── retriever.py        🔨
│   │   ├── reranker.py         ✅
│   │   ├── compressor.py       ✅
│   │   ├── segment_prompter.py 🆕
│   │   └── fact_validator.py   🆕
│   └── schemas/
│       ├── property.py
│       └── marketing.py
├── client_app.py               # Streamlit
├── evaluate_ragas.py           ✅
├── tests/
├── docs/
│   ├── property-schema.md
│   ├── metadata-schema.md
│   ├── retrieval-strategy.md
│   └── evaluation.md
├── rag_versions/               # v1~v22 구현 이력 (보존)
├── change_logs/                # 버전별 변경 기록 (보존)
└── requirements.txt
```

---

## 핵심 설계 원칙

1. **SSoT** — 숙소 정보의 원천은 Property DB. RAG가 없는 정보를 만들어내지 않는다
2. **Metadata First** — 벡터 검색 전에 업무 조건으로 후보를 좁힌다
3. **Retrieval Before Generation** — LLM의 기억이 아니라 검색된 필드로만 생성한다
4. **Fact Consistency** — 생성물은 원본 필드와 대조되어야 나간다
5. **No Silent Fallback** — 검색이 부족하면 임의 생성 대신 재검색하거나 실패를 반환한다
6. **Evaluation Driven** — 청킹·검색·프롬프트 변경은 지표로 검증하고 버전별로 비교한다

---

## 다른 레포와의 연결

| 방향 | 내용 |
|------|------|
| **→ Agent-Customer-Support** | 검색 엔진을 **정책 문서 검색**에 재사용 (취소·환불 규정 조회) |
| **ML-Product →** | 비수기 예상 숙소를 프로모션 콘텐츠 생성 대상으로 수신 |
| **→ Data-Growth** | 생성된 콘텐츠를 A/B 테스트 소재로 제공 |

플랫폼 전체 구성은 [프로필 README](https://github.com/MoonSuhyeon)를 참고하세요.
