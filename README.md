# RAG-Marketing

> Airbnb형 숙박 플랫폼의 **숙소 상세정보를 활용한 RAG 기반 Marketing Intelligence & Content Generation 시스템**
>
> 숙소명·객실·편의시설·가격·이용규칙·지역 정보 등 상품 상세정보를 구조화하고, RAG를 통해 고객 세그먼트와 마케팅 목적에 맞는 콘텐츠를 생성하는 **End-to-End RAG 시스템**입니다.

---

# 프로젝트 개요

RAG-Marketing은 Airbnb형 숙박 플랫폼에서 운영되는 다양한 숙소 정보를 AI가 정확하게 검색하고 활용할 수 있도록 **숙소 정보 → 지식베이스 → 검색 → 콘텐츠 생성**의 전체 Pipeline을 구축한 프로젝트입니다.

단순한 문서 기반 Q&A를 넘어 숙소 상세정보를 마케팅 콘텐츠 생성에 활용할 수 있도록 다음 구조를 설계합니다.

```text
숙소 상세정보
      ↓
문서 구조 분석
      ↓
Chunking
      ↓
Embedding
      ↓
Metadata
      ↓
Vector DB
      ↓
Hybrid Retrieval
      ↓
Reranking
      ↓
Context Compression
      ↓
RAG Workflow
      ↓
고객 세그먼트 / 마케팅 목적
      ↓
Marketing Content Generation
```

---

# Problem

숙박 플랫폼에서는 하나의 숙소에도 다양한 정보가 존재합니다.

```text
숙소
├── 숙소명
├── 숙소 설명
├── 객실
├── 가격
├── 편의시설
├── 위치
├── 주변 관광지
├── 이용규칙
├── 체크인 / 체크아웃
├── 취소 / 환불 정책
└── 호스트 정보
```

이 정보가 여러 문서와 필드에 분산되어 있을 경우 LLM이 전체 데이터를 그대로 읽어 콘텐츠를 생성하면 다음과 같은 문제가 발생할 수 있습니다.

* 필요한 숙소 정보를 정확하게 찾지 못함
* 서로 다른 숙소의 정보가 혼합됨
* 가격·객실·편의시설 등 핵심 정보가 누락됨
* 존재하지 않는 편의시설이나 서비스를 생성
* 정책과 실제 숙소 정보가 불일치
* 고객 세그먼트에 맞지 않는 콘텐츠 생성

따라서 **숙소 정보를 검색 가능한 지식 구조로 만들고, 생성 전에 필요한 정보만 정확하게 검색하는 구조**가 필요합니다.

---

# 주요 업무

## 1. 숙소 상세정보 구조 분석 및 RAG 설계

숙소명·객실·편의시설·가격·이용규칙·위치 등 숙소 상품 정보의 구조를 분석하고 RAG 기반 정보 활용 구조를 설계합니다.

```text
Property
│
├── Basic Information
│   ├── name
│   ├── description
│   └── location
│
├── Room
│   ├── room_type
│   ├── capacity
│   ├── price
│   └── amenities
│
├── Policy
│   ├── check_in
│   ├── check_out
│   ├── cancellation
│   └── house_rules
│
└── Local Information
    ├── attractions
    ├── transportation
    └── neighborhood
```

숙소 데이터를 단순 텍스트로 취급하지 않고 **업무 도메인과 검색 목적에 맞는 정보 단위로 구조화**합니다.

---

# 2. Chunking / Embedding / Metadata 설계

숙소 상세정보의 특성에 맞춰 검색 단위를 설계합니다.

예를 들어 단순히 일정한 글자 수로 자르는 것이 아니라 정보의 의미를 보존하도록 구성합니다.

```text
숙소 설명
      ↓
숙소 기본정보 Chunk

객실 정보
      ↓
객실별 Chunk

편의시설
      ↓
Amenity Chunk

이용규칙
      ↓
Policy Chunk

주변 정보
      ↓
Location Chunk
```

각 Chunk에는 검색 및 필터링에 필요한 메타데이터를 함께 저장합니다.

```text
property_id
property_type
room_id
region
amenity_type
price_range
language
document_type
updated_at
```

이를 통해 단순한 의미 검색뿐 아니라 **숙소·지역·객실·정보 유형 단위의 정밀 검색**이 가능하도록 설계합니다.

---

# 3. 숙소 지식베이스 구축

숙소 상세정보를 Embedding하고 Vector DB에 적재하여 검색 가능한 Knowledge Base를 구축합니다.

```text
Property Data
      ↓
Document Parser
      ↓
Chunking
      ↓
Metadata Enrichment
      ↓
Embedding
      ↓
Vector DB
```

숙소 정보가 변경될 경우 변경된 데이터만 다시 Indexing할 수 있도록 구성합니다.

```text
숙소 정보 변경
      ↓
변경 감지
      ↓
해당 Property Chunk 재생성
      ↓
Embedding 갱신
      ↓
Vector DB Update
```

이를 통해 가격·편의시설·이용규칙 등의 변경사항이 검색 결과에 반영되도록 합니다.

---

# 4. Indexing Pipeline 구축

숙소 데이터의 초기 적재뿐 아니라 변경사항을 검색 시스템에 반영하는 Indexing Pipeline을 설계합니다.

```text
Property DB
    ↓
Extract
    ↓
Transform
 ┌───────────────┐
 │ Chunking      │
 │ Metadata      │
 │ Embedding     │
 └───────────────┘
    ↓
Load
    ↓
Vector DB
```

주요 처리 대상:

* 숙소 신규 등록
* 숙소 정보 수정
* 객실 변경
* 가격 변경
* 편의시설 변경
* 이용규칙 변경
* 숙소 삭제 / 비활성화

변경된 숙소만 재처리하여 불필요한 Embedding API 호출과 Indexing 비용을 줄입니다.

---

# 5. Hybrid Retrieval 설계

숙소 검색에서는 의미 기반 검색뿐 아니라 정확한 키워드 검색도 중요합니다.

예:

> "제주도에서 수영장이 있고 반려동물 동반 가능한 숙소"

```text
Query
 ↓
Query Analysis
 ↓
┌─────────────────┬─────────────────┐
│ Dense Search    │ BM25 Search     │
│ 의미 기반 검색    │ 키워드 기반 검색   │
└────────┬────────┴────────┬────────┘
         └─────────┬───────┘
                   ↓
                  RRF
                   ↓
                Reranking
                   ↓
             Relevant Context
```

Dense Retrieval과 BM25를 결합하여 의미적 유사성과 정확한 키워드 매칭을 동시에 확보합니다.

---

# 6. Metadata 기반 검색

숙소 데이터에서는 Metadata Filtering이 중요하기 때문에 Vector Search와 Metadata Filter를 함께 사용합니다.

예:

```text
Query:
"제주도 가족 여행에 좋은 숙소"

Filter:
region = "Jeju"
capacity >= 4
property_status = "active"
```

검색 구조:

```text
User Query
    ↓
Intent / Metadata Extraction
    ↓
Metadata Filtering
    ↓
Vector Search
    ↓
BM25 Search
    ↓
RRF
    ↓
Reranking
```

이를 통해 전체 숙소를 대상으로 검색하는 것보다 **업무적으로 의미 있는 후보군을 먼저 좁힌 뒤 관련 정보를 검색**할 수 있도록 구성합니다.

---

# 7. Marketing RAG Workflow

RAG의 최종 목적을 단순한 질문 응답이 아니라 **숙소 정보를 활용한 마케팅 콘텐츠 생성**으로 확장합니다.

예:

> "20대 커플을 대상으로 제주 바다 근처 숙소를 인스타그램 광고 문구로 만들어줘."

```text
Marketing Request
        ↓
Customer Segment Analysis
        ↓
Marketing Objective
        ↓
Property Filter
        ↓
Relevant Information Retrieval
        ↓
Reranking
        ↓
Context Compression
        ↓
Content Generation
        ↓
Fact Validation
        ↓
Marketing Content
```

---

# 8. 고객 세그먼트별 콘텐츠 생성

동일한 숙소 정보라도 고객 세그먼트에 따라 강조해야 할 정보가 달라지도록 설계합니다.

### 커플

```text
오션뷰
프라이빗 공간
감성적인 인테리어
주변 맛집
```

### 가족

```text
넓은 객실
수용 인원
주방
주차
아이 편의시설
```

### 비즈니스 여행객

```text
교통 접근성
Wi-Fi
업무 공간
체크인 편의성
```

RAG는 실제 숙소 데이터를 검색하고 LLM은 검색된 정보만을 기반으로 콘텐츠를 생성하도록 구성합니다.

---

# 9. Marketing Content Workflow

생성 목적에 따라 출력 형식을 다르게 구성합니다.

```text
숙소 정보
    ↓
RAG
    ↓
┌──────────────┬──────────────┬──────────────┐
│ SNS Content  │ Ad Copy      │ CRM Message  │
│              │              │              │
│ Instagram    │ Meta Ad      │ Email        │
│ Blog         │ Search Ad    │ Push         │
└──────────────┴──────────────┴──────────────┘
```

예:

```text
Input
"제주 가족 여행객 대상 광고 문구"

        ↓

RAG Retrieval

        ↓

숙소명
객실 크기
수용 인원
수영장
주차
제주 관광지 접근성

        ↓

LLM

        ↓

광고 카피
```

---

# 10. 정보 정합성 검증

Marketing 콘텐츠에서 가장 중요한 것은 **실제 숙소 정보와 생성 결과의 일치 여부**입니다.

예를 들어 숙소에 수영장이 없는데,

> "프라이빗 수영장에서 여유로운 시간을..."

과 같은 콘텐츠가 생성되어서는 안 됩니다.

따라서 생성 이후 Fact Validation 단계를 추가합니다.

```text
Retrieved Context
       ↓
Content Generation
       ↓
Fact Checker
       ↓
┌───────────────┬────────────────┐
│ Consistent    │ Inconsistent   │
│               │                │
│ 콘텐츠 반환    │ 재생성 / Reject │
└───────────────┴────────────────┘
```

검증 기준:

* 숙소명 일치
* 객실 정보 일치
* 가격 정보 일치
* 편의시설 존재 여부
* 이용규칙 일치
* 지역 정보 일치
* 정책 정보 일치

---

# Advanced RAG Architecture

기존 RAG 시스템을 단계적으로 고도화하여 검색 품질·추론·관측 가능성·자동 평가·실패 학습·병렬화·Agentic Retrieval까지 확장합니다.

```text
                         Client
                           │
                           ↓
                  Marketing Request
                           │
                           ↓
                  Query Understanding
                           │
                           ↓
                 Metadata / Intent
                           │
                           ↓
              ┌─────────────────────┐
              │ Retrieval Pipeline  │
              │                     │
              │ Dense Search        │
              │ BM25 Search         │
              │ Multi-Vector        │
              │ Query Routing       │
              │ Metadata Filtering  │
              └──────────┬──────────┘
                         ↓
                      Reranking
                         ↓
                 Context Compression
                         ↓
                 Multi-Hop / Self-RAG
                         ↓
                Marketing Generation
                         ↓
                   Fact Validation
                         ↓
                    Final Output
```

---

# 기존 RAG 시스템의 기술적 진화

각 버전은 실제 검색 및 생성 과정에서 발생하는 문제를 해결하는 방향으로 확장합니다.

## v2~v3 — API 기반 RAG

### 문제

로컬 Embedding 모델의 품질과 생성 모델 성능에 한계가 존재합니다.

### 해결

* OpenAI Embedding API
* GPT 기반 Generation
* Cosine Similarity 검색

### 효과

기본적인 검색 및 생성 품질 확보.

---

# v4~v5 — Query Rewrite + Semantic Chunking

### 문제

사용자의 질문을 그대로 검색하면 필요한 정보가 정확하게 검색되지 않고, 고정 길이 Chunking은 숙소 정보의 의미 단위를 분리할 수 있습니다.

### 해결

* Query Rewrite
* Overlap Chunking
* Semantic Chunking

### 숙박 플랫폼 적용

```text
"아이랑 제주도에서 갈 만한 숙소"

        ↓

"제주 지역 가족 단위 이용객에게 적합한
숙박시설 및 가족 편의시설"
```

검색 의도를 명확하게 변환합니다.

---

# v6~v8 — Multi-document Retrieval

### 문제

숙소 정보가 여러 데이터 영역에 분산되어 있습니다.

```text
Property
Room
Amenity
Policy
Location
```

단일 Chunk만 검색해서는 충분한 정보를 확보하기 어렵습니다.

### 해결

Multi-document Retrieval 및 Multi-document Chain을 적용합니다.

```text
숙소 기본정보
      +
객실정보
      +
편의시설
      +
지역정보
      ↓
통합 Context
      ↓
Marketing Content
```

---

# v9~v11 — Observability

### 문제

어떤 검색 결과가 콘텐츠 품질에 영향을 미쳤는지 확인하기 어렵습니다.

### 해결

* Retrieval latency
* Reranking latency
* Token usage
* Search score
* Generation latency
* LLM evaluation

등을 추적합니다.

이를 통해 RAG Pipeline을 **블랙박스가 아닌 관측 가능한 시스템**으로 전환합니다.

---

# v12~v14 — Hybrid Search + Routing

### 문제

Dense Search만 사용하면 숙소명·지역명·편의시설명과 같은 정확한 키워드 검색에서 Recall이 떨어질 수 있습니다.

### 해결

```text
Dense Search
      +
BM25
      ↓
RRF
      ↓
Reranking
```

또한 Query Routing을 통해 질문 유형에 따라 검색 전략을 다르게 적용합니다.

```text
숙소 정보 질문
→ Metadata + Dense

편의시설 질문
→ Keyword + Dense

지역 추천
→ Metadata + Dense

복합 마케팅 요청
→ Multi-step Retrieval
```

---

# v15~v17 — Quality Optimization

### Self-Refinement

```text
Draft
 ↓
Critique
 ↓
Refine
```

생성된 마케팅 콘텐츠를 다시 검토하여 품질을 개선합니다.

### Multi-Vector

```text
Chunk Vector
Sentence Vector
Keyword Vector
```

숙소 정보의 검색 단위를 세분화합니다.

### Context Compression

검색된 전체 정보를 그대로 LLM에 전달하지 않고 관련성이 낮은 문장을 제거하여 Context를 압축합니다.

---

# v18~v19 — Cache + Fallback

### 문제

동일한 숙소 정보에 대해 반복적인 요청이 발생할 경우 API 비용과 응답 지연이 증가합니다.

### 해결

```text
Query
 ↓
Cache Check
 ↓
Hit → 기존 결과 반환
 ↓
Miss
 ↓
RAG Retrieval
```

검색 품질이 일정 기준 이하인 경우 자동으로 재검색합니다.

```text
Retrieval
 ↓
Quality Check
 ↓
Low Score
 ↓
Query Rewrite
 ↓
Re-retrieval
```

---

# v20 — Failure Learning

검색이나 생성 과정에서 발생한 실패를 자동으로 수집합니다.

```text
Failure
 ↓
Classification
 ↓
Failure Dataset
 ↓
JSONL
 ↓
Analysis / Fine-tuning
```

주요 실패 유형:

```text
low_accuracy
hallucination
incomplete_answer
retrieval_failure
low_relevance
```

숙소 정보 기반 콘텐츠 생성에서 어떤 정보 유형이 반복적으로 실패하는지 분석할 수 있습니다.

---

# v21~v22 — Parallel Retrieval + Production

여러 검색 채널을 병렬 실행하여 검색 지연시간을 줄입니다.

```text
              Query
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
     Dense    BM25    Keyword
       │        │        │
       └────────┼────────┘
                ↓
               RRF
```

또한:

* JWT Authentication
* Rate Limiting
* P50 / P95 / P99 Latency
* Monitoring
* Alert

등을 추가하여 서비스 운영 환경을 고려합니다.

---

# v23 — Architecture Separation

RAG Engine과 API, Client를 분리합니다.

```text
rag_engine.py
     │
     │
server_api.py
     │
     │
client_app.py
```

| 파일              | 역할                      |
| --------------- | ----------------------- |
| `rag_engine.py` | 검색·Reranking·Generation |
| `server_api.py` | FastAPI API             |
| `client_app.py` | Streamlit Client        |

각 계층의 책임을 분리하여 독립적인 테스트와 유지보수가 가능하도록 구성합니다.

---

# v24 — Config + Service Layer

환경 설정과 API Router를 분리합니다.

```text
config.py
deps.py

routers/
├── auth.py
├── properties.py
├── search.py
├── marketing.py
└── metrics.py
```

숙소·검색·마케팅 콘텐츠 생성 등 도메인별 API 책임을 분리합니다.

---

# v25 — Agentic Retrieval

### 문제

복합적인 마케팅 요청은 단일 검색으로 처리하기 어렵습니다.

예:

> "제주도에서 4인 가족이 머물기 좋고 수영장이 있는 숙소를 찾아서 인스타그램 광고 문구를 만들어줘."

### 해결

질문을 여러 작업으로 분해합니다.

```text
User Request
      ↓
Task Decomposition
      ↓
┌─────────────────────────┐
│ 1. 지역 = 제주            │
│ 2. 수용인원 >= 4          │
│ 3. 수영장 존재             │
│ 4. 숙소 정보 검색           │
│ 5. 마케팅 콘텐츠 생성       │
└────────────┬────────────┘
             ↓
       Retrieval
             ↓
       Content Generation
```

Self-RAG를 통해 검색 결과가 충분한지 판단하고 부족할 경우 검색을 반복합니다.

---

# v26 — Streaming + RAGAS Evaluation

### Streaming

마케팅 콘텐츠 생성 결과를 SSE를 통해 실시간으로 전달합니다.

```text
Request
 ↓
Retrieval
 ↓
Generation
 ↓
Token Streaming
 ↓
Client
```

### RAGAS

RAG 시스템의 품질을 정량적으로 평가합니다.

* Faithfulness
* Answer Relevancy
* Context Precision

이를 통해 버전별 검색·생성 품질을 비교할 수 있습니다.

---

# Airbnb형 플랫폼 적용 Architecture

최종적으로 RAG-Marketing은 Airbnb형 플랫폼의 Marketing Team에서 다음과 같이 활용됩니다.

```text
                    Airbnb Platform
                           │
                    Property Database
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
     Property            Room             Amenity
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                    Indexing Pipeline
                           ↓
                     Vector DB
                           ↓
                    RAG Retrieval
                           ↓
                Customer / Marketing Intent
                           ↓
                    Content Generation
                           ↓
                    Fact Validation
                           ↓
              ┌────────────┼────────────┐
              ↓            ↓            ↓
           SNS Copy     Ad Copy      CRM Copy
```

---

# 전체 플랫폼과의 연결

RAG-Marketing은 Airbnb형 숙박 플랫폼에서 **Marketing Team의 AI 업무 자동화**를 담당합니다.

```text
                         Airbnb Platform
                                │
          ┌─────────────────────┼─────────────────────┐
          ↓                     ↓                     ↓
       Product               Marketing             Growth
          │                     │                     │
    ML-Product             RAG-Marketing          Data-Growth
          │                     │                     │
     수요 예측             숙소 정보 활용          CRO Analytics
     예약 수요             콘텐츠 생성             Dashboard
          │                     │                     │
          └─────────────────────┼─────────────────────┘
                                ↓
                         Customer Support
                                │
                      Agent-Customer-Support
                                │
                         CS 업무 자동화
```

### Product

```text
숙소·지역별 예약 수요
        ↓
ML
        ↓
수요 예측
```

### Marketing

```text
숙소 상세정보
        ↓
RAG
        ↓
고객 세그먼트별 콘텐츠
```

### Growth

```text
고객 행동 데이터
        ↓
Analytics
        ↓
CRO
        ↓
전환율 개선
```

### Customer Support

```text
고객 문의
        ↓
Agent
        ↓
예약 / 숙소 / 정책 조회
        ↓
취소 / 환불 / 문의 처리
```

---

# 기술 스택

### AI / RAG

* Python
* OpenAI API
* Embedding
* FAISS
* BM25
* Hybrid Search
* Reranking
* Multi-Vector Retrieval
* Query Routing
* Multi-Hop Retrieval
* Self-RAG
* RAGAS

### Backend

* FastAPI
* Pydantic
* JWT
* REST API
* SSE

### Frontend

* Streamlit
* HTML / CSS

### Data

* PostgreSQL
* Vector DB
* JSONL
* Metadata Schema

---

# 프로젝트 구조

```text
RAG-Marketing/
│
├── rag_v23/
│   ├── rag_engine.py
│   ├── server_api.py
│   └── client_app.py
│
├── rag_v24/
│   ├── config.py
│   ├── deps.py
│   ├── rag_engine.py
│   ├── server_api.py
│   ├── client_app.py
│   └── routers/
│
├── rag_v25/
│   └── ...
│
├── rag_v26/
│   ├── evaluate_ragas.py
│   └── ...
│
├── rag_versions/
├── change_logs/
├── docs/
│   ├── property-schema.md
│   ├── metadata-schema.md
│   ├── retrieval-strategy.md
│   ├── marketing-workflow.md
│   └── evaluation.md
│
└── requirements.txt
```

---

# 핵심 설계 원칙

## SSoT

숙소 정보의 원천을 명확하게 정의하고 RAG가 존재하지 않는 숙소 정보를 생성하지 않도록 합니다.

```text
Property DB
    ↓
Knowledge Base
    ↓
RAG
    ↓
Content
```

---

## Metadata First

숙소·객실·지역·편의시설 등 도메인 정보를 Metadata로 구조화하여 검색 정확도를 높입니다.

---

## Retrieval Before Generation

LLM이 기억이나 추론만으로 숙소 정보를 생성하지 않고 필요한 정보를 먼저 검색한 뒤 콘텐츠를 생성합니다.

---

## Fact Consistency

생성된 마케팅 콘텐츠가 실제 숙소 정보와 일치하는지 검증합니다.

---

## Failure Handling

검색 결과가 부족하거나 신뢰할 수 없는 경우 임의의 콘텐츠를 생성하지 않고 재검색·재생성 또는 실패 상태를 반환합니다.

---

## Evaluation Driven

검색 및 생성 품질을 정량적으로 평가하고 실험 결과를 기반으로 Chunking·Retrieval·Prompt 전략을 개선합니다.

---

# 기대 효과

기존의 단순한 RAG Q&A 시스템을 다음과 같은 **Marketing Intelligence Pipeline**으로 확장합니다.

```text
숙소 정보
   ↓
구조화
   ↓
Indexing
   ↓
Retrieval
   ↓
고객 세그먼트 분석
   ↓
Marketing Intent
   ↓
Content Generation
   ↓
Fact Validation
   ↓
Marketing Asset
```

이를 통해 Marketing Team이 숙소 정보를 직접 확인하고 콘텐츠를 작성하는 반복 작업을 줄이고, **실제 숙소 데이터를 근거로 고객 세그먼트별 마케팅 콘텐츠를 생성하는 AI 기반 업무 자동화 구조**를 구현합니다.

---

# 프로젝트 한 줄 요약

> **Airbnb형 숙박 플랫폼의 숙소 상세정보를 구조화하고 RAG 기반으로 검색·조합하여 고객 세그먼트와 마케팅 목적에 맞는 콘텐츠를 생성하는 End-to-End RAG Marketing 시스템**
