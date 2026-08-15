"""Streamlit 클라이언트 — 생성물과 **검증 결과를 같이** 본다.

문구만 보여주는 화면은 이 프로젝트의 요점을 가린다. 여기서 확인해야 하는 것은
"문장이 그럴듯한가"가 아니라 **"이 주장이 원본 필드에서 왔는가"** 다.
그래서 통과·위반을 문구 옆에 나란히 놓는다.

    streamlit run api/client.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

SEGMENTS = ["COUPLE", "FAMILY", "BUSINESS"]
FORMATS = ["SNS", "AD_COPY", "CRM"]

st.set_page_config(page_title="숙소 콘텐츠 생성", layout="wide")
st.title("숙소 콘텐츠 생성")
st.caption("생성물의 모든 주장을 원본 레코드와 대조한다. 통과하지 못하면 내보내지 않는다.")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=60, **kw)
        return r.status_code, r.json()
    except requests.RequestException as e:
        return 0, {"detail": f"API 에 연결할 수 없습니다 ({API}) — {e}"}


# ─────────────────────────────────────────────────── 사이드바: 색인·지표
with st.sidebar:
    st.subheader("색인")
    count = st.number_input("숙소 수", 10, 500, 100, step=10)
    if st.button("색인 만들기", use_container_width=True):
        code, body = api("POST", "/index", json={"count": int(count)})
        if code == 200:
            st.success(f"숙소 {body['properties']}개 · 임베딩 {body['embedded']}건")
            st.caption("같은 내용으로 다시 누르면 임베딩 건수가 줄어든다 — 증분 색인이다.")
        else:
            st.error(body.get("detail", body))

    code, m = api("GET", "/metrics")
    if code == 200:
        st.subheader("지표")
        st.metric("청크", m["chunks"])
        st.metric("숙소", m["properties"])
        st.caption(f"생성 백엔드: `{m['backend']}`")
        st.caption(
            f"기권 임계 score ≥ {m['thresholds']['min_score']} · "
            f"margin ≥ {m['thresholds']['min_margin']}"
        )

tab_search, tab_generate = st.tabs(["검색", "생성"])

# ─────────────────────────────────────────────────── 검색
with tab_search:
    q = st.text_input("질의", "수영장 있는 제주 숙소")
    c1, c2, c3 = st.columns(3)
    region = c1.text_input("지역", "")
    cap = c2.number_input("최소 인원", 0, 20, 0)
    price = c3.number_input("최대 가격", 0, 1_000_000, 0, step=10_000)

    if st.button("검색", use_container_width=True):
        payload = {"query": q, "top_k": 8}
        if region:
            payload["region"] = region
        if cap:
            payload["min_capacity"] = int(cap)
        if price:
            payload["max_price"] = int(price)
        code, body = api("POST", "/search", json=payload)

        if code != 200:
            st.error(body.get("detail", body))
        else:
            a, b, c = st.columns(3)
            a.metric("필터 전 후보", body["candidates_before_filter"])
            b.metric("필터 후 후보", body["candidates_after_filter"])
            c.metric("축소율", f"{body['filter_reduction'] * 100:.1f}%")
            st.caption("비싼 단계(벡터 검색)에 넘어가는 양이 이만큼 줄었다.")

            if body["grounded"]:
                st.success("근거 충분")
            else:
                st.warning(f"기권 — {body['reason']}")
                st.caption("검색은 언제나 뭔가를 돌려준다. 쓸 만한지는 따로 판정한다.")

            for h in body["hits"]:
                with st.expander(f"{h['property_id']} · {h['doc_type']} · {h['score']:.4f}"):
                    st.write(h["text"])

# ─────────────────────────────────────────────────── 생성
with tab_generate:
    pid = st.text_input("숙소 ID", "P0001")
    c1, c2 = st.columns(2)
    seg = c1.selectbox("세그먼트", SEGMENTS)
    fmt = c2.selectbox("형식", FORMATS)

    if st.button("문구 생성", use_container_width=True):
        code, body = api("POST", "/generate",
                         json={"property_id": pid, "segment": seg, "format": fmt})

        # 검증 실패는 422 로 오고, 본문에 위반 내역이 들어 있다
        detail = body.get("detail") if code == 422 else body

        if code == 404:
            st.error(body.get("detail"))
        elif code not in (200, 422):
            st.error(body.get("detail", body))
        else:
            left, right = st.columns([3, 2])
            with left:
                st.text_area("생성물", detail["content"], height=220)
            with right:
                if detail["valid"]:
                    st.success(f"검증 통과 · 시도 {detail['attempts']}회")
                    st.caption(f"백엔드: `{detail['backend']}`")
                else:
                    st.error(f"거부 — 위반 {len(detail['violations'])}건")
                    for v in detail["violations"]:
                        st.write(f"- **{v['type']}** — {v['detail']}")
                    st.caption("재생성 후에도 어긋나면 내보내지 않는다.")
