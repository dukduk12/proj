import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta, date
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from src.config import settings
from src.logging_config import setup_logger
from src.email_client import fetch_emails_list, download_pdf_for_email
from src.pdf_parser import extract_text_from_pdf
from src.summarizer import summarize_text
from src.word_cloud_gen import generate_word_cloud
from src.db_client import ChromaClient
from src.tfidf_analyzer import extract_tfidf_keywords, generate_tfidf_chart
from src.network_viz import build_and_render_network
from src.ngram_analyzer import (
    extract_frequency, extract_ngrams,
    generate_frequency_chart, generate_ngram_chart,
)

setup_logger()

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="업무 메일 PDF 분석기",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: Deep Navy + Clean White 디자인 시스템 ────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 쿨 그레이 */
[data-testid="stAppViewContainer"] {
    background-color: #F1F5F9;
}
[data-testid="stHeader"] {
    background-color: #F1F5F9;
}

/* 카드 컴포넌트 */
.card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border: 1px solid #E2E8F0;
}

/* 섹션 배지 (STEP 번호) */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: #1D4ED8;
    color: white;
    border-radius: 50%;
    font-size: 13px;
    font-weight: 700;
    margin-right: 10px;
    vertical-align: middle;
}
.step-title {
    font-size: 18px;
    font-weight: 700;
    color: #1E293B;
    vertical-align: middle;
}

/* 헤더 히어로 카드 */
.hero-card {
    background: linear-gradient(135deg, #1D4ED8 0%, #0EA5E9 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 28px;
    color: white;
}
.hero-card h1 { color: white; margin: 0 0 6px 0; font-size: 28px; }
.hero-card p  { color: rgba(255,255,255,0.82); margin: 0; font-size: 14px; }

/* 메일 결과 카드 */
.email-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border-left: 4px solid #1D4ED8;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
}
.email-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 4px;
}
.email-card-meta {
    font-size: 12px;
    color: #64748B;
    margin-bottom: 0;
}

/* 캐시됨 뱃지 */
.badge-cached {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 99px;
    margin-left: 10px;
    vertical-align: middle;
}

/* 구분선 */
.section-divider {
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 24px 0;
}

/* Streamlit 기본 버튼 포인트 컬러 오버라이드 */
[data-testid="stButton"] > button[kind="primary"] {
    background-color: #1D4ED8;
    border-color: #1D4ED8;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #1E40AF;
    border-color: #1E40AF;
}
</style>
""", unsafe_allow_html=True)

# ── 히어로 헤더 ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
  <h1>📧 업무 메일 PDF 분석기</h1>
  <p>기간을 설정하고 말머리를 필터링해 메일을 선택하면, 요약·워드클라우드·TF-IDF·N-gram·관계 네트워크를 자동으로 분석합니다.</p>
</div>
""", unsafe_allow_html=True)

# ── 상태 관리 ─────────────────────────────────────────────────────────────────
PROCESSED_FILE = settings.output_dir / "processed_emails.json"


def load_processed_data():
    if PROCESSED_FILE.exists():
        try:
            with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}


def save_processed_data(msg_id, results):
    data = load_processed_data()
    serialized_results = []
    for item in results:
        new_item = item.copy()
        if 'wc_path' in new_item and new_item['wc_path']:
            new_item['wc_path'] = str(new_item['wc_path'])
        if 'tfidf_path' in new_item and new_item['tfidf_path']:
            new_item['tfidf_path'] = str(new_item['tfidf_path'])
        serialized_results.append(new_item)
    data[msg_id] = serialized_results
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if "email_list" not in st.session_state:
    st.session_state.email_list = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "unique_tags" not in st.session_state:
    st.session_state.unique_tags = []
if "selected_tags" not in st.session_state:
    st.session_state.selected_tags = []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — 기간 설정 및 메일 목록 조회
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<span class="step-badge">1</span><span class="step-title">기간 설정 및 메일 목록 조회</span>', unsafe_allow_html=True)

with st.container():
    # 날짜 2개 + 버튼을 한 행으로 배치 — 버튼이 날짜 바로 옆에 있어 연관성 명확
    col_s, col_e, col_btn = st.columns([2, 2, 1])
    with col_s:
        start_date = st.date_input("시작일", value=date.today() - timedelta(days=7))
    with col_e:
        end_date = st.date_input("종료일", value=date.today())
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)  # 라벨 높이 맞춤
        fetch_clicked = st.button("목록 조회", type="primary", use_container_width=True)

if fetch_clicked:
    with st.spinner("메일 목록을 가져오는 중..."):
        all_emails = fetch_emails_list(start_date, end_date)
        st.session_state.email_list = all_emails
        st.session_state.analysis_results = {}

        tags = set()
        for e in all_emails:
            for t in e.get("tags", []):
                tags.add(t)
        st.session_state.unique_tags = sorted(list(tags))

        if "[업무 협조]" in st.session_state.unique_tags:
            st.session_state.selected_tags = ["[업무 협조]"]
        else:
            st.session_state.selected_tags = st.session_state.unique_tags[:1] if st.session_state.unique_tags else []

    if not st.session_state.email_list:
        st.warning("해당 기간에 말머리([...])가 포함된 메일이 없습니다.")
    else:
        st.success(f"말머리가 있는 **{len(st.session_state.email_list)}개**의 메일을 찾았습니다.")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 메일 선택 및 필터
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.email_list is not None and len(st.session_state.email_list) > 0:

    st.markdown('<span class="step-badge">2</span><span class="step-title">메일 선택 및 분석</span>', unsafe_allow_html=True)
    st.caption("분석할 메일을 테이블에서 선택한 뒤 **선택한 메일 분석하기** 버튼을 클릭하세요.")

    # 필터와 체크박스를 가로 배치 — 같은 '필터' 성격임을 시각적으로 그룹화
    col_filter, col_check = st.columns([3, 1])
    with col_filter:
        st.session_state.selected_tags = st.multiselect(
            "말머리 필터",
            options=st.session_state.unique_tags,
            default=st.session_state.selected_tags,
        )
    with col_check:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        show_only_pdf = st.checkbox("PDF 첨부만 보기", value=True)

    filtered_emails = []
    for e in st.session_state.email_list:
        if any(tag in st.session_state.selected_tags for tag in e.get("tags", [])):
            if show_only_pdf and not e.get("has_attachment", False):
                continue
            filtered_emails.append(e)

    if not filtered_emails:
        st.info("조건에 맞는 메일이 없습니다.")
    else:
        df_data = []
        for e in filtered_emails:
            df_data.append({
                "선택": False,
                "id": e["id"],
                "날짜": e["date"],
                "보낸사람": e["sender"],
                "제목": e["subject"],
                "PDF 개수": e.get("pdf_count", 0),
                "첨부파일명": ", ".join(e.get("pdf_names", [])),
                "본문 미리보기": e.get("body_snippet", ""),
            })

        df = pd.DataFrame(df_data)
        edited_df = st.data_editor(
            df,
            column_config={
                "선택": st.column_config.CheckboxColumn("선택", default=False),
                "id": None,
            },
            disabled=["날짜", "보낸사람", "제목", "PDF 개수", "첨부파일명", "본문 미리보기"],
            hide_index=True,
            use_container_width=True,
        )

        selected_ids = edited_df[edited_df["선택"] == True]["id"].tolist()

        # 버튼을 오른쪽 끝 정렬 — 테이블 액션임을 명확히
        _, col_analyze = st.columns([4, 1])
        with col_analyze:
            if st.button("선택한 메일 분석하기", type="primary",
                         disabled=len(selected_ids) == 0, use_container_width=True):
                st.session_state.analysis_triggered = True
                st.session_state.selected_email_ids = selected_ids

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — 개별 메일 분석 결과
    # ══════════════════════════════════════════════════════════════════════════
    if getattr(st.session_state, 'analysis_triggered', False):

        st.markdown('<span class="step-badge">3</span><span class="step-title">개별 메일 분석 결과</span>', unsafe_allow_html=True)

        processed_data = load_processed_data()
        selected_emails = [e for e in filtered_emails if e["id"] in st.session_state.selected_email_ids]

        for email_meta in selected_emails:
            e_id = email_meta['id']
            msg_id = email_meta.get('message_id', e_id)
            has_att = email_meta.get('has_attachment', False)
            is_cached = msg_id in processed_data

            # 메일 카드 헤더 — 제목·메타·캐시뱃지를 하나의 카드로 묶음
            cached_badge = '<span class="badge-cached">✓ 캐시됨</span>' if is_cached else ""
            st.markdown(f"""
            <div class="email-card">
              <div class="email-card-title">
                {"📎 " if has_att else ""}📄 {email_meta['subject']}{cached_badge}
              </div>
              <div class="email-card-meta">
                📅 {email_meta['date']} &nbsp;|&nbsp; 👤 {email_meta['sender']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if is_cached:
                results = processed_data[msg_id]

                # 요약·워드클라우드·TF-IDF를 탭으로 묶어 세로 스크롤 절감
                tab_sum, tab_wc, tab_tfidf, tab_raw = st.tabs(
                    ["💡 요약", "☁️ 워드클라우드", "📊 TF-IDF", "📄 원본 텍스트"]
                )
                for item in results:
                    if "error" in item:
                        st.warning(f"**{item.get('file', '문서')}**: {item['error']}")
                        continue

                    with tab_sum:
                        if item.get('file'):
                            st.caption(f"파일: `{item['file']}`")
                        st.write(item['summary'])

                    with tab_wc:
                        if item.get('wc_path') and Path(item['wc_path']).exists():
                            st.image(item['wc_path'], caption=item.get('file', ''), use_container_width=True)
                        else:
                            st.info("워드클라우드 이미지가 없습니다.")

                    with tab_tfidf:
                        if item.get('tfidf_path') and Path(item['tfidf_path']).exists():
                            st.image(item['tfidf_path'], caption=item.get('file', ''), use_container_width=True)
                        else:
                            st.info("TF-IDF 차트가 없습니다.")

                    with tab_raw:
                        st.text(item.get('text', ''))

            else:
                with st.spinner("해당 메일의 데이터를 다운로드하고 분석하는 중..."):
                    results = []
                    if has_att:
                        pdf_paths = download_pdf_for_email(e_id)
                        if not pdf_paths:
                            results.append({"error": "이 메일에는 PDF 첨부파일이 없습니다."})
                        else:
                            for pdf_path in pdf_paths:
                                text = extract_text_from_pdf(pdf_path)
                                if not text.strip():
                                    results.append({"file": pdf_path.name, "error": "텍스트를 추출할 수 없습니다."})
                                    continue

                                summary = summarize_text(text)

                                wc_filename = f"wordcloud_{e_id}_{pdf_path.name}.png"
                                wc_path = generate_word_cloud(text, output_filename=wc_filename)

                                tfidf_filename = f"tfidf_{e_id}_{pdf_path.name}.png"
                                keywords_dict = extract_tfidf_keywords({pdf_path.name: text})
                                keywords = keywords_dict.get(pdf_path.name, [])
                                tfidf_path = generate_tfidf_chart(keywords, title=pdf_path.name, output_filename=tfidf_filename)

                                results.append({
                                    "file": pdf_path.name,
                                    "text": text,
                                    "summary": summary,
                                    "wc_path": wc_path,
                                    "tfidf_path": tfidf_path,
                                })
                    else:
                        text = email_meta.get("body_snippet", "")
                        if not text:
                            results.append({"error": "분석할 텍스트가 없습니다."})
                        else:
                            summary = summarize_text(text)
                            results.append({
                                "file": "메일 본문",
                                "text": text,
                                "summary": summary,
                                "wc_path": "",
                            })

                    if results:
                        save_processed_data(msg_id, results)
                        st.rerun()

            # 유사 메일 추천 — expander로 접어서 카드 높이 절감
            if is_cached:
                results = processed_data[msg_id]
                if results and 'summary' in results[0]:
                    with st.expander("💡 과거 유사 업무 메일 추천 보기"):
                        search_text = f"제목: {email_meta['subject']}\n요약: {results[0]['summary']}"
                        with st.spinner("유사한 과거 메일을 검색하는 중..."):
                            chroma_client = ChromaClient()
                            similar_emails = chroma_client.query_similar(search_text, n_results=3)

                        has_recommendation = False
                        if similar_emails:
                            for sim in similar_emails:
                                has_recommendation = True
                                distance = sim.get('distance', 0.0)
                                similarity = max(0, 100 - (distance * 100))
                                st.info(
                                    f"**{sim['metadata'].get('title', '제목 없음')}** (유사도: {similarity:.1f}%)\n\n"
                                    f"📅 *날짜:* {sim['metadata'].get('date', '')} | 👤 *보낸사람:* {sim['metadata'].get('sender', '')}\n\n"
                                    f"{sim['metadata'].get('summary', '')}"
                                )
                        if not has_recommendation:
                            st.write("유사한 과거 메일이 없습니다.")

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


        # ══════════════════════════════════════════════════════════════════════
        # STEP 4 — 전체 비교 분석 (TF-IDF · 빈도·N-gram · 네트워크)
        # ══════════════════════════════════════════════════════════════════════
        all_texts = {}
        sender_docs_net = {}
        for em in selected_emails:
            mid = em.get("message_id", em["id"])
            for item in processed_data.get(mid, []):
                if item.get("text") and "error" not in item:
                    key = f"{em['subject'][:25]} / {item.get('file', '본문')}"
                    all_texts[key] = item["text"]
                    sender_docs_net.setdefault(em["sender"], []).append({
                        "file": item.get("file", "본문"),
                        "text": item["text"],
                        "email_subject": em["subject"],
                        "email_date": em["date"],
                    })

        if all_texts:
            st.markdown('<span class="step-badge">4</span><span class="step-title">전체 비교 분석</span>', unsafe_allow_html=True)
            st.caption(f"선택된 {len(all_texts)}개 문서 기반 비교 분석")

            tfidf_results = extract_tfidf_keywords(all_texts)
            combined_text = " ".join(all_texts.values())

            # 4개 분석 섹션을 탭 하나로 통합 — 페이지 세로 스크롤 대폭 절감
            tab_tfidf, tab_freq, tab_bi, tab_tri, tab_net = st.tabs([
                "📊 TF-IDF 비교", "📈 단어 빈도", "2-gram", "3-gram", "🕸️ 관계 네트워크"
            ])

            with tab_tfidf:
                if tfidf_results:
                    st.caption(f"{len(tfidf_results)}개 문서 동시 비교 — 각 문서에서만 특징적인 단어 추출")
                    cols = st.columns(min(len(tfidf_results), 2))
                    for idx, (doc_name, keywords) in enumerate(tfidf_results.items()):
                        with cols[idx % 2]:
                            safe = "".join(c if c.isalnum() else "_" for c in doc_name[:20])
                            chart_path = generate_tfidf_chart(
                                keywords, title=doc_name,
                                output_filename=f"tfidf_cmp_{idx}_{safe}.png",
                            )
                            if chart_path and Path(chart_path).exists():
                                st.image(str(chart_path), caption=doc_name, use_container_width=True)
                else:
                    st.info("TF-IDF를 계산할 텍스트가 부족합니다.")

            with tab_freq:
                freq = extract_frequency(combined_text, top_n=30)
                if freq:
                    col_tbl, col_chart = st.columns([1, 2])
                    with col_tbl:
                        st.dataframe(
                            pd.DataFrame(freq, columns=["키워드", "빈도"]),
                            use_container_width=True, hide_index=True,
                        )
                    with col_chart:
                        chart_path = generate_frequency_chart(
                            freq[:20], title="상위 20 키워드 빈도",
                            output_filename="freq_all.png",
                        )
                        if chart_path and Path(chart_path).exists():
                            st.image(str(chart_path), use_container_width=True)

            with tab_bi:
                bigrams = extract_ngrams(combined_text, n=2, top_n=20)
                if bigrams:
                    col_tbl, col_chart = st.columns([1, 2])
                    with col_tbl:
                        st.dataframe(
                            pd.DataFrame(bigrams, columns=["2-gram", "빈도"]),
                            use_container_width=True, hide_index=True,
                        )
                    with col_chart:
                        chart_path = generate_ngram_chart(
                            bigrams, n=2,
                            output_filename="ngram2_all.png", color="#DD8452",
                        )
                        if chart_path and Path(chart_path).exists():
                            st.image(str(chart_path), use_container_width=True)
                else:
                    st.info("2-gram을 추출할 텍스트가 부족합니다.")

            with tab_tri:
                trigrams = extract_ngrams(combined_text, n=3, top_n=20)
                if trigrams:
                    col_tbl, col_chart = st.columns([1, 2])
                    with col_tbl:
                        st.dataframe(
                            pd.DataFrame(trigrams, columns=["3-gram", "빈도"]),
                            use_container_width=True, hide_index=True,
                        )
                    with col_chart:
                        chart_path = generate_ngram_chart(
                            trigrams, n=3,
                            output_filename="ngram3_all.png", color="#55A868",
                        )
                        if chart_path and Path(chart_path).exists():
                            st.image(str(chart_path), use_container_width=True)
                else:
                    st.info("3-gram을 추출할 텍스트가 부족합니다.")

            with tab_net:
                if sender_docs_net:
                    st.caption("● 발신자  ■ PDF — 점선은 내용 유사도, 색깔은 업무 클러스터 | 첫 실행 시 모델 다운로드 ~400MB")
                    with st.spinner("임베딩 계산 중... (문서 수에 따라 10~30초)"):
                        net_fig = build_and_render_network(sender_docs_net, tfidf_results)
                    if net_fig:
                        st.plotly_chart(net_fig, use_container_width=True)
                    else:
                        st.info("네트워크를 생성하기에 데이터가 부족합니다.")
                else:
                    st.info("네트워크를 생성하기에 데이터가 부족합니다.")
