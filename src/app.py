import streamlit as st
from pathlib import Path
from datetime import timedelta, date
import sys
import json
from collections import OrderedDict

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

# ── 우선순위 설정 ─────────────────────────────────────────────────────────────
PRIORITY_CONFIG = OrderedDict([
    ("긴급",    {"emoji": "🚨", "color": "#DC2626", "bg": "#FEE2E2", "text": "#991B1B"}),
    ("필독",    {"emoji": "📌", "color": "#EA580C", "bg": "#FFEDD5", "text": "#9A3412"}),
    ("보고",    {"emoji": "📋", "color": "#2563EB", "bg": "#DBEAFE", "text": "#1E40AF"}),
    ("요청",    {"emoji": "📤", "color": "#7C3AED", "bg": "#EDE9FE", "text": "#5B21B6"}),
    ("공유",    {"emoji": "🔗", "color": "#0D9488", "bg": "#CCFBF1", "text": "#115E59"}),
])
_DEFAULT_PRIORITY = {"emoji": "📧", "color": "#1D4ED8", "bg": "#DBEAFE", "text": "#1E40AF"}


def _priority_of(email: dict) -> dict:
    tags_str = " ".join(email.get("tags", []))
    for key, cfg in PRIORITY_CONFIG.items():
        if key in tags_str:
            return {"key": key, **cfg}
    return {"key": "기타", **_DEFAULT_PRIORITY}


# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="업무 메일 PDF 분석기",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 글로벌 CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] { background: #F1F5F9; }
[data-testid="stHeader"]           { background: #F1F5F9; }
[data-testid="stMainBlockContainer"] { padding-top: 24px; }

/* ── 히어로 ── */
.hero {
    background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 50%, #0EA5E9 100%);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "📧";
    position: absolute;
    right: 40px; top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.12;
}
.hero h1 { color: #fff; margin: 0 0 8px; font-size: 26px; font-weight: 700; letter-spacing: -0.5px; }
.hero p  { color: rgba(255,255,255,0.78); margin: 0; font-size: 13.5px; line-height: 1.6; }

/* ── STEP 배지 ── */
.step-row { display: flex; align-items: center; gap: 10px; margin: 28px 0 14px; }
.step-badge {
    width: 26px; height: 26px;
    background: #1D4ED8;
    color: #fff;
    border-radius: 50%;
    font-size: 12px; font-weight: 700;
    display: inline-flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.step-title { font-size: 16px; font-weight: 700; color: #1E293B; }

/* ── 구분선 ── */
.divider { border: none; border-top: 1px solid #E2E8F0; margin: 28px 0; }

/* ── 우선순위 탭 라벨 안 count 칩 ── */
.cnt {
    display: inline-block;
    background: rgba(0,0,0,0.08);
    border-radius: 99px;
    padding: 1px 8px;
    font-size: 11px;
    font-weight: 700;
    margin-left: 4px;
}

/* ── 우선순위 인라인 배지 (테이블·카드용) ── */
.pri-badge {
    display: inline-block;
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 11px; font-weight: 700;
    white-space: nowrap;
}

/* ── 메일 분석 카드 ── */
.mail-card {
    background: #fff;
    border-radius: 14px;
    padding: 18px 22px 14px;
    margin-bottom: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border-top: 3px solid #1D4ED8;   /* JS로 per-email 색상 오버라이드 불가 → CSS var 활용 */
}
.mail-card-title { font-size: 15px; font-weight: 700; color: #0F172A; margin-bottom: 5px; }
.mail-card-meta  { font-size: 12px; color: #64748B; }

/* 캐시 배지 */
.badge-ok {
    display: inline-block;
    background: #DCFCE7; color: #166534;
    font-size: 10px; font-weight: 700;
    padding: 2px 9px; border-radius: 99px;
    margin-left: 8px; vertical-align: middle;
}

/* ── Streamlit 탭 강화 ── */
[data-testid="stTabs"] > div:first-child {
    background: #fff;
    border-radius: 12px 12px 0 0;
    padding: 0 8px;
    border-bottom: 2px solid #E2E8F0;
    gap: 2px;
}
[data-testid="stTabs"] button[role="tab"] {
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 16px !important;
    border-radius: 8px 8px 0 0 !important;
    color: #64748B !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.15s ease !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: #1D4ED8 !important; background: #F8FAFC !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #1D4ED8 !important;
    border-bottom-color: #1D4ED8 !important;
    background: #F0F7FF !important;
}
[data-testid="stTabsContent"] {
    background: #fff;
    border-radius: 0 0 12px 12px;
    border: 1px solid #E2E8F0;
    border-top: none;
    padding: 20px 20px 16px;
}

/* ── 버튼 ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #1D4ED8 !important; border-color: #1D4ED8 !important;
    border-radius: 8px !important; font-weight: 600 !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1E40AF !important; border-color: #1E40AF !important;
}

/* ── data_editor 헤더 ── */
[data-testid="stDataFrameResizable"] thead th {
    background: #F8FAFC !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    color: #475569 !important;
}

/* ── 분석 전체 탭 (STEP 4) ── */
.analysis-tab-wrap [data-testid="stTabs"] > div:first-child { background: #F8FAFC; }
</style>
""", unsafe_allow_html=True)


# ── 히어로 헤더 ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>업무 메일 PDF 분석기</h1>
  <p>기간을 설정하고 말머리 탭을 선택해 메일을 고르면<br>
     요약 · 워드클라우드 · TF-IDF · N-gram · 관계 네트워크를 자동으로 분석합니다.</p>
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


for _k, _v in [
    ("email_list", None), ("analysis_results", {}),
    ("unique_tags", []), ("selected_tags", []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — 기간 설정 및 메일 목록 조회
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="step-row"><span class="step-badge">1</span><span class="step-title">기간 설정 및 메일 목록 조회</span></div>', unsafe_allow_html=True)

col_s, col_e, col_btn = st.columns([2, 2, 1])
with col_s:
    start_date = st.date_input("시작일", value=date.today() - timedelta(days=7))
with col_e:
    end_date = st.date_input("종료일", value=date.today())
with col_btn:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
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

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — 우선순위 탭 + 메일 선택
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.email_list is not None and len(st.session_state.email_list) > 0:

    st.markdown('<div class="step-row"><span class="step-badge">2</span><span class="step-title">메일 선택 및 분석</span></div>', unsafe_allow_html=True)

    # PDF 필터 — 탭 위에 글로벌 토글로 배치
    _, col_toggle = st.columns([5, 1])
    with col_toggle:
        show_only_pdf = st.checkbox("PDF 첨부만", value=True)

    # ── 탭 스펙 빌드 ─────────────────────────────────────────────────────────
    all_emails_list = st.session_state.email_list

    def _tab_emails(key: str) -> list:
        base = all_emails_list if key == "전체" else [
            e for e in all_emails_list
            if any(key in tag for tag in e.get("tags", []))
        ]
        return [e for e in base if not show_only_pdf or e.get("has_attachment")]

    # 전체 탭 + 존재하는 우선순위 탭만 + 그 외 태그
    known_keys = set(PRIORITY_CONFIG.keys())
    tab_specs = []  # (key, emoji, color)

    # 전체
    tab_specs.append(("전체", "📥", "#475569"))

    # 우선순위 순서대로 — PDF 필터 적용 후 메일이 있는 탭만 표시
    for key, cfg in PRIORITY_CONFIG.items():
        if len(_tab_emails(key)) > 0:
            tab_specs.append((key, cfg["emoji"], cfg["color"]))

    # 나머지 말머리 — 동일하게 PDF 필터 적용 후 있는 것만
    for tag in st.session_state.unique_tags:
        if not any(known in tag for known in known_keys):
            if len(_tab_emails(tag)) > 0:
                tab_specs.append((tag, "📧", "#1D4ED8"))

    # 탭 라벨: "🚨 긴급  3"
    def _label(key, emoji, color):
        cnt = len(_tab_emails(key))
        return f"{emoji} {key}  {cnt}"

    tab_labels = [_label(k, e, c) for k, e, c in tab_specs]
    tabs = st.tabs(tab_labels)

    for (key, emoji, color), tab in zip(tab_specs, tabs):
        with tab:
            tab_email_list = _tab_emails(key)

            if not tab_email_list:
                st.markdown("""
                <div style="padding:32px;text-align:center;color:#94A3B8;font-size:14px;">
                  해당 말머리의 메일이 없습니다.
                </div>""", unsafe_allow_html=True)
                continue

            # 우선순위 배지 컬럼 추가
            df_data = []
            for e in tab_email_list:
                p = _priority_of(e)
                df_data.append({
                    "선택": False,
                    "id": e["id"],
                    "말머리": f"{p['emoji']} {p['key']}",
                    "날짜": e["date"],
                    "보낸사람": e["sender"],
                    "제목": e["subject"],
                    "PDF": e.get("pdf_count", 0),
                    "첨부파일": ", ".join(e.get("pdf_names", [])),
                })

            df = pd.DataFrame(df_data)
            edited_df = st.data_editor(
                df,
                column_config={
                    "선택": st.column_config.CheckboxColumn("선택", default=False, width="small"),
                    "id": None,
                    "말머리": st.column_config.TextColumn("말머리", width="small"),
                    "PDF": st.column_config.NumberColumn("PDF", width="small"),
                },
                disabled=["말머리", "날짜", "보낸사람", "제목", "PDF", "첨부파일"],
                hide_index=True,
                use_container_width=True,
                key=f"editor_{key}",
            )

            selected_ids = edited_df[edited_df["선택"] == True]["id"].tolist()

            _, col_analyze = st.columns([4, 1])
            with col_analyze:
                if st.button(
                    f"선택한 메일 분석하기",
                    type="primary",
                    disabled=len(selected_ids) == 0,
                    use_container_width=True,
                    key=f"btn_analyze_{key}",
                ):
                    st.session_state.analysis_triggered = True
                    st.session_state.selected_email_ids = selected_ids

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — 개별 메일 분석 결과
    # ══════════════════════════════════════════════════════════════════════════
    if getattr(st.session_state, 'analysis_triggered', False):

        st.markdown('<div class="step-row"><span class="step-badge">3</span><span class="step-title">개별 메일 분석 결과</span></div>', unsafe_allow_html=True)

        processed_data = load_processed_data()
        # 전체 메일 목록에서 선택된 ID 매핑 — 어느 탭에서 선택했어도 동작
        selected_emails = [
            e for e in st.session_state.email_list
            if e["id"] in st.session_state.selected_email_ids
        ]

        for email_meta in selected_emails:
            e_id      = email_meta['id']
            msg_id    = email_meta.get('message_id', e_id)
            has_att   = email_meta.get('has_attachment', False)
            is_cached = msg_id in processed_data
            p         = _priority_of(email_meta)

            # 우선순위별 top border 색상 + 배지
            cached_badge = '<span class="badge-ok">✓ 캐시됨</span>' if is_cached else ""
            pri_badge = (
                f'<span class="pri-badge" '
                f'style="background:{p["bg"]};color:{p["text"]}">'
                f'{p["emoji"]} {p["key"]}</span>'
            )
            st.markdown(f"""
            <div class="mail-card" style="border-top-color:{p['color']}">
              <div class="mail-card-title">
                {"📎 " if has_att else ""}
                {email_meta['subject']}
                &nbsp;{pri_badge}{cached_badge}
              </div>
              <div class="mail-card-meta">
                📅 {email_meta['date']} &nbsp;·&nbsp; 👤 {email_meta['sender']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if is_cached:
                results = processed_data[msg_id]
                tab_sum, tab_wc, tab_tfidf, tab_raw = st.tabs(
                    ["💡 요약", "☁️ 워드클라우드", "📊 TF-IDF", "📄 원본"]
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
                            st.image(item['wc_path'], use_container_width=True)
                        else:
                            st.info("워드클라우드 이미지가 없습니다.")
                    with tab_tfidf:
                        if item.get('tfidf_path') and Path(item['tfidf_path']).exists():
                            st.image(item['tfidf_path'], use_container_width=True)
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
                                summary    = summarize_text(text)
                                wc_path    = generate_word_cloud(text, output_filename=f"wordcloud_{e_id}_{pdf_path.name}.png")
                                kw_dict    = extract_tfidf_keywords({pdf_path.name: text})
                                tfidf_path = generate_tfidf_chart(
                                    kw_dict.get(pdf_path.name, []),
                                    title=pdf_path.name,
                                    output_filename=f"tfidf_{e_id}_{pdf_path.name}.png",
                                )
                                results.append({
                                    "file": pdf_path.name, "text": text,
                                    "summary": summary, "wc_path": wc_path, "tfidf_path": tfidf_path,
                                })
                    else:
                        text = email_meta.get("body_snippet", "")
                        if not text:
                            results.append({"error": "분석할 텍스트가 없습니다."})
                        else:
                            summary = summarize_text(text)
                            results.append({"file": "메일 본문", "text": text, "summary": summary, "wc_path": ""})
                    if results:
                        save_processed_data(msg_id, results)
                        st.rerun()

            # 유사 메일 추천
            if is_cached:
                results = processed_data[msg_id]
                if results and 'summary' in results[0]:
                    with st.expander("💡 과거 유사 업무 메일 추천 보기"):
                        search_text = f"제목: {email_meta['subject']}\n요약: {results[0]['summary']}"
                        with st.spinner("유사한 과거 메일을 검색하는 중..."):
                            chroma_client = ChromaClient()
                            similar_emails = chroma_client.query_similar(search_text, n_results=3)
                        has_rec = False
                        if similar_emails:
                            for sim in similar_emails:
                                has_rec = True
                                distance   = sim.get('distance', 0.0)
                                similarity = max(0, 100 - (distance * 100))
                                st.info(
                                    f"**{sim['metadata'].get('title', '제목 없음')}** (유사도: {similarity:.1f}%)\n\n"
                                    f"📅 *{sim['metadata'].get('date', '')}* · "
                                    f"👤 *{sim['metadata'].get('sender', '')}*\n\n"
                                    f"{sim['metadata'].get('summary', '')}"
                                )
                        if not has_rec:
                            st.write("유사한 과거 메일이 없습니다.")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)


        # ══════════════════════════════════════════════════════════════════════
        # STEP 4 — 전체 비교 분석
        # ══════════════════════════════════════════════════════════════════════
        all_texts      = {}
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
            st.markdown('<div class="step-row"><span class="step-badge">4</span><span class="step-title">전체 비교 분석</span></div>', unsafe_allow_html=True)
            st.caption(f"선택된 {len(all_texts)}개 문서 기반 비교 분석")

            tfidf_results = extract_tfidf_keywords(all_texts)
            combined_text = " ".join(all_texts.values())

            tab_tfidf, tab_freq, tab_bi, tab_tri, tab_net = st.tabs([
                "📊 TF-IDF 비교", "📈 단어 빈도", "2-gram", "3-gram", "🕸️ 관계 네트워크",
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
                        st.dataframe(pd.DataFrame(freq, columns=["키워드", "빈도"]),
                                     use_container_width=True, hide_index=True)
                    with col_chart:
                        cp = generate_frequency_chart(freq[:20], title="상위 20 키워드 빈도",
                                                      output_filename="freq_all.png")
                        if cp and Path(cp).exists():
                            st.image(str(cp), use_container_width=True)

            with tab_bi:
                bigrams = extract_ngrams(combined_text, n=2, top_n=20)
                if bigrams:
                    col_tbl, col_chart = st.columns([1, 2])
                    with col_tbl:
                        st.dataframe(pd.DataFrame(bigrams, columns=["2-gram", "빈도"]),
                                     use_container_width=True, hide_index=True)
                    with col_chart:
                        cp = generate_ngram_chart(bigrams, n=2,
                                                  output_filename="ngram2_all.png", color="#DD8452")
                        if cp and Path(cp).exists():
                            st.image(str(cp), use_container_width=True)
                else:
                    st.info("2-gram을 추출할 텍스트가 부족합니다.")

            with tab_tri:
                trigrams = extract_ngrams(combined_text, n=3, top_n=20)
                if trigrams:
                    col_tbl, col_chart = st.columns([1, 2])
                    with col_tbl:
                        st.dataframe(pd.DataFrame(trigrams, columns=["3-gram", "빈도"]),
                                     use_container_width=True, hide_index=True)
                    with col_chart:
                        cp = generate_ngram_chart(trigrams, n=3,
                                                  output_filename="ngram3_all.png", color="#55A868")
                        if cp and Path(cp).exists():
                            st.image(str(cp), use_container_width=True)
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
