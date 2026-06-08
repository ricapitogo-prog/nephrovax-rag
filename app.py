"""NephroVax RAG Streamlit app.

Thin UI layer. All RAG logic lives in the `nephrovax` package.
Run locally:
    streamlit run app.py
"""

import os
import streamlit as st

from nephrovax import config, loader, chunker
from nephrovax.retrieval import Retriever
from nephrovax.generation import Generator
from nephrovax.ui_helpers import humanize_breadcrumb


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NephroVax RAG",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Theme: clean neutral palette + Inter font
# ---------------------------------------------------------------------------
# Palette references (Tailwind "slate" + indigo accent):
#   bg:           #ffffff
#   text:         #0f172a  (dark slate)
#   text-muted:   #64748b  (slate)
#   border:       #e2e8f0  (slate-200)
#   surface:      #f8fafc  (slate-50)
#   accent:       #4f46e5  (indigo-600)
#   accent-soft:  #eef2ff  (indigo-50)

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* === Hide Streamlit's default chrome === */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }

    /* === Typography: Inter everywhere === */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0f172a;
    }

    /* === Heading sizes — tight and modern === */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.25rem !important;
        color: #0f172a !important;
    }
    h2 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em !important;
        color: #64748b !important;
        margin-top: 2rem !important;
        margin-bottom: 0.75rem !important;
    }
    h3 {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #0f172a !important;
    }

    /* === Container — wider now that we use a two-column layout === */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1180px !important;
    }

    /* === Body text === */
    .stMarkdown p {
        color: #0f172a;
        line-height: 1.6;
    }
    .stMarkdown strong {
        font-weight: 600;
        color: #0f172a;
    }

    /* === Captions: muted, smaller === */
    [data-testid="stCaptionContainer"] {
        font-size: 0.8125rem !important;
        color: #64748b !important;
    }

    /* === Disclaimer banner: warm, subtle === */
    [data-testid="stAlert"] {
        background-color: #fffbeb !important;
        border: 1px solid #fde68a !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stAlert"] p {
        font-size: 0.875rem !important;
        color: #78350f !important;
    }

    /* === Compact disclaimer (always shown) === */
    .compact-disclaimer {
        font-size: 0.8125rem;
        color: #78350f;
        text-align: center;
        padding: 10px 14px;
        border: 1px solid #fde68a;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        background-color: #fffbeb;
        font-weight: 500;
    }

    /* === About / context block === */
    .about-block {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #4f46e5;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 1.75rem;
    }
    .about-block .about-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 8px;
    }
    .about-block .about-lede {
        font-size: 0.875rem;
        line-height: 1.6;
        color: #334155;
    }
    .about-block .about-sources {
        font-size: 0.8125rem;
        line-height: 1.55;
        color: #64748b;
        margin-top: 9px;
    }
    .about-block .about-sources strong {
        color: #475569;
    }
    .vaccine-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 11px 0 4px;
    }
    .vaccine-chips span {
        font-size: 0.75rem;
        background-color: #eef2ff;
        color: #4f46e5;
        font-weight: 500;
        padding: 3px 10px;
        border-radius: 999px;
    }

    /* === Right-column empty-state hint (pre-query) === */
    .refs-placeholder {
        font-size: 0.8125rem;
        color: #94a3b8;
        line-height: 1.55;
        padding: 14px 16px;
        border: 1px dashed #e2e8f0;
        border-radius: 8px;
        background-color: #f8fafc;
    }

    /* === Buttons === */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 8px 14px !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
    }
    .stButton > button:hover {
        border-color: #4f46e5 !important;
        background-color: #f8fafc !important;
    }
    /* Primary button — the "Get recommendation" one */
    .stButton > button[kind="primary"] {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        text-align: center !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #4338ca !important;
    }
    .stButton > button[kind="primary"]:disabled {
        background-color: #e2e8f0 !important;
        color: #94a3b8 !important;
    }

    /* === Text area === */
    .stTextArea textarea {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9375rem !important;
        padding: 12px !important;
    }
    .stTextArea textarea:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
    }

    /* === Expanders === */
    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: #f8fafc !important;
    }

    /* === Tables: better borders, alternating rows === */
    .stMarkdown table {
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        margin: 1rem 0 !important;
        font-size: 0.875rem !important;
    }
    .stMarkdown thead th {
        background-color: #f8fafc !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        padding: 10px 14px !important;
        text-align: left !important;
        border-bottom: 1px solid #e2e8f0 !important;
    }
    .stMarkdown tbody td {
        padding: 10px 14px !important;
        border-bottom: 1px solid #f1f5f9 !important;
        color: #0f172a !important;
    }
    .stMarkdown tbody tr:last-child td {
        border-bottom: none !important;
    }

    /* === References list (the inline footnote table) === */
    .footnote-refs {
        font-size: 0.875rem;
        color: #475569;
        padding: 12px 16px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    .footnote-refs > div {
        padding: 4px 0;
        line-height: 1.5;
    }
    .footnote-refs .ref-num {
        font-weight: 600;
        color: #4f46e5;
        margin-right: 4px;
    }
    .footnote-refs .ref-score {
        color: #94a3b8;
        font-size: 0.8125rem;
        margin-left: 6px;
    }

    /* === Spinner === */
    .stSpinner > div {
        color: #4f46e5 !important;
    }

    /* === Reduce vertical gap between consecutive markdown blocks === */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }

    /* === Footer === */
    .app-footer {
        font-size: 0.75rem;
        color: #94a3b8;
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #f1f5f9;
    }
    .app-footer code {
        background-color: #f8fafc;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7rem;
    }
</style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Secret loading
# ---------------------------------------------------------------------------

def _get_secret(name: str) -> str:
    if name in st.secrets:
        return st.secrets[name]
    value = os.environ.get(name)
    if not value:
        st.error(
            f"Missing required secret: {name}. "
            f"Set it in Streamlit Cloud's Secrets panel or as a local env var."
        )
        st.stop()
    return value


VOYAGE_API_KEY = _get_secret("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")


# ---------------------------------------------------------------------------
# Optional passcode gate
# ---------------------------------------------------------------------------

def _check_passcode():
    expected = st.secrets.get("APP_PASSCODE") if "APP_PASSCODE" in st.secrets else None
    if not expected:
        return
    if st.session_state.get("authed"):
        return

    st.title("NephroVax RAG")
    st.write("This prototype is restricted. Enter the passcode to continue.")
    pw = st.text_input("Passcode", type="password")
    if pw == expected:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect passcode.")
    st.stop()


_check_passcode()


# ---------------------------------------------------------------------------
# Set up the RAG pipeline once and cache it
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading guidelines and building index...")
def setup_pipeline():
    documents = loader.load_all_guidelines(config.GUIDELINES_DIR)
    chunks = chunker.chunk_documents(documents)
    retriever = Retriever(
        chunks,
        voyage_api_key=VOYAGE_API_KEY,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )
    generator = Generator(anthropic_api_key=ANTHROPIC_API_KEY)
    return documents, chunks, retriever, generator


documents, chunks, retriever, generator = setup_pipeline()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "has_interacted" not in st.session_state:
    st.session_state["has_interacted"] = False
if "question" not in st.session_state:
    st.session_state["question"] = ""


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# NephroVax RAG")
st.caption("Vaccination decision support for patients on complement inhibitor therapy")


# ---------------------------------------------------------------------------
# Disclaimer — always compact, with expander for full text
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="compact-disclaimer">'
    '⚠ Research prototype — not for clinical use. '
    'Verify all recommendations against current CDC/ACIP guidelines.'
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Background / context block — explains the population, vaccines, and sources.
# Built from the loaded guideline frontmatter so it stays accurate if the
# underlying guidelines change.
# ---------------------------------------------------------------------------

_VACCINE_DISPLAY = {
    "pneumococcal": "Pneumococcal",
    "meningococcal": "Meningococcal",
    "hib": "Hib",
}


def _build_about_block(docs: list[dict]) -> str:
    classes, publishers, versions = [], set(), set()
    for d in docs:
        fm = d["frontmatter"]
        vc = fm.get("vaccine_class")
        if vc and vc not in classes:
            classes.append(vc)
        if fm.get("publisher"):
            publishers.add(fm["publisher"])
        if fm.get("version"):
            versions.add(str(fm["version"]))

    # Display in conventional clinical order, not alphabetical filename order.
    _ORDER = ["pneumococcal", "meningococcal", "hib"]
    classes.sort(key=lambda c: _ORDER.index(c) if c in _ORDER else len(_ORDER))

    chips = "".join(
        f"<span>{_VACCINE_DISPLAY.get(c, c.title())}</span>" for c in classes
    )
    publisher = ", ".join(sorted(publishers)) or "GlomCon Foundation"
    version = sorted(versions)[-1] if versions else "2026-02"

    return (
        '<div class="about-block">'
        '<div class="about-label">About this tool</div>'
        '<div class="about-lede">'
        "Answers vaccination questions for <strong>patients on complement "
        "inhibitor therapy</strong> (e.g. eculizumab, ravulizumab), who face "
        "elevated infection risk and need protection across the vaccine classes "
        "below."
        "</div>"
        f'<div class="vaccine-chips">{chips}</div>'
        '<div class="about-sources">'
        f"<strong>Sources:</strong> {publisher} NephroVax guideline "
        f"(v{version}), derived from CDC / ACIP recommendations. Answers are "
        "generated only from these indexed guideline sections — never from "
        "general medical knowledge."
        "</div>"
        "</div>"
    )


st.markdown(_build_about_block(documents), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Two-column layout: query + answer on the left, references on the right.
# ---------------------------------------------------------------------------

col_main, col_refs = st.columns([1.55, 1], gap="large")


# ---------------------------------------------------------------------------
# Sample questions — shorter labels, stacked vertically
# ---------------------------------------------------------------------------

# Result holders — populated in the main column, read by the references column.
retrieval_result = None
hits: list[dict] = []
answer_text = ""

# ---------------------------------------------------------------------------
# LEFT COLUMN — example questions, query input, and the generated answer
# ---------------------------------------------------------------------------

with col_main:
    with st.expander("Example questions", expanded=not st.session_state["has_interacted"]):
        # Pair each (short display label, full question)
        sample_questions = [
            ("New adult on eculizumab, no prior vaccines",
             "30-year-old patient just started eculizumab. No prior pneumococcal vaccines. What do I give?"),
            ("Young child starting therapy, meningococcal",
             "5-year-old initiating complement inhibitor therapy. No prior meningococcal vaccines. What's recommended?"),
            ("Adult with prior PCV20, what's next?",
             "Adult patient on Soliris who received PCV20 last year. Do they need additional pneumococcal vaccines?"),
            ("Adolescent, full vaccination plan",
             "16-year-old being started on ravulizumab. No prior meningococcal vaccines. Full vaccination plan?"),
        ]
        for i, (label, full_q) in enumerate(sample_questions):
            if st.button(label, key=f"sample_{i}", use_container_width=True):
                st.session_state["question"] = full_q

    question = st.text_area(
        "Clinical question",
        value=st.session_state["question"],
        height=80,
        placeholder="e.g., 'Adult on eculizumab with prior PCV13 only, what's next?'",
        label_visibility="collapsed",
    )

    submit = st.button(
        "Get recommendation",
        type="primary",
        disabled=not question.strip(),
        use_container_width=True,
    )

    if submit:
        st.session_state["has_interacted"] = True

        with st.spinner("Searching guidelines and generating answer..."):
            retrieval_result = retriever.retrieve_smart(question)
            hits = retrieval_result["hits"]
            answer_text = generator.answer(question, hits)

        # Answer
        st.markdown("## Answer")
        st.markdown(answer_text)

        # Copy
        with st.expander("📋 Copy answer text"):
            st.code(answer_text, language=None)


# ---------------------------------------------------------------------------
# RIGHT COLUMN — references overview + source excerpts (or a pre-query hint)
# ---------------------------------------------------------------------------

with col_refs:
    if submit and retrieval_result is not None:
        st.markdown("## References")

        # Mode indicator (subtle)
        if retrieval_result["classifier_error"]:
            st.caption(
                f"⚠ Query classifier failed ({retrieval_result['classifier_error']}); "
                f"using single-vaccine retrieval."
            )
        elif retrieval_result["mode"] == "multi":
            classes_str = ", ".join(retrieval_result["classes"])
            st.caption(f"Multi-vaccine query — retrieved across: {classes_str}")

        if hits:
            top_match = humanize_breadcrumb(hits[0]["doc_id"], hits[0]["breadcrumb"])
            st.caption(f"Top match: {top_match}")

        # Numbered reference list as a compact card (overview)
        ref_lines = []
        for i, hit in enumerate(hits, 1):
            label = humanize_breadcrumb(hit["doc_id"], hit["breadcrumb"])
            ref_lines.append(
                f'<div><span class="ref-num">[{i}]</span> {label}'
                f'<span class="ref-score">{hit["score"]:.2f}</span></div>'
            )
        st.markdown(
            f'<div class="footnote-refs">{"".join(ref_lines)}</div>',
            unsafe_allow_html=True,
        )

        # Source excerpts — each reference is itself the expander
        st.markdown("## Source excerpts")
        for i, hit in enumerate(hits, 1):
            label = humanize_breadcrumb(hit["doc_id"], hit["breadcrumb"])
            with st.expander(f"[{i}]  {label}"):
                st.caption(
                    f"`{hit['doc_id']}` · {hit['breadcrumb_str']} · "
                    f"similarity {hit['score']:.3f}"
                )
                st.markdown(hit["text"])
    else:
        st.markdown("## References")
        st.markdown(
            '<div class="refs-placeholder">'
            "References and source excerpts will appear here once you ask a "
            "question — so you can see exactly which guideline sections the "
            "answer draws on."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    f'<div class="app-footer">'
    f'{len(chunks)} guideline sections indexed · '
    f'Embedding: <code>{config.EMBEDDING_MODEL}</code> · '
    f'Generation: <code>{config.GENERATION_MODEL}</code>'
    f'</div>',
    unsafe_allow_html=True,
)
