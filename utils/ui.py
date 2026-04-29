"""
utils/ui.py
===========
Shared UI helpers: CSS injection, card components, state helpers.
"""
import os
import streamlit as st


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
PRIMARY = "#4F6BED"          # indigo-blue accent
PRIMARY_LIGHT = "#EEF1FD"    # very light tint for hover/selected states
BG = "#F8F9FC"               # page background
SURFACE = "#FFFFFF"          # card/panel background
BORDER = "#E4E7EF"           # subtle border
TEXT_MAIN = "#1A1D23"        # near-black
TEXT_MUTED = "#6B7280"       # medium grey
TEXT_LABEL = "#374151"       # slightly lighter than main
SUCCESS = "#16A34A"
WARNING = "#D97706"
ERROR = "#DC2626"


def inject_css() -> None:
    """Inject global CSS once per page load."""
    st.markdown(f"""
    <style>
    /* ── Global reset ─────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {TEXT_MAIN};
    }}

    /* Page background */
    .stApp {{
        background-color: {BG};
    }}

    /* Remove default top padding */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }}

    /* ── Sidebar ───────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] .stMarkdown p {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
    }}

    /* ── Headings ─────────────────────────────────── */
    h1 {{ font-size: 1.55rem !important; font-weight: 700 !important; color: {TEXT_MAIN} !important; }}
    h2 {{ font-size: 1.2rem !important;  font-weight: 600 !important; color: {TEXT_MAIN} !important; }}
    h3 {{ font-size: 1rem !important;    font-weight: 600 !important; color: {TEXT_LABEL} !important; }}

    /* ── Cards ────────────────────────────────────── */
    .dam-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.04);
        transition: box-shadow 0.18s;
    }}
    .dam-card:hover {{
        box-shadow: 0 4px 12px rgba(79,107,237,.10);
    }}
    .dam-card-title {{
        font-weight: 600;
        font-size: 0.95rem;
        color: {TEXT_MAIN};
        margin-bottom: 0.2rem;
    }}
    .dam-card-meta {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-bottom: 0.5rem;
    }}
    .dam-card-excerpt {{
        font-size: 0.83rem;
        color: {TEXT_LABEL};
        line-height: 1.55;
    }}

    /* ── Stat tiles ───────────────────────────────── */
    .dam-stat {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        text-align: center;
        box-shadow: 0 1px 2px rgba(0,0,0,.04);
    }}
    .dam-stat-value {{
        font-size: 1.9rem;
        font-weight: 700;
        color: {PRIMARY};
        line-height: 1.1;
    }}
    .dam-stat-label {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        margin-top: 0.25rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    /* ── Badges ───────────────────────────────────── */
    .badge {{
        display: inline-block;
        padding: 0.2em 0.6em;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-blue  {{ background: {PRIMARY_LIGHT}; color: {PRIMARY}; }}
    .badge-green {{ background: #DCFCE7; color: {SUCCESS}; }}
    .badge-grey  {{ background: #F3F4F6; color: {TEXT_MUTED}; }}

    /* ── Buttons ──────────────────────────────────── */
    .stButton > button {{
        border-radius: 8px !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        border: 1px solid {BORDER} !important;
        background: {SURFACE} !important;
        color: {TEXT_MAIN} !important;
        padding: 0.4rem 0.9rem !important;
        transition: all 0.15s !important;
    }}
    .stButton > button:hover {{
        border-color: {PRIMARY} !important;
        color: {PRIMARY} !important;
        background: {PRIMARY_LIGHT} !important;
    }}
    /* Primary button — wrap with .primary-btn div */
    .primary-btn > button {{
        background: {PRIMARY} !important;
        color: #fff !important;
        border-color: {PRIMARY} !important;
        font-weight: 600 !important;
    }}
    .primary-btn > button:hover {{
        background: #3d56d6 !important;
        border-color: #3d56d6 !important;
        color: #fff !important;
    }}

    /* ── Inputs ───────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {{
        border-radius: 8px !important;
        border-color: {BORDER} !important;
        font-size: 0.9rem !important;
    }}
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px rgba(79,107,237,.12) !important;
    }}

    /* ── File uploader ────────────────────────────── */
    [data-testid="stFileUploader"] {{
        border: 1.5px dashed {BORDER} !important;
        border-radius: 10px !important;
        background: {BG} !important;
        padding: 0.5rem !important;
    }}

    /* ── Expander ─────────────────────────────────── */
    .streamlit-expanderHeader {{
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: {TEXT_MUTED} !important;
        border-radius: 8px !important;
    }}

    /* ── Chat messages ────────────────────────────── */
    [data-testid="stChatMessage"] {{
        border-radius: 10px !important;
        border: 1px solid {BORDER} !important;
        background: {SURFACE} !important;
    }}
    [data-testid="stChatMessage"][data-testid*="user"] {{
        background: {PRIMARY_LIGHT} !important;
    }}

    /* ── Divider ──────────────────────────────────── */
    hr {{ border-color: {BORDER}; margin: 1rem 0; }}

    /* ── Alerts / info boxes ──────────────────────── */
    .stAlert {{ border-radius: 10px !important; }}

    /* ── Section header ───────────────────────────── */
    .section-header {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {TEXT_MUTED};
        margin: 1.2rem 0 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid {BORDER};
    }}

    /* ── Chat input area ──────────────────────────── */
    [data-testid="stChatInput"] textarea {{
        border-radius: 10px !important;
        border: 1.5px solid {BORDER} !important;
        font-size: 0.9rem !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {PRIMARY} !important;
        box-shadow: 0 0 0 3px rgba(79,107,237,.12) !important;
    }}

    /* ── Empty state ──────────────────────────────── */
    .empty-state {{
        text-align: center;
        padding: 3rem 1rem;
        color: {TEXT_MUTED};
    }}
    .empty-state-icon {{
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
    }}
    .empty-state-title {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {TEXT_LABEL};
        margin-bottom: 0.4rem;
    }}
    .empty-state-desc {{
        font-size: 0.85rem;
        line-height: 1.6;
    }}

    /* ── Doc-chat header ──────────────────────────── */
    .doc-chat-header {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .doc-chat-header-icon {{
        font-size: 1.5rem;
    }}
    .doc-chat-header-name {{
        font-weight: 600;
        font-size: 0.95rem;
        color: {TEXT_MAIN};
    }}
    .doc-chat-header-meta {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
    }}

    /* ── Score pill ───────────────────────────────── */
    .score-pill {{
        display: inline-block;
        background: {PRIMARY_LIGHT};
        color: {PRIMARY};
        border-radius: 99px;
        padding: 0.15em 0.55em;
        font-size: 0.72rem;
        font-weight: 600;
    }}

    /* Sidebar nav active link */
    .nav-active {{
        background: {PRIMARY_LIGHT};
        color: {PRIMARY};
        border-radius: 8px;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)


def file_icon(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return {"pdf": "📄", "docx": "📝", "doc": "📝", "xlsx": "📊",
            "png": "🖼", "jpg": "🖼", "jpeg": "🖼"}.get(ext, "📁")


def fmt_size(bytes_: int) -> str:
    if bytes_ < 1024:
        return f"{bytes_} B"
    if bytes_ < 1024 * 1024:
        return f"{bytes_ // 1024} KB"
    return f"{bytes_ / (1024*1024):.1f} MB"


def render_empty_state(icon: str, title: str, desc: str) -> None:
    st.markdown(f"""
    <div class="empty-state">
      <div class="empty-state-icon">{icon}</div>
      <div class="empty-state-title">{title}</div>
      <div class="empty-state-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(label: str) -> None:
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)


def render_stat(value: str | int, label: str) -> None:
    st.markdown(f"""
    <div class="dam-stat">
      <div class="dam-stat-value">{value}</div>
      <div class="dam-stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def get_doc_list(metadata: list[dict]) -> list[dict]:
    """Build per-document summary from FAISS metadata, keyed by stable doc_id."""
    docs: dict[str, dict] = {}
    for m in metadata:
        fn = m.get("file", "?")
        if fn in ("image_ocr", "image_visual"):
            continue
        doc_id = m.get("doc_id") or m.get("hash") or fn
        if doc_id not in docs:
            fpath = os.path.join("uploads", fn)
            size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
            docs[doc_id] = {
                "doc_id": doc_id,
                "file": fn,
                "hash": m.get("hash", ""),
                "size": size,
                "chunks": 0,
                "ext": fn.rsplit(".", 1)[-1].lower() if "." in fn else "",
            }
        docs[doc_id]["chunks"] += 1
    return sorted(docs.values(), key=lambda x: x["file"])


def resolve_selected_document(doc_identifier: str | dict | None, metadata: list[dict]) -> dict | None:
    """
    Resolve a document identifier (doc_id, hash, filename, or dict) to its
    canonical document record: {'doc_id': ..., 'file': ..., 'hash': ..., 'chunks': ...}.
    """
    if not doc_identifier:
        return None

    if isinstance(doc_identifier, dict):
        ident = doc_identifier.get("doc_id") or doc_identifier.get("hash") or doc_identifier.get("file")
    else:
        ident = str(doc_identifier)

    # First: attempt exact match on doc_id or hash
    for m in metadata:
        if m.get("doc_id") == ident or m.get("hash") == ident:
            fn = m.get("file", "")
            fpath = os.path.join("uploads", fn)
            size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
            doc_id = m.get("doc_id") or m.get("hash") or fn
            chunks_count = sum(1 for x in metadata if (x.get("doc_id") == doc_id or x.get("hash") == doc_id))
            return {
                "doc_id": doc_id,
                "file": fn,
                "hash": m.get("hash", ""),
                "size": size,
                "chunks": chunks_count,
            }

    # Second: fallback match on filename
    for m in metadata:
        if m.get("file") == ident:
            fn = m.get("file", "")
            fpath = os.path.join("uploads", fn)
            size = os.path.getsize(fpath) if os.path.exists(fpath) else 0
            doc_id = m.get("doc_id") or m.get("hash") or fn
            chunks_count = sum(1 for x in metadata if x.get("file") == fn)
            return {
                "doc_id": doc_id,
                "file": fn,
                "hash": m.get("hash", ""),
                "size": size,
                "chunks": chunks_count,
            }

    return None


def init_session_defaults() -> None:
    """Ensure all required session-state keys exist exactly once."""
    defaults = {
        "page": "dashboard",
        "selected_doc": None,
        "selected_doc_id": None,
        "doc_chats": {},      # {doc_id: [(role, text), ...]}
        "search_query": "",
        "search_results": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def navigate_to(page: str, doc: str | dict | None = None) -> None:
    st.session_state.page = page
    if doc is not None:
        st.session_state.selected_doc = doc
        if isinstance(doc, dict):
            st.session_state.selected_doc_id = doc.get("doc_id") or doc.get("hash") or doc.get("file")
        else:
            st.session_state.selected_doc_id = str(doc)
    st.rerun()
