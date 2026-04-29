"""
AI Digital Asset Manager (AI DAM)
==================================
Main entry point.

Pages
-----
  Dashboard         — stats, quick search, document library
  My Documents      — upload, browse, filter, per-doc actions
  Search            — global semantic search with Chat CTA
  Document Chat     — per-document dedicated chat interface
  Image Recognition — CLIP label detection + Tesseract OCR
  Settings          — API config, model info, rebuild instructions
"""

import os
import logging

import torch
import clip
import streamlit as st

from utils.db import load_index
from utils.ui import inject_css, init_session_defaults, navigate_to, get_doc_list, PRIMARY, BORDER, TEXT_MUTED

# Page modules
import pages.dashboard as pg_dashboard
import pages.documents as pg_documents
import pages.search as pg_search
import pages.doc_chat as pg_doc_chat
import pages.image_recognition as pg_image
import pages.settings as pg_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Digital Asset Manager",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inject CSS + initialise session state
# ---------------------------------------------------------------------------
inject_css()
init_session_defaults()

# ---------------------------------------------------------------------------
# Heavy models — cached so they load only once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models…")
def _load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    return clip_model, preprocess, device


@st.cache_resource(show_spinner="Loading vector index…")
def _load_index():
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("database", exist_ok=True)
    return load_index()


clip_model, preprocess, device = _load_models()
index, metadata = _load_index()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0 1.2rem;">
      <div style="background:{PRIMARY};color:#fff;border-radius:10px;
           width:36px;height:36px;display:flex;align-items:center;
           justify-content:center;font-size:1.1rem;font-weight:700;
           flex-shrink:0;">A</div>
      <div>
        <div style="font-weight:700;font-size:0.95rem;color:#1A1D23;line-height:1.1">AI DAM</div>
        <div style="font-size:0.7rem;color:{TEXT_MUTED}">Digital Asset Manager</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Doc count badge
    doc_count = len(get_doc_list(metadata))
    st.markdown(
        f'<div style="font-size:0.72rem;color:{TEXT_MUTED};margin-bottom:0.8rem;">'
        f'{doc_count} document{"s" if doc_count != 1 else ""} indexed</div>',
        unsafe_allow_html=True,
    )

    # Nav items — (page_key, label, icon)
    NAV = [
        ("dashboard",  "Dashboard",         "◻"),
        ("documents",  "My Documents",      "📄"),
        ("search",     "Search",            "🔍"),
        ("doc_chat",   "Document Chat",     "💬"),
        ("image",      "Image Recognition", "🖼"),
        ("settings",   "Settings",          "⚙"),
    ]

    current_page = st.session_state.page

    st.markdown(
        '<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:{TEXT_MUTED};margin-bottom:0.4rem;">Navigation</div>',
        unsafe_allow_html=True,
    )

    # Inject per-button active styles via CSS class trick
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        padding: 0.45rem 0.75rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 500;
        border: none !important;
        background: transparent !important;
        color: #374151 !important;
        box-shadow: none !important;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background: #EEF1FD !important;
        color: #4F6BED !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for page_key, label, icon in NAV:
        is_active = current_page == page_key

        if is_active:
            # Active: render a styled HTML block + button on top
            st.markdown(f"""
            <div style="background:#EEF1FD;border-radius:8px;margin-bottom:2px;padding:0.45rem 0.75rem;
                 display:flex;align-items:center;gap:0.5rem;font-size:0.875rem;font-weight:600;
                 color:#4F6BED;">
              <span>{icon}</span><span>{label}</span>
              <span style="margin-left:auto;width:6px;height:6px;border-radius:50%;
                   background:#4F6BED;display:inline-block;"></span>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{page_key}", use_container_width=True):
                st.session_state.page = page_key
                st.rerun()

    st.markdown(f'<hr style="border-color:{BORDER};margin:1rem 0">', unsafe_allow_html=True)

    # Quick upload in sidebar
    st.markdown(
        '<div style="font-size:0.68rem;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.07em;color:{TEXT_MUTED};margin-bottom:0.4rem;">Quick Upload</div>',
        unsafe_allow_html=True,
    )
    quick_up = st.file_uploader(
        "Quick upload",
        type=["pdf", "docx"],
        key="sidebar_upload",
        label_visibility="collapsed",
    )
    if quick_up:
        navigate_to("documents")

# ---------------------------------------------------------------------------
# Main content — route to the active page
# ---------------------------------------------------------------------------
page = st.session_state.page

if page == "dashboard":
    pg_dashboard.render(metadata, index)

elif page == "documents":
    pg_documents.render(metadata, index)

elif page == "search":
    pg_search.render(metadata, index)

elif page == "doc_chat":
    pg_doc_chat.render(metadata, index)

elif page == "image":
    pg_image.render(metadata, index, clip_model, preprocess, device)

elif page == "settings":
    pg_settings.render(metadata, index)

else:
    pg_dashboard.render(metadata, index)