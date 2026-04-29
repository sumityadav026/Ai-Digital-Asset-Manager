"""pages/dashboard.py — Dashboard page."""
import os
import streamlit as st

from utils.ui import (
    render_stat, render_section_header, render_empty_state,
    get_doc_list, file_icon, fmt_size, navigate_to, PRIMARY, SURFACE, BORDER, TEXT_MUTED, TEXT_MAIN
)


def render(metadata, index):
    docs = get_doc_list(metadata)
    total_docs = len(docs)
    total_chunks = sum(d["chunks"] for d in docs)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## Dashboard")
    st.markdown(
        '<p style="color:#6B7280;font-size:0.88rem;margin-top:-0.5rem;">'
        'Your AI-powered document workspace</p>',
        unsafe_allow_html=True,
    )

    # ── Stats row ─────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        render_stat(total_docs, "Documents")
    with c2:
        render_stat(total_chunks, "Indexed Chunks")
    with c3:
        render_stat(index.ntotal, "Vector Embeddings")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick search ──────────────────────────────────────────────────────────
    render_section_header("Quick Search")
    col_search, col_btn = st.columns([5, 1])
    with col_search:
        q = st.text_input(
            "Search", placeholder="Search across all documents…",
            key="dash_search", label_visibility="collapsed"
        )
    with col_btn:
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        if st.button("Search", key="dash_go"):
            if q:
                st.session_state.search_query = q
                navigate_to("search")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Upload ────────────────────────────────────────────────────────────────
    render_section_header("Upload a Document")
    up = st.file_uploader(
        "Drop a PDF or DOCX here", type=["pdf", "docx"],
        key="dash_upload", label_visibility="collapsed"
    )
    if up:
        navigate_to("documents")   # hand off to documents page for processing

    # ── Recent documents ──────────────────────────────────────────────────────
    render_section_header("Document Library")

    if not docs:
        render_empty_state(
            "📂", "No documents yet",
            "Upload your first PDF or DOCX to get started."
        )
        return

    for d in docs[:8]:
        icon = file_icon(d["file"])
        ext = d["ext"].upper() if d["ext"] else "FILE"
        chunks = d["chunks"]
        size = fmt_size(d["size"])

        with st.container():
            st.markdown(f"""
            <div class="dam-card">
              <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
                <span style="font-size:1.3rem">{icon}</span>
                <div>
                  <div class="dam-card-title">{d['file']}</div>
                  <div class="dam-card-meta">
                    <span class="badge badge-blue">{ext}</span>&nbsp;
                    {size} &middot; {chunks} chunks
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            d_id = d.get("doc_id", d["file"])
            btn_col1, btn_col2, _ = st.columns([1, 1, 4])
            with btn_col1:
                if st.button("Chat", key=f"dash_chat_{d_id}"):
                    navigate_to("doc_chat", d)
            with btn_col2:
                if st.button("Search", key=f"dash_srch_{d_id}"):
                    st.session_state.search_query = d["file"]
                    navigate_to("search")
