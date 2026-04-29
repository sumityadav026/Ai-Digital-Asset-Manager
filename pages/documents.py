"""pages/documents.py — My Documents page with upload + library."""
import os
import numpy as np
import streamlit as st

from utils.extract import extract_text
from utils.embeddings import get_embedding
from utils.db import add_to_index, save_index
from utils.ui import (
    render_section_header, render_empty_state, get_doc_list,
    file_icon, fmt_size, navigate_to, PRIMARY, BORDER, TEXT_MUTED
)

import hashlib


def _get_file_hash(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def render(metadata: list, index):
    st.markdown("## My Documents")
    st.markdown(
        '<p style="color:#6B7280;font-size:0.88rem;margin-top:-0.5rem;">'
        'Upload, browse, and manage your indexed documents.</p>',
        unsafe_allow_html=True,
    )

    # ── Upload panel ──────────────────────────────────────────────────────────
    render_section_header("Upload New Document")
    uploaded_file = st.file_uploader(
        "Drop a PDF or DOCX here",
        type=["pdf", "docx"],
        key="docs_upload",
        label_visibility="collapsed",
    )

    if uploaded_file:
        file_bytes = bytes(uploaded_file.getbuffer())
        file_hash = _get_file_hash(file_bytes)
        existing_hashes = [item.get("hash") for item in metadata]

        if file_hash in existing_hashes:
            st.warning("This file has already been indexed.")
        else:
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            text = extract_text(file_path)

            if not text.strip():
                st.warning("No text could be extracted from this file.")
            else:
                with st.status(f"Processing {uploaded_file.name}…", expanded=True) as status:
                    chunks = _chunk_text(text)
                    st.write(f"Splitting into {len(chunks)} chunks…")

                    for chunk_idx, chunk in enumerate(chunks):
                        emb = get_embedding(chunk).astype("float32")
                        emb = emb / (np.linalg.norm(emb) + 1e-10)
                        add_to_index(
                            index, metadata, emb,
                            uploaded_file.name, chunk, file_hash,
                            chunk_index=chunk_idx,
                            doc_id=file_hash,
                        )

                    save_index(index, metadata)
                    status.update(label="Done!", state="complete")

                st.success(
                    f"**{uploaded_file.name}** indexed successfully — {len(chunks)} chunks."
                )

                with st.expander("Inspect extracted text", expanded=False):
                    st.text(text[:3000] + ("…" if len(text) > 3000 else ""))

    # ── Search / filter bar ───────────────────────────────────────────────────
    render_section_header("Document Library")
    docs = get_doc_list(metadata)

    col_filter, col_sort = st.columns([3, 1])
    with col_filter:
        filter_text = st.text_input(
            "Filter", placeholder="Filter by filename…",
            key="docs_filter", label_visibility="collapsed"
        )
    with col_sort:
        sort_by = st.selectbox(
            "Sort", ["Name", "Size ↓", "Chunks ↓"],
            key="docs_sort", label_visibility="collapsed"
        )

    # Apply filter
    if filter_text:
        docs = [d for d in docs if filter_text.lower() in d["file"].lower()]

    # Apply sort
    if sort_by == "Size ↓":
        docs = sorted(docs, key=lambda x: x["size"], reverse=True)
    elif sort_by == "Chunks ↓":
        docs = sorted(docs, key=lambda x: x["chunks"], reverse=True)

    # ── Docs table ────────────────────────────────────────────────────────────
    if not docs:
        render_empty_state(
            "📂", "No documents found",
            "Upload a PDF or DOCX above to get started."
        )
        return

    for d in docs:
        icon = file_icon(d["file"])
        ext = d["ext"].upper() if d["ext"] else "FILE"
        size = fmt_size(d["size"])
        d_id = d.get("doc_id", d["file"])

        # Card HTML
        st.markdown(f"""
        <div class="dam-card">
          <div style="display:flex;align-items:center;gap:0.75rem;">
            <span style="font-size:1.6rem;line-height:1;">{icon}</span>
            <div style="flex:1;min-width:0;">
              <div class="dam-card-title" style="white-space:nowrap;overflow:hidden;
                text-overflow:ellipsis;">{d['file']}</div>
              <div class="dam-card-meta">
                <span class="badge badge-blue">{ext}</span>&nbsp;
                {size} &middot; {d['chunks']} chunks indexed
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Action buttons rendered just below the card
        b1, b2, b3, _ = st.columns([1, 1.2, 1.5, 4])
        with b1:
            if st.button("Chat", key=f"doc_chat_{d_id}"):
                navigate_to("doc_chat", d)
        with b2:
            if st.button("Search", key=f"doc_search_{d_id}"):
                st.session_state.search_query = d["file"]
                navigate_to("search")
        with b3:
            if st.button("View Details", key=f"doc_detail_{d_id}"):
                st.session_state[f"show_detail_{d_id}"] = not st.session_state.get(
                    f"show_detail_{d_id}", False
                )

        # Inline detail expander
        if st.session_state.get(f"show_detail_{d_id}", False):
            chunks_for_doc = [
                m for m in metadata
                if (m.get("doc_id") == d_id or m.get("hash") == d.get("hash") or m.get("file") == d["file"])
            ]
            with st.expander(f"Details — {d['file']}", expanded=True):
                col1, col2 = st.columns(2)
                col1.metric("File size", size)
                col2.metric("Indexed chunks", d["chunks"])
                st.markdown("**Sample chunk (first):**")
                if chunks_for_doc:
                    st.text(chunks_for_doc[0]["text"][:400])

        st.markdown('<div style="height:0.1rem"></div>', unsafe_allow_html=True)
