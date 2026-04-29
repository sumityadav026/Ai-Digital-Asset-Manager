"""pages/search.py — Global semantic search across all documents."""
import numpy as np
import streamlit as st

from utils.rag import extract_query_keywords, score_chunk, hybrid_score, highlight_text
from utils.embeddings import get_embedding
from utils.ui import (
    render_section_header, render_empty_state, file_icon, navigate_to,
    PRIMARY, SURFACE, BORDER, TEXT_MUTED, TEXT_LABEL
)


def render(metadata: list, index):
    st.markdown("## Search")
    st.markdown(
        '<p style="color:#6B7280;font-size:0.88rem;margin-top:-0.5rem;">'
        'Search across all your indexed documents using semantic similarity.</p>',
        unsafe_allow_html=True,
    )

    # ── Search input ──────────────────────────────────────────────────────────
    render_section_header("Search Query")

    # Pre-fill if navigated here with a query
    default_q = st.session_state.get("search_query", "")

    query = st.text_input(
        "Query",
        value=default_q,
        placeholder="e.g. What is the invoice amount?  |  Unity game developer  |  CGPA",
        key="search_input",
        label_visibility="collapsed",
    )

    st.markdown('<div class="primary-btn" style="display:inline-block">', unsafe_allow_html=True)
    search_clicked = st.button("Search Documents", key="search_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    # Store query in session so other pages can pre-fill
    if query:
        st.session_state.search_query = query

    # ── Run search ────────────────────────────────────────────────────────────
    run_search = search_clicked or (query and query == default_q and st.session_state.get("search_results"))

    if (search_clicked or query) and index.ntotal > 0 and query:
        q_vec = get_embedding(query).astype("float32")
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

        k = min(25, index.ntotal)
        D, I = index.search(np.array([q_vec]), k=k)

        keywords = extract_query_keywords(query)

        # Aggregate best chunk per file
        file_best: dict = {}
        for i, idx in enumerate(I[0]):
            if idx >= len(metadata):
                continue
            m = metadata[idx]
            fn = m.get("file", "")
            if fn in ("image_ocr", "image_visual"):
                continue
            sem = float(D[0][i])
            kw = score_chunk(m["text"], keywords)
            score = hybrid_score(sem, kw)
            doc_id = m.get("doc_id") or m.get("hash") or fn
            if fn not in file_best or score > file_best[fn]["score"]:
                file_best[fn] = {
                    "doc_id": doc_id,
                    "file": fn,
                    "text": m["text"],
                    "score": score,
                    "sem": sem,
                    "chunk_index": m.get("chunk_index", 0),
                }

        results = sorted(file_best.values(), key=lambda x: x["score"], reverse=True)
        st.session_state.search_results = results

        # ── Results ───────────────────────────────────────────────────────────
        render_section_header(f"Results — {len(results)} document(s) found")

        if not results:
            render_empty_state(
                "🔍", "No results",
                "Try different keywords or upload more documents."
            )
            return

        for res in results[:8]:
            icon = file_icon(res["file"])
            score_pct = int(res["score"] * 100)
            excerpt = res["text"][:280].replace("\n", " ")
            highlighted = highlight_text(excerpt, query)
            r_id = res.get("doc_id", res["file"])

            st.markdown(f"""
            <div class="dam-card">
              <div style="display:flex;align-items:flex-start;gap:0.7rem;">
                <span style="font-size:1.4rem;line-height:1.2">{icon}</span>
                <div style="flex:1;min-width:0">
                  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.2rem">
                    <span class="dam-card-title">{res['file']}</span>
                    <span class="score-pill">{score_pct}%</span>
                  </div>
                  <div class="dam-card-excerpt">{highlighted}…</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            b1, b2, _ = st.columns([1.4, 2, 4])
            with b1:
                if st.button("Open Doc", key=f"srch_open_{r_id}"):
                    navigate_to("documents")
            with b2:
                st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                if st.button("Chat with this doc", key=f"srch_chat_{r_id}"):
                    navigate_to("doc_chat", res)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div style="height:0.2rem"></div>', unsafe_allow_html=True)

    elif index.ntotal == 0:
        render_empty_state(
            "📂", "Nothing indexed yet",
            "Upload documents first so you can search across them."
        )
    elif not query:
        render_empty_state(
            "🔍", "Enter a query above",
            "Type anything — a keyword, a question, a topic — and hit Search."
        )
