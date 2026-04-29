"""
pages/doc_chat.py
=================
Per-document dedicated chat interface with strict document isolation.

Each document gets its own conversation stored at:
    st.session_state.doc_chats[doc_id] = [(role, text), ...]

Every retrieval operation is strictly scoped to the selected document ID,
guaranteeing zero cross-document data leakage.
"""
import os
import logging
import streamlit as st

from utils.rag import (
    retrieve_document_context,
    validate_document_context,
    call_llm,
    highlight_text,
)
from utils.ui import (
    render_empty_state,
    file_icon,
    fmt_size,
    navigate_to,
    resolve_selected_document,
    PRIMARY,
    SURFACE,
    BORDER,
    TEXT_MUTED,
    TEXT_LABEL,
    TEXT_MAIN,
    PRIMARY_LIGHT,
)

logger = logging.getLogger(__name__)

# Suggested prompts shown in the empty state
_SUGGESTIONS = [
    "Summarize this document",
    "What are the main points?",
    "What qualifications or skills are mentioned?",
    "Extract important dates or numbers",
    "Who is this document about?",
]


def _get_chat(doc_id: str) -> list:
    """Return (creating if needed) the per-doc message list keyed by doc_id."""
    if "doc_chats" not in st.session_state:
        st.session_state.doc_chats = {}
    if doc_id not in st.session_state.doc_chats:
        st.session_state.doc_chats[doc_id] = []
    return st.session_state.doc_chats[doc_id]


def _file_ext(fn: str) -> str:
    return fn.rsplit(".", 1)[-1].upper() if "." in fn else "FILE"


def render(metadata: list, index):
    selected_raw = st.session_state.get("selected_doc") or st.session_state.get("selected_doc_id")

    # ── Resolve selected document canonical identity ──────────────────────────
    doc_info = resolve_selected_document(selected_raw, metadata)

    # ── No doc selected or doc not found in index ─────────────────────────────
    if not doc_info:
        render_empty_state(
            "💬", "No document selected",
            "Go to My Documents or Search, then click \"Chat\" on a document."
        )
        st.markdown('<div class="primary-btn" style="display:inline-block;margin-top:1rem">',
                    unsafe_allow_html=True)
        if st.button("Browse Documents", key="chat_goto_docs"):
            navigate_to("documents")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    doc_id = doc_info["doc_id"]
    doc_name = doc_info["file"]
    doc_chunks_count = doc_info["chunks"]

    # ── Header ────────────────────────────────────────────────────────────────
    icon = file_icon(doc_name)
    ext = _file_ext(doc_name)
    fpath = os.path.join("uploads", doc_name)
    size_str = fmt_size(os.path.getsize(fpath)) if os.path.exists(fpath) else fmt_size(doc_info.get("size", 0))

    st.markdown(f"""
    <div class="doc-chat-header">
      <span class="doc-chat-header-icon">{icon}</span>
      <div style="flex:1;min-width:0;">
        <div class="doc-chat-header-name" title="{doc_name}">{doc_name}</div>
        <div class="doc-chat-header-meta">
          <span class="badge badge-blue">{ext}</span>
          &nbsp;{size_str}&nbsp;&middot;&nbsp;{doc_chunks_count} chunks indexed
          &nbsp;&middot;&nbsp;<span style="font-size:0.7rem;color:{TEXT_MUTED};">ID: {doc_id[:12]}…</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Back + Clear buttons
    hcol1, hcol2, _ = st.columns([1.2, 1.5, 5])
    with hcol1:
        if st.button("← Back", key=f"chat_back_{doc_id}"):
            navigate_to("documents")
    with hcol2:
        if st.button("Clear Chat", key=f"chat_clear_{doc_id}"):
            st.session_state.doc_chats[doc_id] = []
            st.rerun()

    st.markdown('<hr style="margin:0.5rem 0 1rem">', unsafe_allow_html=True)

    # ── Conversation area ─────────────────────────────────────────────────────
    messages = _get_chat(doc_id)

    # Empty state with suggested prompts
    if not messages:
        st.markdown(f"""
        <div style="text-align:center;padding:2rem 1rem 1rem;">
          <div style="font-size:2rem;margin-bottom:0.5rem">{icon}</div>
          <div style="font-weight:600;font-size:1rem;color:{TEXT_MAIN};margin-bottom:0.3rem">
            Chat with {doc_name}
          </div>
          <div style="font-size:0.84rem;color:{TEXT_MUTED};max-width:420px;margin:0 auto 1.2rem;">
            Ask anything about this document. The assistant will answer using only the
            content strictly from <strong>{doc_name}</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggested prompts as clickable chips
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:0.5rem;justify-content:center;margin-bottom:1.5rem">',
            unsafe_allow_html=True
        )
        cols = st.columns(len(_SUGGESTIONS))
        for i, suggestion in enumerate(_SUGGESTIONS):
            with cols[i % len(cols)]:
                if st.button(suggestion, key=f"suggest_{i}_{doc_id}"):
                    st.session_state[f"_pending_q_{doc_id}"] = suggestion
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Display existing messages
    for role, text in messages:
        with st.chat_message(role):
            st.markdown(text)

    # ── Chat input ────────────────────────────────────────────────────────────
    pending = st.session_state.pop(f"_pending_q_{doc_id}", None)

    question = st.chat_input(
        f"Ask about {doc_name}…",
        key=f"chat_input_{doc_id}",
    ) or pending

    if question:
        # Append user message
        messages.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        # Strict document-isolated retrieval
        if index.ntotal == 0 or doc_chunks_count == 0:
            answer = (
                f"No indexed content found for {doc_name}. "
                "Please ensure the document has been indexed."
            )
            sources_used = []
            deduped = []
        else:
            deduped, context_chunks = retrieve_document_context(
                question=question,
                index=index,
                metadata=metadata,
                document_id=doc_id,
                k=25,
            )

            # Defensive verification: assert all context chunks belong to doc_id
            try:
                validate_document_context(context_chunks, doc_id)
            except ValueError as err:
                logger.critical("[doc_chat] Cross-document leak caught defensively: %s", err)
                answer = "Security Error: Cross-document retrieval leakage detected. LLM request blocked."
                context_chunks = []
                sources_used = []

            if not context_chunks:
                answer = (
                    "The information is not available in the provided document. "
                    "The document may not contain content relevant to your question."
                )
                sources_used = []
            else:
                try:
                    api_key = os.getenv("GROQ_API_KEY")
                    if not api_key:
                        answer = "GROQ_API_KEY is not configured. Please set the environment variable."
                        sources_used = []
                    else:
                        with st.spinner("Thinking…"):
                            answer = call_llm(
                                question=question,
                                context_chunks=context_chunks,
                                expected_doc_id=doc_id,
                            )
                        sources_used = context_chunks
                except Exception as e:
                    answer = f"Error calling LLM: {str(e)}"
                    sources_used = []

        # Append assistant message
        messages.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

        # ── Sources ───────────────────────────────────────────────────────────
        if sources_used:
            with st.expander("Sources used", expanded=False):
                for src in sources_used:
                    src_file = src.get("file", doc_name)
                    chunk_n = src.get("chunk_index", "?")
                    sem = src.get("sem_score", 0)
                    hybrid = src.get("score", 0)
                    excerpt = src.get("text", "")[:200].replace("\n", " ")
                    highlighted_exc = highlight_text(excerpt, question)

                    st.markdown(f"""
                    <div style="border-left:3px solid {PRIMARY};padding:0.5rem 0.75rem;
                         margin-bottom:0.6rem;background:{PRIMARY_LIGHT};border-radius:0 6px 6px 0;">
                      <div style="font-size:0.78rem;font-weight:600;color:{TEXT_LABEL};
                           margin-bottom:0.2rem;">
                        {file_icon(src_file)} {src_file} — Chunk {chunk_n}
                        <span class="score-pill" style="margin-left:0.4rem">
                          {int(hybrid*100)}%
                        </span>
                      </div>
                      <div style="font-size:0.82rem;color:{TEXT_LABEL};line-height:1.5">
                        {highlighted_exc}…
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Debug expander ────────────────────────────────────────────────────
        if deduped:
            with st.expander("Retrieval debug (document-isolated)", expanded=False):
                st.caption(f"Top retrieved candidates for doc '{doc_name}' (ID: {doc_id}):")
                rows = []
                for c in deduped[:12]:
                    rows.append({
                        "file": c["file"],
                        "chunk": c.get("chunk_index", "?"),
                        "sem": round(c.get("sem_score", 0), 4),
                        "kw": round(c.get("kw_coverage", 0), 3),
                        "hybrid": round(c.get("score", 0), 4),
                        "preview": c["text"][:70].replace("\n", " "),
                    })
                st.table(rows)
