"""pages/settings.py — Settings / info page."""
import os
import streamlit as st
from utils.ui import render_section_header, SURFACE, BORDER, TEXT_MUTED, PRIMARY


def render(metadata: list, index):
    st.markdown("## Settings & Info")
    st.markdown(
        '<p style="color:#6B7280;font-size:0.88rem;margin-top:-0.5rem;">'
        'Application configuration and status.</p>',
        unsafe_allow_html=True,
    )

    # ── API Status ────────────────────────────────────────────────────────────
    render_section_header("API Configuration")
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        masked = api_key[:6] + "•" * (len(api_key) - 10) + api_key[-4:]
        st.success(f"Groq API key detected: `{masked}`")
    else:
        st.error("GROQ_API_KEY environment variable is not set. Chat with Documents will not work.")
        st.code("export GROQ_API_KEY=your_key_here", language="bash")

    # ── Model info ────────────────────────────────────────────────────────────
    render_section_header("Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
             padding:1rem;margin-bottom:0.5rem">
          <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;
               color:{TEXT_MUTED};margin-bottom:0.3rem">LLM</div>
          <div style="font-weight:600;font-size:0.9rem">openai/gpt-oss-120b</div>
          <div style="font-size:0.78rem;color:{TEXT_MUTED}">via Groq API</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
             padding:1rem;margin-bottom:0.5rem">
          <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;
               color:{TEXT_MUTED};margin-bottom:0.3rem">Embedding Model</div>
          <div style="font-weight:600;font-size:0.9rem">all-MiniLM-L6-v2</div>
          <div style="font-size:0.78rem;color:{TEXT_MUTED}">384-dim · sentence-transformers</div>
        </div>
        """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
             padding:1rem">
          <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;
               color:{TEXT_MUTED};margin-bottom:0.3rem">Vector Store</div>
          <div style="font-weight:600;font-size:0.9rem">FAISS IndexFlatIP</div>
          <div style="font-size:0.78rem;color:{TEXT_MUTED}">{index.ntotal} vectors · cosine similarity</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;
             padding:1rem">
          <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;
               color:{TEXT_MUTED};margin-bottom:0.3rem">Vision Model</div>
          <div style="font-weight:600;font-size:0.9rem">CLIP ViT-B/32</div>
          <div style="font-size:0.78rem;color:{TEXT_MUTED}">OpenAI · image classification</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Database ──────────────────────────────────────────────────────────────
    render_section_header("Database Files")
    for fname in ("database/faiss_index.bin", "database/metadata.pkl"):
        exists = os.path.exists(fname)
        size = os.path.getsize(fname) if exists else 0
        icon = "✅" if exists else "❌"
        size_str = f"{size // 1024} KB" if size else "—"
        st.markdown(f"`{fname}` — {icon} {'exists' if exists else 'missing'} &nbsp;·&nbsp; {size_str}",
                    unsafe_allow_html=True)

    # ── Retrieval config ──────────────────────────────────────────────────────
    render_section_header("Retrieval Configuration")
    st.markdown("""
    | Parameter | Value | Notes |
    |---|---|---|
    | FAISS k | 25 | Candidates fetched per query |
    | Semantic weight | 0.65–0.80 | Hybrid score weight |
    | Min score threshold | 0.10 | Chunks below this are discarded |
    | Context chunks | 6 | Max chunks sent to LLM |
    | Max context chars | 4 000 | Character limit for LLM context |
    | Chunk size | 500 chars | Document splitting |
    | Chunk overlap | 100 chars | Overlap between adjacent chunks |
    """)

    # ── Re-index ──────────────────────────────────────────────────────────────
    render_section_header("Re-index Documents")
    st.info(
        "If you need to rebuild the entire index (e.g. after changing the embedding model), "
        "run the rebuild script from the terminal:"
    )
    st.code(
        "cd /Users/sumityadav/Documents/AI_DAM_project\n"
        "source venv/bin/activate\n"
        "python rebuild_index.py",
        language="bash",
    )
