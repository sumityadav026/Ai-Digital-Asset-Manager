"""
utils/rag.py
============
Shared RAG (Retrieval-Augmented Generation) helpers.
All retrieval logic is preserved exactly from the previous implementation.
Pages import from here so there is no duplication.
"""

import os
import re
import logging

import numpy as np
from groq import Groq

from utils.embeddings import get_embedding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------
STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "at", "to", "for",
    "and", "or", "but", "with", "by", "from", "that", "this", "it", "its",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "which", "who", "whom", "whose", "how", "when", "where", "why",
    "their", "them", "they", "we", "our", "your", "his", "her", "my",
    "please", "tell", "me", "about", "give", "find", "show", "list",
}

# ---------------------------------------------------------------------------
# Synonym map
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, list[str]] = {
    "college": ["university", "institution", "school", "institute"],
    "university": ["college", "institution", "school", "institute"],
    "institution": ["college", "university", "school", "institute"],
    "school": ["college", "university", "institution", "institute"],
    "institute": ["college", "university", "institution", "school"],
    "degree": ["b.tech", "bachelor", "btech", "engineering", "graduation"],
    "bachelor": ["degree", "b.tech", "btech", "engineering", "graduation"],
    "education": ["degree", "college", "university", "qualification", "cgpa"],
}


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def extract_query_keywords(query: str) -> list[str]:
    query_norm = re.sub(r"'s\b", "", query, flags=re.IGNORECASE)
    tokens = re.findall(r"[a-zA-Z]+", query_norm.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _expand_keywords(keywords: list[str]) -> list[str]:
    expanded = list(keywords)
    for kw in keywords:
        for syn in _SYNONYMS.get(kw, []):
            if syn not in expanded:
                expanded.append(syn)
    return expanded


def score_chunk(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text_lower = text.lower()
    matched = sum(
        1 for kw in keywords
        if kw in text_lower or any(syn in text_lower for syn in _SYNONYMS.get(kw, []))
    )
    return matched / len(keywords)


def hybrid_score(semantic_score: float, keyword_coverage: float,
                 semantic_weight: float = 0.80) -> float:
    sem = float(np.clip(semantic_score, 0.0, 1.0))
    kw = float(np.clip(keyword_coverage, 0.0, 1.0))
    return semantic_weight * sem + (1 - semantic_weight) * kw


def highlight_text(text: str, query: str) -> str:
    words = query.split()
    for word in words:
        pattern = re.compile(f"({re.escape(word)})", re.IGNORECASE)
        text = pattern.sub(r"**\1**", text)
    return text


# ---------------------------------------------------------------------------
# Core retrieval — unchanged from previous implementation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Strict Document Isolation & Matching Helpers
# ---------------------------------------------------------------------------

def chunk_matches_doc(chunk_meta: dict, document_id: str) -> bool:
    """
    Check whether a chunk metadata dictionary strictly belongs to the specified document.
    Prioritizes stable document identifiers (doc_id, hash) before falling back to filename.
    """
    if not document_id or not chunk_meta:
        return False
    doc_id_val = str(document_id).strip()
    
    # 1. Match against stable doc_id
    if chunk_meta.get("doc_id") and str(chunk_meta["doc_id"]).strip() == doc_id_val:
        return True
    
    # 2. Match against content hash
    if chunk_meta.get("hash") and str(chunk_meta["hash"]).strip() == doc_id_val:
        return True
        
    # 3. Match against filename (backward compatibility)
    if chunk_meta.get("file") and str(chunk_meta["file"]).strip() == doc_id_val:
        return True
        
    return False


def validate_document_context(context_chunks: list[dict], expected_doc_id: str) -> None:
    """
    Defensive validation: asserts that 100% of the candidate chunks in context
    belong to the expected document.
    Raises ValueError immediately if cross-document contamination is detected.
    """
    if not expected_doc_id:
        return
    for c in context_chunks:
        if not chunk_matches_doc(c, expected_doc_id):
            leak_file = c.get("file", "unknown")
            leak_doc_id = c.get("doc_id", "unknown")
            err_msg = (
                f"CROSS-DOCUMENT RETRIEVAL LEAK DETECTED! "
                f"Expected document '{expected_doc_id}', but found chunk from "
                f"file='{leak_file}' (doc_id='{leak_doc_id}'). Aborting LLM request."
            )
            logger.error("[rag] %s", err_msg)
            raise ValueError(err_msg)


# ---------------------------------------------------------------------------
# Document-Scoped Retrieval (Strict Document Isolation)
# ---------------------------------------------------------------------------

def retrieve_document_context(
    question: str,
    index,
    metadata: list[dict],
    document_id: str,
    k: int = 25,
) -> tuple[list[dict], list[dict]]:
    """
    Retrieve candidate chunks strictly scoped to a specific document.
    Guarantees that NO chunks from other documents can ever enter the candidate pool.

    Parameters
    ----------
    question    : user question
    index       : FAISS index
    metadata    : list of chunk metadata dicts
    document_id : stable document ID or filename
    k           : maximum number of chunks to return

    Returns
    -------
    (deduped_scored_chunks, context_chunks)
    """
    if not document_id or index.ntotal == 0 or len(metadata) == 0:
        return [], []

    # 1. Identify all chunk indices belonging exclusively to this document
    doc_chunk_indices = [
        i for i, m in enumerate(metadata)
        if chunk_matches_doc(m, document_id)
    ]

    doc_display_name = (
        metadata[doc_chunk_indices[0]].get("file", document_id)
        if doc_chunk_indices else document_id
    )

    if not doc_chunk_indices:
        logger.warning(
            "[rag] No chunks found matching document_id='%s'. Returning empty context without fallback.",
            document_id
        )
        return [], []

    # 2. Compute query embedding
    q_vec = get_embedding(question).astype("float32")
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    # 3. Compute semantic similarity strictly for this document's vectors
    try:
        doc_vectors = np.array([index.reconstruct(int(i)) for i in doc_chunk_indices], dtype="float32")
        sem_scores = np.dot(doc_vectors, q_vec)
    except Exception as e:
        logger.warning("[rag] index.reconstruct failed (%s), falling back to index search", e)
        # Fallback for mock indices or indices without reconstruct:
        k_search = min(index.ntotal, len(metadata))
        D, I = index.search(np.array([q_vec]), k=k_search)
        doc_indices_set = set(doc_chunk_indices)
        scores_map = {}
        for rank, idx in enumerate(I[0]):
            if idx in doc_indices_set:
                scores_map[idx] = float(D[0][rank])
        sem_scores = np.array([scores_map.get(i, 0.0) for i in doc_chunk_indices], dtype="float32")

    keywords = extract_query_keywords(question)
    scored: list[dict] = []

    for rank, (chunk_pos, sem_score) in enumerate(zip(doc_chunk_indices, sem_scores)):
        m = metadata[chunk_pos]
        # Strict defensive check: verify chunk belongs to document_id
        if not chunk_matches_doc(m, document_id):
            continue

        sem = float(sem_score)
        kw_coverage = score_chunk(m["text"], keywords)
        score = hybrid_score(sem, kw_coverage, semantic_weight=0.65)

        scored.append({
            "doc_id": m.get("doc_id") or m.get("hash") or m.get("file"),
            "file": m["file"],
            "hash": m.get("hash", ""),
            "text": m["text"],
            "chunk_index": m.get("chunk_index", -1),
            "sem_score": sem,
            "kw_coverage": kw_coverage,
            "score": score,
            "faiss_idx": int(chunk_pos),
        })

    # Sort descending by score
    scored.sort(key=lambda x: x["score"], reverse=True)

    # De-duplicate
    seen: set = set()
    deduped: list[dict] = []
    for c in scored:
        key = (c["file"], c["chunk_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    # Build context strictly from this document's chunks
    CONTEXT_CHUNKS = 6
    MAX_CONTEXT_CHARS = 4000
    MIN_SCORE_THRESHOLD = 0.10

    eligible = [c for c in deduped if c["score"] >= MIN_SCORE_THRESHOLD]
    context_chunks: list[dict] = []
    total_chars = 0
    for c in eligible[:CONTEXT_CHUNKS]:
        tag = f"[Source: {c['file']} | Chunk {c['chunk_index']}]"
        chunk_str = f"{tag}\n{c['text']}"
        if total_chars + len(chunk_str) > MAX_CONTEXT_CHARS:
            break
        context_chunks.append(c)
        total_chars += len(chunk_str)

    # Defensive validation: verify every single context chunk
    validate_document_context(context_chunks, document_id)

    # Diagnostic logging (Section 7)
    logger.info(
        "[rag] Document retrieval stats: "
        "selected_doc_id=%s, selected_file=%s, "
        "total_doc_candidates=%d, remaining_scored=%d, "
        "final_context_chunks=%d, source_doc_ids=%s, source_files=%s",
        document_id, doc_display_name,
        len(doc_chunk_indices), len(scored),
        len(context_chunks),
        list(set(c.get("doc_id", "") for c in context_chunks)),
        list(set(c.get("file", "") for c in context_chunks)),
    )

    return deduped, context_chunks


# ---------------------------------------------------------------------------
# Global Retrieval / Multi-document search
# ---------------------------------------------------------------------------

def retrieve_chunks(
    question: str,
    index,
    metadata: list[dict],
    k: int = 25,
    document_filter: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Run hybrid semantic + keyword retrieval.
    If document_filter is specified, strictly delegates to retrieve_document_context.
    Otherwise, runs global multi-document search across FAISS candidates.
    """
    if document_filter:
        return retrieve_document_context(
            question=question,
            index=index,
            metadata=metadata,
            document_id=document_filter,
            k=k,
        )

    if index.ntotal == 0:
        return [], []

    k_actual = min(k, index.ntotal)
    q_vec = get_embedding(question).astype("float32")
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    D, I = index.search(np.array([q_vec]), k=k_actual)
    keywords = extract_query_keywords(question)
    logger.info("[rag] global search keywords: %s, candidates: %d", keywords, k_actual)

    scored: list[dict] = []
    for rank, idx in enumerate(I[0]):
        if idx >= len(metadata):
            continue
        m = metadata[idx]

        sem_score = float(D[0][rank])
        kw_coverage = score_chunk(m["text"], keywords)
        score = hybrid_score(sem_score, kw_coverage, semantic_weight=0.65)

        scored.append({
            "doc_id": m.get("doc_id") or m.get("hash") or m.get("file"),
            "file": m["file"],
            "hash": m.get("hash", ""),
            "text": m["text"],
            "chunk_index": m.get("chunk_index", -1),
            "sem_score": sem_score,
            "kw_coverage": kw_coverage,
            "score": score,
            "faiss_idx": int(idx),
            "faiss_rank": rank + 1,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # De-duplicate
    seen: set = set()
    deduped: list[dict] = []
    for c in scored:
        key = (c["file"], c["chunk_index"])
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    # Build context
    CONTEXT_CHUNKS = 6
    MAX_CONTEXT_CHARS = 4000
    MIN_SCORE_THRESHOLD = 0.10

    eligible = [c for c in deduped if c["score"] >= MIN_SCORE_THRESHOLD]
    context_chunks: list[dict] = []
    total_chars = 0
    for c in eligible[:CONTEXT_CHUNKS]:
        tag = f"[Source: {c['file']} | Chunk {c['chunk_index']}]"
        chunk_str = f"{tag}\n{c['text']}"
        if total_chars + len(chunk_str) > MAX_CONTEXT_CHARS:
            break
        context_chunks.append(c)
        total_chars += len(chunk_str)

    return deduped, context_chunks


def call_llm(
    question: str,
    context_chunks: list[dict],
    expected_doc_id: str | None = None,
) -> str:
    """
    Call Groq LLM with the assembled context.
    If expected_doc_id is provided, defensively asserts all chunks belong to expected_doc_id.
    Returns the answer string or raises on API error.
    """
    if expected_doc_id:
        validate_document_context(context_chunks, expected_doc_id)

    context_parts = []
    for c in context_chunks:
        tag = f"[Source: {c['file']} | Chunk {c['chunk_index']}]"
        context_parts.append(f"{tag}\n{c['text']}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a precise document question-answering assistant.\n"
        "You will be given excerpts from one or more documents, each clearly labelled "
        "with its source filename and chunk number.\n\n"
        "RULES:\n"
        "1. Answer ONLY using information explicitly stated in the provided excerpts.\n"
        "2. If the answer spans multiple sources, cite each source by its filename.\n"
        "3. Do NOT infer, guess, or use external knowledge.\n"
        "4. Do NOT transfer facts from one named person's document to another person.\n"
        "5. If the answer is genuinely not present, reply: "
        "\"The information is not available in the provided documents.\"\n"
        "6. Keep your answer concise and factual."
    )

    user_message = f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}"

    client = Groq()
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content
