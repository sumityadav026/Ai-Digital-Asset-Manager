import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "database/faiss_index.bin"
META_PATH = "database/metadata.pkl"
DIMENSION = 384

# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------

def load_index():
    """Load the FAISS index and metadata list from disk, or create fresh ones."""
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)

        # Sanity check: if vectors and metadata rows are mismatched, reset.
        if index.ntotal != len(metadata):
            print(
                f"[db] WARNING: index has {index.ntotal} vectors but metadata has "
                f"{len(metadata)} entries — resetting to empty index."
            )
            index = faiss.IndexFlatIP(DIMENSION)
            metadata = []

        # Ensure all metadata records have a stable doc_id (backward compatibility migration)
        for m in metadata:
            if "doc_id" not in m:
                m["doc_id"] = m.get("hash") or m.get("file")

        # Detect a stale index built with L2 (IndexFlatL2) and migrate to IP.
        # IndexFlatIP is correct for pre-normalised embeddings (cosine similarity).
        if isinstance(index, faiss.IndexFlatL2):
            print("[db] Migrating index from L2 to Inner-Product (cosine similarity).")
            new_index = faiss.IndexFlatIP(DIMENSION)
            if index.ntotal > 0:
                vecs = np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype="float32")
                # Re-normalise just in case
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / (norms + 1e-10)
                new_index.add(vecs)
            index = new_index
    else:
        # Fresh start — use Inner-Product for cosine similarity on normalised vectors
        index = faiss.IndexFlatIP(DIMENSION)
        metadata = []

    return index, metadata


def save_index(index, metadata):
    """Persist the FAISS index and metadata to disk."""
    os.makedirs("database", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def add_to_index(index, metadata, embedding, file_name, text, file_hash,
                 chunk_index: int = 0, doc_id: str | None = None):
    """
    Add a single normalised embedding and its metadata to the index.

    Parameters
    ----------
    index       : faiss.Index
    metadata    : list[dict]
    embedding   : np.ndarray, shape (384,), already L2-normalised
    file_name   : str  – original filename
    text        : str  – chunk text
    file_hash   : str  – MD5 of the source file
    chunk_index : int  – position of this chunk within its source file
    doc_id      : str  – stable document ID (defaults to file_hash)
    """
    if doc_id is None:
        doc_id = file_hash
    index.add(np.array([embedding]))
    metadata.append(
        {
            "doc_id": doc_id,
            "file": file_name,
            "text": text,
            "hash": file_hash,
            "chunk_index": chunk_index,
        }
    )


def rebuild_index_from_metadata(metadata, embeddings: np.ndarray):
    """
    Create a brand-new IndexFlatIP and populate it from pre-computed embeddings.
    Used during re-indexing workflows.
    """
    index = faiss.IndexFlatIP(DIMENSION)
    if len(embeddings):
        index.add(embeddings)
    return index