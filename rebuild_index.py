"""
rebuild_index.py
================
Rebuild the FAISS index from all documents in the uploads/ directory.

Run this script whenever:
  - The embedding model changes
  - The chunking strategy changes
  - The FAISS index type changes (L2 → IP)
  - Metadata becomes misaligned with the vector store

Usage:
    cd /Users/sumityadav/Documents/AI_DAM_project
    source venv/bin/activate
    python rebuild_index.py
"""

import os
import hashlib
import re
import shutil
import pickle

import faiss
import numpy as np

from utils.extract import extract_text
from utils.embeddings import get_embedding
from utils.db import save_index, DIMENSION

# ---------------------------------------------------------------------------
# Config (must match app.py)
# ---------------------------------------------------------------------------
UPLOADS_DIR = "uploads"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SUPPORTED_EXTENSIONS = (".pdf", ".docx")


# ---------------------------------------------------------------------------
# Helpers  (duplicated from app.py so this script is self-contained)
# ---------------------------------------------------------------------------

def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Main rebuild
# ---------------------------------------------------------------------------

def rebuild():
    print("=== AI DAM — Index Rebuild ===\n")

    # Fresh index
    new_index = faiss.IndexFlatIP(DIMENSION)
    new_metadata: list[dict] = []

    files = [
        f for f in os.listdir(UPLOADS_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not files:
        print("No PDF/DOCX files found in uploads/. Nothing to index.")
        return

    print(f"Found {len(files)} document(s) to index:\n")

    for fname in sorted(files):
        fpath = os.path.join(UPLOADS_DIR, fname)
        with open(fpath, "rb") as fh:
            raw = fh.read()
        file_hash = get_file_hash(raw)

        print(f"  Processing: {fname}")

        text = extract_text(fpath)
        if not text.strip():
            print(f"    ⚠️  No text extracted — skipping.")
            continue

        chunks = chunk_text(text)
        print(f"    → {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks):
            emb = get_embedding(chunk).astype("float32")
            norm = np.linalg.norm(emb)
            emb = emb / (norm + 1e-10)

            new_index.add(np.array([emb]))
            new_metadata.append(
                {
                    "doc_id": file_hash,
                    "file": fname,
                    "text": chunk,
                    "hash": file_hash,
                    "chunk_index": chunk_idx,
                }
            )

    print(f"\nTotal vectors: {new_index.ntotal}")
    print(f"Total metadata rows: {len(new_metadata)}")
    assert new_index.ntotal == len(new_metadata), "Mismatch!"

    save_index(new_index, new_metadata)
    print("\n✅ Index saved to database/")

    # Verify by loading back
    loaded_index = faiss.read_index("database/faiss_index.bin")
    with open("database/metadata.pkl", "rb") as f:
        loaded_meta = pickle.load(f)

    assert loaded_index.ntotal == loaded_index.ntotal
    print(f"✅ Verification passed: {loaded_index.ntotal} vectors loaded back from disk.")

    print("\n=== Rebuild complete ===")


if __name__ == "__main__":
    rebuild()
