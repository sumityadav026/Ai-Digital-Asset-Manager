import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "database/faiss_index.bin"
META_PATH = "database/metadata.pkl"
DIMENSION = 384

def load_index():
    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(DIMENSION)
        metadata = []

    return index, metadata
# drdt

def save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


def add_to_index(index, metadata, embedding, file_name, text, file_hash):
    index.add(np.array([embedding]))
    metadata.append({
        "file": file_name,
        "text": text,
        "hash": file_hash
    })