import streamlit as st
import os
import numpy as np
import hashlib

from utils.extract import extract_text
from utils.embeddings import get_embedding
from utils.db import load_index, save_index, add_to_index

def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# Chunking helper
def chunk_text(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("database", exist_ok=True)

# Load DB
index, metadata = load_index()

# UI
st.set_page_config(page_title="AI DAM", layout="wide")
st.title("📂 AI Digital Asset Manager")

# ---------------- UPLOAD ----------------
st.header("Upload File")

uploaded_file = st.file_uploader("Upload PDF or DOCX")

if uploaded_file:
    file_bytes = uploaded_file.getbuffer()
    file_hash = get_file_hash(file_bytes)

    # Check duplicates
    existing_hashes = [item.get("hash") for item in metadata]

    if file_hash in existing_hashes:
        st.warning("⚠️ Duplicate file detected! Already uploaded.")
    else:
        file_path = os.path.join("uploads", uploaded_file.name)

        # Save file
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        st.success("File uploaded!")

        # Extract text
        text = extract_text(file_path)

        if text.strip() == "":
            st.warning("No text found in file!")
        else:
            # Split into chunks
            chunks = chunk_text(text)

            for chunk in chunks:
                embedding = get_embedding(chunk)
                add_to_index(index, metadata, embedding, uploaded_file.name, chunk, file_hash)

            save_index(index, metadata)

            st.success(f"File processed & stored in {len(chunks)} chunks!")

# ---------------- SEARCH ----------------
st.header("Search Files")

query = st.text_input("Search anything (semantic search)")

if query:
    query_vec = get_embedding(query)

    D, I = index.search(np.array([query_vec]), k=5)

    st.subheader("Results")

    for i, idx in enumerate(I[0]):
        if idx < len(metadata):
            file_data = metadata[idx]
            score = D[0][i]

            st.write(f"📄 {file_data['file']}")
            st.caption(f"Score: {score:.4f}")

            # Preview snippet
            st.caption(file_data['text'][:200] + "...")