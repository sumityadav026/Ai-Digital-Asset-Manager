import torch
import clip
from PIL import Image
import pytesseract
import streamlit as st
import os
import numpy as np
import hashlib

from groq import Groq

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

import re

# Highlight helper
def highlight_text(text, query):
    words = query.split()
    for word in words:
        pattern = re.compile(f"({word})", re.IGNORECASE)
        text = pattern.sub(r"**\1**", text)
    return text

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("database", exist_ok=True)

# Load DB
index, metadata = load_index()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

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
                embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
                add_to_index(index, metadata, embedding, uploaded_file.name, chunk, file_hash)

            save_index(index, metadata)

            st.success(f"File processed & stored in {len(chunks)} chunks!")

# ---------------- SEARCH ----------------
st.header("Search Files")

query = st.text_input("Search anything (semantic search)")

if query:
    query_vec = get_embedding(query)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    D, I = index.search(np.array([query_vec]), k=10)

    file_best = {}
    query_lower = query.lower()

    for i, idx in enumerate(I[0]):
        if idx < len(metadata):
            file_data = metadata[idx]
            base_score = D[0][i]

            text = file_data['text'].lower()

            # Keyword bonus
            keyword_bonus = 0
            for word in query_lower.split():
                if word in text:
                    keyword_bonus -= 0.05

            final_score = base_score + keyword_bonus

            file_name = file_data["file"]

            # Keep only best chunk per file
            if file_name not in file_best or final_score < file_best[file_name]["score"]:
                file_best[file_name] = {
                    "file": file_name,
                    "text": file_data["text"],
                    "score": final_score
                }

    # Convert to list and sort
    results = list(file_best.values())
    results = sorted(results, key=lambda x: x["score"])

    st.subheader("Results")

    for res in results[:5]:
        st.write(f"📄 {res['file']}")
        st.caption(f"Score: {res['score']:.4f}")
        preview = res['text'][:200]
        highlighted = highlight_text(preview, query)
        st.markdown(highlighted + "...")
        st.markdown("---")


# ---------------- CHAT WITH DOCUMENTS ----------------
#
# ---------------- MULTIMODAL IMAGE PROCESSING ----------------
st.header("Image Recognition (Multimodal)")

image_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png","jpg","jpeg"], key="image_upload")

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", width=700)

    try:
        extracted_text = pytesseract.image_to_string(image)

        if extracted_text.strip() == "":
            st.warning("No text detected in image.")
        else:
            st.subheader("Extracted Text")
            st.write(extracted_text)

            # OCR text embedding
            chunks = chunk_text(extracted_text)
            for chunk in chunks:
                emb = get_embedding(chunk)
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                add_to_index(index, metadata, emb, "image_ocr", chunk, get_file_hash(chunk.encode()))

        # CLIP visual embedding
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        img_embedding = image_features.cpu().numpy().astype("float32")[0]

        # ⚠️ Skip adding CLIP embedding to FAISS (dimension mismatch with text embeddings)
        # Instead, store only as metadata note (future extension)
        metadata.append({
            "file": "image_visual",
            "text": "This image likely contains visual objects (CLIP embedding stored separately)",
            "hash": get_file_hash(image_file.getvalue())
        })

        save_index(index, metadata)

        st.success("Image added with OCR + CLIP embeddings!")

    except Exception as e:
        st.error(f"Image processing error: {repr(e)}")
st.header("Chat with Documents")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

question = st.text_input("Ask something about your documents", key="chat_input")

if question:
    q_vec = get_embedding(question)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    D, I = index.search(np.array([q_vec]), k=3)

    confidence_scores = []

    context = ""
    sources = []

    query_words = question.lower().split()

    for idx in I[0]:
        if idx < len(metadata):
            text = metadata[idx]["text"]
            file = metadata[idx]["file"]

            text_lower = text.lower()

            # Only include chunks that actually match query keywords
            if any(word in text_lower for word in query_words):
                context += text + "\n"

                sources.append({
                    "file": file,
                    "text": text[:200]
                })

                confidence_scores.append(float(D[0][list(I[0]).index(idx)]))

    if context.strip() == "":
        st.warning("No relevant information found in documents.")
        st.stop()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is not set. Please configure your API key.")
        st.stop()

    client = Groq()

    try:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a strict document QA assistant. Answer ONLY using exact information from the provided context. Do NOT guess or infer. If the answer is not explicitly present, reply exactly: 'Not found in documents'."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            )
        answer_text = response.choices[0].message.content
        st.session_state.chat_history.append((question, answer_text))

        col1, col2 = st.columns([2,1])

        with col1:
            st.subheader("Answer")
            st.write(answer_text)

        with col2:
            if confidence_scores:
                avg_score = sum(confidence_scores) / len(confidence_scores)
                confidence = max(0, 1 - avg_score)
                st.metric("Confidence", f"{confidence:.2f}")

        st.markdown("### 📄 Sources Used")

        st.subheader("Sources")

        for src in sources:
            st.write(f"📄 {src['file']}")
            highlighted_text = highlight_text(src["text"], question)
            st.markdown(highlighted_text)
            st.markdown("---")

        if confidence_scores:
            st.caption(f"Confidence Score: {confidence:.2f}")

        st.subheader("Chat History")
        for q, a in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

    except Exception as e:
        st.error(f"Error: {str(e)}")