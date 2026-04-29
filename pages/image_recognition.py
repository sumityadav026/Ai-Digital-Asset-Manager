"""
pages/image_recognition.py
===========================
Image upload, CLIP label detection, and OCR — all logic preserved exactly.
"""
import os
import numpy as np
import streamlit as st
import torch
import clip
from PIL import Image
import pytesseract
import hashlib

from utils.embeddings import get_embedding
from utils.db import add_to_index, save_index
from utils.ui import render_section_header, render_empty_state, PRIMARY, SURFACE, BORDER, TEXT_MUTED


_LABELS = [
    "car", "invoice", "receipt", "laptop", "mobile phone",
    "document", "watch", "person", "screen", "vehicle",
]

_LABEL_ICONS = {
    "car": "🚗", "invoice": "🧾", "receipt": "🧾", "laptop": "💻",
    "mobile phone": "📱", "document": "📄", "watch": "⌚",
    "person": "👤", "screen": "🖥", "vehicle": "🚙",
}


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


def render(metadata: list, index, clip_model, preprocess, device: str):
    st.markdown("## Image Recognition")
    st.markdown(
        '<p style="color:#6B7280;font-size:0.88rem;margin-top:-0.5rem;">'
        'Upload an image for CLIP-based label detection and Tesseract OCR text extraction.</p>',
        unsafe_allow_html=True,
    )

    render_section_header("Upload Image")
    image_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        key="img_upload",
        label_visibility="collapsed",
    )

    if not image_file:
        render_empty_state(
            "🖼", "No image uploaded",
            "Supported formats: PNG, JPG, JPEG. The image will be analysed with CLIP and OCR."
        )
        return

    # ── Preview ───────────────────────────────────────────────────────────────
    image = Image.open(image_file)
    col_img, col_info = st.columns([1, 1])
    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        with st.status("Analysing image…", expanded=True) as status:
            # ── OCR ───────────────────────────────────────────────────────────
            st.write("Running OCR (Tesseract)…")
            extracted_text = pytesseract.image_to_string(image)

            # ── CLIP ──────────────────────────────────────────────────────────
            st.write("Running CLIP label detection…")
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_tokens = clip.tokenize(_LABELS).to(device)

            with torch.no_grad():
                img_feats = clip_model.encode_image(image_input)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                txt_feats = clip_model.encode_text(text_tokens)
                txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
                similarity = (img_feats @ txt_feats.T).softmax(dim=-1)

            scores = similarity[0].cpu().tolist()
            best_idx = int(similarity.argmax().item())
            best_label = _LABELS[best_idx]

            # ── Embed & store ─────────────────────────────────────────────────
            visual_text = f"This image contains: {best_label}"
            emb = get_embedding(visual_text).astype("float32")
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            add_to_index(
                index, metadata, emb,
                "image_visual", visual_text,
                _get_file_hash(image_file.getvalue()),
                chunk_index=0,
            )

            if extracted_text.strip():
                chunks = _chunk_text(extracted_text)
                for ci, chunk in enumerate(chunks):
                    cemb = get_embedding(chunk).astype("float32")
                    cemb = cemb / (np.linalg.norm(cemb) + 1e-10)
                    add_to_index(
                        index, metadata, cemb,
                        "image_ocr", chunk,
                        _get_file_hash(chunk.encode()),
                        chunk_index=ci,
                    )

            save_index(index, metadata)
            status.update(label="Analysis complete!", state="complete")

        # ── Results ───────────────────────────────────────────────────────────
        with col_info:
            st.markdown("### Detection Result")
            label_icon = _LABEL_ICONS.get(best_label, "🏷")
            confidence = int(scores[best_idx] * 100)

            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
                 padding:1.2rem;text-align:center;margin-bottom:1rem">
              <div style="font-size:2.5rem">{label_icon}</div>
              <div style="font-size:1.1rem;font-weight:700;color:#1A1D23;margin:0.3rem 0">
                {best_label.title()}
              </div>
              <div style="font-size:0.8rem;color:{TEXT_MUTED}">Confidence: {confidence}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Label scores
            st.markdown("**All label scores:**")
            sorted_labels = sorted(zip(_LABELS, scores), key=lambda x: x[1], reverse=True)
            for lbl, sc in sorted_labels:
                pct = int(sc * 100)
                bar_width = max(pct, 2)
                color = PRIMARY if lbl == best_label else BORDER
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;
                     font-size:0.8rem;">
                  <span style="width:90px;color:#374151">{lbl}</span>
                  <div style="flex:1;background:#F3F4F6;border-radius:4px;height:8px;">
                    <div style="width:{bar_width}%;background:{color};border-radius:4px;height:8px"></div>
                  </div>
                  <span style="width:32px;color:{TEXT_MUTED};text-align:right">{pct}%</span>
                </div>
                """, unsafe_allow_html=True)

        # ── OCR text ──────────────────────────────────────────────────────────
        render_section_header("Extracted Text (OCR)")
        if extracted_text.strip():
            st.success(f"OCR found {len(extracted_text.split())} words — stored in search index.")
            with st.expander("View extracted text", expanded=True):
                st.text(extracted_text)
        else:
            st.info("No text was detected in this image.")

    except Exception as e:
        st.error(f"Image processing error: {repr(e)}")
