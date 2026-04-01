from sentence_transformers import SentenceTransformer

import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    if not text or text.strip() == "":
        return np.zeros(384, dtype="float32")
    return model.encode(text).astype("float32")