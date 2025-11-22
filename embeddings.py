import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> np.ndarray:
    vec = model.encode(text)
    return vec.astype("float32")


