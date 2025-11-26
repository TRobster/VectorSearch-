import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

"""embeddings.py

Small, stable wrapper around SentenceTransformer that returns numpy arrays.
Provides:
- `model`: the loaded SentenceTransformer instance
- `embed(texts)`: returns a numpy embedding for a string or list of strings
- `cosine_sim(a, b)`: cosine similarity between two 1-D numpy vectors
"""

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(texts: Union[str, List[str]]) -> np.ndarray:
    """Compute embeddings for a single string or a list of strings.

    Args:
        texts: single string or list of strings

    Returns:
        If `texts` is a single string, returns a 1-D numpy array (dim,).
        If `texts` is a list, returns a 2-D numpy array (num_texts, dim).
    """
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True

    embs = model.encode(texts, convert_to_numpy=True)
    if single:
        return embs[0]
    return embs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D numpy vectors.

    Returns 0.0 when either vector has zero norm.
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1-D numpy arrays")
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
