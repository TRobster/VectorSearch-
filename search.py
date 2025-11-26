import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import time
import logging


# load model locally to avoid depending on the embeddings module
_model = SentenceTransformer("all-MiniLM-L6-v2")
_logger = logging.getLogger(__name__)


def _embed(texts: List[str]) -> np.ndarray:
    start = time.perf_counter()
    embs = _model.encode(texts, convert_to_numpy=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _logger.info("embed: encoded %d texts in %.2f ms", len(texts), elapsed_ms)
    return embs


class VectorStore:
    """In-memory vector store that keeps documents and their embeddings.

    Search is implemented with brute-force cosine similarity for simplicity.
    """

    def __init__(self):
        self.docs: List[Dict[str, Any]] = []
        self.embs: np.ndarray = np.zeros((0,))

    def add_documents(self, documents: List[Dict[str, str]]):
        """Add documents to the store and compute embeddings for them.

        Each document is a dict with at least keys: `id` and `text`.
        """
        texts = [d["text"] for d in documents]
        vectors = _embed(texts)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        if self.embs.size == 0:
            self.embs = vectors
        else:
            self.embs = np.vstack([self.embs, vectors])

        self.docs.extend(documents)

    def search(self, query: str, k: int = 5):
        """Return top-k documents most similar to the query.

        Returns a list of tuples `(doc, score)` ordered by descending score.
        """
        start = time.perf_counter()

        qvec = _embed([query])[0]
        if qvec.ndim != 1:
            qvec = qvec[0]

        # compute cosine similarities
        sims = []
        for v in self.embs:
            sims.append(float(np.dot(qvec, v) / (np.linalg.norm(qvec) * np.linalg.norm(v) + 1e-12)))
        sims = np.array(sims)
        if sims.size == 0:
            return []

        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            results.append((self.docs[int(i)], float(sims[int(i)])))

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _logger.info("search: query='%s' k=%d in %.2f ms", query, k, elapsed_ms)
        return results


def build_store_from_docs(docs: List[Dict[str, str]]) -> VectorStore:
    store = VectorStore()
    store.add_documents(docs)
    return store
