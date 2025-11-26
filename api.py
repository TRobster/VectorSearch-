from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from embeddings import embed
from search import build_store_from_docs
from data import DOCUMENTS


app = FastAPI()

# Build an in-memory vector store at import time (small demo dataset)
store = build_store_from_docs(DOCUMENTS)


class EmbedRequest(BaseModel):
    text: str


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.post("/embed")
def embed_text(req: EmbedRequest):
    vec = embed(req.text)
    # ensure it's serializable
    return {"embedding": vec.tolist()}


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """Return top-k documents for `req.query` from the in-memory store.

    Response structure:
      { "results": [ {"id":..., "text":..., "score":...}, ... ] }
    """
    results = store.search(req.query, k=req.k)
    out: List[Dict[str, Any]] = []
    for doc, score in results:
        out.append({"id": doc.get("id"), "text": doc.get("text"), "score": float(score)})
    return {"results": out}
