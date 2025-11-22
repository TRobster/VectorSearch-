from sentence_transformers import SentenceTransformer
from fastapi import FastAPI

app = FastAPI()

# Load model ONCE at application startup
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/embed")
def embed_text(text: str):
    vec = model.encode([text])[0]      # compute semantic vector
    return {"embedding": vec.tolist()} # send it back as JSON
