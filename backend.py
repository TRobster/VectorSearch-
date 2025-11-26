from data import DOCUMENTS
from search import build_store_from_docs


def run_demo():
    store = build_store_from_docs(DOCUMENTS)
    query = "semantic vector search for embeddings"
    print(f"Query: {query}\n")
    results = store.search(query, k=3)
    for doc, score in results:
        print(f"- id={doc['id']} score={score:.4f}\n  text={doc['text']}")


if __name__ == "__main__":
    run_demo()
