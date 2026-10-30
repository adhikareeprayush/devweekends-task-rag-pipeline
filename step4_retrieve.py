import numpy as np

from step1_load import load_documents
from step2_chunk import chunk_documents
from step3_embed import load_model, embed_texts


class VectorStore:
    """Holds chunks and their vectors, and finds the chunks closest to a query."""

    def __init__(self, model):
        self.model = model
        self.chunks = []        # list of dicts: source, chunk_id, text
        self.vectors = None     # array of shape (num_chunks, 384)

    def add(self, chunks: list[dict]) -> None:
        """Embed all chunks and store them alongside their vectors."""
        self.chunks = chunks
        self.vectors = embed_texts(self.model, [c["text"] for c in chunks])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return the top_k chunks most similar to the query, with scores."""
        query_vector = embed_texts(self.model, [query])[0]   # shape (384,)
        scores = self.vectors @ query_vector                 # shape (num_chunks,)
        best = np.argsort(scores)[::-1][:top_k]              # indices, highest score first
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]


def build_store() -> VectorStore:
    """Run the whole indexing side: load -> chunk -> embed -> store."""
    model = load_model()
    docs = load_documents("docs")
    chunks = chunk_documents(docs)
    store = VectorStore(model)
    store.add(chunks)
    return store


if __name__ == "__main__":
    store = build_store()
    print(f"Store ready: {len(store.chunks)} chunks, vectors shape {store.vectors.shape}\n")
    
    questions = [
        "How much caffeine is in tea?",
        "When should I stop drinking coffee before bed?",
        "Where did coffee come from?",
        "What is the capital of France?",
    ]

    for q in questions:
        print(f"QUESTION: {q}")
        for r in store.search(q, top_k=3):
            print(f"  {r['score']:.3f}  [{r['source']} #{r['chunk_index']}]  {r['text'][:80]}...")
        print()