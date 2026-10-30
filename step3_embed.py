import numpy as np 
from sentence_transformers import SentenceTransformer

from step1_load import load_documents
from step2_chunk import chunk_documents

MODEL_NAME = "all-MiniLM-L6-v2"

def load_model() -> SentenceTransformer:
    """Load the embedding model (downloads it the first time)."""
    return SentenceTransformer(MODEL_NAME)

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.darray:
    """Turn a list of texts into a 2D array: one row (vector) per text."""
    return model.encode(texts, normalize_embeddings=True)

if __name__ == "__main__":
    model = load_model()

    # Part A: embed our real chunks 
    docs = load_documents("docs")
    chunks = chunk_documents(docs)
    vectors = embed_texts(model, [c["text"] for c in chunks])

    print("=== Part A: our chunks ===")
    print(f"Chunks: {len(chunks)}")
    print(f"Vector array shape: {vectors.shape}")
    print(f"First 5 numbers of chunk 0's vector: {vectors[0][:5]}")
    print(f"Length of chunk 0's vector: {np.linalg.norm(vectors[0]):.3f}\n")

    # Part B: see similarity 
    sentences = [
        "How much caffeine is in tea?",
        "A typical cup of black tea contains about 40 milligrams of caffeine.",
        "Avoid caffeine in the hours before you go to bed.",
        "The Eiffel Tower is located in Paris.",
    ]

    sent_vectors = embed_texts(model, sentences)
    similarity = sent_vectors @ sent_vectors.T   # every sentence vs every sentence

    print("=== Part B: similarity matrix ===")
    for i, s in enumerate(sentences):
        print(f"{i}: {s}")
    print()
    print(np.round(similarity, 2))
