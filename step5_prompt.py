from step4_retrieve import build_store

MIN_SCORE = 0.3   # chunks scoring below this are treated as "not relevant"


def retrieve(store, question: str, top_k: int = 3, min_score: float = MIN_SCORE) -> list[dict]:
    """Search, then drop any chunk whose score is below the threshold."""
    results = store.search(question, top_k=top_k)
    return [r for r in results if r["score"] >= min_score]


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Combine instructions, retrieved context, and the question into one prompt."""
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}]\n{c['text']}")
    context = "\n\n".join(context_blocks)

    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say \"I don't know based on the provided documents.\"\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )


if __name__ == "__main__":
    store = build_store()

    for q in ["When should I stop drinking coffee before bed?", "What is the capital of France?"]:
        chunks = retrieve(store, q)
        print("=" * 60)
        print(f"QUESTION: {q}")
        print(f"Chunks kept after threshold: {len(chunks)}")
        if not chunks:
            print("-> Nothing relevant found. We would skip the LLM entirely.\n")
            continue
        print("\n--- PROMPT THE LLM WOULD RECEIVE ---")
        print(build_prompt(q, chunks))
        print()