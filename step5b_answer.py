import ollama

from step4_retrieve import build_store
from step5_prompt import retrieve, build_prompt

MODEL = "qwen3:4b"  
NO_ANSWER = "I don't know based on the provided documents."


def generate(prompt: str) -> str:
    """Send a prompt to the local LLM and return its text answer."""
    response = ollama.generate(
        model=MODEL,
        prompt=prompt,
        options={"temperature": 0},   # 0 = least random, same prompt gives (nearly) the same answer
    )
    return response["response"].strip()


def ask(store, question: str) -> dict:
    """The full RAG pipeline for one question."""
    chunks = retrieve(store, question)
    if not chunks:
        return {"answer": NO_ANSWER, "chunks": []}   # skip the LLM entirely
    prompt = build_prompt(question, chunks)
    return {"answer": generate(prompt), "chunks": chunks}


if __name__ == "__main__":
    store = build_store()

    questions = [
        "When should I stop drinking coffee before bed?",
        "What is the capital of France?",
        "How much caffeine is in coffee?",
    ]

    for q in questions:
        result = ask(store, q)
        print("=" * 60)
        print(f"QUESTION: {q}")
        print(f"ANSWER:   {result['answer']}")
        sources = sorted({c["source"] for c in result["chunks"]})
        print(f"SOURCES:  {sources}")