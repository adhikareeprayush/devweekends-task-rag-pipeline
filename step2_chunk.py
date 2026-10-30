from step1_load import load_documents

def chunk_text(text: str, chunk_size: int = 200, overlap: int=50) -> list[str]:
    """Split text into chunks of chunk_size characters,
    where each chunk starts (chunk_size - overlap) characters after the previous one."""
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks

def chunk_documents(documents: list[dict], chunk_size: int = 200, overlap: int = 50) -> list[dict]:
        """Chunk every document, keeping track of where each chunk came from."""
        chunked_documents = []
        for doc in documents:
            chunks = chunk_text(doc["text"], chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                chunked_documents.append({
                    "source": doc["source"],
                    "chunk_index": i,
                    "text": chunk
                })
        return chunked_documents

if __name__ == "__main__":
    docs = load_documents("docs")
    chunked_docs = chunk_documents(docs)
    print(f"Chunked {len(docs)} documents into {len(chunked_docs)} chunks.")
    for doc in chunked_docs:
        print(f"Source: {doc['source']}, Chunk Index: {doc['chunk_index']}, Chunk Length: {len(doc['text'])} characters")
