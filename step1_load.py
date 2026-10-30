from pathlib import Path

def load_documents(folder: str) -> list[dict]: 
    documents = [] 
    for path in sorted(Path(folder).glob("*.txt")): 
        text = path.read_text(encoding="utf-8")
        documents.append({"source": path.name, "text": text})
    return documents

if __name__ == "__main__":
    docs = load_documents("docs")
    print(f"Loaded {len(docs)} documents from 'docs' folder.")
    for doc in docs:
        print(f"Source: {doc['source']}, Text Length: {len(doc['text'])} characters")

