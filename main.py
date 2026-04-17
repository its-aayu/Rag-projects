from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedding_model
from src.vectordb import create_vector_store
from src.query import query_db
from pathlib import Path


DATA_PATH = Path("data/ConstitutionofIndia.txt")


def main():
    print("Loading documents and building vector store...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Add a text file there before running this app."
        )

    docs = load_documents(str(DATA_PATH))
    chunks = chunk_documents(docs)

    embedding = get_embedding_model()
    db = create_vector_store(chunks, embedding)
    print("Ready. Type your question, or press Ctrl+C to stop.")

    while True:
        try:
            q = input("Ask your question: \n").strip()
        except EOFError:
            break

        if q.lower() in {"exit", "quit"}:
            break

        if not q:
            continue

        results = query_db(db, q)

        if not results:
            print("No matching documents found.")
            continue

        for r in results:
            print("\n", r.page_content)

if __name__ == "__main__":
    main()
