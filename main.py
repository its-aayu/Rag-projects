from pathlib import Path

from src.chunker import chunk_documents
from src.embedder import get_embedding_model
from src.hybrid_retriever import BM25Retriever, hybrid_search
from src.llm import get_llm
from src.loader import load_documents
from src.rag_chain import generate_answer
from src.vectordb import create_vector_store, load_vector_store, vector_store_count
from src.reranker import Reranker

DATA_PATH = Path("data/ConstitutionofIndia.txt")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Add a text file there before running this app."
        )

    print("Loading documents...")
    docs = load_documents(str(DATA_PATH))
    chunks = chunk_documents(docs)
    bm25 = BM25Retriever(chunks)
    reranker = Reranker()

    print("Loading embedding model...")
    embedding = get_embedding_model()
    db = load_vector_store(embedding)

    if vector_store_count(db):
        print("Loaded existing vector store.")
    else:
        print("Building clean vector store for the first time...")
        db = create_vector_store(chunks, embedding)

    client = get_llm()

    print("Ready. Type your question, or press Ctrl+C to stop.")

    while True:
        try:
            q = input("Ask your question:\n").strip()
        except EOFError:
            break

        if q.lower() in {"exit", "quit"}:
            break

        if not q:
            continue

        retrieved = hybrid_search(q, db, bm25, k=10)

        results = reranker.rerank(q, retrieved, top_k=3)

        if not results:
            print("No matching documents found.")
            continue

        answer = generate_answer(client, q, results)

        print("\nRetrieved Context:\n")
        for i, result in enumerate(results, start=1):
            print(f"\n--- Chunk {i} ---")
            print(result.page_content[:500])

        print("\nAnswer:\n")
        print(answer)


if __name__ == "__main__":
    main()
