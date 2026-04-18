from pathlib import Path

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


PERSIST_DIRECTORY = Path("./embeddings_constitution")
COLLECTION_NAME = "constitution_v2"


def load_vector_store(embedding):
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIRECTORY),
        embedding_function=embedding,
    )


def vector_store_count(vector_db):
    return vector_db._collection.count()


def create_vector_store(chunks, embedding):
    return Chroma.from_documents(
        collection_name=COLLECTION_NAME,
        documents=chunks,
        embedding=embedding,
        persist_directory=str(PERSIST_DIRECTORY),
    )
