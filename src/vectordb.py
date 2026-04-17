from langchain_community.vectorstores import Chroma


def create_vector_store (chunks, embedding):
    return Chroma.from_documents(
        documents = chunks,
        embedding = embedding,
        persist_directory = "./embeddings_constitution"
    )
