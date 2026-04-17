def query_db(vector_db, query):
    return vector_db.similarity_search(query, k=3)