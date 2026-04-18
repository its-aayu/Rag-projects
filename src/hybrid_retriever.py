import re

from rank_bm25 import BM25Okapi


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _dedupe_documents(documents):
    seen = set()
    unique_docs = []

    for doc in documents:
        key = doc.page_content.strip()
        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def _article_number(query):
    match = re.search(r"\barticle\s+(\d+[A-Z]?)\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"\bart\.?\s+(\d+[A-Z]?)\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


class BM25Retriever:
    def __init__(self, documents):
        self.docs = documents
        self.tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, k=5):
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [self.docs[index] for index in ranked_indices[:k]]


def exact_article_search(query, documents, k=3):
    article = _article_number(query)
    if not article:
        return []

    article_heading = re.compile(rf"(^|\n)\s*{re.escape(article)}\.\s", re.IGNORECASE)
    matches = []

    for doc in documents:
        text = doc.page_content
        heading = article_heading.search(text)
        if not heading:
            continue

        numbered_headings = re.findall(r"(^|\n)\s*\d+[A-Z]?\.\s", text)
        score = 10 + max(0, 5 - heading.start() // 100)

        if len(numbered_headings) > 1:
            score -= len(numbered_headings) * 3

        if "article pages" in text.lower():
            score -= 4

        matches.append((score, doc))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in matches[:k]]


def hybrid_search(query, vector_db, bm25_retriever, k=10):
    vector_results = vector_db.similarity_search(query, k=k)
    bm25_results = bm25_retriever.search(query, k=k)
    exact_results = exact_article_search(query, bm25_retriever.docs, k=3)

    combined_results = exact_results + bm25_results + vector_results
    return _dedupe_documents(combined_results)[:k]
