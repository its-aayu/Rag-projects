import re

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _article_number(query):
    match = re.search(r"\barticle\s+(\d+[A-Z]?)\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"\bart\.?\s+(\d+[A-Z]?)\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


class Reranker:
    def __init__(self, model_name=MODEL_NAME):
        self.model = None

        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception as exc:
            print(f"Could not load reranker model locally: {exc}")
            print("Using lightweight lexical reranking instead.")

    def rerank(self, query, documents, top_k=3):
        if not documents:
            return []

        if self.model:
            return self._cross_encoder_rerank(query, documents, top_k)

        return self._lexical_rerank(query, documents, top_k)

    def _cross_encoder_rerank(self, query, documents, top_k):
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        scored_docs = []

        for index, (doc, score) in enumerate(zip(documents, scores)):
            scored_docs.append((float(score) + self._article_boost(query, doc) - index * 0.01, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def _lexical_rerank(self, query, documents, top_k):
        query_terms = set(_tokenize(query))
        scored_docs = []

        for index, doc in enumerate(documents):
            doc_terms = set(_tokenize(doc.page_content))
            overlap = len(query_terms & doc_terms)
            score = overlap + self._article_boost(query, doc) - index * 0.01
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

    def _article_boost(self, query, doc):
        article = _article_number(query)
        if not article:
            return 0

        article_heading = re.compile(rf"(^|\n)\s*{re.escape(article)}\.\s", re.IGNORECASE)
        if article_heading.search(doc.page_content):
            return 10

        article_mention = re.compile(rf"\barticle\s+{re.escape(article)}\b", re.IGNORECASE)
        if article_mention.search(doc.page_content):
            return 3

        return 0
