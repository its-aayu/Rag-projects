import re

from langchain_core.documents import Document


def _article_number(query):
    match = re.search(r"\barticle\s+(\d+[A-Z]?)\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"\bart\.?\s+(\d+[A-Z]?)\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _keyword_terms(query):
    return {
        term.lower()
        for term in re.findall(r"[A-Za-z]{4,}", query)
        if term.lower() not in {"what", "which", "where", "when", "does", "about", "article"}
    }


def _exact_article_matches(vector_db, query, limit):
    article = _article_number(query)
    if not article:
        return []

    all_docs = vector_db.get(include=["documents", "metadatas"])
    terms = _keyword_terms(query)
    article_heading = re.compile(rf"(^|\n)\s*{re.escape(article)}\.\s", re.IGNORECASE)
    article_mention = re.compile(rf"\barticle\s+{re.escape(article)}\b", re.IGNORECASE)

    matches = []
    for text, metadata in zip(all_docs.get("documents", []), all_docs.get("metadatas", [])):
        if not text:
            continue

        text_lower = text.lower()
        heading = article_heading.search(text)
        heading_match = bool(heading)
        mention_match = bool(article_mention.search(text))

        if not heading_match and not mention_match:
            continue

        score = 10 if heading_match else 5
        score += sum(1 for term in terms if term in text_lower)

        if heading:
            score += max(0, 5 - heading.start() // 100)

        numbered_headings = re.findall(r"(^|\n)\s*\d+[A-Z]?\.\s", text)
        if len(numbered_headings) > 1:
            score -= len(numbered_headings) * 3

        # A table of contents line often contains article numbers but not the answer.
        if "article pages" in text_lower:
            score -= 4

        matches.append((score, Document(page_content=text, metadata=metadata or {})))

    matches.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in matches[:limit]]


def _dedupe_documents(documents):
    seen = set()
    unique = []

    for doc in documents:
        key = doc.page_content.strip()
        if key in seen:
            continue

        seen.add(key)
        unique.append(doc)

    return unique


def query_db(vector_db, query, k=10):
    exact_matches = _exact_article_matches(vector_db, query, limit=5)
    semantic_matches = vector_db.similarity_search(query, k=k)
    return _dedupe_documents(exact_matches + semantic_matches)[:k]
