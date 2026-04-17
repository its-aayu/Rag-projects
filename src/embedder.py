from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import hashlib
import re


class SimpleHashEmbeddings(Embeddings):
    def __init__(self, size=384):
        self.size = size

    def _embed(self, text):
        vector = [0.0] * self.size
        words = re.findall(r"\w+", text.lower())

        for word in words:
            digest = hashlib.md5(word.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.size
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        length = sum(value * value for value in vector) ** 0.5
        if length:
            vector = [value / length for value in vector]

        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

def get_embedding_model():
    try:
        return HuggingFaceEmbeddings(
            model_name = "all-MiniLM-L6-v2",
            model_kwargs = {"local_files_only": True}
        )
    except Exception as exc:
        print(f"Could not load Hugging Face embeddings: {exc}")
        print("Using offline hash embeddings instead.")
        return SimpleHashEmbeddings()
