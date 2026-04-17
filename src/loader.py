from langchain_community.document_loaders import TextLoader

def load_documents(path):
    loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
    return loader.load()
