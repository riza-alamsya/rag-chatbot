from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
EMBEDDING_MODEL = "thenlper/gte-large"
COLLECTION_NAME = "phis_sds"


class PHISRetriever:
    def __init__(self, verbose: bool = False):
        if verbose:
            print("Loading embedding model...", file=sys.stderr)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        if verbose:
            print("Loading Chroma vector store...", file=sys.stderr)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_PATH,
        )
        if verbose:
            count = self.vectorstore._collection.count()
            print(f"Ready! Total vectors: {count}", file=sys.stderr)

    def search(self, query: str, n: int = 3) -> list:
        return self.vectorstore.similarity_search(query, k=n)

    def search_formatted(self, query: str, n: int = 3) -> str:
        docs = self.search(query, n)
        if not docs:
            return "No relevant documentation found."
        results = []
        for i, doc in enumerate(docs):
            page = doc.metadata.get("page", "N/A")
            results.append(f"[Result {i+1} | Page {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(results)

    @property
    def count(self) -> int:
        return self.vectorstore._collection.count()
