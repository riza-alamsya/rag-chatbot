"""
Add documents to Chroma vector store.

Usage:
    python src/ingest.py --file data/document.pdf
    python src/ingest.py --file data/doc1.pdf data/doc2.pdf
    python src/ingest.py --file data/doc.pdf --chunk-size 700 --chunk-overlap 100
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")
EMBEDDING_MODEL = "thenlper/gte-large"
COLLECTION_NAME = "phis_sds"

SUPPORTED_EXTENSIONS = {".pdf"}


def load_file(path: str) -> list:
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"  Loaded {len(pages)} pages from {os.path.basename(path)}")
    return pages


def chunk_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Chunked into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def ingest(files: list[str], chunk_size: int = 700, chunk_overlap: int = 100):
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Connecting to Chroma vector store...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    count_before = vectorstore._collection.count()
    print(f"Vectors before ingest: {count_before}")

    all_chunks = []
    for file_path in files:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            print(f"  [SKIP] File not found: {abs_path}", file=sys.stderr)
            continue
        print(f"\nProcessing: {abs_path}")
        pages = load_file(abs_path)
        chunks = chunk_documents(pages, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nNo chunks to ingest. Exiting.")
        return

    print(f"\nEmbedding and storing {len(all_chunks)} chunks...")
    vectorstore.add_documents(all_chunks)

    count_after = vectorstore._collection.count()
    print(f"\nDone! Vectors: {count_before} → {count_after} (+{count_after - count_before})")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma vector store")
    parser.add_argument("--file", nargs="+", required=True, metavar="PATH", help="PDF file(s) to ingest")
    parser.add_argument("--chunk-size", type=int, default=700, help="Chunk size in characters (default: 700)")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in characters (default: 100)")
    args = parser.parse_args()

    ingest(args.file, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)


if __name__ == "__main__":
    main()
