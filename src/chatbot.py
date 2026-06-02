"""
PHIS SAM Chatbot CLI — Ollama (phi3:mini) + Chroma local.
100% offline.

Usage:
    python src/chatbot.py
    python src/chatbot.py --model phi3:mini --n-results 5
"""
import argparse
import sys
import os
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from retriever import PHISRetriever

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3:mini"
SYSTEM_PROMPT = (
    "You are a helpful assistant for PHIS (Pharmacy Information System) SAM documentation. "
    "Answer based ONLY on the context provided. "
    "If the answer is not in the context, say: \"I don't have information about that in the documentation.\""
)


def check_ollama(model: str) -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        if model not in models:
            print(f"[ERROR] Model '{model}' not found. Available: {models}", file=sys.stderr)
            return False
        return True
    except Exception:
        print("[ERROR] Ollama not running. Start Ollama first.", file=sys.stderr)
        return False


def build_prompt(query: str, context: str, history: list) -> str:
    history_text = ""
    if history:
        for h in history[-3:]:
            history_text += f"User: {h['user']}\nAssistant: {h['assistant']}\n\n"

    history_section = f"Previous conversation:\n{history_text}\n" if history_text else ""

    return f"""{SYSTEM_PROMPT}

{history_section}Context:
{context}

Question: {query}

Answer:"""


def ask_ollama(prompt: str, model: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False,"options": {
            "num_predict": 256,  # max 256 token output
            "temperature": 0.1   # lebih fokus, kurang random
        }},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"]


def chat_loop(model: str, n_results: int):
    print("Loading retriever...", file=sys.stderr)
    retriever = PHISRetriever(verbose=True)

    print(f"\nPHIS SAM Chatbot ({model} + Chroma)")
    print(f"Vectors loaded: {retriever.count}")
    print("Type 'exit' or Ctrl+C to quit.")
    print("=" * 55)

    history = []

    while True:
        try:
            query = input("\nKamu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        docs = retriever.search(query, n=n_results)
        context = "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        )

        prompt = build_prompt(query, context, history)

        try:
            answer = ask_ollama(prompt, model)
        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}", file=sys.stderr)
            continue

        history.append({"user": query, "assistant": answer})

        print(f"\nAssistant: {answer}")
        print("-" * 55)
        print("Sources:", ", ".join(
            f"Page {doc.metadata.get('page', 'N/A')}" for doc in docs
        ))


def main():
    parser = argparse.ArgumentParser(description="PHIS SAM Chatbot CLI")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--n-results", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    if not check_ollama(args.model):
        sys.exit(1)

    chat_loop(args.model, args.n_results)


if __name__ == "__main__":
    main()
