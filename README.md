# RAG Chatbot

**Offline Retrieval-Augmented Generation untuk Dokumentasi PHIS SAM**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=flat-square)](https://trychroma.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=flat-square)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> 100% offline. No API key. No data leaves your machine.

---

## Tentang Project

Chatbot berbasis **RAG** untuk query dokumentasi **PHIS SAM** (Pharmacy Information System — Special Approval Medicine) secara offline.

Masalah: dokumen SDS PHIS ratusan halaman, sulit cari informasi spesifik. Solusi: embed seluruh dokumen ke vector store, lalu pakai semantic search + LLM lokal untuk jawab pertanyaan dalam bahasa natural.

---

## Fitur

| Fitur | Detail |
|-------|--------|
| Fully Offline | Zero cloud dependency |
| Semantic Search | Embedding `thenlper/gte-large` 384-dim |
| Multi-Document | Ingest beberapa PDF ke satu vector store |
| Conversation History | Konteks 3 turn terakhir |
| MCP Integration | Expose `search_phis_docs` tool ke Claude AI |
| Source Citations | Setiap jawaban + nomor halaman sumber |

---

## Arsitektur

```
PDF Files → Chunking → Embedding → ChromaDB
                                       |
                            Query → Search → Top-N Chunks
                                                  |
                                         LLM (Ollama) → Answer + Sources
```

**Tech stack:**
- **LangChain** — pipeline orchestration
- **ChromaDB** — vector store persisten
- **HuggingFace** — embedding (`thenlper/gte-large`)
- **Ollama** — local LLM inference (phi3:mini, llama3, dll)
- **FastMCP** — MCP server untuk Claude integration

---

## Struktur Project

```
phis-rag-chatbot/
├── src/
│   ├── chatbot.py        # CLI chatbot (Ollama + Chroma)
│   ├── ingest.py         # PDF ingestion pipeline
│   ├── retriever.py      # PHISRetriever class
│   └── mcp_server.py     # MCP server untuk Claude AI
├── notebooks/
│   ├── 01_chunking.ipynb       # PDF → chunks
│   ├── 02_embedding.ipynb      # Generate embeddings
│   ├── 03_vectorstore.ipynb    # Chroma setup
│   ├── 04_retrieval.ipynb      # Search testing
│   ├── 06_chatbot_local.ipynb  # End-to-end demo
│   └── 07_merge_colab_chroma.ipynb
├── data/
│   └── chunks.json       # Pre-processed chunks (metadata)
├── vector_db/            # Chroma persisted store (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Instalasi

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.ai) installed

```bash
# Pull LLM model dulu
ollama pull phi3:mini
```

```bash
# Clone & install
git clone https://github.com/YOUR_USERNAME/phis-rag-chatbot.git
cd phis-rag-chatbot
pip install -r requirements.txt
```

**Ingest dokumen:**

```bash
python src/ingest.py --file data/your_document.pdf
# Multiple files
python src/ingest.py --file data/doc1.pdf data/doc2.pdf --chunk-size 700 --chunk-overlap 100
```

---

## Cara Pakai

### CLI Chatbot

```bash
python src/chatbot.py
python src/chatbot.py --model llama3:8b --n-results 5
```

### MCP Server (Claude Integration)

Tambahkan ke Claude Code config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "phis-search": {
      "command": "python",
      "args": ["src/mcp_server.py"],
      "cwd": "/path/to/phis-rag-chatbot"
    }
  }
}
```

Claude akan punya tool `search_phis_docs(query, n_results)`.

---

## Notebooks

Ikuti urutan notebook untuk memahami pipeline dari awal:

| Notebook | Deskripsi |
|----------|-----------|
| `01_chunking` | Load PDF, split jadi chunks |
| `02_embedding` | Generate vector embeddings |
| `03_vectorstore` | Init & test ChromaDB |
| `04_retrieval` | Test semantic search |
| `06_chatbot_local` | End-to-end chatbot demo |
| `07_merge_colab_chroma` | Merge vectors dari multiple sources |

---

## Environment Variables

```bash
cp .env.example .env
```

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini
EMBEDDING_MODEL=thenlper/gte-large
VECTOR_DB_PATH=./vector_db
N_RESULTS=3
```

---

## Pengembangan ke Depan

- [ ] Web UI (Streamlit / Gradio)
- [ ] Support dokumen Word & Excel
- [ ] Multi-language support (EN/ID)
- [ ] Evaluasi kualitas retrieval (RAGAS)
- [ ] Docker compose setup

---

## License

[MIT](LICENSE) — bebas digunakan, dimodifikasi, dan didistribusikan.

---

<div align="center">
Made with LangChain + ChromaDB + Ollama
</div>
