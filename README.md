# 🌐 DocuSphere
### Dual-Mode RAG Intelligence Powered by Endee Vector Database

> Upload any document or drop a topic — DocuSphere researches, stores, and lets you converse with knowledge.

![Python](https://img.shields.io/badge/Python-3.11.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.43+-red)
![Endee](https://img.shields.io/badge/Vector%20DB-Endee-purple)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview & Problem Statement

### The Problem
In today's information-heavy world, two key challenges exist:

1. **Document Intelligence**: People need to interact with large PDFs and Word documents — research papers, reports, contracts — but reading through hundreds of pages to find a specific answer is inefficient and time-consuming.

2. **Dynamic Knowledge Q&A**: When someone wants to understand a topic deeply, they search across multiple sources, manually piece together information, and still can't have a natural conversation with that knowledge.

### The Solution — DocuSphere
DocuSphere solves both problems through a unified dual-mode RAG (Retrieval-Augmented Generation) system:

- **Mode 1 — Document Mode**: Upload any PDF or Word document. DocuSphere processes it, stores its knowledge in Endee Vector Database, and enables natural conversational Q&A over the document's content.

- **Mode 2 — Research Mode**: Enter any topic. DocuSphere autonomously fetches content from Wikipedia and the web, builds a live knowledge base in Endee, and lets you have an intelligent conversation with the research data.

Both modes are powered by **Endee as the central vector database**, ensuring fast, persistent, and production-ready vector storage and retrieval.

---

## 🏗️ System Design & Technical Approach

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DocuSphere                               │
│                    (Streamlit Frontend)                         │
└──────────────────┬───────────────────────────┬─────────────────┘
                   │                           │
         ┌─────────▼─────────┐     ┌──────────▼──────────┐
         │   MODE 1          │     │   MODE 2             │
         │  Document Mode    │     │  Research Mode       │
         │  PDF / DOCX       │     │  Any Topic           │
         └─────────┬─────────┘     └──────────┬──────────┘
                   │                           │
         ┌─────────▼─────────┐     ┌──────────▼──────────┐
         │  PyMuPDF /        │     │  Wikipedia API +     │
         │  python-docx      │     │  DuckDuckGo Search   │
         │  (Text Extraction)│     │  (Web Research)      │
         └─────────┬─────────┘     └──────────┬──────────┘
                   │                           │
                   └─────────────┬─────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Text Chunking         │
                    │  (1000 chars / 150 overlap)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   HuggingFace           │
                    │   Embeddings            │
                    │  (all-MiniLM-L6-v2)     │
                    │   384 dimensions        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │                         │
                    │    ENDEE VECTOR DB      │◄──── Core Component
                    │    (localhost:8080)      │
                    │    Persistent Storage   │
                    │    Cosine Similarity    │
                    │                         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   User Question         │
                    │   → Embed Question      │
                    │   → Query Endee (top 5) │
                    │   → Retrieve Chunks     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Groq LLM              │
                    │  (LLaMA 3.3 70B)        │
                    │   Grounded Answer       │
                    └─────────────────────────┘
```

### Technical Approach

**Why RAG instead of sending everything to LLM?**

A typical 50-page PDF contains ~100,000 tokens — far beyond LLM context limits and expensive. RAG solves this by:
1. Storing all content as vectors in Endee
2. At query time, only retrieving the top 5 most relevant chunks (~5,000 tokens)
3. Sending only those chunks to the LLM for grounded, accurate answers

**Why Endee over FAISS or ChromaDB?**

| Feature              | FAISS     | ChromaDB      | **Endee** |
|----------------------|-----------|---------------|-----------|
| Persistence          |    ❌     |    ✅        |     ✅    |
| Production-ready API |    ❌     |    ⚠️        |     ✅    |
| Docker deployment    |    ❌     |    ⚠️        |     ✅    |
| REST + Python SDK    |    ❌     |    ⚠️        |     ✅    |
| Advanced filtering   |    ❌     |    ⚠️        |     ✅    |
| Hybrid Search        |    ❌     |    ❌        |     ✅    |

FAISS is in-memory — restart the app and all vectors are gone. Endee is a dedicated vector database server with persistent disk storage, a clean REST API, and an official Python SDK — making it genuinely production-ready for real applications.

---

## 🔍 How Endee is Used

Endee is the **central intelligence layer** of DocuSphere. Here's exactly how it's integrated:

### 1. Index Creation
Each document or research topic gets its own named index in Endee:
```python
client.create_index(
    name="research_quantum_computing",
    dimension=384,          # Matches all-MiniLM-L6-v2 output
    space_type="cosine",    # Cosine similarity for semantic search
    precision="float32"
)
```

### 2. Vector Storage (Upsert)
After chunking and embedding, all vectors are stored in Endee with rich metadata:
```python
index.upsert([
    {
        "id": "vec_0",
        "vector": [0.231, -0.445, ...],   # 384-dimensional embedding
        "meta": {
            "text": "original chunk text",  # Retrieved at query time
            "source": "document.pdf",
            "chunk_id": 0
        }
    },
    ...
])
```

### 3. Semantic Search at Query Time
When a user asks a question, it's embedded and sent to Endee:
```python
results = index.query(
    vector=question_embedding,   # Question converted to 384-dim vector
    top_k=5,                     # Return 5 most semantically similar chunks
    include_vectors=False        # Only need metadata, not raw vectors
)
```
Endee performs **Approximate Nearest Neighbor (ANN) search** using the HNSW algorithm — finding the most semantically similar chunks in milliseconds regardless of how many vectors are stored.

### 4. Grounded Answer Generation
The retrieved chunks (metadata text) are passed to Groq LLM as context, ensuring answers are grounded in the actual document/research content rather than LLM hallucinations.

### Complete RAG Flow in DocuSphere:
```
"What are the key features of quantum computing?"
        ↓
Embed question → [0.219, -0.441, ...] (384 numbers)
        ↓
Endee cosine similarity search across all stored vectors
        ↓
Returns top 5 most relevant chunks from knowledge base
        ↓
Groq LLaMA 3.3 70B reads chunks → generates grounded answer
        ↓
"Quantum computing leverages qubits which can exist in superposition..."
```

---

## 📁 Project Structure

```
docusphere/
│
├── app.py                    # Streamlit UI — main entry point
├── endee_client.py           # Endee Python SDK wrapper
│                             # (create_index, insert_vectors, query_index)
├── embedder.py               # HuggingFace sentence-transformers
│                             # (all-MiniLM-L6-v2, 384 dimensions)
├── document_processor.py     # PDF + DOCX parsing & chunking
│                             # (PyMuPDF, python-docx, 50 page limit)
├── web_researcher.py         # Agentic web research module
│                             # (Wikipedia + DuckDuckGo)
├── llm_handler.py            # Groq LLM integration
│                             # (LLaMA 3.3 70B, prompt engineering)
│
├── docker-compose.yml        # Endee Vector DB Docker setup
├── requirements.txt          # Pinned Python dependencies
├── runtime.txt               # Python 3.11 for deployment
├── .env.example              # Environment variables template
└── .gitignore                # Excludes .env, venv, __pycache__
```

---

## ⚙️ Tech Stack

| Component         | Technology                     |
|-------------------|--------------------------------|
| Frontend UI       | Streamlit                      |
| Vector Database   | **Endee** (Docker)             |
| Embedding Model   | HuggingFace `all-MiniLM-L6-v2` |
| LLM               | Groq `llama-3.3-70b-versatile` |
| PDF Parsing       | PyMuPDF (fitz)                 |
| DOCX Parsing      | python-docx                    |
| Web Research      | DuckDuckGo Search + Wikipedia  |
| Text Chunking     | LangChain Text Splitters       |
| Containerization  | Docker + Docker Compose        |
| Language          | Python 3.11.9                  |

---

## 🚀 Setup & Execution Instructions

### Prerequisites

Before starting, ensure you have:
- Python 3.11.x installed
- Docker Desktop installed and running
- Git installed
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### Step 1 — Clone the Repository

```bash
git clone https://github.com/17Atishay/docusphere.git
cd docusphere
```

### Step 2 — Start Endee Vector Database

Endee runs as a Docker container. Start it first:

```bash
docker compose up -d
```

Verify it's running:
```bash
docker ps
```
You should see `endee-server` container running. Access the Endee dashboard at `http://localhost:8080`.

### Step 3 — Create Virtual Environment

```bash
# Windows
py -3.11 -m venv DocuSphereVenv
DocuSphereVenv\Scripts\activate

# macOS/Linux
python3.11 -m venv DocuSphereVenv
source DocuSphereVenv/bin/activate
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
GROQ_API_KEY=your_groq_api_key_here
ENDEE_URL=http://localhost:8080
ENDEE_TOKEN=
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### Step 6 — Run DocuSphere

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` — DocuSphere is ready!

---

## 📖 Usage Guide

### Mode 1 — Document Mode
1. Select **📄 Document Mode** from the sidebar
2. Upload a PDF or DOCX file (max 10MB, 50 pages)
3. Click **🚀 Process & Store in Endee**
4. Wait for embeddings to be generated and stored
5. Ask questions in the chat box on the right
6. Optionally click **📝 Summarize Document** for a quick overview

### Mode 2 — Research Mode
1. Select **🔍 Research Mode** from the sidebar
2. Type any topic (e.g., "Quantum Computing", "Climate Change")
3. Click **🌐 Research & Store in Endee**
4. DocuSphere automatically fetches Wikipedia + web content
5. Ask questions in the chat box once research is complete
6. Optionally click **📝 Summarize Research** for an overview

---

## 🧠 Model Configuration

| Setting           | Value                        | Reason                                       |
|-------------------|------------------------------|----------------------------------------------|
| Embedding Model   | `all-MiniLM-L6-v2`           | Free, fast, 384-dim, runs locally            |
| LLM               | `llama-3.3-70b-versatile`    | Groq free tier, 14,400 req/day               |
| Temperature       | `0.3`                        | Low = factual, grounded answers              |
| Max Output Tokens | `1024`                       | Sufficient for detailed answers              |
| Chunk Size        | `1000` chars                 | ~150-200 words, full paragraph context       |
| Chunk Overlap     | `150` chars                  | Prevents context loss at boundaries          |
| Top-K Retrieval   | `5` chunks                   | Balanced context vs token efficiency         |
| Endee Precision   | `float32`                    | Compatible across SDK and API                |
| Endee Space Type  | `cosine`                     | Best for semantic text similarity            |

---

## 🔒 Safety & Limits

- **Max file size**: 10MB per upload
  *Set conservatively for stable performance on local machines. Can be increased in `app.py` (`max_file_size > 10`) based on available RAM.*

- **Max pages processed**: 50 pages (PDF)
  *Processes first 50 pages by default to keep embedding time under 30 seconds. Can be increased in `document_processor.py` (`max_pages=50`) — tested up to 200 pages without issues.*

- **Max chunks per document**: 500
  *Safety cap to prevent memory overload during batch embedding. Adjustable in `document_processor.py` (`max_chunks > 500`). Endee itself handles millions of vectors with no performance degradation.*

- **API keys**: Stored in `.env` — never committed to GitHub

> **Note**: These limits are conservative defaults optimized for local development stability, not hard technical constraints. Endee's vector storage and retrieval performance remains fast regardless of index size — the limits exist purely on the ingestion side.

---
