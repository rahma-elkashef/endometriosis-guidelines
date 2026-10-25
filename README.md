# EndoGuide: Endometriosis Guideline Assistant

A retrieval-augmented generation (RAG) system that answers questions about endometriosis using only the text of NICE clinical guidelines. Each answer is grounded in retrieved guideline passages and includes source information such as the guideline, section, page number, recommendation number, and a heuristic confidence score.

EndoGuide uses hybrid retrieval, cross-encoder reranking, and a relevance gate to retrieve relevant evidence and to decline questions when the available context does not meet the configured relevance thresholds.

> **Disclaimer:** This project is a research and educational prototype. It is not a medical device and does not provide medical advice. Its answers may be incomplete or incorrect. Always consult a qualified healthcare professional for diagnosis and treatment decisions.

## Source Guidelines

| Code | Title | Publisher |
|---|---|---|
| NG73 | Endometriosis: diagnosis and management | NICE |
| QS172 | Endometriosis (quality standard) | NICE |

The source PDFs are stored in `backend/data/pdfs/`.

## Key Features

- **Hybrid retrieval:** dense vector search (BGE) combined with BM25 keyword search.
- **Cross-encoder reranking:** reorders candidates by relevance to the query.
- **Relevance gate:** declines queries when the best result fails configurable thresholds.
- **Source-grounded answers:** returns supporting passages with guideline, section, page and recommendation metadata.
- **Section-aware embeddings:** section context is used for embedding, while clean passage text is kept for display.
- **Local language model:** a lightweight Hugging Face model generates answers on CPU.
- **Heuristic confidence score:** a relevance signal, not a probability of correctness.
- **Chat interface:** a Streamlit UI backed by a FastAPI service.

## Architecture

```text
NICE Guideline PDFs
        |
        v
    ingest.py
        |
        v
     chunk.py
        |
        v
  build_index.py
        |
        v
  +--------------------+
  |  ChromaDB + BM25   |
  +--------------------+
        |
        v
      rag.py
        |
        +--> Dense retrieval
        +--> BM25 retrieval
        +--> Candidate merging
        +--> Cross-encoder reranking
        +--> Relevance gate
        +--> Context-constrained generation
        |
        v
    main.py (FastAPI)
        |
        v
 Streamlit chat interface
```

## How It Works

### 1. PDF Ingestion

**File:** `backend/ingest.py`

- Extracts text page by page using PyMuPDF.
- Removes NICE headers and footers.
- Detects section headings using PDF font metadata.
- Extracts section numbers and recommendation numbers (e.g. `1.5.10`).
- Attaches document metadata (guideline code, title, publisher, page number).

**Output:** `backend/data/json/nice_documents.json`

### 2. Text Chunking

**File:** `backend/chunk.py`

- Splits extracted text into sentence-aware chunks.
- Overlaps consecutive chunks to preserve context across boundaries.
- Builds a separate `embedding_text` field that includes section context.
- Keeps the original chunk text clean for retrieval results and source display.

**Default settings:** chunk size of 500 characters with an overlap of 100 characters. These are small chunks, so they are worth tuning (`CHUNK_SIZE`, `CHUNK_OVERLAP` in `chunk.py`) if retrieval returns passages that lack context.

**Output:** `backend/data/json/nice_chunks.json`

### 3. Index Building

**File:** `backend/build_index.py`

- Generates dense embeddings with `BAAI/bge-base-en-v1.5`.
- Stores embeddings and metadata in a ChromaDB collection (`nice_guidelines`) configured for cosine similarity.
- Builds a BM25 keyword index over the clean passage text.

**Output:**
- `backend/data/vector_store/chromadb/`
- `backend/data/vector_store/nice_bm25.pkl`

### 4. Retrieval and Answer Generation

**File:** `backend/rag.py`

1. **Dense retrieval:** finds semantically similar passages using BGE embeddings.
2. **Keyword retrieval:** uses BM25 to find passages containing relevant terms.
3. **Candidate merging:** combines candidates from both methods.
4. **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` reorders candidates by relevance.
5. **Relevance gate:** rejects the query if the best result fails the reranker-score or Chroma-distance threshold.
6. **Context-constrained generation:** `Qwen/Qwen2.5-0.5B-Instruct` writes a concise answer using only the retrieved context, guided by a strict system prompt.

Constraining generation to retrieved evidence and declining low-relevance queries reduces unsupported answers, but it does not guarantee that every generated statement is correct or fully supported.

### 5. API

**File:** `backend/main.py`

A FastAPI service exposing a health-check endpoint and a chat endpoint.

### 6. Frontend

**File:** `frontend/RAG_and_Roll_redesigned.py`

A Streamlit interface where users submit questions and view the generated answer, supporting passages, source metadata and confidence.

## Project Structure

```text
endometriosis-guidelines/
├── backend/
│   ├── data/
│   │   └── pdfs/                       # Source NICE guideline PDFs
│   ├── ingest.py                       # PDF text extraction and metadata
│   ├── chunk.py                        # Sentence-aware text chunking
│   ├── build_index.py                  # Embeddings, ChromaDB and BM25
│   ├── rag.py                          # Retrieval, reranking and generation
│   ├── main.py                         # FastAPI application
│   └── requirements.txt
├── frontend/
│   ├── RAG_and_Roll_redesigned.py      # Streamlit application
│   └── logo.png
├── README.md
└── .gitignore
```

The folders `backend/data/json/`, `backend/data/vector_store/` and `backend/uploads/` are created automatically when the scripts run.

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git
- Internet access for installation and the first-time download of the Hugging Face models
- Enough RAM and disk space for the embedding model, reranker, language model and indexes

CPU execution is supported, but model loading and inference can be slow on machines with limited memory.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd endometriosis-guidelines
```

Replace `<your-repo-url>` with your repository URL.

### 2. Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

From the repository root:

```bash
pip install -r backend/requirements.txt
```

### 4. Build the Index

Run the scripts in order from the `backend/` directory:

```bash
cd backend

python ingest.py
python chunk.py
python build_index.py

cd ..
```

Rebuild the index whenever you change the source documents, the chunking settings or the embedding model.

### 5. Start the FastAPI Backend

From the repository root:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

- Health check: `http://127.0.0.1:8000/`
- Interactive API docs: `http://127.0.0.1:8000/docs`

The health-check response reports whether the index is ready, the indexed chunk count, and any startup error.

### 6. Start the Streamlit Frontend

In a second terminal, with the same virtual environment activated, run from the repository root:

```bash
streamlit run frontend/RAG_and_Roll_redesigned.py
```

By default the frontend connects to `http://127.0.0.1:8000`. To use a different backend address, set `RAG_BACKEND_URL` before launching Streamlit.

**Windows PowerShell:**

```powershell
$env:RAG_BACKEND_URL = "http://127.0.0.1:8000"
```

**Linux / macOS:**

```bash
export RAG_BACKEND_URL="http://127.0.0.1:8000"
```

## API Reference

### `GET /`

Returns the service status, whether the index is ready, the number of indexed chunks, and any startup error.

### `POST /chat`

**Request:**

```json
{
  "question": "What imaging should be offered to someone with suspected endometriosis?"
}
```

**Illustrative response structure:**

```json
{
  "answer": "Generated answer based on retrieved guideline evidence.",
  "sources": [
    {
      "text": "Retrieved guideline passage.",
      "guideline": "NG73",
      "section": "Relevant section",
      "page": 12,
      "recommendation_number": "1.2.5",
      "rerank_score": 3.1,
      "chroma_distance": 0.31
    }
  ],
  "prompt": "The prompt sent to the language model.",
  "confidence": 0.81,
  "chunk_count": 0
}
```

The values above illustrate the structure only; they are not real output.

| HTTP Status | Meaning |
|---|---|
| `200` | Request processed successfully |
| `400` | Empty question |
| `503` | Index unavailable or failed to load |
| `500` | Error during retrieval or generation |

## Configuration

Main settings in `backend/rag.py`:

| Setting | Default | Purpose |
|---|---|---|
| `EMBED_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | Dense embedding model |
| `RERANKER_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Candidate reranking model |
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Answer generation model |
| `MIN_RERANK_SCORE` | `0.0` | Minimum accepted reranker score |
| `MAX_CHROMA_DISTANCE` | `0.65` | Maximum accepted vector distance |

These are initial values, not clinically validated thresholds. If you change the embedding model, rebuild the index so the stored vectors match the model.

### Confidence and Relevance

The confidence score is a heuristic derived from the top result's reranker score and Chroma distance. It is not a calibrated probability that an answer is correct, safe, or fully supported by the guidelines.

Tune the relevance thresholds against a set of test questions that includes:

- questions clearly answered by the indexed guidelines
- questions with relevant terminology but insufficient evidence
- questions unrelated to endometriosis
- questions requiring information absent from the indexed documents


## Tech Stack

- **Language:** Python
- **PDF processing:** PyMuPDF
- **Embeddings and reranking:** Sentence Transformers
- **Vector database:** ChromaDB
- **Keyword retrieval:** rank-bm25
- **Language model:** Hugging Face Transformers
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit

