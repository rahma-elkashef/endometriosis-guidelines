import json
import pickle
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
JSON_DIR = BASE_DIR / "data" / "json"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
UPLOAD_DIR = BASE_DIR / "uploads"

for _dir in (PDF_DIR, JSON_DIR, VECTOR_DIR, UPLOAD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

INPUT_FILE = JSON_DIR / "nice_chunks.json"
CHROMA_PATH = VECTOR_DIR / "chromadb"
COLLECTION_NAME = "nice_guidelines"
BM25_FILE = VECTOR_DIR / "nice_bm25.pkl"
MODEL_NAME = "BAAI/bge-base-en-v1.5"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks.")
    return chunks


# ============================================================
# PREPARE TWO TYPES OF TEXT
# ============================================================
# embedding_text: used ONLY for generating embeddings (includes section
#                 context so retrieval understands where a chunk sits).
# text:           clean chunk stored as the actual Chroma document — this
#                 prevents Section / Section title / Recommendation
#                 metadata from leaking into retrieved evidence text.

def prepare_texts(chunks):
    embedding_texts = []
    documents = []

    for chunk in chunks:
        embedding_text = chunk.get("embedding_text") or chunk.get("text", "")
        embedding_texts.append(embedding_text)
        documents.append(chunk.get("text", ""))

    print(f"Prepared {len(embedding_texts)} embedding texts.")
    print(f"Prepared {len(documents)} clean documents.")

    return embedding_texts, documents


# ============================================================
# BUILD CHROMADB COLLECTION
# ============================================================

def build_chroma_collection(chunks, embedding_texts, documents, model):
    print("\nGenerating embeddings...")
    embeddings = model.encode(
        embedding_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Embedding shape: {embeddings.shape}")

    print("\nInitializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # IMPORTANT: delete any existing collection first. get_or_create_collection()
    # will NOT replace stale documents from a previous indexing run.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"✓ Deleted old collection: {COLLECTION_NAME}")
    except Exception:
        print("No existing collection to delete.")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "NICE clinical guideline embeddings",
            "embedding_model": MODEL_NAME,
            "hnsw:space": "cosine",
        },
    )
    print(f"✓ Created ChromaDB collection: {COLLECTION_NAME}")

    # --- IDs ---
    ids = [str(chunk.get("chunk_id") or f"chunk_{i}") for i, chunk in enumerate(chunks)]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ChromaDB IDs detected.")
    print(f"✓ Verified {len(ids)} unique IDs.")

    # --- Metadata ---
    metadatas = []
    for chunk in chunks:
        metadatas.append({
            "guideline": chunk.get("guideline") or "",
            "title": chunk.get("title") or "",
            "publisher": chunk.get("publisher") or "",
            "source": chunk.get("source") or "",
            "page": int(chunk.get("page", 0) or 0),
            "section": chunk.get("section") or "",
            "section_number": chunk.get("section_number") or "",
            "section_title": chunk.get("section_title") or "",
            "recommendation_number": chunk.get("recommendation_number") or "",
        })

    # --- Add in batches ---
    print("\nAdding embeddings to ChromaDB...")
    BATCH_SIZE = 500

    for start in range(0, len(chunks), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(chunks))

        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],   # clean text, NOT embedding_text
            metadatas=metadatas[start:end],
        )

        print(f"Added {start} - {end}")

    print(f"\nChromaDB collection '{COLLECTION_NAME}' contains {collection.count()} documents.")

    verification = collection.get(limit=1, include=["documents", "metadatas"])
    print("\n" + "=" * 60)
    print("CHROMADB VERIFICATION")
    print("=" * 60)
    print("\nStored document:")
    print(verification["documents"][0][:1000])
    print("\nStored metadata:")
    print(verification["metadatas"][0])

    return collection


# ============================================================
# BUILD BM25 INDEX
# ============================================================
# BM25 also indexes the CLEAN document text — otherwise it would match
# metadata artifacts like "Section: 1.4" instead of actual evidence.

def build_bm25_index(documents):
    tokenized_corpus = [document.lower().split() for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)

    print(f"\nSaved BM25 index to:\n{BM25_FILE}")
    return bm25


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    chunks = load_chunks()
    embedding_texts, documents = prepare_texts(chunks)

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Embedding model loaded.")

    print("\nExample embedding text:")
    print(embedding_texts[0][:1000])
    print("\nExample clean document:")
    print(documents[0][:1000])

    build_chroma_collection(chunks, embedding_texts, documents, model)
    build_bm25_index(documents)

    print("\n" + "=" * 60)
    print("Embedding and ChromaDB indexing completed!")
    print("=" * 60)
