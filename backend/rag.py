import re
import pickle
import io
from pathlib import Path

import numpy as np
import chromadb
import pymupdf
import pytesseract
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
JSON_DIR = BASE_DIR / "data" / "json"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
UPLOAD_DIR = BASE_DIR / "uploads"

CHROMA_PATH = VECTOR_DIR / "chromadb"
COLLECTION_NAME = "nice_guidelines"
BM25_FILE = VECTOR_DIR / "nice_bm25.pkl"

for _dir in (PDF_DIR, JSON_DIR, VECTOR_DIR, UPLOAD_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# BGE query instruction — must match what was used at index time
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Fallback question used when the user attaches a PDF but doesn't type a question.
DEFAULT_PDF_ONLY_QUESTION = (
    "Based on the attached document, what are the main clinical points or recommendations?"
)

# Relevance gate thresholds.
MIN_RERANK_SCORE = -1.5
MAX_CHROMA_DISTANCE = 0.75
USE_RERANK_GATE = True
USE_CHROMA_GATE = True


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-grounded clinical decision support assistant.

Answer the user's question using ONLY the retrieved context.

============================================================
CORE RULES
============================================================

1. Use only information explicitly supported by the retrieved context.

2. Do not use general medical knowledge or assumptions.

3. If the retrieved context is insufficient, say:
   "Evidence is insufficient in the retrieved documents to answer this question."

4. Do not provide patient-specific diagnosis, treatment, or medical advice.

5. Every clinical or factual claim must have a citation using a SOURCE ID
   that exists in the retrieved context.

6. Never invent SOURCE IDs, recommendations, guideline numbers, or page numbers.

7. Preserve the population, condition, circumstances, and strength of the
   original guideline recommendation.

8. Do not change "consider" to "offer", "recommend", or "should" unless
   the retrieved evidence explicitly supports that wording.

============================================================
QUESTION INTENT
============================================================

First determine what the user is asking, then answer that specific question.

- "What is..." -> provide a definition.
- "What are the symptoms..." -> provide symptoms/features.
- "Why..." -> provide the stated rationale.
- "When should..." -> provide the relevant circumstances/indications.
- "Should..." / "What should..." -> provide the guideline recommendation.
- "What imaging/treatment/investigation..." -> provide the relevant option
  or recommendation.
- "How..." -> provide the relevant process explicitly supported by the evidence.

Do NOT force every question into a recommendation.

============================================================
EVIDENCE SELECTION
============================================================

1. Select the retrieved source that most directly answers the question.

2. Question relevance is more important than retrieval score.

3. Do not combine unrelated recommendations merely because they concern
   the same disease.

4. If the question asks about one intervention, do not substitute another
   intervention from the retrieved context.

5. If the question asks for a definition, prioritize an explicit definition
   over diagnostic, imaging, treatment, or management information.

6. Use additional sources only when they are necessary to answer the
   specific question.

============================================================
ANSWER RULES
============================================================

1. Give ONE concise answer.

2. Answer only what the user asked.

3. Do not add unrelated symptoms, causes, treatments, background information,
   or explanations.

4. Do not repeat the answer.

5. Do not repeat citations unnecessarily.

6. Do not reproduce retrieved evidence verbatim unless a short excerpt is
   necessary.

7. For definition questions, normally use ONE direct definition.

8. For recommendation questions, provide ONE concise recommendation.

9. Keep the answer preferably under 40 words unless more information is
   required to accurately answer the question.

============================================================
OUTPUT FORMAT
============================================================

Return EXACTLY ONE section and nothing else:

Answer:
[One concise answer] [S#]

============================================================
STRICT OUTPUT RESTRICTIONS
============================================================

DO NOT output:
- Supporting Evidence
- Medical Disclaimer
- Evidence
- Explanation
- Additional Information
- Final Answer
- Retrieved Context
- Source Descriptions
- The user's question
- Internal reasoning
- Analysis
- Multiple versions of the answer
- Repeated sections

STOP generating immediately after the Answer block.
"""


# ============================================================
# LOAD RESOURCES
# ============================================================

_client = None
collection = None
bm25 = None
embed_model = None
reranker = None
tokenizer = None
llm = None

INDEX_READY = False
STARTUP_ERROR = None

try:
    print("Loading ChromaDB...")
    _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        collection = _client.get_collection(name=COLLECTION_NAME)
        print(f"✓ ChromaDB loaded ({collection.count()} documents)")
    except Exception as exc:
        STARTUP_ERROR = f"ChromaDB collection '{COLLECTION_NAME}' is missing: {exc}"
        print(f"⚠ {STARTUP_ERROR}")

    print("Loading BM25 index...")
    if BM25_FILE.exists():
        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)
        print("✓ BM25 loaded")
    else:
        if STARTUP_ERROR is None:
            STARTUP_ERROR = f"BM25 index not found at {BM25_FILE}"
        print(f"⚠ {STARTUP_ERROR}")

    print("Loading embedding model...")
    try:
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print("✓ Embedding model loaded")
    except Exception as exc:
        if STARTUP_ERROR is None:
            STARTUP_ERROR = f"Embedding model failed to load: {exc}"
        print(f"⚠ {STARTUP_ERROR}")

    print("Loading reranker...")
    try:
        reranker = CrossEncoder(RERANKER_NAME)
        print(f"✓ Reranker loaded: {RERANKER_NAME}")
    except Exception as exc:
        if STARTUP_ERROR is None:
            STARTUP_ERROR = f"Reranker failed to load: {exc}"
        print(f"⚠ {STARTUP_ERROR}")

    print("Loading LLM...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, torch_dtype=torch.float32)
        llm.eval()
        print("✓ LLM loaded")
    except Exception as exc:
        if STARTUP_ERROR is None:
            STARTUP_ERROR = f"LLM failed to load: {exc}"
        print(f"⚠ {STARTUP_ERROR}")

    INDEX_READY = collection is not None and bm25 is not None
except Exception as exc:
    STARTUP_ERROR = str(exc)
    print(f"⚠ Startup failed: {STARTUP_ERROR}")


# ============================================================
# EXTRACTION & RETRIEVAL
# ============================================================

def tokenize(text):
    if not text:
        return []
    return re.findall(r"\b[\w.]+\b", text.lower())


def _split_words_into_chunks(words, max_words=220, overlap=40):
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_uploaded_pdf_chunks(pdf_bytes, filename, max_words=220, overlap=40):
    """Extract chunkable text from an uploaded PDF."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Unable to read uploaded PDF: {exc}") from exc

    stem = Path(filename or "uploaded.pdf").stem or "uploaded_pdf"
    chunks = []

    for page_number, page in enumerate(document, start=1):
        page_text = re.sub(r"\s+", " ", page.get_text("text", sort=True) or "").strip()

        if not page_text:
            block_text = " ".join(
                str(block[4])
                for block in page.get_text("blocks", sort=True)
                if len(block) > 4 and block[4]
            )
            page_text = re.sub(r"\s+", " ", block_text).strip()

        # Fall back to OCR for scanned/image-only PDF pages.
        if not page_text:
            try:
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                page_text = re.sub(r"\s+", " ", pytesseract.image_to_string(image)).strip()
            except Exception:
                page_text = ""

        if not page_text:
            continue

        words = page_text.split()
        for chunk_index, chunk_text in enumerate(_split_words_into_chunks(words, max_words=max_words, overlap=overlap), start=1):
            chunks.append({
                "id": f"{stem}-p{page_number}-c{chunk_index}",
                "document": chunk_text,
                "metadata": {
                    "guideline": stem,
                    "title": stem,
                    "publisher": "Uploaded PDF",
                    "source": filename or stem,
                    "page": page_number,
                    "section": "Uploaded PDF",
                    "section_number": "",
                    "section_title": "Uploaded PDF",
                    "recommendation_number": "",
                    "source_type": "uploaded_pdf",
                },
            })
    return chunks


def retrieve_from_pdf(query, pdf_bytes, filename, top_k=5):
    """Extract and rank chunks from an uploaded PDF."""
    chunks = extract_uploaded_pdf_chunks(pdf_bytes, filename)
    if not chunks:
        raise ValueError(
            "No readable text was found in the uploaded PDF. "
            "For scanned PDFs, install the Tesseract OCR application and ensure it is on PATH."
        )

    # Use the cross-encoder to rank the PDF chunks against the user's question
    if reranker is not None:
        pairs = [[query, c["document"]] for c in chunks]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
    else:
        # Fallback if reranker fails
        ranked = list(zip(chunks, [0.0] * len(chunks)))

    results = []
    for chunk, score in ranked[:top_k]:
        meta = chunk["metadata"].copy()
        meta["text"] = chunk["document"]
        meta["chunk_id"] = chunk["id"]
        meta["rerank_score"] = float(score)
        meta["source"] = filename or "Uploaded Document"
        meta["guideline"] = "Attached PDF" 
        results.append(meta)

    return results


def retrieve_merged(query, pdf_bytes=None, pdf_name=None, top_k=5):
    """Retrieve from the guideline database and optional uploaded PDF."""
    guideline_results = retrieve(query, top_k=top_k)
    pdf_results = []

    if pdf_bytes:
        pdf_results = retrieve_from_pdf(query, pdf_bytes, pdf_name, top_k=top_k)

    merged = guideline_results + pdf_results
    if not merged:
        return []

    # Rerank both source types together so the answer receives the strongest
    # passages regardless of which source produced them.
    if reranker is not None and len(merged) > 1:
        pairs = [[query, item.get("text", "")] for item in merged]
        scores = reranker.predict(pairs)
        for item, score in zip(merged, scores):
            item["rerank_score"] = float(score)

    merged.sort(key=lambda item: float(item.get("rerank_score", 0.0)), reverse=True)
    return merged[:top_k]


def retrieve(query, top_k=5, candidate_k=50, min_score=None, max_distance=None, debug=False):
    """Hybrid retrieval pipeline from ChromaDB (used when NO PDF is attached)."""
    if min_score is None:
        min_score = MIN_RERANK_SCORE
    if max_distance is None:
        max_distance = MAX_CHROMA_DISTANCE

    if collection is None or bm25 is None or embed_model is None or reranker is None:
        return []

    # --- 1. Dense retrieval ---
    retrieval_query = QUERY_INSTRUCTION + query

    query_embedding = embed_model.encode(
        [retrieval_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    chroma_results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    dense_candidates = []
    if chroma_results["ids"]:
        ids = chroma_results["ids"][0]
        docs = chroma_results["documents"][0]
        metas = chroma_results["metadatas"][0]
        dists = chroma_results["distances"][0]
        for i in range(len(ids)):
            dense_candidates.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                "distance": dists[i],
            })

    # --- 2. BM25 retrieval ---
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

    bm25_candidates = []
    if len(bm25_indices) > 0:
        all_docs = collection.get(include=["documents", "metadatas"])
        all_ids = all_docs["ids"]
        all_texts = all_docs["documents"]
        all_metas = all_docs["metadatas"]

        for idx in bm25_indices:
            idx = int(idx)
            if idx >= len(all_ids):
                continue
            bm25_candidates.append({
                "id": all_ids[idx],
                "document": all_texts[idx],
                "metadata": all_metas[idx],
                "bm25_score": float(bm25_scores[idx]),
            })

    # --- 3. Merge dense + BM25 candidates ---
    candidate_map = {}
    for c in dense_candidates:
        candidate_map[c["id"]] = c
    for c in bm25_candidates:
        cid = c["id"]
        if cid in candidate_map:
            candidate_map[cid]["bm25_score"] = c["bm25_score"]
        else:
            candidate_map[cid] = c

    candidates = list(candidate_map.values())
    if not candidates:
        return []

    # --- 4. Cross-encoder rerank ---
    pairs = [[query, c["document"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: float(x[1]), reverse=True)

    # --- 5. Build final filtered, top-k results ---
    results = []
    for candidate, score in ranked:
        score = float(score)

        if USE_RERANK_GATE and score < min_score:
            continue

        distance = candidate.get("distance", None)
        if USE_CHROMA_GATE and distance is not None and distance > max_distance:
            continue

        metadata = candidate["metadata"].copy()
        metadata["rerank_score"] = score
        if distance is not None:
            metadata["chroma_distance"] = float(distance)
        if "bm25_score" in candidate:
            metadata["bm25_score"] = float(candidate["bm25_score"])
        metadata["text"] = candidate["document"]
        metadata["chunk_id"] = candidate["id"]

        results.append(metadata)
        if len(results) >= top_k:
            break

    return results


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_context(results):
    """Turn retrieved chunks into a citation-tagged context block."""
    context_parts = []

    for i, result in enumerate(results, start=1):
        text = result.get("text", "")
        source = result.get("source", "Unknown")
        page = result.get("page", "Unknown")
        section = result.get("section_title", result.get("section", "Unknown"))
        guideline = result.get("guideline", "Source Document")
        section_number = result.get("section_number", "Not specified")
        publisher = result.get("publisher", "NICE")
        chunk_id = result.get("chunk_id", "Unknown")
        citation_id = f"S{i}"

        context_parts.append(f"""
Citation:
SOURCE {i}
Citation ID: [{citation_id}]

Publisher: {publisher}
Guideline: {guideline}
Section Number: {section_number}
Chunk ID: {chunk_id}
Retrieval Score: {result.get("rerank_score", 0):.4f}
Section: {section}
Page: {page}
Source file: {source}

Evidence:
{text}
""")
    return "\n".join(context_parts)


def build_prompt(query, results):
    evidence_context = build_context(results)
    
    prompt_parts = []
    prompt_parts.append(f"Retrieved Evidence Context:\n{evidence_context}")
    prompt_parts.append(f"Question: {query}")
    
    return "\n\n".join(prompt_parts)


# ============================================================
# GENERATION
# ============================================================

def generate_answer(query, results):
    disclaimer = (
        "\n\n**Medical Disclaimer:**\n"
        "This information is based on the retrieved documents and is provided for "
        "informational and clinical decision-support purposes only. It does not replace "
        "professional medical judgment, diagnosis, or individualized medical advice."
    )
    
    if not results:
        return "Answer:\nEvidence is insufficient in the retrieved documents to answer this question." + disclaimer

    if tokenizer is None or llm is None:
        return "Answer:\nThe generation model is unavailable in this session." + disclaimer

    user_prompt = build_prompt(query, results)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Safely append the disclaimer 
    return raw_answer + disclaimer


# ============================================================
# PIPELINE ENTRY POINT
# ============================================================

def rag_pipeline(query=None, top_k=5, uploaded_pdf_bytes=None, uploaded_pdf_name=None):
    """
    Retrieve from the guideline database, optionally merged with an uploaded PDF.
    """
    query = (query or "").strip()
    has_pdf = uploaded_pdf_bytes is not None

    if not query and not has_pdf:
        raise ValueError("Provide a question, an attached PDF, or both.")

    if has_pdf and not query:
        query = DEFAULT_PDF_ONLY_QUESTION

    results = retrieve_merged(
        query,
        pdf_bytes=uploaded_pdf_bytes,
        pdf_name=uploaded_pdf_name,
        top_k=top_k,
    )
    print(f"\n[DEBUG] Merged retrieval active. Query: '{query}'")
    print(f"[DEBUG] Guideline/PDF evidence chunks returned: {len(results)}")

    # 2. Build and generate from the merged evidence set.
    prompt = build_prompt(query, results)
    answer = generate_answer(query, results) 
    
    return answer, prompt, results

if __name__ == "__main__":
    # Quick manual smoke test
    test_query = "What imaging should be offered to someone with suspected endometriosis?"
    ans, ctx, results = rag_pipeline(test_query)
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(ans)