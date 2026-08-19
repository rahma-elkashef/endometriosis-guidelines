import re
import pickle
from pathlib import Path

import numpy as np
import chromadb
import torch
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

# Relevance gate thresholds.
# These are INITIAL values — tune against your 15-20 test queries,
# not final scientifically validated cutoffs.
MIN_RERANK_SCORE = 0.0
MAX_CHROMA_DISTANCE = 0.65
USE_RERANK_GATE = True
USE_CHROMA_GATE = True


# ============================================================
# SYSTEM PROMPT
# ============================================================
# This is the prompt actually wired into generation (previously
# system_prompt.py). The longer, excerpt-heavy version that used
# to live in build_context.py was never imported by generate_answer.py
# and has been dropped as dead code — restore it here if you decide
# you want excerpt-style answers instead of the concise 3-section format.

SYSTEM_PROMPT = """
You are an evidence-grounded clinical decision support assistant.

Answer the user's question using ONLY the retrieved guideline context.

============================================================
CORE RULES
============================================================

1. Use only information explicitly supported by the retrieved context.

2. Do not use general medical knowledge or assumptions.

3. If the retrieved context is insufficient, say:
   "Evidence is insufficient in the retrieved guidelines to answer this question."

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
CONFIDENCE
============================================================

Use:

High:
A direct and explicit guideline statement answers the question.

Moderate:
Relevant evidence exists but is indirect or requires limited interpretation.

Low:
Evidence is incomplete, weak, conflicting, or insufficient.

Do not claim high confidence merely because many sources were retrieved.

============================================================
OUTPUT FORMAT
============================================================

Return EXACTLY these three sections and nothing else:

Answer:
[One concise answer] [S#]

Confidence and Safety:
[High / Moderate / Low]
[One short sentence explaining the confidence.]

Medical Disclaimer:
This information is based on the retrieved guidelines and is provided for
informational and clinical decision-support purposes only. It does not replace
professional medical judgment, diagnosis, or individualized medical advice.

============================================================
STRICT OUTPUT RESTRICTIONS
============================================================

DO NOT output:
- Supporting Evidence
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
- Any section other than the three specified above

After the Medical Disclaimer, STOP generating immediately.

Do not continue writing after the disclaimer.
"""


# ============================================================
# LOAD RESOURCES (module-level, loaded once on import)
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
# RETRIEVAL
# ============================================================

def tokenize(text):
    """
    Tokenize text while preserving clinically important recommendation
    numbers such as 1.5.10, 1.5.11, plus normal words/abbreviations
    (MRI, TVUS, ultrasound, ...).
    """
    if not text:
        return []
    return re.findall(r"\b[\w.]+\b", text.lower())


def retrieve(query, top_k=5, candidate_k=50, min_score=None, max_distance=None, debug=False):
    """
    Hybrid retrieval pipeline:
        1. BGE dense semantic retrieval
        2. BM25 lexical retrieval
        3. Merge candidates
        4. Cross-encoder reranking
        5. Relevance gate (best result must pass Chroma distance + rerank score)
        6. Return top-k results as a list of metadata dicts (each includes
           'text', 'chunk_id', 'rerank_score', 'chroma_distance', ...)
    """
    if min_score is None:
        min_score = MIN_RERANK_SCORE
    if max_distance is None:
        max_distance = MAX_CHROMA_DISTANCE

    if collection is None or bm25 is None or embed_model is None or reranker is None:
        return []

    # --- 1. Dense retrieval ---
    query_embedding = embed_model.encode(
        [QUERY_INSTRUCTION + query],
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

    # --- 3. Merge dense + BM25 candidates (dedup by chunk id) ---
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

    if debug:
        print("\n" + "=" * 80)
        print("RERANKING DIAGNOSTICS")
        print("=" * 80)
        print(f"Query: {query}")
        print(f"Candidates: {len(ranked)}")
        for rank, (candidate, score) in enumerate(ranked, start=1):
            meta = candidate["metadata"]
            print(f"\n{rank}. Rerank={float(score):.4f} | "
                  f"Chroma distance={candidate.get('distance', 'N/A')} | "
                  f"BM25={candidate.get('bm25_score', 'N/A')}")
            print(f"   Section: {meta.get('section', '')}")
            print(f"   Page: {meta.get('page', '')}")

    # --- 5. Relevance gate on the best candidate ---
    best_candidate, best_score = ranked[0][0], float(ranked[0][1])
    best_distance = best_candidate.get("distance", None)

    if USE_RERANK_GATE and best_score < min_score:
        if debug:
            print(f"\n❌ Relevance gate failed: best rerank score {best_score:.4f} < {min_score:.4f}")
        return []

    if USE_CHROMA_GATE and best_distance is not None and best_distance > max_distance:
        if debug:
            print(f"\n❌ Semantic gate failed: best distance {best_distance:.4f} > {max_distance:.4f}")
        return []

    # --- 6. Build final filtered, top-k results ---
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
        guideline = result.get("guideline", "NICE NG73")
        section_number = result.get("section_number", "Not specified")
        publisher = result.get("publisher", "NICE")
        chunk_id = result.get("chunk_id")
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
    context = build_context(results)
    return f"""

Retrieved NICE guideline context:

{context}
"""


# ============================================================
# GENERATION
# ============================================================

def generate_answer(query, results):
    if not results:
        return """
Recommendation:
Evidence is insufficient in the retrieved guidelines.

Confidence and safety:
Low confidence. No sufficiently relevant guideline evidence was retrieved.

Citation:
None.
"""

    if tokenizer is None or llm is None:
        return """
Recommendation:
Evidence is insufficient in the retrieved guidelines.

Confidence and safety:
Low confidence. The generation model is unavailable in this session.

Citation:
None.
"""

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
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ============================================================
# PIPELINE ENTRY POINT
# ============================================================

def rag_pipeline(query, top_k=5):
    """
    Run the full pipeline: retrieve -> build prompt -> generate answer.

    Returns:
        (answer, prompt, results) tuple.
    """
    results = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, results)
    answer = generate_answer(query, results)
    return answer, prompt, results


if __name__ == "__main__":
    # Quick manual smoke test: python rag.py
    test_query = "What imaging should be offered to someone with suspected endometriosis?"
    ans, ctx, results = rag_pipeline(test_query)
    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(ans)
