import re
import json
from pathlib import Path


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

INPUT_FILE = JSON_DIR / "nice_documents.json"
OUTPUT_FILE = JSON_DIR / "nice_chunks.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print(f"Loaded {len(documents)} documents/pages.")
    return documents


# ============================================================
# CLEAN CHUNK TEXT
# ============================================================

def clean_chunk_text(text, section_number=None, section_title=None):
    """
    Final defensive cleaning step. Removes section metadata that may have
    accidentally remained inside the extracted document text.

    Applied ONLY to the actual chunk text — section info is preserved
    separately in metadata and embedding_text.
    """
    if not text:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\t", " ")

    # Explicit section labels, e.g. "Section: 1.4"
    text = re.sub(r"(?im)^\s*Section\s*:\s*\d+\.\d+\s*$", "", text)

    # "Section title: Pharmacological treatment..."
    text = re.sub(r"(?im)^\s*Section\s+title\s*:\s*.*$", "", text)

    # Collapsed same-line form: "Section: 1.4 Section title: ..."
    text = re.sub(
        r"(?is)Section\s*:\s*\d+\.\d+\s*Section\s+title\s*:\s*.*?(?=\n|$)",
        "",
        text,
    )

    if (
        section_number
        and section_title
        and section_number != "Not Found"
        and section_title != "Not Found"
    ):
        number = re.escape(str(section_number).strip())
        title = re.escape(str(section_title).strip())

        # "1.4 Pharmacological treatment..."
        text = re.sub(rf"(?im)^\s*{number}\s+{title}\s*$", "", text)

        # "Section: 1.4 Pharmacological treatment..."
        text = re.sub(rf"(?im)^\s*Section\s*:\s*{number}\s+{title}\s*$", "", text)

        # "Section title: ..."
        text = re.sub(rf"(?im)^\s*Section\s+title\s*:\s*{title}\s*$", "", text)

    # Generic heading removal — catches "1.4 Diagnosis..." even without
    # matching metadata, but does NOT strip recommendation numbers like
    # "1.4.7" (those have three numeric components).
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        if re.match(r"^\d+\.\d+\s+.+$", stripped) and not re.match(r"^\d+\.\d+\.\d+", stripped):
            continue

        if re.match(r"^Section\s*:", stripped, flags=re.IGNORECASE):
            continue

        if re.match(r"^Section\s+title\s*:", stripped, flags=re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # NICE headers/footers
    patterns = [
        r"© NICE \d{4}\. All rights reserved\.",
        r"National Institute of Health and Care Excellence",
        r"National Institute for Health and Care Excellence",
        r"www\.nice\.org\.uk",
        r"nice\.org\.uk",
        r"Subject to Notice of rights.*",
        r"Page\s+\d+\s+of\s+\d+",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# ============================================================
# SENTENCE SPLITTING
# ============================================================

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_into_sentences(text):
    if not text:
        return []

    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ============================================================
# CREATE SENTENCE-AWARE CHUNKS
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1

        if current and current_len + sentence_len > chunk_size:
            chunks.append(" ".join(current))

            # Build overlap from the tail of the current chunk
            overlap_sentences = []
            overlap_len = 0

            for s in reversed(current):
                s_len = len(s) + 1
                if overlap_len + s_len > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += s_len

            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


# ============================================================
# CHROMA-SAFE UNIQUE ID
# ============================================================

def create_chunk_id(document, global_index):
    guideline = document.get("guideline") or "UNKNOWN"
    source = document.get("source") or "unknown"
    page = document.get("page") or 0
    section_number = document.get("section_number") or "unknown"

    guideline = re.sub(r"[^a-zA-Z0-9_-]", "_", str(guideline))
    source = re.sub(r"[^a-zA-Z0-9_-]", "_", str(source))
    section_number = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(section_number))

    return f"{guideline}_{source}_page_{page}_section_{section_number}_chunk_{global_index}"


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def chunk_documents(documents):
    chunked_documents = []
    global_chunk_index = 0

    for document in documents:
        original_text = document.get("text", "")
        if not original_text:
            continue

        section_number = document.get("section_number") or ""
        section_title = document.get("section_title") or ""
        # NOTE: intentionally NOT falling back to document["section"] here —
        # that field can contain "1.4 Pharmacological treatment..." which
        # would contaminate the embedding/context text.
        recommendation_number = document.get("recommendation_number") or ""

        cleaned_text = clean_chunk_text(original_text, section_number, section_title)
        if not cleaned_text:
            continue

        chunks = create_chunks(cleaned_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        for chunk_index, chunk in enumerate(chunks):
            # Clean again defensively in case a heading survived into a chunk boundary
            chunk = clean_chunk_text(chunk, section_number, section_title)
            if not chunk:
                continue

            chunk_id = create_chunk_id(document, global_chunk_index)
            global_chunk_index += 1

            # Build embedding_text: section context + actual chunk text.
            # `text` stays clean (no section context) so retrieved evidence
            # shown to the user/LLM never contains metadata noise.
            context_parts = []
            if section_number:
                context_parts.append(f"Section: {section_number}")
            if section_title:
                context_parts.append(f"Section title: {section_title}")
            if recommendation_number:
                context_parts.append(f"Recommendation: {recommendation_number}")

            context = "\n".join(context_parts)
            embedding_text = f"{context}\n{chunk}" if context else chunk

            chunk_document = {
                "guideline": document.get("guideline"),
                "title": document.get("title"),
                "publisher": document.get("publisher"),
                "source": document.get("source"),
                "page": document.get("page"),
                "section": f"{section_title}",
                "section_number": section_number,
                "section_title": section_title,
                "recommendation_number": recommendation_number,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "text": chunk,
                "embedding_text": embedding_text,
            }

            chunked_documents.append(chunk_document)

    return chunked_documents


# ============================================================
# VALIDATION
# ============================================================

def validate_chunk_ids(chunks):
    ids = [chunk["chunk_id"] for chunk in chunks]
    unique_ids = set(ids)

    print(f"Total chunk IDs: {len(ids)}")
    print(f"Unique chunk IDs: {len(unique_ids)}")

    if len(ids) != len(unique_ids):
        print("❌ Duplicate chunk IDs detected!")
        duplicates = {id_ for id_ in ids if ids.count(id_) > 1}
        print("Duplicates:", duplicates)
        raise ValueError("Chunk IDs must be unique for ChromaDB.")

    print("✓ All chunk IDs are unique.")


def validate_no_section_contamination(chunks):
    contamination_patterns = [r"Section\s*:", r"Section\s+title\s*:"]
    contaminated = []

    for chunk in chunks:
        text = chunk.get("text", "")

        for pattern in contamination_patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                contaminated.append(chunk)
                break

        if re.search(r"(?m)^\s*\d+\.\d+\s+[A-Za-z]", text):
            contaminated.append(chunk)

    unique_contaminated = {c["chunk_id"]: c for c in contaminated}
    contaminated = list(unique_contaminated.values())

    print("\nText contamination check:")

    if not contaminated:
        print("✓ No section heading contamination detected in chunk text.")
        return

    print(f"⚠ Found {len(contaminated)} potentially contaminated chunks.")
    for chunk in contaminated[:10]:
        print("\n------------------------------------------------")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("Text:")
        print(chunk["text"][:500])


def save_chunks(chunks):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(chunks)} chunks to:\n{OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("NICE DOCUMENT CHUNKING")
    print("=" * 70)
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")

    documents = load_documents()
    chunks = chunk_documents(documents)

    validate_chunk_ids(chunks)
    validate_no_section_contamination(chunks)
    save_chunks(chunks)

    print("\n" + "=" * 70)
    print("Chunking completed!")
    print("=" * 70)
