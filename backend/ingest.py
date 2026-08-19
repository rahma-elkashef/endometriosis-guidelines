
import re
import json
from pathlib import Path

import pymupdf


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

OUTPUT_FILE = JSON_DIR / "nice_documents.json"


# ============================================================
# PDF METADATA
# ============================================================
# Add an entry here for every new guideline PDF you drop into PDF_DIR.

PDF_METADATA = {
    "endometriosis-pdf-75545657547973.pdf": {
        "guideline": "QS172",
        "title": "Endometriosis",
        "publisher": "National Institute of Health and Care Excellence (NICE)",
    },
    "endometriosis-diagnosis-and-management-pdf-1837632548293.pdf": {
        "guideline": "NG73",
        "title": "Endometriosis: diagnosis and management",
        "publisher": "National Institute of Health and Care Excellence (NICE)",
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize PDF-extracted text while preserving paragraph/line boundaries."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix words split across lines: "recom-\nmendation" -> "recommendation"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    text = text.replace("\t", " ")
    text = re.sub(r"\.{2,}", " ", text)          # repeated dots
    text = re.sub(r"[ ]{2,}", " ", text)          # repeated spaces
    text = re.sub(r" *\n *", "\n", text)          # spaces around newlines
    text = re.sub(r"\n{3,}", "\n\n", text)        # excessive blank lines

    return text.strip()


def remove_nice_headers_footers(text):
    """Remove common NICE PDF headers and footers."""
    if not text:
        return ""

    patterns = [
        r"©\s*NICE\s*\d{4}\.\s*All rights reserved\.",
        r"National Institute of Health and Care Excellence",
        r"National Institute for Health and Care Excellence",
        r"www\.nice\.org\.uk",
        r"nice\.org\.uk",
        r"Subject to Notice of rights.*",
        r"Page\s+\d+\s+of\s+\d+",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text


def clean_text(text):
    """General cleaning on extracted PDF text."""
    if not text:
        return ""

    text = normalize_text(text)
    text = remove_nice_headers_footers(text)
    text = normalize_text(text)

    return text.strip()


# ============================================================
# SECTION DETECTION
# ============================================================

def extract_section_from_block(block):
    """
    Detect a NICE main section heading (e.g. "1.4 Pharmacological treatment...").
    Recommendation numbers like "1.5.10" are NOT treated as main sections.
    """
    if "lines" not in block:
        return None

    all_spans = []
    for line in block["lines"]:
        for span in line.get("spans", []):
            span_text = span.get("text", "").strip()
            if span_text:
                all_spans.append(span)

    if not all_spans:
        return None

    block_text = " ".join(span["text"].strip() for span in all_spans)
    block_text = re.sub(r"\s+", " ", block_text).strip()

    match = re.match(r"^\s*(\d+\.\d+)\s+(.+?)\s*$", block_text)
    if not match:
        return None

    section_number = match.group(1).strip()
    section_title = match.group(2).strip()

    font_names = [span.get("font", "") for span in all_spans]
    font_sizes = [span.get("size", 0) for span in all_spans]
    font_sizes = [s for s in font_sizes if isinstance(s, (int, float))]

    if not font_sizes:
        return None

    average_size = sum(font_sizes) / len(font_sizes)
    is_lora = any("lora" in font.lower() for font in font_names)
    is_large = average_size >= 18

    if not (is_lora and is_large):
        return None

    return {
        "section_number": section_number,
        "section_title": section_title,
        "section": f"{section_title}",
    }


def detect_section_from_page(page, previous_section=None):
    """Detect the latest main section heading on the page, falling back to previous."""
    data = page.get_text("dict", sort=True)
    detected_section = None

    for block in data.get("blocks", []):
        section = extract_section_from_block(block)
        if section is not None:
            detected_section = section

    return detected_section if detected_section is not None else previous_section


# ============================================================
# SECTION / SECTION-TITLE ARTIFACT REMOVAL
# ============================================================
# NICE pages often carry a literal running header like:
#
#   Section: 1.10
#   Section title: Management if fertility is a priority
#
# ...sometimes collapsed onto one line with no separator before the body
# text starts. This is decomposed into independent, tolerant steps so a
# partial mismatch between the detected section_title and the literal
# header text doesn't block removal of the labels.

def _strip_known_title_prefix(text, title):
    """
    Remove a leading occurrence of `title` from `text`, tolerating minor
    differences by matching word-by-word instead of an exact substring.
    Only strips if a strong majority of the title's words match, in order,
    at the very start of `text`.
    """
    if not text or not title:
        return text

    title_words = re.findall(r"\w+", title)
    if not title_words:
        return text

    pos = 0
    matched_words = 0

    for target_word in title_words:
        match = re.match(r"\s*([^\s]+)", text[pos:])
        if not match:
            break

        candidate = re.sub(r"^\W+|\W+$", "", match.group(1))

        if candidate.lower() == target_word.lower():
            pos += match.end()
            matched_words += 1
        else:
            break

    required = max(1, int(round(len(title_words) * 0.6)))
    if matched_words < required:
        return text

    remainder = text[pos:]
    remainder = re.sub(r"^[\s:.\-–—]+", "", remainder)

    return remainder


def remove_section_artifacts(text, section_number=None, section_title=None):
    """
    Robustly remove section-heading artifacts from extracted text, handling:
        Section: 1.4
        Section title: Pharmacological treatment...
        Section: 1.4 Section title: Pharmacological treatment...
        1.4 Pharmacological treatment...
    Section info is preserved separately in metadata; only the redundant
    in-text header is stripped here.
    """
    if not text:
        return ""

    text = normalize_text(text)
    if not text:
        return ""

    number = str(section_number).strip() if section_number else None
    title = str(section_title).strip() if section_title else None

    # Step 1: strip leading "Section: <number>" label
    if number:
        specific_pattern = rf"^\s*Section\s*:\s*{re.escape(number)}\s*"
        stripped = re.sub(specific_pattern, "", text, count=1, flags=re.IGNORECASE)
        if stripped != text:
            text = stripped
        else:
            text = re.sub(r"^\s*Section\s*:\s*\d+(?:\.\d+)+\s*", "", text, count=1, flags=re.IGNORECASE)
    else:
        text = re.sub(r"^\s*Section\s*:\s*\d+(?:\.\d+)+\s*", "", text, count=1, flags=re.IGNORECASE)

    # Step 2: strip leading "Section title:" label
    text = re.sub(r"^\s*Section\s+title\s*:\s*", "", text, count=1, flags=re.IGNORECASE)

    # Step 3: strip bare leading heading number, e.g. "1.10 Management..."
    if number:
        text = re.sub(rf"^\s*{re.escape(number)}\s+", "", text, count=1)

    # Step 4: strip actual title text via fuzzy word matching
    if title:
        text = _strip_known_title_prefix(text, title)

    return normalize_text(text).strip()


def remove_any_section_labels(text):
    """Final safety net for leftover section labels when detection failed."""
    if not text:
        return ""

    text = re.sub(r"^\s*Section\s*:\s*\d+(?:\.\d+)+\s*", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"^\s*Section\s+title\s*:\s*.{1,200}?(?=\n|\.\s+[A-Z]|$)",
        "",
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(r"^\s*\d+\.\d+\s+[^\n]+(?=\n|$)", "", text, flags=re.IGNORECASE)

    return normalize_text(text).strip()


def detect_recommendation(text):
    """Detect NICE recommendation numbers, e.g. 1.5.10."""
    if not text:
        return None

    match = re.search(r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)", text)
    return match.group(1) if match else None


# ============================================================
# EXTRACT ONE PDF
# ============================================================

def extract_pdf(pdf_path):
    """Extract one PDF page-by-page. Section info stored in metadata, stripped from text."""
    documents = []
    filename = pdf_path.name

    metadata = PDF_METADATA.get(filename)
    if metadata is None:
        print(f"⚠ Warning: No metadata found for {filename}")
        metadata = {
            "guideline": None,
            "title": None,
            "publisher": "National Institute of Health and Care Excellence (NICE)",
        }

    pdf = pymupdf.open(pdf_path)
    current_section = None

    for page_number, page in enumerate(pdf, start=1):
        print(f"  Processing page {page_number}/{len(pdf)}", end="\r")

        new_section = detect_section_from_page(page, current_section)
        if new_section is not None:
            current_section = new_section

        raw_text = page.get_text("text", sort=True)
        if not raw_text:
            continue

        text = clean_text(raw_text)
        if not text:
            continue

        if current_section:
            text = remove_section_artifacts(
                text,
                section_number=current_section.get("section_number"),
                section_title=current_section.get("section_title"),
            )

        text = remove_any_section_labels(text)
        if not text:
            continue

        recommendation_number = detect_recommendation(text)

        document = {
            "guideline": metadata.get("guideline"),
            "title": metadata.get("title"),
            "publisher": metadata.get("publisher"),
            "source": filename,
            "page": page_number,
            "section_number": current_section["section_number"] if current_section else "Not Found",
            "section_title": current_section["section_title"] if current_section else "Not Found",
            "section": current_section["section"] if current_section else "Not Found",
            "recommendation_number": recommendation_number,
            "text": text,
        }

        documents.append(document)

    print()
    pdf.close()

    return documents


def process_all_pdfs():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    if not pdf_files:
        print(f"⚠ No PDF files found in {PDF_DIR}")
        return []

    all_documents = []

    for pdf_path in pdf_files:
        print("\n" + "=" * 70)
        print(f"Processing: {pdf_path.name}")
        print("=" * 70)

        try:
            documents = extract_pdf(pdf_path)
            all_documents.extend(documents)
            print(f"✓ Extracted {len(documents)} pages")
        except Exception as e:
            print(f"✗ Failed to process {pdf_path.name}: {e}")

    return all_documents


def validate_documents(documents):
    """Check whether section labels are still leaking into the text field."""
    print("\n" + "=" * 70)
    print("VALIDATING CLEAN TEXT")
    print("=" * 70)

    problematic = []
    patterns = [r"Section\s*:", r"Section\s+title\s*:", r"^\s*\d+\.\d+\s+[A-Z]"]

    for index, document in enumerate(documents):
        text = document.get("text", "")
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                problematic.append((index, document.get("source"), document.get("page"), text[:200]))
                break

    if problematic:
        print(f"⚠ Found {len(problematic)} documents that may still contain section artifacts.")
        for item in problematic[:10]:
            print("\nProblem:")
            print(f"Index: {item[0]}")
            print(f"Source: {item[1]}")
            print(f"Page: {item[2]}")
            print(f"Text preview: {item[3]}")
    else:
        print("✓ No section artifacts detected in text fields.")


def save_json(documents):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(documents, file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(documents)} pages to:\n{OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    documents = process_all_pdfs()
    validate_documents(documents)
    save_json(documents)

    print("\n" + "=" * 70)
    print("PDF processing completed!")
    print("=" * 70)
