
import re
import json
from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "data" / "pdfs"
JSON_DIR = BASE_DIR / "data" / "json"
VECTOR_DIR = BASE_DIR / "data" / "vector_store"
UPLOAD_DIR = BASE_DIR / "uploads"

#============================================================
# PDF METADATA
# ============================================================
PDF_METADATA = {

    "endometriosis-pdf-75545657547973.pdf": {
        "guideline": "QS172",
        "title": "Endometriosis",
        "publisher": (
            "National Institute of Health "
            "and Care Excellence (NICE)"
        )
    },

    "endometriosis-diagnosis-and-management-pdf-1837632548293.pdf": {
        "guideline": "NG73",
        "title": "Endometriosis: diagnosis and management",
        "publisher": (
            "National Institute of Health "
            "and Care Excellence (NICE)"
        )
    },
}

# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):
    """
    Normalize PDF-extracted text while preserving useful
    paragraph/line boundaries.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Fix words split across lines
    #
    # recom-
    # mendation
    #
    # -> recommendation
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Replace tabs
    text = text.replace(
        "\t",
        " "
    )

    # Remove repeated dots
    text = re.sub(
        r"\.{2,}",
        " ",
        text
    )

    # Normalize spaces inside lines
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text
    )

    # Remove spaces surrounding newlines
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()

# ============================================================
# REMOVE NICE HEADERS / FOOTERS
# ============================================================

def remove_nice_headers_footers(text):
    """
    Remove common NICE PDF headers and footers.
    """

    if not text:
        return ""

    patterns = [

        # Copyright
        r"©\s*NICE\s*\d{4}\.\s*All rights reserved\.",

        # NICE organization name
        r"National Institute of Health and Care Excellence",

        r"National Institute for Health and Care Excellence",

        # Websites
        r"www\.nice\.org\.uk",

        r"nice\.org\.uk",

        # Notice of rights
        r"Subject to Notice of rights.*",

        # Page numbering
        r"Page\s+\d+\s+of\s+\d+",

    ]

    for pattern in patterns:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    return text

# ============================================================
# GENERAL TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Perform general cleaning on extracted PDF text.
    """

    if not text:
        return ""

    # Normalize
    text = normalize_text(text)

    # Remove NICE headers/footers
    text = remove_nice_headers_footers(
        text
    )

    # Normalize again after removal
    text = normalize_text(text)

    return text.strip()

# ============================================================
# SECTION DETECTION
# ============================================================

def extract_section_from_block(block):
    """
    Detect a NICE main section heading.

    Examples:

        1.4 Pharmacological treatment...

        1.5 Diagnosis and referral...

    Main section numbers:

        1.1
        1.4
        1.5
        1.10

    Recommendation numbers such as:

        1.5.10
        1.5.11

    are NOT considered main sections.
    """

    if "lines" not in block:
        return None

    all_spans = []

    for line in block["lines"]:

        for span in line.get(
            "spans",
            []
        ):

            span_text = span.get(
                "text",
                ""
            ).strip()

            if span_text:

                all_spans.append(
                    span
                )

    if not all_spans:
        return None
 # --------------------------------------------------------
    # Combine block text
    # --------------------------------------------------------

    block_text = " ".join(

        span["text"].strip()

        for span in all_spans

    )

    block_text = re.sub(
        r"\s+",
        " ",
        block_text
    ).strip()

    # --------------------------------------------------------
    # Detect section number + title
    # --------------------------------------------------------

    match = re.match(
        r"^\s*(\d+\.\d+)\s+(.+?)\s*$",
        block_text
    )

    if not match:
        return None

    section_number = match.group(
        1
    ).strip()

    section_title = match.group(
        2
    ).strip()

    # --------------------------------------------------------
    # Font information
    # --------------------------------------------------------

    font_names = [

        span.get(
            "font",
            ""
        )

        for span in all_spans

    ]

    font_sizes = [

        span.get(
            "size",
            0
        )

        for span in all_spans

    ]

    font_sizes = [

        size

        for size in font_sizes

        if isinstance(
            size,
            (int, float)
        )

    ]

    if not font_sizes:
        return None

    average_size = (
        sum(font_sizes)
        /
        len(font_sizes)
    )

    # --------------------------------------------------------
    # Lora heading detection
    # --------------------------------------------------------

    is_lora = any(

        "lora" in font.lower()

        for font in font_names

    )

    # --------------------------------------------------------
    # Large heading detection
    # --------------------------------------------------------

    is_large = (
        average_size >= 18
    )

    # --------------------------------------------------------
    # Accept heading
    # --------------------------------------------------------

    if not (
        is_lora
        and is_large
    ):
        return None

    return {

        "section_number":
            section_number,

        "section_title":
            section_title,

        "section":
            (
                f"{section_number} "
                f"{section_title}"
            )
    }


# ============================================================
# DETECT SECTION FROM PAGE
# ============================================================

def detect_section_from_page(
    page,
    previous_section=None
):
    """
    Detect the latest main section heading on the page.

    If no new section is found, retain the previous section.
    """

    data = page.get_text(
        "dict",
        sort=True
    )

    detected_section = None

    for block in data.get(
        "blocks",
        []
    ):

        section = (
            extract_section_from_block(
                block
            )
        )

        if section is not None:

            detected_section = section

    if detected_section is not None:

        return detected_section

    return previous_section


# ============================================================
# SECTION / SECTION-TITLE ARTIFACT REMOVAL
# ============================================================
#
# NICE pages often carry a literal running header like:
#
#   Section: 1.10
#   Section title: Management if fertility is a priority
#
# ...sometimes on separate lines, sometimes collapsed onto one
# line with no separator before the actual body text starts,
# e.g.:
#
#   "Section: 1.10 Section title: Management if fertility is a
#    priority Based on this, the committee agreed..."
#
# The previous implementation tried to strip this using a
# single regex that required BOTH the section number AND the
# full escaped section title to match in one shot. If the
# detected `section_title` (from font/heading analysis)
# differed from the literal header text in *any* way -- extra
# whitespace, a stray character, different trailing
# punctuation, etc. -- the whole pattern failed to match and
# NOTHING was stripped, which is exactly the artifact seen in
# the sample output ("Section: 1.10 Section title: ..." showing
# up inside "text").
#
# The rewrite below decomposes the removal into independent,
# tolerant steps so a partial mismatch in the title text no
# longer blocks removal of the labels, and the title text
# itself is stripped via fuzzy word-by-word matching rather
# than a brittle exact-string regex.
# ============================================================

def _strip_known_title_prefix(text, title):
    """
    Remove a leading occurrence of `title` from `text`, tolerating
    minor differences (extra/missing punctuation, whitespace,
    case) by matching word-by-word instead of requiring an exact
    substring match.

    Only strips if a strong majority of the title's words are
    found, in order, at the very start of `text` -- this avoids
    accidentally eating into real body text.
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
            # allow the match to continue past isolated glue
            # words like "and"/"of" that sometimes get merged
            # oddly, but stop on a genuine mismatch
            break

    # Require most of the title to have actually matched before
    # we trust this is the heading and not real body text.
    required = max(1, int(round(len(title_words) * 0.6)))

    if matched_words < required:
        return text

    remainder = text[pos:]

    # Clean up any leftover separators / punctuation glue
    remainder = re.sub(r"^[\s:.\-–—]+", "", remainder)

    return remainder


def remove_section_artifacts(
    text,
    section_number=None,
    section_title=None
):
    """
    Robustly remove section-heading artifacts from extracted
    text, independent of whether the detected section metadata
    matches the literal header text character-for-character.

    Handles, in any combination:

        Section: 1.4
        Section title: Pharmacological treatment...

        Section: 1.4 Section title: Pharmacological treatment...

        1.4 Pharmacological treatment...

    The section information itself is preserved separately in
    document metadata -- only the redundant in-text header is
    stripped here.
    """

    if not text:
        return ""

    text = normalize_text(text)

    if not text:
        return ""

    number = (
        str(section_number).strip()
        if section_number
        else None
    )

    title = (
        str(section_title).strip()
        if section_title
        else None
    )

    # --------------------------------------------------------
    # Step 1: strip a leading "Section: <number>" label.
    #
    # First try to match the *known* number exactly; if that
    # doesn't hit (e.g. stale/mismatched section tracking),
    # fall back to stripping ANY "Section: N.N" label so we
    # don't leave the raw label behind.
    # --------------------------------------------------------

    if number:

        specific_pattern = (
            rf"^\s*Section\s*:\s*{re.escape(number)}\s*"
        )

        stripped = re.sub(
            specific_pattern,
            "",
            text,
            count=1,
            flags=re.IGNORECASE
        )

        if stripped != text:
            text = stripped
        else:
            text = re.sub(
                r"^\s*Section\s*:\s*\d+(?:\.\d+)+\s*",
                "",
                text,
                count=1,
                flags=re.IGNORECASE
            )
    else:
        text = re.sub(
            r"^\s*Section\s*:\s*\d+(?:\.\d+)+\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE
        )

    # --------------------------------------------------------
    # Step 2: strip a leading "Section title:" label (the title
    # text itself, if any, is handled separately in Step 4 so a
    # mismatch here doesn't block label removal).
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*Section\s+title\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE
    )

    # --------------------------------------------------------
    # Step 3: strip a bare leading heading number with no label,
    # e.g. "1.10 Management if fertility is a priority ..."
    # --------------------------------------------------------

    if number:
        text = re.sub(
            rf"^\s*{re.escape(number)}\s+",
            "",
            text,
            count=1
        )

    # --------------------------------------------------------
    # Step 4: strip the actual title text using fuzzy word
    # matching, wherever it landed after steps 1-3.
    # --------------------------------------------------------

    if title:
        text = _strip_known_title_prefix(text, title)

    text = normalize_text(text)

    return text.strip()

# ============================================================
# REMOVE SECTION INFORMATION GENERICALLY (SAFETY NET)
# ============================================================

def remove_any_section_labels(text):
    """
    Final safety layer for any leftover artificial section
    labels not tied to a known section_number/section_title
    (e.g. because section detection failed for that page).
    """

    if not text:
        return ""

    # Remove:
    #
    # Section: 1.4
    #
    text = re.sub(
        r"^\s*Section\s*:\s*\d+(?:\.\d+)+\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove:
    #
    # Section title: XXXXX
    #
    # Bounded so it can never eat the whole rest of the page:
    # stops at the first newline, or the first ". <Capital>"
    # sentence boundary, or after at most 200 characters.
    #
    text = re.sub(
        r"^\s*Section\s+title\s*:\s*.{1,200}?(?=\n|\.\s+[A-Z]|$)",
        "",
        text,
        count=1,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Remove standalone main section heading:
    #
    # 1.4 Pharmacological treatment...
    #
    text = re.sub(
        r"^\s*\d+\.\d+\s+[^\n]+(?=\n|$)",
        "",
        text,
        flags=re.IGNORECASE
    )

    return normalize_text(
        text
    ).strip()


# ============================================================
# RECOMMENDATION DETECTION
# ============================================================

def detect_recommendation(text):
    """
    Detect NICE recommendation numbers.

    Examples:

        1.5.1
        1.5.10
        1.10.4
    """

    if not text:
        return None

    match = re.search(
        r"(?<!\d)(\d+\.\d+\.\d+)(?!\d)",
        text
    )

    if match:

        return match.group(
            1
        )

    return None


# ============================================================
# EXTRACT ONE PDF
# ============================================================

def extract_pdf(pdf_path):
    """
    Extract one PDF page-by-page.

    Section information is stored as metadata but is removed
    from the actual text field.
    """

    documents = []

    filename = pdf_path.name

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = PDF_METADATA.get(
        filename
    )

    if metadata is None:

        print(
            f"⚠ Warning: No metadata found "
            f"for {filename}"
        )

        metadata = {

            "guideline": None,

            "title": None,

            "publisher":
                "National Institute of Health "
                "and Care Excellence (NICE)"
        }

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    pdf = pymupdf.open(
        pdf_path
    )

    # --------------------------------------------------------
    # Current section
    # --------------------------------------------------------

    current_section = None

    # --------------------------------------------------------
    # Process pages
    # --------------------------------------------------------

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        print(
            f"  Processing page "
            f"{page_number}/{len(pdf)}",
            end="\r"
        )

        # ====================================================
        # DETECT SECTION FROM PDF STRUCTURE
        # ====================================================

        new_section = (
            detect_section_from_page(
                page,
                current_section
            )
        )

        if new_section is not None:

            current_section = (
                new_section
            )

        # ====================================================
        # EXTRACT RAW TEXT
        # ====================================================

        raw_text = page.get_text(
            "text",
            sort=True
        )

        if not raw_text:
            continue

        # ====================================================
        # CLEAN RAW TEXT
        # ====================================================

        text = clean_text(
            raw_text
        )

        if not text:
            continue

        # ====================================================
        # REMOVE SECTION HEADING
        # ====================================================

        if current_section:

            text = remove_section_artifacts(

                text,

                section_number=
                    current_section.get(
                        "section_number"
                    ),

                section_title=
                    current_section.get(
                        "section_title"
                    )
            )

        # ====================================================
        # FINAL SAFETY CLEANUP
        # ====================================================

        text = remove_any_section_labels(
            text
        )

        if not text:
            continue

        # ====================================================
        # DETECT RECOMMENDATION
        # ====================================================

        recommendation_number = (
            detect_recommendation(
                text
            )
        )

        # ====================================================
        # CREATE DOCUMENT
        # ====================================================

        document = {

            # ------------------------------------------------
            # Document metadata
            # ------------------------------------------------

            "guideline":
                metadata.get(
                    "guideline"
                ),

            "title":
                metadata.get(
                    "title"
                ),

            "publisher":
                metadata.get(
                    "publisher"
                ),

            "source":
                filename,

            "page":
                page_number,

            # ------------------------------------------------
            # Section metadata
            # ------------------------------------------------

            "section_number":
                (
                    current_section[
                        "section_number"
                    ]

                    if current_section

                    else "Not Found"
                ),

            "section_title":
                (
                    current_section[
                        "section_title"
                    ]

                    if current_section

                    else "Not Found"
                ),

            "section":
                (
                    current_section[
                        "section"
                    ]

                    if current_section

                    else "Not Found"
                ),

            # ------------------------------------------------
            # Recommendation metadata
            # ------------------------------------------------

            "recommendation_number":
                recommendation_number,

            # ------------------------------------------------
            # CLEAN TEXT
            # ------------------------------------------------

            "text":
                text
        }

        documents.append(
            document
        )

    print()

    pdf.close()

    return documents


# ============================================================
# PROCESS ALL PDF FILES
# ============================================================

def process_all_pdfs():

    pdf_files = list(
        Path(PDF_FOLDER).glob(
            "*.pdf"
        )
    )

    print(
        f"Found {len(pdf_files)} PDF files."
    )

    if not pdf_files:

        print(
            f"⚠ No PDF files found in "
            f"{PDF_FOLDER}"
        )

        return []

    all_documents = []

    for pdf_path in pdf_files:

        print(
            "\n" + "=" * 70
        )

        print(
            f"Processing: "
            f"{pdf_path.name}"
        )

        print(
            "=" * 70
        )

        try:

            documents = extract_pdf(
                pdf_path
            )

            all_documents.extend(
                documents
            )

            print(
                f"✓ Extracted "
                f"{len(documents)} pages"
            )

        except Exception as e:

            print(
                f"✗ Failed to process "
                f"{pdf_path.name}: {e}"
            )

    return all_documents


# ============================================================
# VALIDATE SECTION ARTIFACTS
# ============================================================

def validate_documents(documents):
    """
    Check whether section labels are still appearing
    in the actual text field.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "VALIDATING CLEAN TEXT"
    )

    print(
        "=" * 70
    )

    problematic = []

    patterns = [

        r"Section\s*:",

        r"Section\s+title\s*:",

        r"^\s*\d+\.\d+\s+[A-Z]"
    ]

    for index, document in enumerate(
        documents
    ):

        text = document.get(
            "text",
            ""
        )

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE
            ):

                problematic.append(
                    (
                        index,
                        document.get(
                            "source"
                        ),
                        document.get(
                            "page"
                        ),
                        text[:200]
                    )
                )

                break

    if problematic:

        print(
            f"⚠ Found "
            f"{len(problematic)} "
            f"documents that may still contain "
            f"section artifacts."
        )

        for item in problematic[:10]:

            print(
                "\nProblem:"
            )

            print(
                f"Index: {item[0]}"
            )

            print(
                f"Source: {item[1]}"
            )

            print(
                f"Page: {item[2]}"
            )

            print(
                f"Text preview: {item[3]}"
            )

    else:

        print(
            "✓ No section artifacts detected "
            "in text fields."
        )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(documents):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved {len(documents)} pages to:"
        f"\n{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    documents = process_all_pdfs()

    # --------------------------------------------------------
    # Validate before saving
    # --------------------------------------------------------

    validate_documents(
        documents
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_json(
        documents
    )
