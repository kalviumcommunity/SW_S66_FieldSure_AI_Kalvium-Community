"""Text extraction and semantic cleaning pipeline for FieldSure AI (Module #11).

Cleans raw extracted document text across the corpus:
- Normalizes Unicode (NFKC) and character artifacts
- Removes boilerplate, repeated headers, footers, and page numbers
- Repairs broken hyphenated words across line wraps
- Normalizes whitespace and line breaks while preserving clinical structure
- Generates before/after diff audit metrics
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

from src.ingestion.loader import LoadedDocument, load_corpus

logger = logging.getLogger("fieldsure.cleaner")

OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"
LOG_FILE_PATH = OUTPUTS_DIR / "text_cleaning_sample.log"


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass
class CleaningReport:
    """Audit report comparing raw vs cleaned document text."""

    doc_id: str
    source_file: str
    original_char_count: int
    cleaned_char_count: int
    original_word_count: int
    cleaned_word_count: int
    reduction_chars: int
    reduction_percentage: float
    transformations_applied: List[str]
    before_snippet: str
    after_snippet: str


# ------------------------------------------------------------------------------
# Cleaning Pipeline Steps
# ------------------------------------------------------------------------------

def normalize_unicode(text: str) -> Tuple[str, bool]:
    """Normalize text using Unicode NFKC and clean special whitespace characters."""
    if not text:
        return "", False

    # 1. NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Clean zero-width, non-breaking, and odd whitespace
    cleaned = (
        normalized.replace("\u200b", "")      # zero-width space
        .replace("\ufeff", "")                # BOM
        .replace("\u00a0", " ")               # non-breaking space
        .replace("\u202f", " ")               # narrow no-break space
        .replace("\u2018", "'")               # smart single quotes
        .replace("\u2019", "'")
        .replace("\u201c", '"')               # smart double quotes
        .replace("\u201d", '"')
        .replace("\u2013", "-")               # en-dash
        .replace("\u2014", " - ")             # em-dash
    )

    changed = cleaned != text
    return cleaned, changed


def fix_line_wrap_hyphenation(text: str) -> Tuple[str, bool]:
    """Rejoin words split across line breaks by hyphens (e.g. 'de-\\nhydration' -> 'dehydration')."""
    # Match letters, a hyphen, newline, optional whitespace, and continuing letters
    pattern = r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)"
    cleaned, count = re.subn(pattern, r"\1\2", text)
    return cleaned, (count > 0)


def remove_boilerplate_patterns(text: str) -> Tuple[str, List[str]]:
    """Strip repeated page numbering, document footers, and navigation text."""
    removed_categories = []

    # 1. "Page X of Y" or "Page X"
    p_page = r"(?i)^\s*(?:---\s*)?Page\s+\d+(\s+of\s+\d+)?(?:\s*---)?\s*$"
    cleaned, count_page = re.subn(p_page, "", text, flags=re.MULTILINE)
    if count_page > 0:
        removed_categories.append(f"page_numbers_removed({count_page})")

    # 2. Standalone page number lines like " - 12 - " or "[12]"
    p_num = r"^\s*(?:-\s*\d+\s*-|\[\d+\])\s*$"
    cleaned, count_num = re.subn(p_num, "", cleaned, flags=re.MULTILINE)
    if count_num > 0:
        removed_categories.append(f"isolated_numbers_removed({count_num})")

    # 3. Common document footer boilerplate
    p_boilerplate = r"(?i)^\s*(?:Confidential\s*-\s*Internal\s+Use\s+Only|All\s+[Rr]ights\s+[Rr]eserved\.?|Draft\s+for\s+Review\s+Only)\s*$"
    cleaned, count_bp = re.subn(p_boilerplate, "", cleaned, flags=re.MULTILINE)
    if count_bp > 0:
        removed_categories.append(f"boilerplate_headers_removed({count_bp})")

    return cleaned, removed_categories


def strip_html_tags(text: str) -> Tuple[str, bool]:
    """Strip HTML markup while converting structural elements to newlines."""
    if "<" not in text or ">" not in text:
        return text, False

    # Convert block tags to line breaks to preserve separation
    converted = re.sub(r"(?i)<br\s*/?>", "\n", text)
    converted = re.sub(r"(?i)</?(?:p|div|h[1-6]|li|tr|blockquote)[^>]*>", "\n", converted)

    # Strip remaining tags
    cleaned = re.sub(r"<[^>]+>", " ", converted)
    # Unescape common HTML entities
    cleaned = (
        cleaned.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&ge;", ">=")
        .replace("&le;", "<=")
        .replace("&deg;", "°")
    )
    return cleaned, True


def normalize_whitespace_and_layout(text: str) -> Tuple[str, bool]:
    """Normalize line endings, horizontal spaces, and consecutive blank lines.

    Preserves markdown headers ('#', '##'), list bullets ('-', '*', '1.'), and table pipes ('|').
    """
    # 1. Normalize line endings
    step1 = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Clean each line: trim runaway horizontal spaces but keep list/header indent
    lines = []
    for line in step1.split("\n"):
        # Collapse multiple spaces and tabs into single space
        collapsed_line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(collapsed_line)

    recombined = "\n".join(lines)

    # 3. Collapse 3 or more consecutive newlines into 2 (clean paragraph break)
    step3 = re.sub(r"\n{3,}", "\n\n", recombined).strip()

    changed = step3 != text
    return step3, changed


# ------------------------------------------------------------------------------
# Core Cleaning Pipeline API
# ------------------------------------------------------------------------------

def clean_text(text: str, is_html: bool = False) -> Tuple[str, List[str]]:
    """Execute the end-to-end cleaning pipeline on a raw text string.

    Args:
        text: Raw document text.
        is_html: Whether text contains HTML markup requiring tag stripping.

    Returns:
        Tuple of (cleaned_text, list_of_transformations_applied).
    """
    if not text or not text.strip():
        return "", ["empty_input"]

    applied: List[str] = []

    # Step 1: Unicode Normalization
    current, changed = normalize_unicode(text)
    if changed:
        applied.append("unicode_nfkc_normalized")

    # Step 2: HTML tag stripping if applicable
    if is_html or ("<html" in text.lower() or "<body" in text.lower()):
        current, changed_html = strip_html_tags(current)
        if changed_html:
            applied.append("html_tags_stripped")

    # Step 3: Repair broken hyphenated words
    current, changed_hyphen = fix_line_wrap_hyphenation(current)
    if changed_hyphen:
        applied.append("line_wrap_hyphens_repaired")

    # Step 4: Remove boilerplate & page numbers
    current, bp_categories = remove_boilerplate_patterns(current)
    applied.extend(bp_categories)

    # Step 5: Normalize whitespace and collapse excessive line breaks
    current, changed_ws = normalize_whitespace_and_layout(current)
    if changed_ws:
        applied.append("whitespace_collapsed")

    return current, applied


def clean_document(doc: LoadedDocument) -> Tuple[LoadedDocument, CleaningReport]:
    """Clean a single LoadedDocument and generate an audit report."""
    orig_chars = doc.char_count
    orig_words = doc.word_count
    orig_snippet = doc.text[:120].replace("\n", " ") + ("..." if orig_chars > 120 else "")

    is_html = doc.file_type in {"html", "htm"}
    cleaned_str, applied = clean_text(doc.text, is_html=is_html)

    new_chars = len(cleaned_str)
    new_words = len(cleaned_str.split())
    diff_chars = max(0, orig_chars - new_chars)
    reduction_pct = round((diff_chars / orig_chars * 100), 2) if orig_chars else 0.0

    cleaned_snippet = cleaned_str[:120].replace("\n", " ") + ("..." if new_chars > 120 else "")

    # Create updated LoadedDocument container
    cleaned_doc = LoadedDocument(
        doc_id=doc.doc_id,
        source_file=doc.source_file,
        file_path=doc.file_path,
        file_type=doc.file_type,
        text=cleaned_str,
        char_count=new_chars,
        word_count=new_words,
        page_count=doc.page_count,
        metadata={
            **doc.metadata,
            "cleaned": True,
            "cleaning_timestamp": datetime.now(timezone.utc).isoformat(),
            "character_reduction": diff_chars,
            "reduction_percentage": reduction_pct,
            "transformations": applied,
        },
    )

    report = CleaningReport(
        doc_id=doc.doc_id,
        source_file=doc.source_file,
        original_char_count=orig_chars,
        cleaned_char_count=new_chars,
        original_word_count=orig_words,
        cleaned_word_count=new_words,
        reduction_chars=diff_chars,
        reduction_percentage=reduction_pct,
        transformations_applied=applied,
        before_snippet=orig_snippet,
        after_snippet=cleaned_snippet,
    )

    return cleaned_doc, report


def clean_corpus(docs: List[LoadedDocument]) -> Tuple[List[LoadedDocument], List[CleaningReport]]:
    """Uniformly apply semantic cleaning across an entire document corpus (Task 3).

    Args:
        docs: List of raw LoadedDocument objects.

    Returns:
        Tuple of (cleaned_documents, cleaning_reports).
    """
    cleaned_docs: List[LoadedDocument] = []
    reports: List[CleaningReport] = []

    for d in docs:
        cd, rep = clean_document(d)
        cleaned_docs.append(cd)
        reports.append(rep)
        logger.info(
            "Cleaned %s: %d -> %d chars (%.2f%% reduction)",
            d.source_file,
            rep.original_char_count,
            rep.cleaned_char_count,
            rep.reduction_percentage,
        )

    return cleaned_docs, reports


# ------------------------------------------------------------------------------
# CLI Runner & Demonstration (Tasks 4 & 5)
# ------------------------------------------------------------------------------

def run_cleaning_pipeline(corpus_dir: Optional[Path] = None) -> int:
    """Execute cleaning across corpus and print detailed before/after comparisons."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    sample_dir = corpus_dir or (repo_root / "data" / "sample_advisories")
    outputs_dir = repo_root / "outputs"
    log_file = LOG_FILE_PATH

    print("\n" + "=" * 80)
    print("FIELDSURE AI — TEXT EXTRACTION & CLEANING PIPELINE")
    print(f"Target Corpus: {sample_dir}")
    print("=" * 80)

    # 1. Ingest raw corpus
    raw_docs, load_errors = load_corpus(sample_dir, recursive=True, skip_bad_files=True)
    if not raw_docs:
        print("\n[ERROR] No valid documents found in corpus directory.", file=sys.stderr)
        return 1

    # 2. Clean corpus uniformly (Task 3)
    cleaned_docs, reports = clean_corpus(raw_docs)

    print(f"\nProcessing Summary: {len(cleaned_docs)} documents cleaned uniformly.\n")

    # 3. Print Before/After Comparison Samples (Task 4)
    audit_reports_json = []

    for idx, rep in enumerate(reports, 1):
        print(f"► Document [{idx}]: {rep.source_file} (ID: {rep.doc_id})")
        print(f"  Transformations:    {', '.join(rep.transformations_applied) if rep.transformations_applied else 'None'}")
        print(f"  Length Reduction:   {rep.original_char_count:,} chars -> {rep.cleaned_char_count:,} chars (-{rep.reduction_percentage}%)")
        print(f"  Word Count:         {rep.original_word_count:,} words -> {rep.cleaned_word_count:,} words")
        print("  ┌─ BEFORE (Raw Extracted Sample) " + "─" * 47)
        print(f"  │ \"{rep.before_snippet}\"")
        print("  ├─ AFTER (Cleaned Normalized Sample) " + "─" * 43)
        print(f"  │ \"{rep.after_snippet}\"")
        print("  └" + "─" * 78)
        print("-" * 80)

        audit_reports_json.append({
            "source_file": rep.source_file,
            "doc_id": rep.doc_id,
            "original_char_count": rep.original_char_count,
            "cleaned_char_count": rep.cleaned_char_count,
            "original_word_count": rep.original_word_count,
            "cleaned_word_count": rep.cleaned_word_count,
            "reduction_chars": rep.reduction_chars,
            "reduction_percentage": rep.reduction_percentage,
            "transformations_applied": rep.transformations_applied,
            "before_snippet": rep.before_snippet,
            "after_snippet": rep.after_snippet,
        })

    # 4. Save Structured Audit Log (Task 5)
    audit_data = {
        "module": "Module #11 — Text Extraction & Cleaning Pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_documents_cleaned": len(cleaned_docs),
        "corpus_reports": audit_reports_json,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n--- Structured Text Cleaning Audit ---\n")
        f.write(json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n")

    print(f"\nCleaning pipeline sample audit successfully saved to: {log_file}\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_cleaning_pipeline())
