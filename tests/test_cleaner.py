"""Unit and regression tests for the text extraction and cleaning pipeline (Module #11)."""

from pathlib import Path
from unittest.mock import patch
import pytest

from src.ingestion.cleaner import (
    clean_corpus,
    clean_document,
    clean_text,
    fix_line_wrap_hyphenation,
    normalize_unicode,
    normalize_whitespace_and_layout,
    remove_boilerplate_patterns,
    run_cleaning_pipeline,
    strip_html_tags,
)
from src.ingestion.loader import LoadedDocument


def test_normalize_unicode_artifacts():
    """Verify Unicode NFKC normalization and cleanup of special whitespace and smart quotes."""
    raw = "“Dengue fever” – severe cases\u00a0require immediate\u200b attention."
    cleaned, changed = normalize_unicode(raw)

    assert changed is True
    assert '"Dengue fever"' in cleaned
    assert "\u00a0" not in cleaned
    assert "\u200b" not in cleaned


def test_fix_line_wrap_hyphenation():
    """Verify repair of broken words split across lines by soft/hard hyphens (Task 2)."""
    raw = (
        "Early signs of severe de-\n  hydration include sunken eyes.\n"
        "Oral rehydration salts were adminis-\ntered promptly."
    )
    cleaned, changed = fix_line_wrap_hyphenation(raw)

    assert changed is True
    assert "dehydration" in cleaned
    assert "administered" in cleaned
    assert "de-\n" not in cleaned
    assert "adminis-\n" not in cleaned


def test_remove_boilerplate_headers_footers():
    """Verify removal of page numbers, disclaimers, and repetitive footers (Task 1)."""
    raw = (
        "# Clinical Protocol Header\n\n"
        "Page 1 of 12\n\n"
        "Assess patient hydration status immediately.\n\n"
        "Confidential - Internal Use Only\n"
        "--- Page 2 of 12 ---\n\n"
        "Administer zinc tablets daily.\n\n"
        "All Rights Reserved.\n"
    )
    cleaned, removed = remove_boilerplate_patterns(raw)

    assert "Page 1 of 12" not in cleaned
    assert "--- Page 2 of 12 ---" not in cleaned
    assert "Confidential - Internal Use Only" not in cleaned
    assert "All Rights Reserved." not in cleaned
    assert "Assess patient hydration status immediately." in cleaned
    assert "Administer zinc tablets daily." in cleaned
    assert len(removed) >= 2


def test_strip_html_markup():
    """Verify HTML tags are stripped while retaining line breaks and entity values."""
    raw_html = (
        "<h3>Malaria Triage</h3>\n"
        "<p>Patient temperature &ge; 38.5&deg;C requires mRDT testing.</p>\n"
        "<ul><li>Step 1</li><li>Step 2</li></ul>"
    )
    cleaned, changed = strip_html_tags(raw_html)

    assert changed is True
    assert "<p>" not in cleaned
    assert "<ul>" not in cleaned
    assert "Malaria Triage" in cleaned
    assert ">= 38.5°C" in cleaned
    assert "Step 1" in cleaned


def test_normalize_whitespace_and_blank_lines():
    """Verify runaway horizontal spaces and excessive blank lines are collapsed."""
    raw = "Clinical    assessment    step.\n\n\n\n\nFollow-up   in  4  hours."
    cleaned, changed = normalize_whitespace_and_layout(raw)

    assert changed is True
    assert "    " not in cleaned
    assert "Clinical assessment step." in cleaned
    assert "\n\n\n" not in cleaned
    assert "Follow-up in 4 hours." in cleaned


def test_clean_document_end_to_end():
    """Verify clean_document produces updated LoadedDocument and audit report (Task 4)."""
    raw_text = (
        "Page 1 of 5\n\n"
        "“Cholera” out-\nbreak protocol.\n\n\n\n"
        "Confidential - Internal Use Only\n\n"
        "Wash   hands    thoroughly."
    )
    doc = LoadedDocument(
        doc_id="test_cholera",
        source_file="cholera.txt",
        file_path="/data/cholera.txt",
        file_type="txt",
        text=raw_text,
        char_count=len(raw_text),
        word_count=len(raw_text.split()),
    )

    cleaned_doc, report = clean_document(doc)

    assert cleaned_doc.char_count < doc.char_count
    assert report.reduction_chars > 0
    assert report.reduction_percentage > 0.0
    assert "outbreak" in cleaned_doc.text
    assert "Page 1 of 5" not in cleaned_doc.text
    assert "Wash hands thoroughly." in cleaned_doc.text
    assert cleaned_doc.metadata["cleaned"] is True


def test_clean_corpus_uniformity():
    """Verify all documents in corpus are cleaned uniformly with matching reports (Task 3)."""
    doc1 = LoadedDocument(
        doc_id="d1", source_file="d1.txt", file_path="d1.txt", file_type="txt",
        text="Doc  1   text  with   spaces.", char_count=30, word_count=5,
    )
    doc2 = LoadedDocument(
        doc_id="d2", source_file="d2.md", file_path="d2.md", file_type="md",
        text="Page 1 of 2\n# Doc 2 heading.", char_count=30, word_count=5,
    )

    cleaned_list, reports = clean_corpus([doc1, doc2])

    assert len(cleaned_list) == 2
    assert len(reports) == 2
    assert cleaned_list[0].text == "Doc 1 text with spaces."
    assert "Page 1 of 2" not in cleaned_list[1].text


def test_run_cleaning_pipeline_runner(tmp_path):
    """Verify run_cleaning_pipeline executes and saves before/after sample logs (Task 5)."""
    test_log = tmp_path / "test_cleaning_sample.log"

    with patch("src.ingestion.cleaner.LOG_FILE_PATH", test_log):
        exit_code = run_cleaning_pipeline()
        assert exit_code == 0
        assert test_log.is_file()

        log_content = test_log.read_text(encoding="utf-8")
        assert "Module #11 — Text Extraction & Cleaning Pipeline" in log_content
        assert "corpus_reports" in log_content
