"""Unit and integration tests for multi-format document loading and corpus intake (Module #10)."""

from pathlib import Path
import pytest

from src.ingestion.loader import (
    CorruptDocumentError,
    DocumentNotFoundError,
    LoadedDocument,
    UnsupportedFileError,
    load_corpus,
    load_document,
    run_intake_confirmation,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_advisories"


def test_load_markdown_document():
    """Verify loading Markdown document preserves text, character counts, and source."""
    md_file = SAMPLE_DIR / "who_dengue_clinical_guidance.md"
    assert md_file.is_file(), f"Missing test fixture: {md_file}"

    doc = load_document(md_file)

    assert isinstance(doc, LoadedDocument)
    assert doc.source_file == "who_dengue_clinical_guidance.md"
    assert doc.file_type == "md"
    assert doc.char_count > 0
    assert doc.word_count > 0
    assert "Dengue" in doc.text
    assert "Febrile Phase" in doc.text


def test_load_plain_text_document():
    """Verify loading plain text document extracts all lines and metadata."""
    txt_file = SAMPLE_DIR / "cdc_cholera_water_safety.txt"
    assert txt_file.is_file(), f"Missing test fixture: {txt_file}"

    doc = load_document(txt_file)

    assert doc.source_file == "cdc_cholera_water_safety.txt"
    assert doc.file_type == "txt"
    assert "CHOLERA OUTBREAK" in doc.text
    assert doc.char_count > 100


def test_load_html_document():
    """Verify loading HTML document into common text representation."""
    html_file = SAMPLE_DIR / "moh_malaria_rapid_triage.html"
    assert html_file.is_file(), f"Missing test fixture: {html_file}"

    doc = load_document(html_file)

    assert doc.source_file == "moh_malaria_rapid_triage.html"
    assert doc.file_type == "html"
    assert "Malaria Rapid Diagnostic Testing" in doc.text
    assert doc.char_count > 100


def test_source_identity_preserved():
    """Verify source provenance is immutable and retained for citations (Task 3)."""
    md_file = SAMPLE_DIR / "who_dengue_clinical_guidance.md"
    doc = load_document(md_file)

    assert doc.doc_id == "who_dengue_clinical_guidance"
    assert doc.source_file == "who_dengue_clinical_guidance.md"
    assert str(md_file.resolve()) == doc.file_path
    assert "file_size_bytes" in doc.metadata
    assert doc.metadata["file_size_bytes"] > 0


def test_missing_file_raises_error():
    """Verify non-existent file path raises DocumentNotFoundError (Task 2)."""
    fake_path = SAMPLE_DIR / "non_existent_file_12345.pdf"
    with pytest.raises(DocumentNotFoundError, match="not found"):
        load_document(fake_path)


def test_unsupported_format_raises_error():
    """Verify unsupported file format raises UnsupportedFileError (Task 2)."""
    bad_file = SAMPLE_DIR / "unsupported_format.xyz"
    assert bad_file.is_file()

    with pytest.raises(UnsupportedFileError, match="Unsupported file format"):
        load_document(bad_file)


def test_load_corpus_gracefully_survives_bad_files(tmp_path):
    """Verify load_corpus skips corrupt/unsupported files without crashing (Task 2)."""
    # Create temporary mixed corpus: 2 valid files, 1 unsupported file
    (tmp_path / "valid1.txt").write_text("Valid text content 1", encoding="utf-8")
    (tmp_path / "valid2.md").write_text("# Valid Markdown 2", encoding="utf-8")
    (tmp_path / "corrupt.badext").write_text("Unknown binary blob", encoding="utf-8")

    docs, errors = load_corpus(tmp_path, recursive=False, skip_bad_files=True)

    # Valid files must be loaded
    assert len(docs) == 2
    loaded_names = {d.source_file for d in docs}
    assert "valid1.txt" in loaded_names
    assert "valid2.md" in loaded_names

    # Bad files must be caught in error ledger
    assert len(errors) == 1
    assert errors[0]["file"] == "corrupt.badext"
    assert errors[0]["error_type"] == "UnsupportedFileError"


def test_intake_confirmation_runner_execution():
    """Verify intake confirmation runner executes successfully (Task 4 & 5)."""
    exit_code = run_intake_confirmation(SAMPLE_DIR)
    assert exit_code == 0
