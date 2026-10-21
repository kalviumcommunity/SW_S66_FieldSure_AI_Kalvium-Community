"""Unit and end-to-end integration tests for Corpus Ingestion and Validation Pipeline (Module #15)."""

import json
from pathlib import Path
import pytest

from src.ingestion.pipeline import (
    CompletenessAudit,
    IngestionPipelineReport,
    discover_corpus_files,
    run_ingestion_pipeline,
)


@pytest.fixture
def corpus_directory() -> Path:
    """Path to verified sample public health corpus."""
    root = Path(__file__).resolve().parent.parent
    return root / "data" / "sample_advisories"


def test_discover_corpus_files(corpus_directory: Path):
    """Verify discovery finds all candidate files while ignoring hidden/readme files."""
    files = discover_corpus_files(corpus_directory)
    assert len(files) >= 4
    file_names = {f.name for f in files}
    assert "who_dengue_clinical_guidance.md" in file_names
    assert "cdc_cholera_water_safety.txt" in file_names
    assert "moh_malaria_rapid_triage.html" in file_names
    assert "unsupported_format.xyz" in file_names
    assert "README.md" not in file_names


def test_run_ingestion_pipeline_end_to_end(tmp_path: Path, corpus_directory: Path):
    """Task 1 & 2: Test full end-to-end pipeline execution and summary reporting."""
    output_jsonl = tmp_path / "test_ingested.jsonl"
    report = run_ingestion_pipeline(
        input_dir=corpus_directory,
        output_jsonl=output_jsonl,
        target_token_budget=500,
        overlap_tokens=50,
    )

    assert isinstance(report, IngestionPipelineReport)
    audit = report.audit
    assert audit.total_source_files_discovered >= 4
    assert audit.documents_successfully_ingested >= 3
    assert audit.documents_failed_or_skipped >= 1
    assert audit.total_chunks_created > 0
    assert audit.zero_empty_chunks is True
    assert audit.token_budget_compliant is True
    assert audit.metadata_contract_compliant is True


def test_reconciliation_audit_completeness(tmp_path: Path, corpus_directory: Path):
    """Task 3: Prove mathematical completeness reconciliation (no document silently dropped)."""
    output_jsonl = tmp_path / "test_reconciliation.jsonl"
    report = run_ingestion_pipeline(input_dir=corpus_directory, output_jsonl=output_jsonl)
    audit = report.audit

    # Equation: Discovered == Ingested + Failures
    assert audit.reconciliation_passed is True
    assert audit.total_source_files_discovered == (
        audit.documents_successfully_ingested + audit.documents_failed_or_skipped
    )
    assert len(audit.issues) == 0

    # Verify failure ledger contains unsupported_format.xyz
    failed_names = {f.file_name for f in report.failed_files}
    assert "unsupported_format.xyz" in failed_names


def test_jsonl_export_validity(tmp_path: Path, corpus_directory: Path):
    """Task 4 & 5: Verify exported JSONL matches schema contract and contains valid chunks."""
    output_jsonl = tmp_path / "test_export.jsonl"
    report = run_ingestion_pipeline(input_dir=corpus_directory, output_jsonl=output_jsonl)

    assert output_jsonl.exists()
    lines = output_jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == report.audit.total_chunks_created

    expected_metadata_fields = {
        "chunk_id",
        "doc_id",
        "source_file",
        "document_title",
        "document_version",
        "agency",
        "publication_date",
        "section_heading",
        "chunk_index",
        "total_chunks",
        "char_length",
        "token_count",
        "page_number",
        "extra_metadata",
    }

    # Inspect each chunk in JSONL
    for line in lines:
        chunk_obj = json.loads(line)
        assert "chunk_id" in chunk_obj
        assert "text" in chunk_obj
        assert "metadata" in chunk_obj
        assert len(chunk_obj["text"].strip()) > 0

        meta = chunk_obj["metadata"]
        assert set(meta.keys()) == expected_metadata_fields
        assert meta["token_count"] <= 550  # Allows tag margin
        assert 0 <= meta["chunk_index"] < meta["total_chunks"]
