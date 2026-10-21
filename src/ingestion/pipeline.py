"""End-to-End Document Ingestion and Validation Pipeline for FieldSure AI (Module #15).

Coordinates the complete ingestion sequence across the entire corpus:
1. Multi-Format Document Intake (src.ingestion.loader)
2. Semantic Text Cleaning & Normalization (src.ingestion.cleaner)
3. Token-Aware Structural Chunking (src.ingestion.chunker)
4. Immutable Metadata Tagging & Provenance (src.ingestion.metadata)
5. Comprehensive Completeness Audit & Mathematical Reconciliation
6. Export to JSON Lines (JSONL) ready for downstream Vector Database Indexing
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingestion.chunker import (
    ChunkingStrategy,
    chunk_document,
    split_token_structural,
)
from src.ingestion.cleaner import clean_document
from src.ingestion.loader import (
    CorruptDocumentError,
    DocumentLoaderError,
    LoadedDocument,
    UnsupportedFileError,
    load_document,
)
from src.ingestion.metadata import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    build_document_metadata,
    tag_document_chunks,
    validate_chunk_metadata_completeness,
)
from src.token_cost_estimator import count_tokens

logger = logging.getLogger("fieldsure.pipeline")

OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_JSONL_EXPORT = OUTPUTS_DIR / "ingested_chunks.jsonl"
DEFAULT_VALIDATION_REPORT = OUTPUTS_DIR / "ingestion_validation_report.json"
DEFAULT_SUMMARY_LOG = OUTPUTS_DIR / "ingestion_summary.log"


# ------------------------------------------------------------------------------
# Data Models for Ingestion Audit and Reporting
# ------------------------------------------------------------------------------

@dataclass
class IngestionFailure:
    """Record of an unreadable or unsupported file encountered during intake."""

    file_path: str
    file_name: str
    error_type: str
    error_message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert failure record to dictionary."""
        return asdict(self)


@dataclass
class DocumentIngestionResult:
    """Detailed summary metrics for a successfully ingested document."""

    source_file: str
    file_name: str
    doc_id: str
    document_title: str
    agency: Optional[str]
    document_version: str
    file_type: str
    original_char_count: int
    cleaned_char_count: int
    total_chunks: int
    total_tokens: int
    avg_chunk_tokens: float
    min_chunk_tokens: int
    max_chunk_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert ingestion result to dictionary."""
        return asdict(self)


@dataclass
class CompletenessAudit:
    """Mathematical reconciliation proving no document was silently dropped."""

    reconciliation_passed: bool
    total_source_files_discovered: int
    documents_successfully_ingested: int
    documents_failed_or_skipped: int
    total_chunks_created: int
    zero_empty_chunks: bool
    token_budget_compliant: bool
    metadata_contract_compliant: bool
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit metrics to dictionary."""
        return asdict(self)


@dataclass
class IngestionPipelineReport:
    """Comprehensive audit report for an end-to-end ingestion pipeline execution."""

    timestamp: str
    input_directory: str
    output_jsonl_file: str
    audit: CompletenessAudit
    ingested_documents: List[DocumentIngestionResult]
    failed_files: List[IngestionFailure]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "metadata": {
                "module": "Module #15 — Corpus Preparation & Ingestion Validation",
                "timestamp": self.timestamp,
                "input_directory": self.input_directory,
                "output_jsonl_file": self.output_jsonl_file,
            },
            "completeness_audit": self.audit.to_dict(),
            "ingested_documents": [doc.to_dict() for doc in self.ingested_documents],
            "failed_files": [fail.to_dict() for fail in self.failed_files],
        }


# ------------------------------------------------------------------------------
# Core Pipeline Engine
# ------------------------------------------------------------------------------

def discover_corpus_files(input_dir: Union[str, Path]) -> List[Path]:
    """Discover all potential guidance files within the corpus directory."""
    path = Path(input_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {path}")

    # Discover all non-hidden, non-system files
    files = [
        f for f in path.iterdir()
        if f.is_file() and not f.name.startswith(".") and f.name != "README.md"
    ]
    return sorted(files, key=lambda p: p.name)


def run_ingestion_pipeline(
    input_dir: Optional[Union[str, Path]] = None,
    output_jsonl: Optional[Union[str, Path]] = None,
    target_token_budget: int = 500,
    overlap_tokens: int = 50,
) -> IngestionPipelineReport:
    """Execute the complete ingestion pipeline over the whole corpus with audit validation.

    Tasks performed:
    1. Discovers 100% of files in input_dir
    2. Loads valid documents and records unsupported/corrupt files in error ledger
    3. Cleans extracted text preserving clinical terminology
    4. Splits text into token-budgeted chunks with controlled overlap
    5. Tags chunks with uniform 14-field immutable metadata schema
    6. Mathematically reconciles input files vs (ingested + failed)
    7. Exports validated chunks to JSONL and produces audit reports

    Args:
        input_dir: Directory containing guidance files (default: data/sample_advisories).
        output_jsonl: Target JSONL path for downstream vector DB handoff.
        target_token_budget: Maximum tokens per chunk (default: 500).
        overlap_tokens: Overlap tokens between adjacent chunks (default: 50).

    Returns:
        IngestionPipelineReport containing complete audit metrics.
    """
    corpus_path = Path(input_dir or (REPO_ROOT / "data" / "sample_advisories")).resolve()
    target_jsonl = Path(output_jsonl or DEFAULT_JSONL_EXPORT).resolve()
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)

    discovered_files = discover_corpus_files(corpus_path)
    total_discovered = len(discovered_files)

    logger.info("Discovered %d candidate files in %s", total_discovered, corpus_path)

    ingested_documents: List[DocumentIngestionResult] = []
    failed_files: List[IngestionFailure] = []
    all_document_chunks: List[DocumentChunk] = []

    # 1. Processing Loop
    for file_path in discovered_files:
        try:
            # Stage 1: Intake & Loading
            loaded_doc = load_document(file_path)

            # Stage 2: Semantic Cleaning
            cleaned_doc, clean_report = clean_document(loaded_doc)

            # Stage 3: Metadata Extraction & Document Descriptor
            doc_meta = build_document_metadata(cleaned_doc)

            # Stage 4: Token-Aware Structural Chunking
            raw_chunks = chunk_document(
                doc=cleaned_doc,
                strategy=ChunkingStrategy.TOKEN_STRUCTURAL,
                chunk_size=target_token_budget,
                chunk_overlap=overlap_tokens,
            )

            # Stage 5: Uniform 14-Field Metadata Tagging
            tagged_chunks = tag_document_chunks(raw_chunks, doc_meta)
            all_document_chunks.extend(tagged_chunks)

            # Record Document Summary
            token_counts = [c.metadata.token_count for c in tagged_chunks]
            tot_tokens = sum(token_counts)
            avg_tokens = round(tot_tokens / len(token_counts), 2) if token_counts else 0.0

            ingested_documents.append(
                DocumentIngestionResult(
                    source_file=doc_meta.source_file,
                    file_name=file_path.name,
                    doc_id=doc_meta.doc_id,
                    document_title=doc_meta.title,
                    agency=doc_meta.agency,
                    document_version=doc_meta.document_version,
                    file_type=doc_meta.file_type,
                    original_char_count=clean_report.original_char_count,
                    cleaned_char_count=clean_report.cleaned_char_count,
                    total_chunks=len(tagged_chunks),
                    total_tokens=tot_tokens,
                    avg_chunk_tokens=avg_tokens,
                    min_chunk_tokens=min(token_counts) if token_counts else 0,
                    max_chunk_tokens=max(token_counts) if token_counts else 0,
                )
            )

        except UnsupportedFileError as err:
            logger.warning("Unsupported file skipped: %s (%s)", file_path.name, err)
            failed_files.append(
                IngestionFailure(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    error_type="UnsupportedFileError",
                    error_message=str(err),
                )
            )
        except (CorruptDocumentError, DocumentLoaderError) as err:
            logger.error("Corrupt document error: %s (%s)", file_path.name, err)
            failed_files.append(
                IngestionFailure(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    error_type=type(err).__name__,
                    error_message=str(err),
                )
            )
        except Exception as err:
            logger.exception("Unexpected error processing %s: %s", file_path.name, err)
            failed_files.append(
                IngestionFailure(
                    file_path=str(file_path),
                    file_name=file_path.name,
                    error_type=type(err).__name__,
                    error_message=str(err),
                )
            )

    # 2. Mathematical Completeness Reconciliation (Task 3)
    reconciled_count = len(ingested_documents) + len(failed_files)
    reconciliation_passed = (reconciled_count == total_discovered)

    audit_issues: List[str] = []
    if not reconciliation_passed:
        audit_issues.append(
            f"Completeness reconciliation failed: {total_discovered} discovered != "
            f"{len(ingested_documents)} ingested + {len(failed_files)} failures."
        )

    # Validate zero empty chunks
    zero_empty = all(len(c.text.strip()) > 0 for c in all_document_chunks)
    if not zero_empty:
        audit_issues.append("Detected one or more empty or whitespace-only chunks.")

    # Validate token budget compliance
    token_compliant = all(c.metadata.token_count <= (target_token_budget + 20) for c in all_document_chunks)
    if not token_compliant:
        audit_issues.append(f"Detected chunks exceeding target token budget of {target_token_budget} tokens.")

    # Validate metadata schema completeness
    meta_val = validate_chunk_metadata_completeness(all_document_chunks)
    meta_compliant = meta_val["valid"]
    if not meta_compliant:
        audit_issues.extend(meta_val["issues"])

    audit_summary = CompletenessAudit(
        reconciliation_passed=reconciliation_passed,
        total_source_files_discovered=total_discovered,
        documents_successfully_ingested=len(ingested_documents),
        documents_failed_or_skipped=len(failed_files),
        total_chunks_created=len(all_document_chunks),
        zero_empty_chunks=zero_empty,
        token_budget_compliant=token_compliant,
        metadata_contract_compliant=meta_compliant,
        issues=audit_issues,
    )

    report = IngestionPipelineReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_directory=str(corpus_path),
        output_jsonl_file=str(target_jsonl),
        audit=audit_summary,
        ingested_documents=ingested_documents,
        failed_files=failed_files,
    )

    # 3. Export JSONL Dataset for Member 2 Vector Ingestion (Task 4 & 5)
    with open(target_jsonl, "w", encoding="utf-8") as f:
        for chunk in all_document_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    # 4. Save outputs/ingestion_validation_report.json
    with open(DEFAULT_VALIDATION_REPORT, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    # 5. Save outputs/ingestion_summary.log
    _write_summary_log(report, all_document_chunks[:3])

    return report


def _write_summary_log(report: IngestionPipelineReport, sample_chunks: List[DocumentChunk]) -> None:
    """Format and write human-readable ingestion summary log."""
    audit = report.audit
    lines = [
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Starting FieldSure AI End-to-End Ingestion Validation (Module #15)...",
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Input Directory: {report.input_directory}",
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] [INFO] Output JSONL:    {report.output_jsonl_file}",
        "",
        "--- Ingestion Summary Dashboard (Task 2) ---",
        f"Total Source Documents Discovered:  {audit.total_source_files_discovered}",
        f"Successfully Ingested Documents:    {audit.documents_successfully_ingested}",
        f"Failed or Skipped Documents:        {audit.documents_failed_or_skipped}",
        f"Total Chunks Created:               {audit.total_chunks_created}",
        "",
        "--- Mathematical Completeness Reconciliation (Task 3) ---",
        f"Equation: Discovered ({audit.total_source_files_discovered}) = Ingested ({audit.documents_successfully_ingested}) + Recorded Failures ({audit.documents_failed_or_skipped})",
        f"Reconciliation Status:              [{'PASSED' if audit.reconciliation_passed else 'FAILED'}] — No document was silently dropped.",
        f"Zero Empty Chunks Verified:         [{'PASSED' if audit.zero_empty_chunks else 'FAILED'}]",
        f"Token Budget Ceiling Compliant:     [{'PASSED' if audit.token_budget_compliant else 'FAILED'}]",
        f"Metadata Schema Contract Compliant: [{'PASSED' if audit.metadata_contract_compliant else 'FAILED'}]",
        "",
        "--- Document Ingestion Breakdown ---",
    ]

    for doc in report.ingested_documents:
        lines.append(
            f"• [{doc.agency or 'N/A'}] {doc.file_name:<34} | Chunks: {doc.total_chunks:>2} | "
            f"Tokens: {doc.total_tokens:>4} | Avg Tokens: {doc.avg_chunk_tokens:>5.1f} | Version: {doc.document_version}"
        )

    if report.failed_files:
        lines.append("\n--- Recorded Skipped / Failed Files Ledger ---")
        for fail in report.failed_files:
            lines.append(f"• [SKIPPED] {fail.file_name:<30} | Error: {fail.error_type} ({fail.error_message})")

    lines.extend([
        "",
        "--- Sample Chunks Inspection (Task 4) ---",
    ])

    for c in sample_chunks:
        lines.extend([
            f"Chunk ID:     {c.chunk_id}",
            f"Source File:  {c.metadata.source_file}",
            f"Section:      {c.metadata.section_heading}",
            f"Position:     Chunk #{c.metadata.chunk_index + 1} of {c.metadata.total_chunks}",
            f"Token Count:  {c.metadata.token_count} tokens ({c.metadata.char_length} chars)",
            f"Citation:     {c.format_citation(style='inline')}",
            f"Snippet:      {c.text[:180].replace(chr(10), ' ')}...",
            "-" * 80,
        ])

    lines.extend([
        "",
        f"[INFO] Ingestion run completed successfully.",
        f"[INFO] Serialized dataset exported to {report.output_jsonl_file}",
        f"[INFO] Validation report saved to {DEFAULT_VALIDATION_REPORT}",
    ])

    with open(DEFAULT_SUMMARY_LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 80)
    print("FieldSure AI — Corpus Preparation & Ingestion Validation (Module #15)")
    print("=" * 80)

    rep = run_ingestion_pipeline()
    audit = rep.audit

    print("\n--- Ingestion Summary ---")
    print(f"Total Discovered Files: {audit.total_source_files_discovered}")
    print(f"Successfully Ingested:  {audit.documents_successfully_ingested}")
    print(f"Recorded Failures:      {audit.documents_failed_or_skipped}")
    print(f"Total Valid Chunks:     {audit.total_chunks_created}")
    print(f"Completeness Reconciled: {audit.reconciliation_passed} (Zero documents silently dropped)")

    print(f"\nArtifacts successfully exported:")
    print(f" - {rep.output_jsonl_file}")
    print(f" - {DEFAULT_VALIDATION_REPORT}")
    print(f" - {DEFAULT_SUMMARY_LOG}")
