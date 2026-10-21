"""Multi-format document loader for FieldSure AI (Module #10).

Ingests public health guidance documents across Markdown, Plain Text, PDF, DOCX,
and HTML formats into a unified plain-text representation while preserving source identity
and handling unreadable or corrupted files gracefully.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("fieldsure.loader")

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt", ".html", ".htm"}


# ------------------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------------------

class DocumentLoaderError(Exception):
    """Base exception for all document loading failures."""


class DocumentNotFoundError(DocumentLoaderError):
    """Raised when the specified file path does not exist."""


class UnsupportedFileError(DocumentLoaderError):
    """Raised when an unrecognized file format is encountered."""


class CorruptDocumentError(DocumentLoaderError):
    """Raised when a document cannot be decoded or parsed."""


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass
class LoadedDocument:
    """Standardized plain-text representation of an ingested document."""

    doc_id: str
    source_file: str
    file_path: str
    file_type: str
    text: str
    char_count: int
    word_count: int
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        source_path: Path,
        text: str,
        page_count: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> "LoadedDocument":
        """Factory method computing character and word counts."""
        cleaned_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        file_stat = source_path.stat() if source_path.exists() else None

        meta: Dict[str, Any] = {
            "file_size_bytes": file_stat.st_size if file_stat else 0,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra_meta:
            meta.update(extra_meta)

        return cls(
            doc_id=source_path.stem.lower().replace(" ", "_"),
            source_file=source_path.name,
            file_path=str(source_path.resolve()),
            file_type=source_path.suffix.lstrip(".").lower(),
            text=cleaned_text,
            char_count=len(cleaned_text),
            word_count=len(cleaned_text.split()),
            page_count=page_count,
            metadata=meta,
        )


# ------------------------------------------------------------------------------
# Format-Specific Readers
# ------------------------------------------------------------------------------

def _read_plain_text(path: Path) -> Tuple[str, Optional[int], Dict[str, Any]]:
    """Read plain text, markdown, or HTML using standard library."""
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    for enc in encodings:
        try:
            content = path.read_text(encoding=enc)
            return content, None, {"encoding": enc}
        except UnicodeDecodeError:
            continue
    raise CorruptDocumentError(f"Unable to decode text document '{path.name}' with supported encodings.")


def _read_pdf(path: Path) -> Tuple[str, Optional[int], Dict[str, Any]]:
    """Extract text page-by-page from PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed, reading PDF bytes as fallback.")
        content = path.read_bytes().decode("utf-8", errors="replace")
        return content, 1, {"extractor": "fallback_raw"}

    try:
        reader = PdfReader(str(path))
        num_pages = len(reader.pages)
        page_texts: List[str] = []

        for idx, page in enumerate(reader.pages):
            extracted = page.extract_text() or ""
            if extracted.strip():
                page_texts.append(extracted.strip())

        full_text = "\n\n".join(page_texts)
        if not full_text.strip():
            logger.warning("PDF '%s' contains no extractable text layer (possible scanned image).", path.name)

        return full_text, num_pages, {"extractor": "pypdf", "page_count": num_pages}
    except Exception as err:
        raise CorruptDocumentError(f"Failed to parse PDF document '{path.name}': {err}") from err


def _read_docx(path: Path) -> Tuple[str, Optional[int], Dict[str, Any]]:
    """Extract paragraphs and table text from DOCX using python-docx."""
    try:
        import docx
    except ImportError:
        logger.warning("python-docx not installed, fallback text extraction.")
        return path.read_text(encoding="utf-8", errors="replace"), None, {"extractor": "fallback_text"}

    try:
        doc = docx.Document(str(path))
        elements: List[str] = []

        for para in doc.paragraphs:
            if para.text.strip():
                elements.append(para.text.strip())

        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                if row_text:
                    elements.append(row_text)

        full_text = "\n\n".join(elements)
        return full_text, None, {"extractor": "python-docx", "paragraph_count": len(doc.paragraphs)}
    except Exception as err:
        raise CorruptDocumentError(f"Failed to parse Word document '{path.name}': {err}") from err


# ------------------------------------------------------------------------------
# Core Loader API
# ------------------------------------------------------------------------------

def load_document(file_path: Union[str, Path]) -> LoadedDocument:
    """Load a single document from disk into a LoadedDocument container.

    Args:
        file_path: Path to the target document.

    Returns:
        LoadedDocument containing normalized plain text and source provenance.

    Raises:
        DocumentNotFoundError: If the file does not exist.
        UnsupportedFileError: If the extension is not in SUPPORTED_EXTENSIONS.
        CorruptDocumentError: If file reading or parsing fails.
    """
    path = Path(file_path)

    if not path.is_file():
        raise DocumentNotFoundError(f"Document file not found at '{path}'")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file format '{ext}' for file '{path.name}'. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if ext in {".txt", ".md", ".html", ".htm"}:
            text, pages, extra_meta = _read_plain_text(path)
        elif ext == ".pdf":
            text, pages, extra_meta = _read_pdf(path)
        elif ext == ".docx":
            text, pages, extra_meta = _read_docx(path)
        else:
            raise UnsupportedFileError(f"Handler not implemented for '{ext}'")

        return LoadedDocument.create(
            source_path=path,
            text=text,
            page_count=pages,
            extra_meta=extra_meta,
        )
    except (DocumentNotFoundError, UnsupportedFileError, CorruptDocumentError):
        raise
    except Exception as err:
        raise CorruptDocumentError(f"Unexpected error loading document '{path.name}': {err}") from err


def load_corpus(
    directory_path: Union[str, Path],
    recursive: bool = True,
    skip_bad_files: bool = True,
) -> Tuple[List[LoadedDocument], List[Dict[str, str]]]:
    """Ingest a directory of heterogeneous documents.

    Args:
        directory_path: Directory containing corpus files.
        recursive: Whether to scan subdirectories recursively.
        skip_bad_files: If True, survives corrupt/unsupported/missing files gracefully.

    Returns:
        Tuple of (loaded_documents, error_ledger).
    """
    base_dir = Path(directory_path)
    if not base_dir.is_dir():
        raise DocumentNotFoundError(f"Corpus directory not found: '{base_dir}'")

    loaded_documents: List[LoadedDocument] = []
    error_ledger: List[Dict[str, str]] = []

    pattern = "**/*" if recursive else "*"
    candidate_files = [p for p in base_dir.glob(pattern) if p.is_file()]

    for file_path in candidate_files:
        # Ignore git tracking files
        if file_path.name in {".gitkeep", ".DS_Store"}:
            continue

        try:
            doc = load_document(file_path)
            loaded_documents.append(doc)
            logger.info("Successfully loaded: %s (%d chars)", doc.source_file, doc.char_count)
        except Exception as err:
            err_record = {
                "file": file_path.name,
                "path": str(file_path),
                "error_type": type(err).__name__,
                "error_message": str(err),
            }
            error_ledger.append(err_record)
            logger.warning("Skipped unreadable/unsupported file '%s': %s", file_path.name, err)

            if not skip_bad_files:
                raise

    return loaded_documents, error_ledger


# ------------------------------------------------------------------------------
# CLI Intake Confirmation Runner (Task 4 & 5)
# ------------------------------------------------------------------------------

def run_intake_confirmation(corpus_dir: Optional[Path] = None) -> int:
    """Load corpus, verify intake, and print confirmation samples."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    sample_dir = corpus_dir or (repo_root / "data" / "sample_advisories")
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_file = outputs_dir / "document_loading_sample.log"

    print("\n" + "=" * 80)
    print("FIELDSURE AI — MULTI-FORMAT DOCUMENT INTAKE CONFIRMATION")
    print(f"Corpus Directory: {sample_dir}")
    print("=" * 80)

    if not sample_dir.is_dir():
        print(f"\n[ERROR] Sample advisories directory does not exist: {sample_dir}\n", file=sys.stderr)
        return 1

    docs, errors = load_corpus(sample_dir, recursive=True, skip_bad_files=True)

    print(f"\nIntake Results: {len(docs)} documents successfully loaded, {len(errors)} files skipped.\n")

    confirmation_records = []

    for idx, doc in enumerate(docs, 1):
        preview = doc.text[:140].replace("\n", " ") + ("..." if len(doc.text) > 140 else "")
        page_info = f" | Pages: {doc.page_count}" if doc.page_count else ""

        print(f"[{idx}] Source Identity: {doc.source_file}")
        print(f"    Format: {doc.file_type.upper()}{page_info}")
        print(f"    Length: {doc.char_count:,} characters | {doc.word_count:,} words")
        print(f"    Sample Preview: \"{preview}\"")
        print("-" * 80)

        confirmation_records.append({
            "source_file": doc.source_file,
            "file_type": doc.file_type,
            "char_count": doc.char_count,
            "word_count": doc.word_count,
            "page_count": doc.page_count,
            "sample_preview": preview,
        })

    if errors:
        print("\nSkipped / Faulty Files Ledger:")
        for err in errors:
            print(f"  • {err['file']}: [{err['error_type']}] {err['error_message']}")
        print("-" * 80)

    # Save intake confirmation audit log
    audit_data = {
        "module": "Module #10 — Document Loading & Multi-Format Intake",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_loaded": len(docs),
        "total_skipped": len(errors),
        "documents": confirmation_records,
        "error_ledger": errors,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n--- Structured Document Intake Audit ---\n")
        f.write(json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n")

    print(f"\nAudit confirmation log successfully saved to: {log_file}\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_intake_confirmation())
