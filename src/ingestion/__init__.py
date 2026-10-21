"""FieldSure AI Document Ingestion Package.

Handles document loading, multi-format intake, text extraction, cleaning, and chunking.
"""

from src.ingestion.loader import (
    CorruptDocumentError,
    DocumentLoaderError,
    DocumentNotFoundError,
    LoadedDocument,
    UnsupportedFileError,
    load_corpus,
    load_document,
)

__all__ = [
    "CorruptDocumentError",
    "DocumentLoaderError",
    "DocumentNotFoundError",
    "LoadedDocument",
    "UnsupportedFileError",
    "load_corpus",
    "load_document",
]
