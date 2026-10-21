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
from src.ingestion.cleaner import (
    CleaningReport,
    clean_corpus,
    clean_document,
    clean_text,
    fix_line_wrap_hyphenation,
    normalize_unicode,
    remove_boilerplate_patterns,
    strip_html_tags,
)

from src.ingestion.chunker import (
    Chunk,
    ChunkingStats,
    ChunkingStrategy,
    chunk_document,
    chunk_text_by_strategy,
    compare_strategies_on_document,
    compute_chunking_stats,
    run_chunking_comparison,
    split_fixed_size,
    split_into_sentences,
    split_paragraph_structural,
    split_sentence_chunks,
)
from src.ingestion.metadata import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    SourceTraceReport,
    build_document_metadata,
    process_and_tag_document,
    run_metadata_tagging_and_tracing,
    tag_document_chunks,
    trace_chunk_to_source,
)

__all__ = [
    "CorruptDocumentError",
    "DocumentLoaderError",
    "DocumentNotFoundError",
    "LoadedDocument",
    "UnsupportedFileError",
    "load_corpus",
    "load_document",
    "CleaningReport",
    "clean_corpus",
    "clean_document",
    "clean_text",
    "fix_line_wrap_hyphenation",
    "normalize_unicode",
    "remove_boilerplate_patterns",
    "strip_html_tags",
    "Chunk",
    "ChunkingStats",
    "ChunkingStrategy",
    "chunk_document",
    "chunk_text_by_strategy",
    "compare_strategies_on_document",
    "compute_chunking_stats",
    "run_chunking_comparison",
    "split_fixed_size",
    "split_into_sentences",
    "split_paragraph_structural",
    "split_sentence_chunks",
    "ChunkMetadata",
    "DocumentChunk",
    "DocumentMetadata",
    "SourceTraceReport",
    "build_document_metadata",
    "process_and_tag_document",
    "run_metadata_tagging_and_tracing",
    "tag_document_chunks",
    "trace_chunk_to_source",
]
