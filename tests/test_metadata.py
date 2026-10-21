"""Unit and regression tests for chunk metadata tagging and source tracking (Module #13)."""

from pathlib import Path
import pytest

from src.ingestion.loader import LoadedDocument
from src.ingestion.metadata import (
    ChunkMetadata,
    DocumentChunk,
    DocumentMetadata,
    SourceTraceReport,
    build_document_metadata,
    extract_agency,
    extract_document_title,
    extract_document_version,
    generate_document_id,
    process_and_tag_document,
    tag_document_chunks,
    trace_chunk_to_source,
)
from src.ingestion.chunker import Chunk, ChunkingStrategy


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_dengue_doc() -> LoadedDocument:
    """Fixture for WHO Dengue guidance document."""
    text = (
        "# WHO Clinical Management of Dengue in Primary Health Centers\n\n"
        "## Document Information\n"
        "- Source: World Health Organization\n"
        "- Version: 2024.1 Update\n"
        "- Publication Date: 2024-01-15\n\n"
        "## 1. Case Definition\n"
        "Dengue is an arboviral infection transmitted by Aedes mosquitoes.\n"
        "High fever (>38.5°C) and retro-orbital headache are typical.\n\n"
        "## 2. Danger Signs\n"
        "Look for persistent vomiting and mucosal bleeding.\n"
    )
    return LoadedDocument(
        doc_id="who_dengue_guidance",
        source_file="data/sample_advisories/who_dengue_guidance.md",
        file_path="/virtual/data/sample_advisories/who_dengue_guidance.md",
        file_type="md",
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
        metadata={"category": "vector_borne", "agency": "WHO"},
    )


@pytest.fixture
def sample_cdc_doc() -> LoadedDocument:
    """Fixture for CDC Cholera advisory."""
    text = (
        "CDC EMERGENCY HEALTH BULLETIN: CHOLERA WATER AND SANITATION PROTOCOL\n\n"
        "DOCUMENT IDENTIFIER: CDC-WASH-CHOLERA-2024\n"
        "TARGET AUDIENCE: Community Health Workers\n\n"
        "1. GENERAL OUTBREAK CONTEXT\n"
        "Vibrio cholerae is transmitted via contaminated water.\n"
    )
    return LoadedDocument(
        doc_id="cdc_cholera_protocol",
        source_file="data/sample_advisories/cdc_cholera_protocol.txt",
        file_path="/virtual/data/sample_advisories/cdc_cholera_protocol.txt",
        file_type="txt",
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
        metadata={},
    )


# ------------------------------------------------------------------------------
# Task 1 & 2 Tests: Source Identifier & Additional Metadata Extraction
# ------------------------------------------------------------------------------

def test_extract_metadata_fields(sample_dengue_doc: LoadedDocument):
    """Verify automatic extraction of title, agency, version, and doc_id (Tasks 1 & 2)."""
    doc_meta = build_document_metadata(sample_dengue_doc)

    assert isinstance(doc_meta, DocumentMetadata)
    assert doc_meta.agency == "WHO"
    assert "Dengue" in doc_meta.title
    assert "2024" in doc_meta.document_version
    assert doc_meta.file_type == "md"
    assert doc_meta.source_file == sample_dengue_doc.source_file
    assert doc_meta.doc_id.startswith("doc_") or doc_meta.doc_id == sample_dengue_doc.doc_id


def test_deterministic_doc_and_chunk_ids():
    """Verify doc_id and chunk_id generation is deterministic and reproducible."""
    content = "Sample medical protocol text"
    file_name = "who_guidance_v1.md"

    id1 = generate_document_id(file_name, content)
    id2 = generate_document_id(file_name, content)
    assert id1 == id2
    assert "who_guidance_v1" in id1


# ------------------------------------------------------------------------------
# Task 3 Tests: Consistent Structure Across Corpus
# ------------------------------------------------------------------------------

def test_consistent_chunk_metadata_schema(sample_dengue_doc: LoadedDocument, sample_cdc_doc: LoadedDocument):
    """Verify that chunks across different documents have identical field structure (Task 3)."""
    _, dengue_chunks = process_and_tag_document(sample_dengue_doc, chunk_size=300)
    _, cdc_chunks = process_and_tag_document(sample_cdc_doc, chunk_size=300)

    assert len(dengue_chunks) > 0
    assert len(cdc_chunks) > 0

    expected_fields = {
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

    # Verify every single chunk from both documents adheres strictly to the exact same schema
    for chunk in dengue_chunks + cdc_chunks:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(chunk.metadata, ChunkMetadata)
        meta_dict = chunk.metadata.to_dict()
        assert set(meta_dict.keys()) == expected_fields
        # Verify chunk_id pattern
        assert re.match(r"^.+_c\d{4}$", chunk.chunk_id)
        # Verify position indexing
        assert 0 <= chunk.metadata.chunk_index < chunk.metadata.total_chunks


# ------------------------------------------------------------------------------
# Task 4 Tests: Tracing Chunk to Exact Source
# ------------------------------------------------------------------------------

def test_trace_chunk_to_source_success(sample_dengue_doc: LoadedDocument):
    """Demonstrate tracing a retrieved chunk back to its source document and span (Task 4)."""
    doc_meta, chunks = process_and_tag_document(sample_dengue_doc, chunk_size=300)
    corpus_map = {sample_dengue_doc.source_file: sample_dengue_doc}

    # Trace each chunk
    for chunk in chunks:
        report = trace_chunk_to_source(chunk, corpus_map)
        assert isinstance(report, SourceTraceReport)
        assert report.traced_successfully is True
        assert report.source_file == sample_dengue_doc.source_file
        assert report.document_title == doc_meta.title
        assert report.agency == "WHO"
        assert report.chunk_index == chunk.metadata.chunk_index
        assert report.match_type in {"exact_substring", "prefix_fingerprint", "normalized_text_match", "metadata_match"}


def test_trace_chunk_missing_source():
    """Verify trace report handles unknown/missing source gracefully."""
    fake_meta = ChunkMetadata(
        chunk_id="doc_missing_c0000",
        doc_id="doc_missing",
        source_file="nonexistent_protocol.md",
        document_title="Ghost Advisory",
        document_version="1.0",
        agency="WHO",
        publication_date=None,
        section_heading="Unknown",
        chunk_index=0,
        total_chunks=1,
        char_length=20,
        token_count=5,
    )
    fake_chunk = DocumentChunk(chunk_id="doc_missing_c0000", text="Missing source text", metadata=fake_meta)
    report = trace_chunk_to_source(fake_chunk, {})

    assert report.traced_successfully is False
    assert report.matched_doc_id == "NOT_FOUND"


# ------------------------------------------------------------------------------
# Citation Formatting Tests
# ------------------------------------------------------------------------------

def test_chunk_citation_formatting(sample_dengue_doc: LoadedDocument):
    """Verify citation formatting contains agency, title, version, section, and chunk ID."""
    _, chunks = process_and_tag_document(sample_dengue_doc)
    first_chunk = chunks[0]

    # Test inline style
    inline_citation = first_chunk.format_citation(style="inline")
    assert "[WHO]" in inline_citation
    assert "Dengue" in inline_citation
    assert "Chunk #1" in inline_citation
    assert first_chunk.chunk_id in inline_citation

    # Test markdown blockquote style
    markdown_citation = first_chunk.format_citation(style="markdown")
    assert "> **Source**:" in markdown_citation
    assert "> **Chunk**:" in markdown_citation

    # Test full audit style
    full_citation = first_chunk.format_citation(style="full")
    assert "Document ID:" in full_citation
    assert "Chunk ID:" in full_citation


# ------------------------------------------------------------------------------
# Filtering & Validation Tests
# ------------------------------------------------------------------------------

def test_filter_chunks_by_metadata(sample_dengue_doc: LoadedDocument, sample_cdc_doc: LoadedDocument):
    """Verify filtering DocumentChunks by agency, section keyword, and token limits."""
    from src.ingestion.metadata import filter_chunks_by_metadata

    _, dengue_chunks = process_and_tag_document(sample_dengue_doc, chunk_size=300)
    _, cdc_chunks = process_and_tag_document(sample_cdc_doc, chunk_size=300)
    combined = dengue_chunks + cdc_chunks

    # Filter by agency
    who_only = filter_chunks_by_metadata(combined, agency="WHO")
    assert len(who_only) == len(dengue_chunks)
    assert all(c.metadata.agency == "WHO" for c in who_only)

    cdc_only = filter_chunks_by_metadata(combined, agency="CDC")
    assert len(cdc_only) == len(cdc_chunks)
    assert all(c.metadata.agency == "CDC" for c in cdc_only)

    # Filter by section keyword
    danger_chunks = filter_chunks_by_metadata(combined, section_keyword="Danger")
    assert len(danger_chunks) >= 1
    assert any("Danger Signs" in c.metadata.section_heading for c in danger_chunks)


def test_validate_chunk_metadata_completeness(sample_dengue_doc: LoadedDocument):
    """Verify batch validation of chunk metadata completeness."""
    from src.ingestion.metadata import validate_chunk_metadata_completeness

    _, valid_chunks = process_and_tag_document(sample_dengue_doc)
    report = validate_chunk_metadata_completeness(valid_chunks)

    assert report["valid"] is True
    assert report["issues_count"] == 0
    assert report["total_chunks"] == len(valid_chunks)
