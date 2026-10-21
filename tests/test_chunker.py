"""Unit and regression tests for document chunking strategies (Module #12)."""

from pathlib import Path
import pytest

from src.ingestion.chunker import (
    Chunk,
    ChunkingStats,
    ChunkingStrategy,
    chunk_document,
    chunk_text_by_strategy,
    compute_chunking_stats,
    compare_strategies_on_document,
    split_fixed_size,
    split_into_sentences,
    split_paragraph_structural,
    split_sentence_chunks,
)
from src.ingestion.loader import LoadedDocument


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_clinical_text() -> str:
    """Sample clinical protocol with headings, bullet lists, and abbreviations."""
    return (
        "# WHO Dengue Clinical Protocol\n\n"
        "## Diagnostic Phases\n"
        "Dengue fever presents in three distinct phases. The febrile phase lasts 2-7 days.\n"
        "Critical phase begins around defervescence at day 3-7. Plasma leakage is common.\n"
        "Recovery phase begins after 48 hours of critical phase.\n\n"
        "## Danger Signs & Interventions\n"
        "Field workers must monitor for severe abdominal pain and persistent vomiting.\n"
        "Administer Oral Rehydration Salts (ORS) solution promptly (10-15 ml/kg/hr).\n"
        "Avoid Ibuprofen and Aspirin due to hemorrhage risk.\n"
    )


@pytest.fixture
def sample_loaded_document(sample_clinical_text: str) -> LoadedDocument:
    """Fixture representing an ingested LoadedDocument from Module #10."""
    return LoadedDocument(
        doc_id="who_dengue_protocol",
        source_file="data/sample_advisories/who_dengue_guidance.md",
        file_path="/dummy/path/who_dengue_guidance.md",
        file_type="md",
        text=sample_clinical_text,
        char_count=len(sample_clinical_text),
        word_count=len(sample_clinical_text.split()),
        metadata={"category": "vector_borne", "agency": "WHO"},
    )


# ------------------------------------------------------------------------------
# Strategy 1 Tests: Fixed-Size with Overlap
# ------------------------------------------------------------------------------

def test_split_fixed_size_basic():
    """Verify fixed-size character windowing and sliding overlap."""
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 26 characters
    chunks = split_fixed_size(text, chunk_size=10, chunk_overlap=2)

    # Step size = 10 - 2 = 8
    # Chunk 0: 0..10 -> "ABCDEFGHIJ"
    # Chunk 1: 8..18 -> "IJKLMNOPQR"
    # Chunk 2: 16..26 -> "QRSTUVWXYZ"
    assert len(chunks) == 3
    assert chunks[0] == "ABCDEFGHIJ"
    assert chunks[1] == "IJKLMNOPQR"
    assert chunks[2] == "QRSTUVWXYZ"


def test_split_fixed_size_empty_and_short():
    """Verify behavior on empty text or text shorter than window."""
    assert split_fixed_size("") == []
    assert split_fixed_size("   ") == []

    short = "Hello world"
    assert split_fixed_size(short, chunk_size=50, chunk_overlap=10) == [short]


def test_split_fixed_size_invalid_args():
    """Verify argument validation for chunk_size and chunk_overlap."""
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_fixed_size("sample", chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        split_fixed_size("sample", chunk_size=100, chunk_overlap=100)


# ------------------------------------------------------------------------------
# Strategy 2 Tests: Sentence-Based Chunking
# ------------------------------------------------------------------------------

def test_split_into_sentences_preserves_abbreviations():
    """Verify that clinical abbreviations like 'e.g.', 'Dr.', '38.5°C' do not break sentences."""
    text = (
        "Patient seen by Dr. Smith with fever > 38.5°C. "
        "Symptoms include nausea, e.g. persistent vomiting. "
        "Immediate referral is required!"
    )
    sentences = split_into_sentences(text)

    assert len(sentences) == 3
    assert "Dr. Smith" in sentences[0]
    assert "38.5°C." in sentences[0]
    assert "e.g. persistent vomiting." in sentences[1]
    assert sentences[2] == "Immediate referral is required!"


def test_split_sentence_chunks_sentence_integrity():
    """Verify sentences are aggregated without splitting individual sentences."""
    text = (
        "First clinical sentence. "
        "Second clinical sentence is somewhat longer. "
        "Third clinical sentence contains dosage details."
    )
    chunks = split_sentence_chunks(text, max_chunk_chars=60, sentence_overlap=0)

    # No chunk should end mid-word or mid-sentence
    for chunk in chunks:
        assert chunk.endswith(".")
        assert "clinical" in chunk


# ------------------------------------------------------------------------------
# Strategy 3 Tests: Paragraph / Structural Recursive Chunking
# ------------------------------------------------------------------------------

def test_split_paragraph_structural_section_tracking(sample_clinical_text: str):
    """Verify section headings are tracked and prepended to chunks."""
    chunks = split_paragraph_structural(
        sample_clinical_text,
        max_chunk_chars=300,
        overlap_chars=30,
        prepend_heading=True,
    )

    assert len(chunks) >= 2
    # Check that sections are identified
    headings = [h for _, h in chunks if h is not None]
    assert any("Diagnostic Phases" in h for h in headings)
    assert any("Danger Signs" in h for h in headings)

    # Check that chunks contain the prepended section tag
    tagged_chunks = [text for text, _ in chunks if text.startswith("[Section:")]
    assert len(tagged_chunks) > 0


def test_split_paragraph_structural_oversized_fallback():
    """Verify that oversized paragraphs gracefully fall back to sentence splitting."""
    long_para = "Clinical sentence number one. " * 30  # ~900 characters
    text = f"## Primary Section\n\n{long_para}"

    chunks = split_paragraph_structural(text, max_chunk_chars=250, prepend_heading=True)
    assert len(chunks) > 1
    for chunk_text, heading in chunks:
        assert len(chunk_text) <= 350  # Allows tag + sentence margin
        assert heading == "Primary Section"


# ------------------------------------------------------------------------------
# Orchestrator & Multi-Strategy Comparison Tests
# ------------------------------------------------------------------------------

def test_chunk_document_end_to_end(sample_loaded_document: LoadedDocument):
    """Verify chunk_document produces fully typed, metadata-rich Chunk objects."""
    chunks = chunk_document(
        doc=sample_loaded_document,
        strategy=ChunkingStrategy.PARAGRAPH_STRUCTURAL,
        chunk_size=400,
        chunk_overlap=50,
    )

    assert len(chunks) > 0
    first = chunks[0]
    assert isinstance(first, Chunk)
    assert first.doc_id == "who_dengue_protocol"
    assert first.chunk_id == "who_dengue_protocol_c0000"
    assert first.total_chunks == len(chunks)
    assert first.char_count == len(first.text)
    assert first.token_count > 0
    assert first.strategy == "paragraph_structural"
    assert first.source_file == sample_loaded_document.source_file


def test_compute_chunking_stats(sample_loaded_document: LoadedDocument):
    """Verify statistical aggregation (count, avg, min, max chars and tokens)."""
    chunks = chunk_document(sample_loaded_document, strategy=ChunkingStrategy.FIXED_SIZE, chunk_size=200, chunk_overlap=20)
    stats = compute_chunking_stats(
        chunks=chunks,
        strategy_name="fixed_size",
        doc_id=sample_loaded_document.doc_id,
        source_file=sample_loaded_document.source_file,
    )

    assert isinstance(stats, ChunkingStats)
    assert stats.total_chunks == len(chunks)
    assert stats.avg_chunk_chars > 0
    assert stats.avg_chunk_tokens > 0
    assert stats.min_chunk_chars <= stats.max_chunk_chars
    assert stats.min_chunk_tokens <= stats.max_chunk_tokens


def test_compare_strategies_on_document(sample_loaded_document: LoadedDocument):
    """Verify running multiple strategies side-by-side produces complete comparisons."""
    chunks_map, stats_map = compare_strategies_on_document(
        doc=sample_loaded_document,
        chunk_size=300,
        chunk_overlap=30,
    )

    assert "fixed_size" in chunks_map
    assert "sentence" in chunks_map
    assert "paragraph_structural" in chunks_map

    assert "fixed_size" in stats_map
    assert "sentence" in stats_map
    assert "paragraph_structural" in stats_map

    # Fixed size splits mechanically
    assert stats_map["fixed_size"].total_chunks > 0
    # Paragraph structural retains headings
    structural_chunks = chunks_map["paragraph_structural"]
    assert any("[Section:" in c.text for c in structural_chunks)
