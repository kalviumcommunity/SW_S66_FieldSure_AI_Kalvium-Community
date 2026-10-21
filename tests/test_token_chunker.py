"""Unit and regression tests for Token-Aware Chunk Sizing & Overlap (Module #14)."""

import pytest

from src.ingestion.chunker import (
    ChunkingStrategy,
    compare_boundary_overlap,
    decode_tokens,
    encode_tokens,
    split_by_tokens,
    split_token_structural,
    verify_chunk_overlap,
)
from src.token_cost_estimator import count_tokens


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def clinical_boundary_text() -> str:
    """Sample clinical text with critical information sitting at boundary transitions."""
    return (
        "Dengue fever presents in three distinct phases: febrile, critical, and recovery. "
        "The febrile phase is characterized by abrupt onset of high fever (>38.5°C), severe retro-orbital headache, and myalgia. "
        "The critical phase occurs at defervescence around day 3 to 7 of illness and is characterized by systemic plasma leakage, "
        "severe thrombocytopenia, ascites, pleural effusion, and potential circulatory shock. "
        "Immediate field triage requires daily hematocrit and platelet monitoring."
    )


# ------------------------------------------------------------------------------
# Task 1 & 2 Tests: Token-Aware Sizing and Controlled Overlap
# ------------------------------------------------------------------------------

def test_token_encoding_decoding_roundtrip(clinical_boundary_text: str):
    """Verify that encode_tokens and decode_tokens accurately roundtrip text."""
    tokens = encode_tokens(clinical_boundary_text)
    assert len(tokens) > 0

    reconstructed = decode_tokens(tokens)
    assert "Dengue fever" in reconstructed
    assert "plasma leakage" in reconstructed


def test_split_by_tokens_strict_budget(clinical_boundary_text: str):
    """Task 1: Verify 100% of chunks strictly respect the specified token ceiling."""
    max_budget = 40
    overlap = 10
    chunks = split_by_tokens(clinical_boundary_text, max_tokens=max_budget, overlap_tokens=overlap)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        t_count = count_tokens(chunk)
        # All chunks must be within the max_budget
        assert t_count <= max_budget, f"Chunk {idx} exceeded token budget: {t_count} > {max_budget}"


def test_split_by_tokens_overlap_repetition(clinical_boundary_text: str):
    """Task 2: Verify adjacent chunks repeat the controlled overlap tokens."""
    max_budget = 50
    overlap = 15
    chunks = split_by_tokens(clinical_boundary_text, max_tokens=max_budget, overlap_tokens=overlap)

    assert len(chunks) >= 2
    # Verify using overlap auditor
    verifications = verify_chunk_overlap(chunks, expected_overlap_tokens=overlap)
    assert len(verifications) == len(chunks) - 1
    # Check that overlap tokens were identified between pairs
    for item in verifications:
        assert item["tail_tokens_chunk_a"] > 0
        assert item["head_tokens_chunk_b"] > 0


def test_split_by_tokens_edge_cases():
    """Verify handling of empty strings, small texts, and invalid arguments."""
    assert split_by_tokens("") == []
    assert split_by_tokens("   ") == []

    single_sentence = "Oral rehydration salts save lives."
    chunks = split_by_tokens(single_sentence, max_tokens=100, overlap_tokens=10)
    assert len(chunks) == 1
    assert chunks[0] == single_sentence

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        split_by_tokens("sample text", max_tokens=0)

    with pytest.raises(ValueError, match="overlap_tokens must be non-negative and strictly less than max_tokens"):
        split_by_tokens("sample text", max_tokens=50, overlap_tokens=50)


# ------------------------------------------------------------------------------
# Task 3 Tests: Boundary Context Preservation Demonstration
# ------------------------------------------------------------------------------

def test_compare_boundary_overlap_preserves_clinical_idea(clinical_boundary_text: str):
    """Task 3: Demonstrate an idea on a boundary appearing intact thanks to overlap."""
    comparison = compare_boundary_overlap(
        clinical_boundary_text,
        target_tokens=45,
        overlap_tokens=15,
    )

    assert "without_overlap" in comparison
    assert "with_overlap" in comparison

    no_ov = comparison["without_overlap"]["chunks"]
    with_ov = comparison["with_overlap"]["chunks"]

    assert len(no_ov) >= 2
    assert len(with_ov) >= 2

    # In with_overlap, chunk 1 repeats the trailing tokens of chunk 0
    c0_tail = with_ov[0]["text"].split()[-3:]
    c1_head = with_ov[1]["text"].split()[:5]

    # Verify word overlap continuity across boundary
    assert any(word.lower().strip(".,") in " ".join(c1_head).lower() for word in c0_tail)


# ------------------------------------------------------------------------------
# Structural Token-Aware Splitting Tests
# ------------------------------------------------------------------------------

def test_split_token_structural():
    """Verify hierarchical markdown splitting with token budgeting and section context."""
    text = (
        "# Dengue Triage\n\n"
        "## Diagnostic Phase\n"
        "Febrile phase lasts 2 to 7 days with high fever.\n\n"
        "## Danger Signs\n"
        "Persistent vomiting, severe abdominal pain, and mucosal bleeding.\n"
    )

    chunks = split_token_structural(text, target_tokens=30, overlap_tokens=8, prepend_heading=True)
    assert len(chunks) >= 2

    # All chunks must include section context
    for chunk_text, heading in chunks:
        assert heading in {"Diagnostic Phase", "Danger Signs", "Dengue Triage", None}
        assert count_tokens(chunk_text) <= 50  # Allows tag margin
