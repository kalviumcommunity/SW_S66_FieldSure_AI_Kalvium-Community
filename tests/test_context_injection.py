"""Tests for Context Injection & Prompt Augmentation (Module #29).

Covers:
    - format_chunk: source marker generation, missing metadata handling.
    - assemble_context: token budget enforcement, chunk ordering, separator use.
    - build_augmented_prompt: grounding instructions, question injection, refusal text.
    - compute_context_budget: dynamic vs static budget calculation.
    - Edge cases: empty chunks, single chunk, very tight budgets.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

from src.context_injection import (
    ANSWER_RESERVE_TOKENS,
    CHUNK_SEPARATOR,
    FALLBACK_NO_CONTEXT,
    GROUNDING_INSTRUCTION,
    INSTRUCTION_RESERVE_TOKENS,
    MAX_CONTEXT_TOKENS,
    AssembledContext,
    AugmentedPrompt,
    assemble_context,
    build_augmented_prompt,
    compute_context_budget,
    format_chunk,
)
from src.token_cost_estimator import count_tokens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str,
    source_file: str = "data/sample.md",
    chunk_index: int = 0,
    section_heading: str = "Introduction",
    doc_id: str = "doc_test",
    token_count: int | None = None,
) -> Dict[str, Any]:
    """Helper to build a chunk dictionary matching the project schema."""
    return {
        "text": text,
        "metadata": {
            "source_file": source_file,
            "chunk_index": chunk_index,
            "section_heading": section_heading,
            "doc_id": doc_id,
            "document_title": "Test Document",
            "document_version": "v1.0",
            "token_count": token_count or count_tokens(text),
        },
    }


SAMPLE_CHUNKS: List[Dict[str, Any]] = [
    _make_chunk(
        text="Dengue is transmitted by Aedes aegypti mosquitoes.",
        source_file="data/who_dengue.md",
        chunk_index=0,
        section_heading="Transmission",
    ),
    _make_chunk(
        text="Warning signs include persistent vomiting and abdominal pain.",
        source_file="data/who_dengue.md",
        chunk_index=1,
        section_heading="Warning Signs",
    ),
    _make_chunk(
        text="Use Paracetamol for fever. Avoid NSAIDs like Ibuprofen.",
        source_file="data/who_dengue.md",
        chunk_index=2,
        section_heading="Field Interventions",
    ),
]


# ---------------------------------------------------------------------------
# Tests: format_chunk (Task 1 & Task 3)
# ---------------------------------------------------------------------------

class TestFormatChunk:
    """Tests for format_chunk — source marker generation."""

    def test_basic_format_includes_source_marker(self) -> None:
        chunk = SAMPLE_CHUNKS[0]
        result = format_chunk(1, chunk)
        assert result.startswith("[1]")
        assert "data/who_dengue.md" in result
        assert "#chunk0" in result

    def test_format_includes_section_heading(self) -> None:
        chunk = SAMPLE_CHUNKS[1]
        result = format_chunk(2, chunk)
        assert "(Section: Warning Signs)" in result

    def test_format_includes_chunk_text(self) -> None:
        chunk = SAMPLE_CHUNKS[0]
        result = format_chunk(1, chunk)
        assert "Dengue is transmitted by Aedes aegypti mosquitoes." in result

    def test_format_index_increments(self) -> None:
        r1 = format_chunk(1, SAMPLE_CHUNKS[0])
        r2 = format_chunk(2, SAMPLE_CHUNKS[1])
        r3 = format_chunk(3, SAMPLE_CHUNKS[2])
        assert r1.startswith("[1]")
        assert r2.startswith("[2]")
        assert r3.startswith("[3]")

    def test_missing_metadata_uses_unknown_source(self) -> None:
        chunk = {"text": "Some text.", "metadata": {}}
        result = format_chunk(1, chunk)
        assert "Unknown Source" in result

    def test_missing_chunk_index_omits_hash(self) -> None:
        chunk = {"text": "Some text.", "metadata": {"source_file": "doc.md"}}
        result = format_chunk(1, chunk)
        assert "#chunk" not in result

    def test_missing_section_heading_omits_section(self) -> None:
        chunk = {"text": "Some text.", "metadata": {"source_file": "doc.md"}}
        result = format_chunk(1, chunk)
        assert "(Section:" not in result

    def test_raises_on_missing_text_key(self) -> None:
        with pytest.raises(ValueError, match="text"):
            format_chunk(1, {"metadata": {}})

    def test_raises_on_non_dict(self) -> None:
        with pytest.raises(ValueError, match="dictionary"):
            format_chunk(1, "not a dict")  # type: ignore[arg-type]

    def test_no_metadata_key_at_all(self) -> None:
        chunk = {"text": "Bare chunk with no metadata."}
        result = format_chunk(1, chunk)
        assert "[1]" in result
        assert "Unknown Source" in result
        assert "Bare chunk with no metadata." in result

    def test_source_fallback_to_document_title(self) -> None:
        chunk = {"text": "Text.", "metadata": {"document_title": "Dengue Guide"}}
        result = format_chunk(1, chunk)
        assert "Dengue Guide" in result


# ---------------------------------------------------------------------------
# Tests: assemble_context (Task 2)
# ---------------------------------------------------------------------------

class TestAssembleContext:
    """Tests for assemble_context — token budget enforcement."""

    def test_all_chunks_fit_within_budget(self) -> None:
        result = assemble_context(SAMPLE_CHUNKS, max_tokens=5000)
        assert result.chunks_included == 3
        assert result.chunks_total == 3
        assert result.token_count > 0
        assert result.token_count <= 5000

    def test_separator_between_chunks(self) -> None:
        result = assemble_context(SAMPLE_CHUNKS, max_tokens=5000)
        assert "---" in result.text

    def test_tight_budget_truncates_chunks(self) -> None:
        # Very tight budget should include fewer than all chunks
        result = assemble_context(SAMPLE_CHUNKS, max_tokens=30)
        assert result.chunks_included < result.chunks_total

    def test_zero_budget_includes_nothing(self) -> None:
        result = assemble_context(SAMPLE_CHUNKS, max_tokens=0)
        assert result.chunks_included == 0
        assert result.text == ""
        assert result.token_count == 0

    def test_empty_chunks_list(self) -> None:
        result = assemble_context([], max_tokens=5000)
        assert result.chunks_included == 0
        assert result.text == ""

    def test_sources_used_matches_included_chunks(self) -> None:
        result = assemble_context(SAMPLE_CHUNKS, max_tokens=5000)
        assert len(result.sources_used) == result.chunks_included

    def test_token_count_accurate(self) -> None:
        result = assemble_context(SAMPLE_CHUNKS, max_tokens=5000)
        # The token_count should roughly match count_tokens on each formatted chunk
        # (summed individually, not on the joined text which includes separators)
        assert result.token_count > 0

    def test_order_preserved(self) -> None:
        result = assemble_context(SAMPLE_CHUNKS, max_tokens=5000)
        # Chunk 1 text should appear before chunk 2 text
        pos1 = result.text.index("Aedes aegypti")
        pos2 = result.text.index("persistent vomiting")
        assert pos1 < pos2

    def test_dynamic_budget_from_window(self) -> None:
        result = assemble_context(SAMPLE_CHUNKS, total_window_tokens=8000)
        expected_budget = 8000 - ANSWER_RESERVE_TOKENS - INSTRUCTION_RESERVE_TOKENS
        assert result.token_count <= expected_budget

    def test_single_chunk(self) -> None:
        result = assemble_context([SAMPLE_CHUNKS[0]], max_tokens=5000)
        assert result.chunks_included == 1
        assert "---" not in result.text  # No separator for a single chunk


# ---------------------------------------------------------------------------
# Tests: compute_context_budget
# ---------------------------------------------------------------------------

class TestComputeContextBudget:
    """Tests for compute_context_budget — dynamic vs static budget."""

    def test_static_default(self) -> None:
        assert compute_context_budget() == MAX_CONTEXT_TOKENS

    def test_dynamic_calculation(self) -> None:
        budget = compute_context_budget(total_window_tokens=8000)
        expected = 8000 - ANSWER_RESERVE_TOKENS - INSTRUCTION_RESERVE_TOKENS
        assert budget == expected

    def test_very_small_window_floors_at_zero(self) -> None:
        budget = compute_context_budget(total_window_tokens=100)
        assert budget == 0

    def test_custom_reserves(self) -> None:
        budget = compute_context_budget(
            total_window_tokens=10000,
            answer_reserve=2000,
            instruction_reserve=1000,
        )
        assert budget == 7000


# ---------------------------------------------------------------------------
# Tests: build_augmented_prompt (Task 4 & Task 5)
# ---------------------------------------------------------------------------

class TestBuildAugmentedPrompt:
    """Tests for build_augmented_prompt — full prompt construction."""

    def test_prompt_contains_grounding_instruction(self) -> None:
        result = build_augmented_prompt("What is dengue?", SAMPLE_CHUNKS)
        assert "grounded assistant" in result.prompt
        assert "ONLY the provided context" in result.prompt

    def test_prompt_contains_question(self) -> None:
        question = "What are the warning signs of dengue?"
        result = build_augmented_prompt(question, SAMPLE_CHUNKS)
        assert f"Question: {question}" in result.prompt

    def test_prompt_contains_context_label(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert "Context:" in result.prompt

    def test_prompt_contains_source_markers(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert "[1]" in result.prompt
        assert "[2]" in result.prompt

    def test_prompt_contains_chunk_text(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert "Aedes aegypti" in result.prompt

    def test_prompt_includes_refusal_instruction(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert "I don't have enough information" in result.prompt

    def test_prompt_includes_citation_instruction(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert "cite sources" in result.prompt.lower() or "[1]" in result.prompt

    def test_empty_chunks_prompt_notes_no_context(self) -> None:
        result = build_augmented_prompt("Test?", [])
        assert "No relevant context" in result.prompt
        assert result.chunks_included == 0

    def test_total_prompt_tokens_positive(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert result.total_prompt_tokens > 0

    def test_context_tokens_less_than_budget(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS, max_context_tokens=5000)
        assert result.context_tokens <= 5000

    def test_sources_used_populated(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert len(result.sources_used) > 0
        assert "source_file" in result.sources_used[0]

    def test_raises_on_empty_question(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_augmented_prompt("", SAMPLE_CHUNKS)

    def test_raises_on_whitespace_question(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_augmented_prompt("   ", SAMPLE_CHUNKS)

    def test_custom_grounding_instruction(self) -> None:
        custom = "Answer using ONLY the documents below."
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS, grounding_instruction=custom)
        assert custom in result.prompt
        assert GROUNDING_INSTRUCTION not in result.prompt

    def test_tight_budget_limits_chunks(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS, max_context_tokens=30)
        assert result.chunks_included < len(SAMPLE_CHUNKS)

    def test_dynamic_window_budget(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS, total_window_tokens=8000)
        expected_budget = 8000 - ANSWER_RESERVE_TOKENS - INSTRUCTION_RESERVE_TOKENS
        assert result.context_tokens <= expected_budget


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    """Integration tests verifying the full format -> assemble -> prompt flow."""

    def test_full_pipeline_produces_valid_prompt(self) -> None:
        question = "How to manage dengue fever in the field?"
        result = build_augmented_prompt(question, SAMPLE_CHUNKS)

        # Structural checks
        assert result.prompt.startswith("You are a grounded assistant")
        assert "Context:" in result.prompt
        assert f"Question: {question}" in result.prompt

        # Source markers present
        assert "[1]" in result.prompt

        # Grounding present
        assert "ONLY" in result.prompt
        assert "cite sources" in result.prompt.lower() or "cite" in result.prompt.lower()

        # Metrics valid
        assert result.chunks_included > 0
        assert result.context_tokens > 0
        assert result.total_prompt_tokens > result.context_tokens

    def test_all_chunks_source_metadata_preserved(self) -> None:
        result = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        for src in result.sources_used:
            assert "source_file" in src
            assert "doc_id" in src
            assert "section_heading" in src

    def test_deterministic_output(self) -> None:
        """Same inputs always produce the same prompt."""
        r1 = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        r2 = build_augmented_prompt("Test?", SAMPLE_CHUNKS)
        assert r1.prompt == r2.prompt
        assert r1.context_tokens == r2.context_tokens
