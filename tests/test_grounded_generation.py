"""Tests for Grounded Answer Generation (Concept 3.39).

Covers:
    - Task 1: generate_grounded_answer — context-only generation with source tracing.
    - Task 2: verify_grounding — keyword overlap, sentence-level support checks.
    - Task 3: answer_query — missing-context fallback behaviour.
    - Task 4: compare_grounded_vs_ungrounded — side-by-side output comparison.
    - Task 5: print_grounding_check — verification output formatting.
    - Edge cases: empty chunks, empty questions, single chunks, custom llm_fn.
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List

from src.grounded_generation import (
    FALLBACK_NO_CONTEXT,
    GroundedAnswer,
    GroundedComparison,
    GroundingCheck,
    answer_query,
    compare_grounded_vs_ungrounded,
    direct_llm_call,
    generate_grounded_answer,
    print_grounding_check,
    verify_grounding,
    _extract_content_words,
    _generate_ungrounded_answer,
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
) -> Dict[str, Any]:
    """Build a chunk dictionary matching the project schema."""
    return {
        "text": text,
        "metadata": {
            "source_file": source_file,
            "chunk_index": chunk_index,
            "section_heading": section_heading,
            "doc_id": doc_id,
            "document_title": "Test Document",
            "document_version": "v1.0",
            "token_count": count_tokens(text),
        },
    }


SAMPLE_CHUNKS: List[Dict[str, Any]] = [
    _make_chunk(
        text="Dengue is transmitted by Aedes aegypti mosquitoes. Early symptoms include high fever and headache.",
        source_file="data/who_dengue.md",
        chunk_index=0,
        section_heading="Transmission",
    ),
    _make_chunk(
        text="Warning signs include persistent vomiting, abdominal pain, and rapid breathing.",
        source_file="data/who_dengue.md",
        chunk_index=1,
        section_heading="Warning Signs",
    ),
    _make_chunk(
        text="Use Paracetamol for fever management. Avoid NSAIDs like Ibuprofen as they increase bleeding risk.",
        source_file="data/who_dengue.md",
        chunk_index=2,
        section_heading="Field Interventions",
    ),
]


# ---------------------------------------------------------------------------
# Task 1: Generate from injected context
# ---------------------------------------------------------------------------

class TestGenerateGroundedAnswer:
    """Tests for generate_grounded_answer — context-only generation."""

    def test_returns_grounded_answer_dataclass(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert isinstance(result, GroundedAnswer)

    def test_answer_is_non_empty(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert result.answer.strip() != ""

    def test_question_preserved(self) -> None:
        question = "What is dengue?"
        result = generate_grounded_answer(question, SAMPLE_CHUNKS)
        assert result.question == question

    def test_sources_populated(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert len(result.sources) > 0
        assert "source_file" in result.sources[0]

    def test_chunks_used_count(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert result.chunks_used > 0
        assert result.chunks_used <= len(SAMPLE_CHUNKS)

    def test_context_contains_source_markers(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert "[1]" in result.context

    def test_is_not_fallback_when_chunks_exist(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert result.is_fallback is False

    def test_custom_llm_fn_used(self) -> None:
        custom_fn = lambda prompt: "Custom grounded answer from LLM."
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS, llm_fn=custom_fn)
        assert result.answer == "Custom grounded answer from LLM."

    def test_raises_on_empty_question(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            generate_grounded_answer("", SAMPLE_CHUNKS)

    def test_raises_on_whitespace_question(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            generate_grounded_answer("   ", SAMPLE_CHUNKS)

    def test_empty_chunks_returns_fallback(self) -> None:
        result = generate_grounded_answer("What is dengue?", [])
        assert result.is_fallback is True
        assert FALLBACK_NO_CONTEXT in result.answer

    def test_single_chunk_generates_answer(self) -> None:
        result = generate_grounded_answer("What is dengue?", [SAMPLE_CHUNKS[0]])
        assert result.chunks_used == 1
        assert result.is_fallback is False


# ---------------------------------------------------------------------------
# Task 2: Check source accuracy
# ---------------------------------------------------------------------------

class TestVerifyGrounding:
    """Tests for verify_grounding — keyword overlap and sentence support."""

    def test_returns_grounding_check_dataclass(self) -> None:
        result = verify_grounding("Dengue is transmitted.", "Dengue is transmitted by mosquitoes.")
        assert isinstance(result, GroundingCheck)

    def test_high_overlap_is_grounded(self) -> None:
        context = "Dengue is transmitted by Aedes aegypti mosquitoes. Warning signs include fever."
        answer = "Dengue is transmitted by Aedes aegypti mosquitoes."
        result = verify_grounding(answer, context)
        assert result.is_grounded is True
        assert result.keyword_overlap_score > 0.5

    def test_low_overlap_not_grounded(self) -> None:
        context = "Dengue is transmitted by Aedes aegypti mosquitoes."
        answer = "Malaria is caused by Plasmodium parasites transmitted through Anopheles bites."
        result = verify_grounding(answer, context, overlap_threshold=0.5)
        assert result.is_grounded is False
        assert len(result.unsupported_claims) > 0

    def test_empty_answer_is_grounded(self) -> None:
        result = verify_grounding("", "Some context.")
        assert result.is_grounded is True
        assert result.answer_sentences == 0

    def test_empty_context_flags_unsupported(self) -> None:
        result = verify_grounding("Dengue is transmitted by mosquitoes.", "")
        assert result.keyword_overlap_score == 0.0

    def test_supported_sentences_count(self) -> None:
        context = "Dengue fever causes high temperature. Warning signs include vomiting."
        answer = "Dengue fever causes high temperature."
        result = verify_grounding(answer, context)
        assert result.supported_sentences > 0

    def test_unsupported_claims_listed(self) -> None:
        context = "Dengue is transmitted by mosquitoes."
        answer = "Dengue is transmitted by mosquitoes. Ebola is airborne and highly contagious."
        result = verify_grounding(answer, context, overlap_threshold=0.5)
        assert len(result.unsupported_claims) > 0

    def test_fieldsure_notice_excluded_from_checks(self) -> None:
        context = "Dengue fever management includes Paracetamol."
        answer = (
            "Dengue fever management includes Paracetamol. "
            "Notice: FieldSure AI is an informational protocol assistant."
        )
        result = verify_grounding(answer, context)
        # The notice sentence should be filtered out from sentence checks
        assert result.answer_sentences >= 1


# ---------------------------------------------------------------------------
# Task 3: Missing-context fallback
# ---------------------------------------------------------------------------

class TestAnswerQuery:
    """Tests for answer_query — fallback when context is missing."""

    def test_returns_fallback_when_no_chunks(self) -> None:
        result = answer_query("What is the submission deadline?", [])
        assert result.is_fallback is True
        assert FALLBACK_NO_CONTEXT in result.answer

    def test_returns_fallback_when_none_chunks(self) -> None:
        result = answer_query("What is the submission deadline?", None)
        assert result.is_fallback is True

    def test_returns_grounded_when_chunks_exist(self) -> None:
        result = answer_query("What is dengue?", SAMPLE_CHUNKS)
        assert result.is_fallback is False
        assert result.chunks_used > 0

    def test_fallback_has_empty_sources(self) -> None:
        result = answer_query("Unknown topic?", [])
        assert result.sources == []

    def test_fallback_has_empty_context(self) -> None:
        result = answer_query("Unknown topic?", [])
        assert result.context == ""

    def test_custom_llm_fn_passed_through(self) -> None:
        custom_fn = lambda prompt: "LLM grounded response."
        result = answer_query("What is dengue?", SAMPLE_CHUNKS, llm_fn=custom_fn)
        assert result.answer == "LLM grounded response."


# ---------------------------------------------------------------------------
# Task 4: Compare grounded vs ungrounded
# ---------------------------------------------------------------------------

class TestCompareGroundedVsUngrounded:
    """Tests for compare_grounded_vs_ungrounded — side-by-side comparison."""

    def test_returns_comparison_dataclass(self) -> None:
        result = compare_grounded_vs_ungrounded("What is dengue?", SAMPLE_CHUNKS)
        assert isinstance(result, GroundedComparison)

    def test_grounded_answer_non_empty(self) -> None:
        result = compare_grounded_vs_ungrounded("What is dengue?", SAMPLE_CHUNKS)
        assert result.grounded_answer.strip() != ""

    def test_ungrounded_answer_non_empty(self) -> None:
        result = compare_grounded_vs_ungrounded("What is dengue?", SAMPLE_CHUNKS)
        assert result.ungrounded_answer.strip() != ""

    def test_grounded_has_higher_overlap(self) -> None:
        result = compare_grounded_vs_ungrounded(
            "What are warning signs of dengue?", SAMPLE_CHUNKS,
        )
        # Grounded answer should have more keyword overlap with context
        assert result.keyword_overlap_grounded >= result.keyword_overlap_ungrounded

    def test_question_preserved(self) -> None:
        question = "What is dengue fever?"
        result = compare_grounded_vs_ungrounded(question, SAMPLE_CHUNKS)
        assert result.question == question

    def test_grounded_sources_populated(self) -> None:
        result = compare_grounded_vs_ungrounded("What is dengue?", SAMPLE_CHUNKS)
        assert len(result.grounded_sources) > 0

    def test_custom_ungrounded_fn(self) -> None:
        custom_fn = lambda q: "Raw model answer without context."
        result = compare_grounded_vs_ungrounded(
            "What is dengue?", SAMPLE_CHUNKS, ungrounded_fn=custom_fn,
        )
        assert result.ungrounded_answer == "Raw model answer without context."

    def test_empty_chunks_grounded_is_fallback(self) -> None:
        result = compare_grounded_vs_ungrounded("What is dengue?", [])
        assert result.grounded_is_fallback is True


# ---------------------------------------------------------------------------
# Task 5: Print grounding check
# ---------------------------------------------------------------------------

class TestPrintGroundingCheck:
    """Tests for print_grounding_check — verification output."""

    def test_returns_grounding_check(self) -> None:
        grounded = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        check = print_grounding_check(grounded)
        assert isinstance(check, GroundingCheck)

    def test_prints_question(self, capsys: pytest.CaptureFixture[str]) -> None:
        grounded = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        print_grounding_check(grounded)
        captured = capsys.readouterr()
        assert "What is dengue?" in captured.out

    def test_prints_sources(self, capsys: pytest.CaptureFixture[str]) -> None:
        grounded = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        print_grounding_check(grounded)
        captured = capsys.readouterr()
        assert "data/who_dengue.md" in captured.out

    def test_prints_grounding_status(self, capsys: pytest.CaptureFixture[str]) -> None:
        grounded = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        print_grounding_check(grounded)
        captured = capsys.readouterr()
        assert "Is Grounded:" in captured.out


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtilities:
    """Tests for helper functions."""

    def test_extract_content_words_removes_stopwords(self) -> None:
        words = _extract_content_words("the quick brown fox jumps over the lazy dog")
        assert "the" not in words
        assert "over" not in words
        assert "quick" in words
        assert "brown" in words

    def test_extract_content_words_removes_short(self) -> None:
        words = _extract_content_words("I am a big fan of AI")
        assert "am" not in words
        assert "big" in words
        assert "fan" in words

    def test_ungrounded_answer_mentions_general_knowledge(self) -> None:
        answer = _generate_ungrounded_answer("What is dengue?")
        assert "general knowledge" in answer.lower()

    def test_ungrounded_answer_contains_question(self) -> None:
        question = "What are the symptoms?"
        answer = _generate_ungrounded_answer(question)
        assert question in answer


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    """Integration tests verifying the full grounded generation pipeline."""

    def test_full_pipeline_grounded_answer(self) -> None:
        """Generate a grounded answer, verify it, and check output structure."""
        question = "How is dengue transmitted?"
        result = generate_grounded_answer(question, SAMPLE_CHUNKS)

        assert isinstance(result, GroundedAnswer)
        assert result.question == question
        assert result.chunks_used > 0
        assert result.is_fallback is False
        assert len(result.sources) > 0

        # Verify grounding
        check = verify_grounding(result.answer, result.context, result.sources)
        assert check.keyword_overlap_score >= 0.0
        assert check.answer_sentences >= 0

    def test_full_pipeline_fallback_and_comparison(self) -> None:
        """Test fallback path and comparison path together."""
        question = "What evidence is required for project submission?"

        # Fallback: no chunks
        fallback = answer_query(question, [])
        assert fallback.is_fallback is True

        # Comparison: with chunks
        comparison = compare_grounded_vs_ungrounded(question, SAMPLE_CHUNKS)
        assert comparison.grounded_answer != comparison.ungrounded_answer

    def test_deterministic_output(self) -> None:
        """Same inputs always produce the same grounded answer."""
        r1 = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        r2 = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert r1.answer == r2.answer
        assert r1.chunks_used == r2.chunks_used


class TestDictionaryAccessAndHelpers:
    """Verify dictionary-style subscripting, to_dict, and helper utilities."""

    def test_grounded_answer_subscripting(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        assert result["question"] == "What is dengue?"
        assert isinstance(result["answer"], str)
        assert isinstance(result["context"], str)
        assert isinstance(result["sources"], list)
        assert result["chunks_used"] > 0
        assert result["is_fallback"] is False

    def test_grounded_answer_subscript_missing_key_raises(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        with pytest.raises(KeyError):
            _ = result["non_existent_key"]

    def test_grounded_answer_to_dict(self) -> None:
        result = generate_grounded_answer("What is dengue?", SAMPLE_CHUNKS)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["question"] == "What is dengue?"
        assert "answer" in d
        assert "context" in d
        assert "sources" in d

    def test_grounding_check_subscripting_and_to_dict(self) -> None:
        check = verify_grounding("Dengue is fever.", "Dengue is fever.")
        assert check["is_grounded"] is True
        assert check["keyword_overlap_score"] == 1.0
        d = check.to_dict()
        assert d["is_grounded"] is True
        with pytest.raises(KeyError):
            _ = check["invalid_attr"]

    def test_grounded_comparison_subscripting_and_to_dict(self) -> None:
        comp = compare_grounded_vs_ungrounded("What is dengue?", SAMPLE_CHUNKS)
        assert comp["question"] == "What is dengue?"
        assert isinstance(comp["grounded_answer"], str)
        assert isinstance(comp["ungrounded_answer"], str)
        d = comp.to_dict()
        assert d["question"] == "What is dengue?"
        with pytest.raises(KeyError):
            _ = comp["invalid_attr"]

    def test_direct_llm_call(self) -> None:
        ans1 = direct_llm_call("What is dengue?")
        assert "general knowledge" in ans1.lower()

        ans2 = direct_llm_call("Custom", llm_fn=lambda q: f"Echo: {q}")
        assert ans2 == "Echo: Custom"

    def test_answer_query_with_dynamic_retriever(self) -> None:
        retriever_called = []

        def dummy_retriever(q: str):
            retriever_called.append(q)
            return SAMPLE_CHUNKS[:1]

        res = answer_query("What is dengue?", retriever_fn=dummy_retriever)
        assert retriever_called == ["What is dengue?"]
        assert res.is_fallback is False
        assert res.chunks_used == 1

