"""Automated test suite for Hallucination Guardrails & Refusal Handling (Concept 3.41)."""

from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.guardrails import (
    DEFAULT_REFUSAL_MESSAGE,
    MIN_SUPPORTING_CHUNKS,
    MIN_TOP_SCORE,
    STATUS_ANSWERED,
    STATUS_REFUSED_WEAK_CONTEXT,
    GuardedAnswer,
    RetrievalQualityResult,
    evaluate_retrieval_quality,
    get_chunk_score,
    guarded_answer,
    retrieval_is_strong,
    retrieve,
    run_guardrails_demo,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strong_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "test_strong_001",
            "text": "Critical warning signs for dengue include persistent vomiting and mucosal bleeding.",
            "score": 0.85,
            "similarity_score": 0.85,
            "metadata": {
                "source": "who_dengue_guidance.pdf",
                "section": "2. Warning Signs",
                "chunk_id": "test_strong_001",
            },
        },
        {
            "chunk_id": "test_strong_002",
            "text": "Administer oral rehydration salts (ORS) immediately at primary triage points.",
            "score": 0.78,
            "similarity_score": 0.78,
            "metadata": {
                "source": "who_dengue_guidance.pdf",
                "section": "3. Immediate Interventions",
                "chunk_id": "test_strong_002",
            },
        },
    ]


@pytest.fixture
def weak_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "test_weak_001",
            "text": "General outpatient clinic administration schedule and facility registration.",
            "score": 0.42,
            "similarity_score": 0.42,
            "metadata": {
                "source": "clinic_admin.pdf",
                "section": "Operations",
                "chunk_id": "test_weak_001",
            },
        }
    ]


@pytest.fixture
def borderline_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "test_borderline_001",
            "text": "Vector control spraying schedule across municipal zones.",
            "score": 0.70,  # Below 0.72 threshold
            "similarity_score": 0.70,
            "metadata": {
                "source": "vector_control.pdf",
                "section": "Schedule",
                "chunk_id": "test_borderline_001",
            },
        }
    ]


# ---------------------------------------------------------------------------
# Task 1 & Task 3: Chunk Score Extraction & Weak Retrieval Detection Tests
# ---------------------------------------------------------------------------

class TestScoreExtractionAndStrength:
    def test_get_chunk_score_extracts_from_score_key(self):
        chunk = {"score": 0.88, "text": "sample text"}
        assert get_chunk_score(chunk) == 0.88

    def test_get_chunk_score_extracts_from_similarity_score_key(self):
        chunk = {"similarity_score": 0.76, "text": "sample text"}
        assert get_chunk_score(chunk) == 0.76

    def test_get_chunk_score_extracts_from_similarity_key(self):
        chunk = {"similarity": 0.81, "text": "sample text"}
        assert get_chunk_score(chunk) == 0.81

    def test_get_chunk_score_extracts_from_cosine_distance(self):
        chunk = {"cosine_distance": 0.15, "text": "sample text"}
        assert round(get_chunk_score(chunk), 2) == 0.85

    def test_get_chunk_score_returns_zero_on_missing_or_invalid(self):
        assert get_chunk_score({}) == 0.0
        assert get_chunk_score({"score": None}) == 0.0
        assert get_chunk_score({"score": "invalid"}) == 0.0

    def test_retrieval_is_strong_returns_false_for_none_and_empty(self):
        assert retrieval_is_strong(None) is False
        assert retrieval_is_strong([]) is False

    def test_retrieval_is_strong_returns_false_when_score_below_threshold(self, weak_chunks):
        assert retrieval_is_strong(weak_chunks, min_top_score=MIN_TOP_SCORE) is False

    def test_retrieval_is_strong_returns_false_for_borderline_score(self, borderline_chunks):
        # 0.70 is below default threshold 0.72
        assert retrieval_is_strong(borderline_chunks, min_top_score=MIN_TOP_SCORE) is False

    def test_retrieval_is_strong_returns_true_when_meeting_threshold(self, strong_chunks):
        assert retrieval_is_strong(strong_chunks, min_top_score=MIN_TOP_SCORE) is True

    def test_retrieval_is_strong_respects_custom_thresholds(self, borderline_chunks):
        # 0.70 meets custom threshold 0.65
        assert retrieval_is_strong(borderline_chunks, min_top_score=0.65) is True
        # 0.70 fails custom threshold 0.80
        assert retrieval_is_strong(borderline_chunks, min_top_score=0.80) is False

    def test_retrieval_is_strong_respects_min_supporting_chunks(self, strong_chunks):
        # strong_chunks has 2 chunks >= 0.72
        assert retrieval_is_strong(strong_chunks, min_supporting_chunks=2) is True
        assert retrieval_is_strong(strong_chunks, min_supporting_chunks=3) is False


class TestEvaluateRetrievalQuality:
    def test_evaluate_quality_empty_results(self):
        res = evaluate_retrieval_quality([])
        assert isinstance(res, RetrievalQualityResult)
        assert res.is_strong is False
        assert res.total_chunks == 0
        assert res.top_score == 0.0
        assert res.reason == "empty_results"

    def test_evaluate_quality_score_below_threshold(self, weak_chunks):
        res = evaluate_retrieval_quality(weak_chunks, min_top_score=0.72)
        assert res.is_strong is False
        assert res.total_chunks == 1
        assert res.top_score == 0.42
        assert res.strong_chunks_count == 0
        assert res.reason == "top_score_below_threshold"

    def test_evaluate_quality_insufficient_supporting_chunks(self, strong_chunks):
        res = evaluate_retrieval_quality(strong_chunks, min_supporting_chunks=3)
        assert res.is_strong is False
        assert res.reason == "insufficient_supporting_chunks"
        assert res.strong_chunks_count == 2

    def test_evaluate_quality_strong_context(self, strong_chunks):
        res = evaluate_retrieval_quality(strong_chunks, min_top_score=0.72, min_supporting_chunks=1)
        assert res.is_strong is True
        assert res.reason == "strong_context"
        assert res.top_score == 0.85
        assert res.strong_chunks_count == 2
        assert len(res.strong_chunks) == 2

    def test_evaluate_quality_dict_subscripting(self, strong_chunks):
        res = evaluate_retrieval_quality(strong_chunks)
        assert res["is_strong"] is True
        assert res["top_score"] == 0.85
        assert res["reason"] == "strong_context"


# ---------------------------------------------------------------------------
# Task 2: Safe Refusal Tests
# ---------------------------------------------------------------------------

class TestSafeRefusal:
    def test_guarded_answer_refuses_when_no_chunks_provided(self):
        res = guarded_answer("What is the treatment protocol?", chunks=[])
        assert res["status"] == STATUS_REFUSED_WEAK_CONTEXT
        assert res["answer"] == DEFAULT_REFUSAL_MESSAGE
        assert res["sources"] == []
        assert res.is_refused is True
        assert res["chunks_used"] == 0

    def test_guarded_answer_refuses_when_retrieval_is_weak(self, weak_chunks):
        res = guarded_answer("What is the treatment protocol?", chunks=weak_chunks)
        assert res["status"] == STATUS_REFUSED_WEAK_CONTEXT
        assert res["answer"] == DEFAULT_REFUSAL_MESSAGE
        assert res["sources"] == []
        assert res.is_refused is True

    def test_guarded_answer_custom_refusal_message(self, weak_chunks):
        custom_refusal = "Protocol information is unavailable for this condition."
        res = guarded_answer(
            "What is the treatment protocol?",
            chunks=weak_chunks,
            refusal_message=custom_refusal,
        )
        assert res["status"] == STATUS_REFUSED_WEAK_CONTEXT
        assert res["answer"] == custom_refusal

    def test_guarded_answer_refuses_out_of_corpus_query(self):
        # Out-of-corpus query returns no relevant chunks
        res = guarded_answer("What is the refund policy for a product not in this corpus?")
        assert res["status"] == STATUS_REFUSED_WEAK_CONTEXT
        assert res["answer"] == DEFAULT_REFUSAL_MESSAGE
        assert res["sources"] == []


# ---------------------------------------------------------------------------
# Task 4: Confident Answer Tests
# ---------------------------------------------------------------------------

class TestConfidentAnswerGeneration:
    def test_guarded_answer_answers_with_strong_context(self, strong_chunks):
        res = guarded_answer(
            "What are the critical warning signs for dengue fever?",
            chunks=strong_chunks,
        )
        assert res["status"] == STATUS_ANSWERED
        assert res["is_refused"] is False
        assert len(res["sources"]) > 0
        assert "vomiting" in res["answer"].lower()
        assert res["top_score"] == 0.85

    def test_guarded_answer_project_submission_benchmark_case(self):
        # Prompt reference test case: What evidence is required for project submission?
        res = guarded_answer("What evidence is required for project submission?")
        assert res["status"] == STATUS_ANSWERED
        assert res["is_refused"] is False
        assert "project submission" in res["answer"].lower()
        assert len(res["sources"]) >= 1

    def test_guarded_answer_clinical_corpus_retrieval(self):
        # Querying real ingested corpus
        res = guarded_answer("What are the critical warning signs for dengue fever?")
        assert res["status"] == STATUS_ANSWERED
        assert res["is_refused"] is False
        assert "vomiting" in res["answer"].lower()

    def test_guarded_answer_with_custom_llm_fn(self, strong_chunks):
        def mock_llm(prompt: str) -> str:
            return "Custom LLM response confirming severe dengue danger signs."

        res = guarded_answer(
            "What are the danger signs?",
            chunks=strong_chunks,
            llm_fn=mock_llm,
        )
        assert res["status"] == STATUS_ANSWERED
        assert "Custom LLM response" in res["answer"]

    def test_guarded_answer_with_custom_retriever_fn(self, strong_chunks):
        called = False

        def mock_retriever(q: str, k: int):
            nonlocal called
            called = True
            return strong_chunks

        res = guarded_answer(
            "What is the protocol?",
            retriever_fn=mock_retriever,
        )
        assert called is True
        assert res["status"] == STATUS_ANSWERED


# ---------------------------------------------------------------------------
# GuardedAnswer Data Structure & API Contract Tests
# ---------------------------------------------------------------------------

class TestGuardedAnswerInterface:
    def test_dict_subscripting_and_attribute_access(self, strong_chunks):
        res = guarded_answer("Dengue warning signs?", chunks=strong_chunks)

        # Dictionary subscripting
        assert res["status"] == STATUS_ANSWERED
        assert isinstance(res["answer"], str)
        assert isinstance(res["sources"], list)
        assert res["top_score"] == 0.85

        # Attribute access
        assert res.status == STATUS_ANSWERED
        assert res.answer == res["answer"]
        assert res.sources == res["sources"]
        assert res.top_score == 0.85
        assert res.is_refused is False

    def test_dict_unpacking_support(self, strong_chunks):
        res = guarded_answer("Dengue warning signs?", chunks=strong_chunks)
        unpacked = {**res, "audit_flag": True}

        assert unpacked["status"] == STATUS_ANSWERED
        assert unpacked["audit_flag"] is True
        assert "answer" in unpacked

    def test_to_dict_returns_plain_dict(self, strong_chunks):
        res = guarded_answer("Dengue warning signs?", chunks=strong_chunks)
        d = res.to_dict()
        assert isinstance(d, dict)
        assert d["status"] == STATUS_ANSWERED

    def test_empty_question_raises_value_error(self):
        with pytest.raises(ValueError, match="Question must be a non-empty string"):
            guarded_answer("")

        with pytest.raises(ValueError, match="Question must be a non-empty string"):
            guarded_answer("   ")


# ---------------------------------------------------------------------------
# Task 5: End-to-End Demo & Report Generation Tests
# ---------------------------------------------------------------------------

class TestDemoExecutionAndArtifacts:
    def test_run_guardrails_demo_generates_markdown_artifact(self, tmp_path):
        out_file = tmp_path / "test_guardrails_sample.md"
        demo_res = run_guardrails_demo(output_file=out_file)

        assert out_file.is_file()
        content = out_file.read_text(encoding="utf-8")

        # Verify key sections and prompt-specified queries
        assert "# Hallucination Guardrails & Refusal Handling — Execution Report" in content
        assert "What evidence is required for project submission?" in content
        assert "What is the refund policy for a product not in this corpus?" in content
        assert "refused_weak_context" in content
        assert "answered" in content
        assert "MIN_TOP_SCORE = 0.72" in content
        assert "I don't have enough reliable context to answer that." in content

        # Verify demo result object
        assert demo_res["answer_case"]["status"] == STATUS_ANSWERED
        assert demo_res["refusal_case"]["status"] == STATUS_REFUSED_WEAK_CONTEXT
        assert demo_res["borderline_case"]["status"] == STATUS_REFUSED_WEAK_CONTEXT
