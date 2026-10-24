"""Automated unit and integration test suite for RAG Evaluation & Answer Quality Scoring (Concept 3.43)."""

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.rag_evaluation import (
    DEFAULT_EVALUATION_CHUNKS,
    DEFAULT_TEST_SET,
    check_citations,
    diagnose_failure,
    evaluate_test_set,
    format_evaluation_summary_md,
    judge_expected_points,
    judge_grounding,
    retrieve_evaluation_context,
    run_rag_evaluation,
    score_answer,
    summarize_evaluations,
    test_set,
)
from src.source_citation import FALLBACK_NO_CONTEXT


# ---------------------------------------------------------------------------
# Task 1: Test Set Preparation & Schema Tests
# ---------------------------------------------------------------------------

class TestTestSetPreparation:
    def test_default_test_set_structure(self):
        assert isinstance(DEFAULT_TEST_SET, list)
        assert len(DEFAULT_TEST_SET) >= 6
        assert test_set is DEFAULT_TEST_SET

        for item in DEFAULT_TEST_SET:
            assert "question" in item
            assert "expected_points" in item
            assert "expected_sources" in item
            assert isinstance(item["question"], str) and item["question"].strip()
            assert isinstance(item["expected_points"], (list, tuple))
            assert isinstance(item["expected_sources"], (set, list, tuple))

    def test_rubric_and_guardrails_examples_match_specification(self):
        rubric_items = [
            it for it in DEFAULT_TEST_SET
            if "project submission" in it["question"].lower()
        ]
        assert len(rubric_items) == 1
        r_item = rubric_items[0]
        assert "PR link" in r_item["expected_points"]
        assert "sample output" in r_item["expected_points"]
        assert "video explanation" in r_item["expected_points"]
        assert "submission-rubric.md" in {Path(s).name for s in r_item["expected_sources"]}

        guardrail_items = [
            it for it in DEFAULT_TEST_SET
            if "context is missing" in it["question"].lower()
        ]
        assert len(guardrail_items) == 1
        g_item = guardrail_items[0]
        assert "refuse" in g_item["expected_points"]
        assert "say not enough information" in g_item["expected_points"]
        assert "guardrails.md" in {Path(s).name for s in g_item["expected_sources"]}

    def test_test_set_serialization_round_trip(self, tmp_path: Path):
        test_file = tmp_path / "test_set.json"
        serializable = []
        for item in DEFAULT_TEST_SET:
            c_item = dict(item)
            c_item["expected_sources"] = list(c_item["expected_sources"])
            serializable.append(c_item)

        test_file.write_text(json.dumps(serializable), encoding="utf-8")
        loaded = json.loads(test_file.read_text(encoding="utf-8"))

        assert len(loaded) == len(DEFAULT_TEST_SET)
        assert loaded[0]["question"] == DEFAULT_TEST_SET[0]["question"]


# ---------------------------------------------------------------------------
# Task 2: Correctness Judging Tests
# ---------------------------------------------------------------------------

class TestCorrectnessJudging:
    def test_judge_all_expected_points_present(self):
        answer = "The submission needs a PR link, sample output logs, and a video explanation walkthrough."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        assert score == 1.0

    def test_judge_partial_expected_points(self):
        answer = "The submission includes a PR link and sample output."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        assert score == pytest.approx(0.6667, abs=1e-3)

    def test_judge_no_expected_points(self):
        answer = "There are no submission requirements mentioned."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        assert score == 0.0

    def test_judge_case_insensitive_matching(self):
        answer = "Ensure a pr link, SAMPLE OUTPUT, and Video Explanation are provided."
        expected = ["PR link", "sample output", "video explanation"]
        score = judge_expected_points(answer, expected)
        assert score == 1.0

    def test_judge_refusal_semantics(self):
        answer = "I don't have enough reliable context to answer that."
        expected = ["refuse", "say not enough information"]
        score = judge_expected_points(answer, expected)
        assert score == 1.0

        fallback_answer = "I don't have enough information in the provided context."
        score2 = judge_expected_points(fallback_answer, expected)
        assert score2 == 1.0

    def test_judge_empty_inputs(self):
        assert judge_expected_points("Any answer", []) == 1.0
        assert judge_expected_points("", ["PR link"]) == 0.0
        assert judge_expected_points("   ", ["PR link"]) == 0.0


# ---------------------------------------------------------------------------
# Task 2: Grounding Judging Tests
# ---------------------------------------------------------------------------

class TestGroundingJudging:
    def test_grounded_answer_from_retrieved_sources_dict(self):
        answer = "Critical warning signs include persistent vomiting and severe abdominal pain."
        sources = {
            "[1]": {
                "source": "who_dengue_guidance.md",
                "text": "Critical warning signs: Persistent vomiting and severe abdominal pain.",
            }
        }
        score = judge_grounding(answer, retrieved_sources=sources)
        assert score == 1.0

    def test_safe_refusal_is_fully_grounded(self):
        score1 = judge_grounding(FALLBACK_NO_CONTEXT, retrieved_sources={})
        assert score1 == 1.0

        score2 = judge_grounding("I don't have enough reliable context to answer that.", retrieved_sources={})
        assert score2 == 1.0

    def test_unsupported_hallucination_penalized(self):
        answer = "Dengue can be treated using experimental nano-robotic surgery and cosmic radiation therapy."
        sources = {
            "[1]": {
                "source": "who_dengue_guidance.md",
                "text": "Administer oral rehydration salts and monitor fever with paracetamol.",
            }
        }
        score = judge_grounding(answer, retrieved_sources=sources)
        assert score == 0.0

    def test_empty_context_with_factual_claims_fails_grounding(self):
        answer = "Project submission requires a PR link and video explanation."
        score = judge_grounding(answer, retrieved_sources={})
        assert score == 0.0

    def test_mandatory_disclaimer_not_penalized(self):
        answer = (
            "Persistent vomiting requires urgent referral.\n\n"
            "Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. "
            "Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC)."
        )
        sources = {
            "[1]": {
                "source": "who_dengue_guidance.md",
                "text": "Persistent vomiting requires urgent hospital referral.",
            }
        }
        score = judge_grounding(answer, retrieved_sources=sources)
        assert score == 1.0

    def test_list_and_string_context_formats(self):
        chunk_list = [{"text": "Target chlorine level is 0.5 to 1.0 mg/L at distribution."}]
        score = judge_grounding("Target chlorine level is 0.5 to 1.0 mg/L.", retrieved_sources=chunk_list)
        assert score == 1.0

        raw_str = "Clean patient fingertip with 70% alcohol swab."
        score2 = judge_grounding("Clean fingertip with alcohol swab.", retrieved_sources=raw_str)
        assert score2 == 1.0


# ---------------------------------------------------------------------------
# Task 3: Citation Accuracy Checking Tests
# ---------------------------------------------------------------------------

class TestCitationAccuracy:
    def test_exact_citation_match(self):
        citations = {
            "[1]": {"source": "submission-rubric.md", "chunk_id": "c1"},
        }
        expected = {"submission-rubric.md"}
        assert check_citations(citations, expected) == 1.0

    def test_path_normalization_matches_basename(self):
        citations = {
            "[1]": {"source": "data/sample_advisories/who_dengue_clinical_guidance.md"},
        }
        expected = {"who_dengue_clinical_guidance.md"}
        assert check_citations(citations, expected) == 1.0

    def test_missing_citations_when_expected(self):
        assert check_citations({}, {"submission-rubric.md"}) == 0.0

    def test_extra_unwanted_citations_penalized(self):
        citations = {
            "[1]": {"source": "submission-rubric.md"},
            "[2]": {"source": "unrelated_policy.pdf"},
        }
        expected = {"submission-rubric.md"}
        score = check_citations(citations, expected)
        assert score == 0.5

    def test_empty_expected_sources_with_empty_citations(self):
        # When an ungrounded or out-of-domain query should refuse without citations
        assert check_citations({}, set()) == 1.0

    def test_empty_expected_sources_with_spurious_citations(self):
        citations = {"[1]": {"source": "random_doc.pdf"}}
        assert check_citations(citations, set()) == 0.0


# ---------------------------------------------------------------------------
# Task 4: Failure Diagnosis & Summarization Tests
# ---------------------------------------------------------------------------

class TestFailureDiagnosisAndSummarization:
    def test_diagnose_weak_retrieval(self):
        scored = {
            "correctness": 0.0,
            "grounding": 1.0,
            "citation_accuracy": 0.0,
            "citations": {},
        }
        example = {"expected_sources": {"solar_standards.pdf"}}
        diag = diagnose_failure(scored, example)
        assert diag["likely_cause"] == "weak_retrieval"
        assert "retrieval failed" in diag["explanation"].lower()

    def test_diagnose_missing_metadata(self):
        scored = {
            "correctness": 0.5,
            "grounding": 1.0,
            "citation_accuracy": 0.0,
            "citations": {"[1]": {"source": "Unknown Source"}},
        }
        diag = diagnose_failure(scored)
        assert diag["likely_cause"] == "missing_metadata"
        assert "metadata" in diag["explanation"].lower()

    def test_diagnose_bad_citations(self):
        scored = {
            "correctness": 1.0,
            "grounding": 1.0,
            "citation_accuracy": 0.5,
            "citations": {"[1]": {"source": "wrong_doc.pdf"}},
        }
        diag = diagnose_failure(scored)
        assert diag["likely_cause"] == "bad_citations"

    def test_diagnose_unsupported_claims(self):
        scored = {
            "correctness": 1.0,
            "grounding": 0.2,
            "citation_accuracy": 1.0,
            "citations": {"[1]": {"source": "correct_doc.pdf"}},
        }
        diag = diagnose_failure(scored)
        assert diag["likely_cause"] == "unsupported_claims"

    def test_diagnose_poor_prompt_instructions(self):
        scored = {
            "correctness": 0.33,
            "grounding": 1.0,
            "citation_accuracy": 1.0,
            "citations": {"[1]": {"source": "correct_doc.pdf"}},
        }
        diag = diagnose_failure(scored)
        assert diag["likely_cause"] == "poor_prompt_instructions"

    def test_diagnose_no_failure(self):
        scored = {
            "correctness": 1.0,
            "grounding": 1.0,
            "citation_accuracy": 1.0,
            "citations": {"[1]": {"source": "doc.md"}},
        }
        diag = diagnose_failure(scored)
        assert diag["likely_cause"] == "none"
        assert diag["failed_dimensions"] == []

    def test_summarize_evaluations_aggregates_metrics(self):
        rows = [
            {"correctness": 1.0, "grounding": 1.0, "citation_accuracy": 1.0, "citations": {"[1]": {"source": "a.md"}}},
            {"correctness": 0.5, "grounding": 1.0, "citation_accuracy": 1.0, "citations": {"[1]": {"source": "b.md"}}},
            {"correctness": 0.0, "grounding": 1.0, "citation_accuracy": 0.0, "citations": {}},
        ]
        summary = summarize_evaluations(rows)

        assert summary["questions"] == 3
        assert summary["avg_correctness"] == pytest.approx(0.5, abs=1e-3)
        assert summary["avg_grounding"] == 1.0
        assert summary["avg_citation_accuracy"] == pytest.approx(0.6667, abs=1e-3)
        assert len(summary["failures"]) == 2
        assert summary["passed_count"] == 1
        assert summary["failure_count"] == 2

    def test_format_evaluation_summary_md_contains_key_sections(self):
        summary = {
            "questions": 3,
            "avg_correctness": 0.8,
            "avg_grounding": 1.0,
            "avg_citation_accuracy": 0.9,
            "overall_quality": 0.9,
            "passed_count": 2,
            "failure_count": 1,
            "failures": [
                {
                    "question": "Sample fail question?",
                    "correctness": 0.5,
                    "grounding": 1.0,
                    "citation_accuracy": 1.0,
                    "likely_cause": "poor_prompt_instructions",
                    "recommendation": "Refine instructions.",
                }
            ],
        }
        md = format_evaluation_summary_md(summary)
        assert "# RAG Evaluation & Answer Quality Scoring Report" in md
        assert "| **Evaluated Questions** | `3` |" in md
        assert "## 2. Dimension Analysis & Weakest Area" in md
        assert "## 4. Notable Failures & Root Cause Analysis" in md
        assert "poor_prompt_instructions" in md


# ---------------------------------------------------------------------------
# Task 5: End-to-End Score Answer & Runner Tests
# ---------------------------------------------------------------------------

class TestEndToEndEvaluation:
    def test_score_answer_custom_callable(self):
        def mock_rag(q: str):
            return {
                "answer": "Submission requires a PR link, sample output, and video explanation [1].",
                "citations": {
                    "[1]": {
                        "source": "submission-rubric.md",
                        "text": "Submission requires PR link, sample output, and video explanation.",
                    }
                },
            }

        example = {
            "question": "What evidence is required for project submission?",
            "expected_points": ["PR link", "sample output", "video explanation"],
            "expected_sources": {"submission-rubric.md"},
        }

        result = score_answer(example, answer_fn=mock_rag)
        assert result["correctness"] == 1.0
        assert result["grounding"] == 1.0
        assert result["citation_accuracy"] == 1.0
        assert "[1]" in result["citations"]
        assert "likely_cause" not in result

    def test_score_answer_default_rag_execution(self):
        example = DEFAULT_TEST_SET[0]  # project submission
        result = score_answer(example)
        assert result["correctness"] == 1.0
        assert result["grounding"] == 1.0
        assert result["citation_accuracy"] == 1.0
        assert "submission-rubric.md" in str(result["citations"])

    def test_evaluate_test_set_runs_all_items(self):
        subset = DEFAULT_TEST_SET[:3]
        results = evaluate_test_set(subset)
        assert len(results) == 3
        for res in results:
            assert "question" in res
            assert "answer" in res
            assert "correctness" in res
            assert "grounding" in res
            assert "citation_accuracy" in res

    def test_run_rag_evaluation_generates_all_artifacts(self, tmp_path: Path):
        report_file = tmp_path / "report.json"
        summary_file = tmp_path / "summary.md"
        test_set_file = tmp_path / "test_set.json"

        summary = run_rag_evaluation(
            output_report_path=report_file,
            output_summary_path=summary_file,
            test_set_export_path=test_set_file,
        )

        assert report_file.is_file()
        assert summary_file.is_file()
        assert test_set_file.is_file()

        # Check JSON report format
        payload = json.loads(report_file.read_text(encoding="utf-8"))
        assert "summary" in payload
        assert "rows" in payload
        assert len(payload["rows"]) == len(DEFAULT_TEST_SET)
        assert payload["summary"]["questions"] == len(DEFAULT_TEST_SET)
        assert payload["summary"]["passed_count"] >= 1
