"""Evaluation module entry point re-exporting RAG evaluation components."""

from src.rag_evaluation import (
    DEFAULT_EVALUATION_CHUNKS,
    DEFAULT_OUTPUT_REPORT,
    DEFAULT_OUTPUT_SUMMARY,
    DEFAULT_TEST_SET,
    DEFAULT_TEST_SET_FILE,
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

__all__ = [
    "DEFAULT_EVALUATION_CHUNKS",
    "DEFAULT_TEST_SET",
    "test_set",
    "DEFAULT_OUTPUT_REPORT",
    "DEFAULT_OUTPUT_SUMMARY",
    "DEFAULT_TEST_SET_FILE",
    "retrieve_evaluation_context",
    "judge_expected_points",
    "judge_grounding",
    "check_citations",
    "diagnose_failure",
    "score_answer",
    "evaluate_test_set",
    "summarize_evaluations",
    "format_evaluation_summary_md",
    "run_rag_evaluation",
]
