"""Citation module entry point re-exporting source citation components."""

from src.source_citation import (
    FALLBACK_NO_CONTEXT,
    AssembledContextResult,
    CitedAnswer,
    answer_with_citations,
    assemble_context,
    build_citation_map,
    build_cited_prompt,
    call_llm,
    detect_fabricated_citations,
    extract_citations,
    print_citation_verification,
    retrieve,
    run_source_citation_demo,
    sanitize_answer_citations,
    verify_all_citations,
    verify_citation,
)

__all__ = [
    "FALLBACK_NO_CONTEXT",
    "AssembledContextResult",
    "CitedAnswer",
    "build_citation_map",
    "assemble_context",
    "build_cited_prompt",
    "retrieve",
    "call_llm",
    "answer_with_citations",
    "extract_citations",
    "detect_fabricated_citations",
    "verify_citation",
    "verify_all_citations",
    "sanitize_answer_citations",
    "print_citation_verification",
    "run_source_citation_demo",
]
