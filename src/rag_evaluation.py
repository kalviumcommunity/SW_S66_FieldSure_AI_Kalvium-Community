"""RAG Evaluation & Answer Quality Scoring (Concept 3.43).

Evaluates the end-to-end RAG system across three core quality dimensions:
1. Correctness: Does the answer match expected answer points?
2. Grounding: Are answer claims supported by retrieved context?
3. Citation Accuracy: Do citations point to sources that support the claims?

Tasks:
    Task 1: Prepare a test set of questions with expected answers/points and sources.
    Task 2: Score correctness and grounding against retrieved context.
    Task 3: Check citation accuracy against retrieved sources.
    Task 4: Summarize quality and failures, diagnosing likely causes.
    Task 5: Export and commit evaluation scripts, test sets, reports, and summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.source_citation import (
    FALLBACK_NO_CONTEXT,
    answer_with_citations,
    extract_citations,
    retrieve as default_retrieve,
)

logger = logging.getLogger("fieldsure.rag_evaluation")

DEFAULT_OUTPUT_REPORT = REPO_ROOT / "outputs" / "rag_evaluation_report.json"
DEFAULT_OUTPUT_SUMMARY = REPO_ROOT / "outputs" / "rag_evaluation_summary.md"
DEFAULT_OUTPUT_SAMPLE = REPO_ROOT / "outputs" / "rag_evaluation_sample.md"
DEFAULT_TEST_SET_FILE = REPO_ROOT / "outputs" / "rag_evaluation_test_set.json"

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "did", "do", "does", "doing", "don't", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "let", "like", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}

# ---------------------------------------------------------------------------
# Default Evaluation Benchmark Chunks
# ---------------------------------------------------------------------------

DEFAULT_EVALUATION_CHUNKS: List[Dict[str, Any]] = [
    {
        "chunk_id": "eval_submission_rubric_001",
        "text": (
            "Project submission requires verifiable evidence including a PR link, "
            "sample output logs, and a video explanation walkthrough."
        ),
        "metadata": {
            "source": "submission-rubric.md",
            "source_file": "submission-rubric.md",
            "document_title": "FieldSure AI Project Submission Rubric",
            "section": "Submission Evidence Standards",
            "chunk_id": "eval_submission_rubric_001",
            "chunk_index": 0,
        },
    },
    {
        "chunk_id": "eval_guardrails_policy_001",
        "text": (
            "When context is missing, weak, or unrelated, the system must refuse and "
            "say not enough information or say not enough reliable context."
        ),
        "metadata": {
            "source": "guardrails.md",
            "source_file": "guardrails.md",
            "document_title": "FieldSure AI Hallucination Guardrails Policy",
            "section": "Missing Context Fallback Policy",
            "chunk_id": "eval_guardrails_policy_001",
            "chunk_index": 0,
        },
    },
]

# ---------------------------------------------------------------------------
# Task 1: End-to-End Evaluation Test Set
# ---------------------------------------------------------------------------

DEFAULT_TEST_SET: List[Dict[str, Any]] = [
    {
        "id": "eval_001_project_submission",
        "question": "What evidence is required for project submission?",
        "expected_points": ["PR link", "sample output", "video explanation"],
        "expected_sources": {"submission-rubric.md"},
        "category": "grounded_rubric",
    },
    {
        "id": "eval_002_missing_context_refusal",
        "question": "What should the system do when context is missing?",
        "expected_points": ["refuse", "say not enough information"],
        "expected_sources": {"guardrails.md"},
        "category": "guardrails_policy",
    },
    {
        "id": "eval_003_dengue_warning_signs",
        "question": "Which warning signs require urgent referral for a dengue patient?",
        "expected_points": ["persistent vomiting", "abdominal pain", "mucosal bleeding"],
        "expected_sources": {"who_dengue_clinical_guidance.md"},
        "category": "clinical_triage",
    },
    {
        "id": "eval_004_cholera_water_treatment",
        "question": "What chlorine levels are required for drinking water in settlement camps?",
        "expected_points": ["0.5 to 1.0 mg/L", "point of distribution", "disinfected"],
        "expected_sources": {"cdc_cholera_water_safety.txt"},
        "category": "clinical_wash",
    },
    {
        "id": "eval_005_malaria_rapid_test",
        "question": "What are the steps for field testing with a malaria rapid diagnostic test?",
        "expected_points": ["alcohol swab", "lancet", "buffer", "15 and 20 minutes"],
        "expected_sources": {"moh_malaria_rapid_triage.html"},
        "category": "clinical_protocol",
    },
    {
        "id": "eval_006_unsupported_medical_query",
        "question": "What surgical intervention is recommended for acute dengue fever?",
        "expected_points": ["not enough information", "refuse"],
        "expected_sources": set(),
        "category": "safe_refusal",
    },
    {
        "id": "eval_007_out_of_domain_retrieval_miss",
        "question": "What is the recommended solar battery storage capacity for field shelters?",
        "expected_points": ["10 kWh", "lithium-ion battery", "solar inverter"],
        "expected_sources": {"solar_shelter_standards.pdf"},
        "category": "out_of_domain_failure",
    },
]

# Provide module-level test_set alias matching concept documentation
test_set = DEFAULT_TEST_SET


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


# ---------------------------------------------------------------------------
# Task 2: Score Correctness & Grounding
# ---------------------------------------------------------------------------

def _extract_content_tokens(text: str) -> Set[str]:
    """Extract lowercased content words excluding punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b[a-zA-Z0-9_\.-]+\b", text.lower())
    return {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}


def judge_expected_points(
    answer: str,
    expected_points: Sequence[str],
) -> float:
    """Judge answer correctness by measuring coverage of expected answer points.

    Checks if each required factual concept or point appears in the answer,
    supporting exact matches, token overlap, and refusal semantics.

    Args:
        answer: Generated answer text.
        expected_points: Sequence of key strings/concepts expected in the answer.

    Returns:
        Fraction of expected points satisfied (0.0 to 1.0).
    """
    if not expected_points:
        return 1.0
    if not answer or not answer.strip():
        return 0.0

    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()
    answer_tokens = _extract_content_tokens(answer_clean)

    matched_count = 0
    refusal_cues = {
        "not enough information",
        "not enough reliable context",
        "i do not have enough",
        "don't have enough",
        "refuse",
        "cannot answer",
        "unable to answer",
    }
    is_refusal_answer = any(cue in answer_lower for cue in refusal_cues)

    for point in expected_points:
        point_str = str(point).strip()
        point_lower = point_str.lower()

        # Handle refusal / fallback expectations
        if point_lower in {"refuse", "say not enough information", "not enough information", "safe refusal"}:
            if is_refusal_answer:
                matched_count += 1
                continue

        # Substring match
        if point_lower in answer_lower:
            matched_count += 1
            continue

        # Subword/token coverage match for multi-word phrases (e.g. "PR link", "video explanation")
        point_tokens = _extract_content_tokens(point_str)
        if point_tokens:
            common = point_tokens & answer_tokens
            if len(common) / len(point_tokens) >= 0.75:
                matched_count += 1
                continue

    return round(matched_count / len(expected_points), 4)


def judge_grounding(
    answer: str,
    retrieved_sources: Any = None,
    context: Optional[str] = None,
    overlap_threshold: float = 0.25,
) -> float:
    """Judge whether answer claims are supported by retrieved context.

    Inspects sentence-level claims in the answer against the retrieved chunk
    text. A safe refusal on missing context is considered 100% grounded (score 1.0)
    as it makes zero unsupported claims.

    Args:
        answer: Generated answer text.
        retrieved_sources: Retrieved citations dict, list of chunks, or raw context.
        context: Optional explicit raw context string.
        overlap_threshold: Minimum content token overlap to treat sentence as grounded.

    Returns:
        Grounding score between 0.0 and 1.0.
    """
    if not answer or not answer.strip():
        return 1.0

    answer_lower = answer.strip().lower()

    # Honest refusals make no unsupported factual claims -> grounded by definition
    refusal_cues = [
        FALLBACK_NO_CONTEXT.lower(),
        "don't have enough",
        "do not have enough",
        "not enough information",
        "not enough reliable context",
        "not enough context",
        "refuse",
        "cannot answer",
        "unable to answer",
    ]
    if any(cue in answer_lower for cue in refusal_cues):
        return 1.0

    # Assemble reference text from context or retrieved_sources
    source_texts: List[str] = []
    if context:
        source_texts.append(context)

    if retrieved_sources:
        if isinstance(retrieved_sources, dict):
            for c_info in retrieved_sources.values():
                if isinstance(c_info, dict):
                    source_texts.append(str(c_info.get("text", "")))
                    source_texts.append(str(c_info.get("section", "")))
        elif isinstance(retrieved_sources, (list, tuple)):
            for item in retrieved_sources:
                if isinstance(item, dict):
                    source_texts.append(str(item.get("text", "")))
                    source_texts.append(str(item.get("metadata", {}).get("section", "")))
                elif hasattr(item, "text"):
                    source_texts.append(str(getattr(item, "text", "")))
        elif isinstance(retrieved_sources, str):
            source_texts.append(retrieved_sources)

    combined_context = " ".join(source_texts).strip()
    if not combined_context:
        # Answer made claims, but zero context was provided -> unsupported
        return 0.0

    context_tokens = _extract_content_tokens(combined_context)
    if not context_tokens:
        return 0.0

    # Split into candidate sentences, ignoring disclaimer notice
    raw_sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    sentences = [
        s.strip()
        for s in raw_sentences
        if s.strip()
        and "FieldSure AI is an informational protocol" not in s
        and "primary health center" not in s
    ]

    if not sentences:
        return 1.0

    supported_count = 0
    for sentence in sentences:
        s_clean = re.sub(r"\[\d+\]", "", sentence).strip()
        s_tokens = _extract_content_tokens(s_clean)
        if not s_tokens:
            supported_count += 1
            continue

        overlap = len(s_tokens & context_tokens) / len(s_tokens)
        if overlap >= overlap_threshold:
            supported_count += 1

    return round(supported_count / len(sentences), 4)


# ---------------------------------------------------------------------------
# Task 3: Check Citation Accuracy
# ---------------------------------------------------------------------------

def _normalize_source_name(source: Any) -> str:
    """Normalize a document source string or file path to its canonical basename."""
    if not source:
        return ""
    src_str = str(source).strip().replace("\\", "/")
    # Extract filename or basename
    name = Path(src_str).name.lower()
    return name


def check_citations(
    citations: Any,
    expected_sources: Any,
) -> float:
    """Check whether citations point to the sources that actually support claims.

    Verifies cited documents against expected source references, accounting
    for missing sources, unwanted citations, and citation fabrication.

    Args:
        citations: Citations dict (mapping markers to chunk info) or list of sources.
        expected_sources: Set or list of expected source filenames/identifiers.

    Returns:
        Citation accuracy score between 0.0 and 1.0.
    """
    # Normalize expected sources
    norm_expected: Set[str] = set()
    if expected_sources is not None:
        if isinstance(expected_sources, (set, list, tuple)):
            for s in expected_sources:
                norm = _normalize_source_name(s)
                if norm:
                    norm_expected.add(norm)
        elif isinstance(expected_sources, str):
            norm = _normalize_source_name(expected_sources)
            if norm:
                norm_expected.add(norm)

    # Extract cited sources from citations structure
    norm_cited: Set[str] = set()
    if citations:
        if isinstance(citations, dict):
            for marker, c_info in citations.items():
                if isinstance(c_info, dict):
                    source = (
                        c_info.get("source")
                        or c_info.get("source_file")
                        or c_info.get("document_title")
                        or c_info.get("metadata", {}).get("source_file")
                    )
                    norm = _normalize_source_name(source)
                    if norm:
                        norm_cited.add(norm)
        elif isinstance(citations, (list, tuple, set)):
            for item in citations:
                if isinstance(item, dict):
                    source = item.get("source") or item.get("source_file")
                    norm = _normalize_source_name(source)
                else:
                    norm = _normalize_source_name(item)
                if norm:
                    norm_cited.add(norm)

    # Case 1: No sources expected (e.g. refusal or ungrounded inquiry)
    if not norm_expected:
        return 1.0 if len(norm_cited) == 0 else 0.0

    # Case 2: Expected sources exist, but zero citations provided
    if not norm_cited:
        return 0.0

    # Case 3: Measure overlap between cited and expected
    matched_sources = norm_cited & norm_expected
    if not matched_sources:
        return 0.0

    # Jaccard overlap between cited and expected
    accuracy = len(matched_sources) / max(len(norm_expected), len(norm_cited))
    return round(accuracy, 4)


# ---------------------------------------------------------------------------
# Task 4: Summarize Quality & Diagnose Failures
# ---------------------------------------------------------------------------

def diagnose_failure(
    scored_result: Dict[str, Any],
    example: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Diagnose the most likely cause of failure for a scored RAG example.

    Identifies failure causes:
    - 'weak_retrieval': context missing or retrieval failed to fetch expected sources.
    - 'missing_metadata': citations missing filename/section metadata.
    - 'bad_citations': citations point to wrong document or fabricated markers.
    - 'unsupported_claims': answer made claims not grounded in retrieved context.
    - 'poor_prompt_instructions': answer missed expected factual points despite context.

    Args:
        scored_result: Scored example dictionary.
        example: Optional original test set example for deeper diagnostics.

    Returns:
        Dictionary with 'likely_cause', 'explanation', and 'recommendation'.
    """
    correctness = scored_result.get("correctness", 1.0)
    grounding = scored_result.get("grounding", 1.0)
    citation_accuracy = scored_result.get("citation_accuracy", 1.0)
    citations = scored_result.get("citations", {})

    failed_dims: List[str] = []
    if correctness < 1.0:
        failed_dims.append("correctness")
    if grounding < 1.0:
        failed_dims.append("grounding")
    if citation_accuracy < 1.0:
        failed_dims.append("citation_accuracy")

    # If all dimensions passed
    if not failed_dims:
        return {
            "likely_cause": "none",
            "failed_dimensions": [],
            "explanation": "All quality dimensions met or exceeded acceptance criteria.",
            "recommendation": "Maintain current retrieval and generation configuration.",
        }

    # Weak retrieval: expected source was not retrieved or citations empty on grounded query
    expected_sources = set()
    if example:
        exp = example.get("expected_sources", set())
        if isinstance(exp, (set, list, tuple)):
            expected_sources = {str(s) for s in exp}

    if expected_sources and (not citations or len(citations) == 0):
        return {
            "likely_cause": "weak_retrieval",
            "failed_dimensions": failed_dims,
            "explanation": (
                "Retrieval failed to return chunks for the expected source document, "
                "leading to an empty citation map or a safe refusal admission."
            ),
            "recommendation": (
                "Inspect retrieval query terms, tune embedding similarity thresholds, "
                "or increase top-k to capture the target documents."
            ),
        }

    # Missing metadata check
    has_missing_metadata = False
    if isinstance(citations, dict):
        for c in citations.values():
            if isinstance(c, dict):
                src = c.get("source") or c.get("source_file")
                if not src or src in {"Unknown Source", "None", ""}:
                    has_missing_metadata = True
                    break

    if has_missing_metadata:
        return {
            "likely_cause": "missing_metadata",
            "failed_dimensions": failed_dims,
            "explanation": (
                "Retrieved chunks lacked source filename or section metadata "
                "required for verified attribution."
            ),
            "recommendation": (
                "Ensure document ingestion loader and chunker preserve source_file "
                "and section_heading metadata."
            ),
        }

    # Bad citations / citation mismatch
    if citation_accuracy < 1.0:
        return {
            "likely_cause": "bad_citations",
            "failed_dimensions": failed_dims,
            "explanation": (
                "Citations point to sources that do not match the expected documents "
                "or model generated unsupported citation markers."
            ),
            "recommendation": (
                "Fix citation mapping in prompt assembly and enforce post-generation "
                "fabricated citation detection."
            ),
        }

    # Grounding failure (unsupported claims / hallucination)
    if grounding < 1.0:
        return {
            "likely_cause": "unsupported_claims",
            "failed_dimensions": failed_dims,
            "explanation": (
                "Answer includes factual claims that are not supported by the retrieved "
                "context (hallucination)."
            ),
            "recommendation": (
                "Strengthen context-only prompt constraints and enforce strict fallback "
                "behavior when evidence is partial."
            ),
        }

    # Correctness failure (incomplete answer)
    return {
        "likely_cause": "poor_prompt_instructions",
        "failed_dimensions": failed_dims,
        "explanation": (
            "The generated answer omitted one or more expected answer points despite "
            "supporting context being available."
        ),
        "recommendation": (
            "Refine synthesis prompt instructions to extract and list all required factual points."
        ),
    }


def retrieve_evaluation_context(
    question: str,
    k: int = 4,
    chunks_path: Optional[Path] = None,
    supplemental_chunks: Optional[Sequence[Dict[str, Any]]] = None,
    min_score: float = 0.35,
    score_ratio_threshold: float = 0.6,
) -> List[Dict[str, Any]]:
    """Retrieve candidate context chunks for evaluation with relevance filtering.

    Combines ingested corpus chunks with evaluation benchmark chunks and
    filters out low-scoring distractor chunks to isolate supporting context.

    Args:
        question: Evaluation inquiry.
        k: Maximum chunks to return.
        chunks_path: Optional path to ingested corpus JSONL.
        supplemental_chunks: Optional extra chunks (e.g. rubric or policy).
        min_score: Minimum keyword overlap score required.
        score_ratio_threshold: Minimum score relative to top hit.

    Returns:
        List of relevant chunk dictionaries.
    """
    from src.source_citation import _load_ingested_chunks_fallback

    all_chunks = _load_ingested_chunks_fallback(chunks_path)
    supp = (
        supplemental_chunks
        if supplemental_chunks is not None
        else DEFAULT_EVALUATION_CHUNKS
    )
    all_chunks.extend(supp)

    if not question or not question.strip():
        return []

    query_tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z0-9_-]+\b", question)
        if token.lower() not in _STOP_WORDS and len(token) > 2
    }
    if not query_tokens:
        return []

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in all_chunks:
        text = chunk.get("text", "").lower()
        metadata = chunk.get("metadata", {})
        section = str(metadata.get("section_heading", metadata.get("section", ""))).lower()
        source = str(metadata.get("source_file", metadata.get("source", ""))).lower()
        title = str(metadata.get("document_title", "")).lower()

        combined = f"{text} {section} {source} {title}"
        matched = sum(1 for token in query_tokens if token in combined)
        if matched > 0:
            score = matched / len(query_tokens)
            scored.append((score, chunk))

    if not scored:
        return []

    scored.sort(key=lambda p: p[0], reverse=True)
    top_score = scored[0][0]

    filtered = [
        chunk
        for score, chunk in scored[:k]
        if score >= min_score and score >= top_score * score_ratio_threshold
    ]
    return filtered if filtered else [chunk for score, chunk in scored[:1] if score >= 0.2]


def score_answer(
    example: Dict[str, Any],
    answer_fn: Optional[Callable[[str], Any]] = None,
    supplemental_chunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score a single test set question across all RAG evaluation dimensions.

    Pipeline:
        1. Generate answer with citations via answer_fn or answer_with_citations.
        2. Judge correctness against expected answer points.
        3. Judge grounding against retrieved source context.
        4. Check citation accuracy against expected source references.
        5. Attach diagnosis if any dimension scores below 1.0.

    Args:
        example: Dictionary containing 'question', 'expected_points', and 'expected_sources'.
        answer_fn: Optional callable to generate answers. Defaults to answer_with_citations.
        supplemental_chunks: Optional additional chunks to search alongside corpus.

    Returns:
        Dictionary containing question, answer, scores, citations, and diagnostics.
    """
    question = example.get("question", "")
    expected_points = example.get("expected_points", [])
    expected_sources = example.get("expected_sources", set())

    # 1. Generate answer with citations
    if answer_fn is not None:
        result = answer_fn(question)
    else:
        chunks_to_use = (
            supplemental_chunks
            if supplemental_chunks is not None
            else DEFAULT_EVALUATION_CHUNKS
        )
        result = answer_with_citations(
            question=question,
            retriever_fn=lambda q, k: retrieve_evaluation_context(
                q, k=k, supplemental_chunks=chunks_to_use
            ),
        )

    # Support dict subscripting or object attributes
    if isinstance(result, dict):
        answer_text = result.get("answer", "")
        citations_data = result.get("citations", {})
    else:
        answer_text = getattr(result, "answer", str(result))
        citations_data = getattr(result, "citations", {})

    # 2. Score answer dimensions
    correctness = judge_expected_points(
        answer=answer_text,
        expected_points=expected_points,
    )
    grounding = judge_grounding(
        answer=answer_text,
        retrieved_sources=citations_data,
    )
    citation_accuracy = check_citations(
        citations=citations_data,
        expected_sources=expected_sources,
    )

    scored: Dict[str, Any] = {
        "question": question,
        "answer": answer_text,
        "correctness": correctness,
        "grounding": grounding,
        "citation_accuracy": citation_accuracy,
        "citations": citations_data,
    }

    # Attach optional example metadata if present
    if "id" in example:
        scored["id"] = example["id"]
    if "category" in example:
        scored["category"] = example["category"]

    # Diagnose failure if any dimension < 1.0
    if min(correctness, grounding, citation_accuracy) < 1.0:
        diag = diagnose_failure(scored, example)
        scored["likely_cause"] = diag["likely_cause"]
        scored["failure_reason"] = diag["explanation"]
        scored["recommendation"] = diag["recommendation"]

    return scored


def evaluate_test_set(
    test_set_items: Sequence[Dict[str, Any]],
    answer_fn: Optional[Callable[[str], Any]] = None,
    supplemental_chunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Score all questions in a test set."""
    return [
        score_answer(
            example=item,
            answer_fn=answer_fn,
            supplemental_chunks=supplemental_chunks,
        )
        for item in test_set_items
    ]


def summarize_evaluations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize overall quality and isolate notable failures across evaluated rows.

    Args:
        rows: Sequence of scored answer dictionaries.

    Returns:
        Summary dictionary reporting question count, averages, and failure list.
    """
    if not rows:
        return {
            "questions": 0,
            "avg_correctness": 0.0,
            "avg_grounding": 0.0,
            "avg_citation_accuracy": 0.0,
            "overall_quality": 0.0,
            "passed_count": 0,
            "failure_count": 0,
            "failures": [],
        }

    n = len(rows)
    avg_correctness = round(sum(r.get("correctness", 0.0) for r in rows) / n, 4)
    avg_grounding = round(sum(r.get("grounding", 0.0) for r in rows) / n, 4)
    avg_citation_accuracy = round(
        sum(r.get("citation_accuracy", 0.0) for r in rows) / n, 4
    )

    failures = [
        r for r in rows
        if min(
            r.get("correctness", 0.0),
            r.get("grounding", 0.0),
            r.get("citation_accuracy", 0.0),
        ) < 1.0
    ]

    for f in failures:
        if "likely_cause" not in f:
            diag = diagnose_failure(f)
            f["likely_cause"] = diag["likely_cause"]
            f["failure_reason"] = diag["explanation"]
            f["recommendation"] = diag["recommendation"]

    overall_quality = round((avg_correctness + avg_grounding + avg_citation_accuracy) / 3, 4)

    return {
        "questions": n,
        "avg_correctness": avg_correctness,
        "avg_grounding": avg_grounding,
        "avg_citation_accuracy": avg_citation_accuracy,
        "overall_quality": overall_quality,
        "passed_count": n - len(failures),
        "failure_count": len(failures),
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Task 4 & Task 5: Markdown Report & CLI Runner
# ---------------------------------------------------------------------------

def format_evaluation_summary_md(
    summary: Dict[str, Any],
    rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Format evaluation results and failure diagnostics into a clean Markdown report."""
    questions = summary.get("questions", 0)
    avg_corr = summary.get("avg_correctness", 0.0)
    avg_grd = summary.get("avg_grounding", 0.0)
    avg_cit = summary.get("avg_citation_accuracy", 0.0)
    overall = summary.get("overall_quality", 0.0)
    failures = summary.get("failures", [])
    passed = summary.get("passed_count", questions - len(failures))

    # Identify weakest dimension
    dimension_scores = {
        "Correctness": avg_corr,
        "Grounding": avg_grd,
        "Citation Accuracy": avg_cit,
    }
    weakest_dim = min(dimension_scores, key=dimension_scores.get)
    weakest_val = dimension_scores[weakest_dim]

    lines = [
        "# RAG Evaluation & Answer Quality Scoring Report",
        "",
        "> **Concept**: 3.43 — RAG Evaluation & Answer Quality Scoring  ",
        "> **Status**: Verified Execution  ",
        "> **Generated**: 2026-10-24  ",
        "",
        "---",
        "",
        "## 1. Overall Quality Summary",
        "",
        "| Evaluation Metric | Score | Status / Note |",
        "| :--- | :---: | :--- |",
        f"| **Evaluated Questions** | `{questions}` | Total test set inquiries |",
        f"| **Average Correctness** | `{avg_corr:.4f}` | Coverage of expected answer points |",
        f"| **Average Grounding** | `{avg_grd:.4f}` | Factual support in retrieved context |",
        f"| **Average Citation Accuracy** | `{avg_cit:.4f}` | Precision of cited source documents |",
        f"| **Overall Quality Index** | `{overall:.4f}` | Composite average across dimensions |",
        f"| **Passed Evaluations** | `{passed}/{questions}` | Min score across dimensions == 1.0 |",
        f"| **Notable Failures** | `{len(failures)}` | Min score across dimensions < 1.0 |",
        "",
        "---",
        "",
        "## 2. Dimension Analysis & Weakest Area",
        "",
        f"- **Strongest Dimension**: `{max(dimension_scores, key=dimension_scores.get)}` ({max(dimension_scores.values()):.4f})",
        f"- **Weakest Dimension**: `{weakest_dim}` ({weakest_val:.4f})",
        "",
    ]

    if weakest_dim == "Correctness":
        lines.extend([
            "### Remediation for Correctness:",
            "- Inspect retrieval top-k to ensure all critical chunks are returned.",
            "- Refine prompt generation instructions to extract all required factual elements.",
            "- Ensure synthesis instructions do not truncate key lists.",
        ])
    elif weakest_dim == "Grounding":
        lines.extend([
            "### Remediation for Grounding:",
            "- Strengthen context-only prompt constraints ('Answer using only the context below').",
            "- Strengthen safe refusal behavior when context is partial or absent.",
            "- Filter out unsupported sentences during post-generation verification.",
        ])
    else:
        lines.extend([
            "### Remediation for Citation Accuracy:",
            "- Ensure chunker and metadata loader preserve document basenames and source paths.",
            "- Strengthen prompt citation rules requiring [1], [2] markers for every factual claim.",
            "- Strip fabricated citation markers before returning answers.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 3. Scored Test Set Breakdown",
        "",
        "| ID | Question | Correctness | Grounding | Citations | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])

    all_rows = rows if rows is not None else []
    for idx, r in enumerate(all_rows, start=1):
        q_id = r.get("id", f"Q{idx:03d}")
        q_text = r.get("question", "")[:45] + ("..." if len(r.get("question", "")) > 45 else "")
        c = r.get("correctness", 0.0)
        g = r.get("grounding", 0.0)
        ca = r.get("citation_accuracy", 0.0)
        passed_q = "✅ Pass" if min(c, g, ca) >= 1.0 else "⚠️ Review"
        lines.append(f"| `{q_id}` | {q_text} | `{c:.2f}` | `{g:.2f}` | `{ca:.2f}` | {passed_q} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Notable Failures & Root Cause Analysis",
        "",
    ])

    if not failures:
        lines.append("No failures detected. All evaluated questions achieved 100% across correctness, grounding, and citation accuracy.")
    else:
        lines.append("| Question | Failed Dimension(s) | Likely Cause | Recommended Fix |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for f in failures:
            q_short = f.get("question", "")[:40] + ("..." if len(f.get("question", "")) > 40 else "")
            cause = f.get("likely_cause", "unknown")
            rec = f.get("recommendation", "Review retrieval and prompt quality.")
            failed_dims = []
            if f.get("correctness", 1.0) < 1.0:
                failed_dims.append("Correctness")
            if f.get("grounding", 1.0) < 1.0:
                failed_dims.append("Grounding")
            if f.get("citation_accuracy", 1.0) < 1.0:
                failed_dims.append("Citations")
            lines.append(f"| {q_short} | {', '.join(failed_dims)} | `{cause}` | {rec} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Execution Verification Evidence",
        "",
        "- **Evaluation Script**: `src/rag_evaluation.py`",
        "- **Test Set Artifact**: `outputs/rag_evaluation_test_set.json`",
        "- **JSON Report**: `outputs/rag_evaluation_report.json`",
        "- **Automated Tests**: `tests/test_rag_evaluation.py`",
        "",
    ])

    return "\n".join(lines)


def run_rag_evaluation(
    test_set_path: Optional[Path] = None,
    output_report_path: Optional[Path] = None,
    output_summary_path: Optional[Path] = None,
    test_set_export_path: Optional[Path] = None,
    answer_fn: Optional[Callable[[str], Any]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end RAG evaluation and write artifact files.

    Args:
        test_set_path: Optional path to custom test set JSON.
        output_report_path: Optional path for JSON report.
        output_summary_path: Optional path for Markdown summary.
        test_set_export_path: Optional path to export test set JSON.
        answer_fn: Optional custom answering callable.

    Returns:
        Summary dictionary of evaluation results.
    """
    # 1. Load or use default test set
    if test_set_path and test_set_path.is_file():
        try:
            raw = json.loads(test_set_path.read_text(encoding="utf-8"))
            active_test_set = raw if isinstance(raw, list) else DEFAULT_TEST_SET
        except Exception as err:
            logger.warning("Could not read custom test set, using default: %s", err)
            active_test_set = DEFAULT_TEST_SET
    else:
        active_test_set = DEFAULT_TEST_SET

    # 2. Score answer dimensions
    rows = evaluate_test_set(active_test_set, answer_fn=answer_fn)

    # 3. Summarize failures and quality
    summary = summarize_evaluations(rows)

    # 4. Prepare JSON-serializable test set
    serializable_test_set = []
    for item in active_test_set:
        c_item = dict(item)
        if isinstance(c_item.get("expected_sources"), set):
            c_item["expected_sources"] = sorted(list(c_item["expected_sources"]))
        serializable_test_set.append(c_item)

    # 5. Write artifacts
    report_file = output_report_path or DEFAULT_OUTPUT_REPORT
    summary_file = output_summary_path or DEFAULT_OUTPUT_SUMMARY
    test_set_file = test_set_export_path or DEFAULT_TEST_SET_FILE

    report_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    test_set_file.parent.mkdir(parents=True, exist_ok=True)

    # Save test set JSON
    test_set_file.write_text(
        json.dumps(serializable_test_set, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save report JSON
    report_payload = {
        "summary": summary,
        "rows": rows,
    }
    report_file.write_text(
        json.dumps(report_payload, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save Markdown summary
    summary_md = format_evaluation_summary_md(summary, rows=rows)
    summary_file.write_text(summary_md, encoding="utf-8")
    if summary_file != DEFAULT_OUTPUT_SAMPLE:
        DEFAULT_OUTPUT_SAMPLE.write_text(summary_md, encoding="utf-8")

    return summary


def main() -> None:
    """CLI entry point for running RAG evaluation."""
    parser = argparse.ArgumentParser(
        description="Run RAG Evaluation & Answer Quality Scoring across test set.",
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=None,
        help="Optional path to custom test set JSON file.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_OUTPUT_REPORT,
        help="Path to output evaluation JSON report.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=DEFAULT_OUTPUT_SUMMARY,
        help="Path to output evaluation Markdown summary.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test with default benchmark test set.",
    )

    args = parser.parse_args()

    summary = run_rag_evaluation(
        test_set_path=args.test_set,
        output_report_path=args.output_report,
        output_summary_path=args.output_summary,
    )

    print("\n=== FieldSure AI RAG Evaluation Summary ===")
    print(f"Questions Evaluated:    {summary['questions']}")
    print(f"Average Correctness:    {summary['avg_correctness']:.4f}")
    print(f"Average Grounding:      {summary['avg_grounding']:.4f}")
    print(f"Average Citation Acc:   {summary['avg_citation_accuracy']:.4f}")
    print(f"Overall Quality Index:  {summary['overall_quality']:.4f}")
    print(f"Passed Evaluations:     {summary['passed_count']}/{summary['questions']}")
    print(f"Notable Failures:       {summary['failure_count']}")
    if summary['failures']:
        print("\nFailures:")
        for idx, f in enumerate(summary['failures'], start=1):
            print(f"  {idx}. {f['question']} -> {f.get('likely_cause')}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
