"""Run known-relevance sanity checks against stored corpus embeddings."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from src.config import LLMConfig, LLMConfigError, load_llm_config
from src.embedding_fundamentals import (
    EmbeddingAPIError,
    generate_embeddings,
)
from src.rank_chunks import load_embedded_chunks, rank_chunks

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "outputs" / "embedded_chunks.jsonl"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "embedding_quality_sanity_report.json"
BORDERLINE_MARGIN = 0.05

KNOWN_RELEVANCE_CASES: tuple[dict[str, str], ...] = (
    {
        "name": "dengue field interventions",
        "query": "What immediate actions and medicines are recommended for dengue?",
        "expected_chunk_id": "doc_who_dengue_guidance_2024_c0003",
    },
    {
        "name": "cholera water safety",
        "query": "How should drinking water be disinfected during a cholera outbreak?",
        "expected_chunk_id": "doc_cdc_cholera_protocol_2024_c0001",
    },
    {
        "name": "malaria urgent treatment",
        "query": "What treatment and urgent referral steps are required for malaria?",
        "expected_chunk_id": "doc_moh_malaria_triage_2024_c0002",
    },
)


def evaluate_quality_cases(
    chunks: Sequence[dict[str, Any]],
    query_embeddings: Sequence[Sequence[float]],
    cases: Sequence[dict[str, str]] = KNOWN_RELEVANCE_CASES,
    borderline_margin: float = BORDERLINE_MARGIN,
) -> dict[str, Any]:
    """Check whether each known relevant chunk ranks first for its query."""
    if not cases:
        raise ValueError("At least one known-relevance case is required.")
    if len(query_embeddings) != len(cases):
        raise ValueError("Each relevance case must have one query embedding.")
    if borderline_margin < 0:
        raise ValueError("Borderline margin must not be negative.")

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    results: list[dict[str, Any]] = []
    for case, query_embedding in zip(cases, query_embeddings):
        expected_id = case.get("expected_chunk_id", "")
        if expected_id not in chunks_by_id:
            raise ValueError(
                f"Expected relevant chunk '{expected_id}' for case "
                f"'{case.get('name', 'unnamed')}' is missing from embedded corpus."
            )

        ranked = rank_chunks(query_embedding, chunks)
        expected_rank = next(
            index
            for index, chunk in enumerate(ranked, start=1)
            if chunk["chunk_id"] == expected_id
        )
        expected_chunk = chunks_by_id[expected_id]
        expected_score = next(
            chunk["similarity_score"]
            for chunk in ranked
            if chunk["chunk_id"] == expected_id
        )
        best_competitor = next(
            chunk for chunk in ranked if chunk["chunk_id"] != expected_id
        )
        margin = expected_score - best_competitor["similarity_score"]
        top_chunk = ranked[0]

        if expected_rank != 1:
            status = "failed"
            note = (
                f"Expected chunk ranked #{expected_rank}; "
                f"'{top_chunk['chunk_id']}' outranked it."
            )
        elif margin <= borderline_margin:
            status = "borderline"
            note = (
                f"Expected chunk ranked first, but its lead "
                f"({margin:.6f}) is within the {borderline_margin:.6f} "
                "borderline threshold."
            )
        else:
            status = "passed"
            note = (
                f"Expected chunk ranked first with a "
                f"{margin:.6f} lead over the next result."
            )

        results.append(
            {
                "name": case["name"],
                "query": case["query"],
                "status": status,
                "expected_chunk_id": expected_id,
                "expected_source": expected_chunk["metadata"].get(
                    "source_file", "unknown"
                ),
                "expected_rank": expected_rank,
                "expected_score": expected_score,
                "top_chunk_id": top_chunk["chunk_id"],
                "top_source": top_chunk["metadata"].get("source_file", "unknown"),
                "top_section": top_chunk["metadata"].get(
                    "section_heading", "unknown"
                ),
                "top_score": top_chunk["similarity_score"],
                "runner_up_chunk_id": best_competitor["chunk_id"],
                "runner_up_score": best_competitor["similarity_score"],
                "margin_over_runner_up": margin,
                "note": note,
            }
        )

    failures = sum(result["status"] == "failed" for result in results)
    borderline = sum(result["status"] == "borderline" for result in results)
    borderline_results = [
        result for result in results if result["status"] == "borderline"
    ]
    if borderline_results:
        first_borderline = borderline_results[0]
        overall_note = (
            f"Borderline case '{first_borderline['name']}': expected chunk "
            f"leads '{first_borderline['runner_up_chunk_id']}' by only "
            f"{first_borderline['margin_over_runner_up']:.6f}; inspect this "
            "near-tie before trusting retrieval."
        )
    else:
        overall_note = "All known-relevance checks ranked their expected chunk first."

    return {
        "metric": "cosine_similarity",
        "metric_justification": (
            "Cosine similarity compares embedding direction, reducing the "
            "effect of vector magnitude when matching semantic representations."
        ),
        "borderline_margin_threshold": borderline_margin,
        "test_count": len(results),
        "pass_count": sum(result["status"] == "passed" for result in results),
        "failure_count": failures,
        "borderline_count": borderline,
        "borderline_cases": [result["name"] for result in borderline_results],
        "results": results,
        "overall_note": overall_note,
    }


def write_sanity_report(report_path: Path, report: dict[str, Any]) -> None:
    """Persist the complete sanity report as formatted JSON."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")


def run_quality_check(
    input_path: Path,
    report_path: Path,
    model: str,
    config: LLMConfig,
    cases: Sequence[dict[str, str]] = KNOWN_RELEVANCE_CASES,
) -> dict[str, Any]:
    """Embed known queries, evaluate ranking, save and print the sanity report."""
    chunks = load_embedded_chunks(input_path)
    queries = [case["query"] for case in cases]
    query_embeddings = generate_embeddings(queries, model, config)
    report = evaluate_quality_cases(chunks, query_embeddings, cases)
    report["model"] = model
    report["query_vector_source"] = "configured embeddings API"
    report["corpus_path"] = str(input_path)
    write_sanity_report(report_path, report)

    print(f"Metric: {report['metric']}")
    print(f"Tests: {report['test_count']}")
    print(f"Passes: {report['pass_count']}")
    print(f"Failures: {report['failure_count']}")
    print(f"Borderline: {report['borderline_count']}")
    print(f"Report: {report_path}")
    for result in report["results"]:
        print(
            f"\n{result['status'].upper()}: {result['name']} | "
            f"expected={result['expected_chunk_id']} "
            f"(rank {result['expected_rank']}, "
            f"score {result['expected_score']:.6f}) | "
            f"top={result['top_chunk_id']} "
            f"({result['top_score']:.6f}) | "
            f"source={result['top_source']}"
        )
        print(f"Note: {result['note']}")
    print(f"\nOverall: {report['overall_note']}")
    return report


def main() -> int:
    """Run known-relevance embedding checks with configured API credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()

    try:
        config = load_llm_config()
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            raise ValueError(
                "EMBEDDING_MODEL is not configured. Set it in your .env file."
            )
        report = run_quality_check(
            arguments.input,
            arguments.report,
            model,
            config,
        )
    except (EmbeddingAPIError, LLMConfigError, OSError, ValueError) as err:
        print(f"Embedding quality check failed: {err}", file=sys.stderr)
        return 1
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
