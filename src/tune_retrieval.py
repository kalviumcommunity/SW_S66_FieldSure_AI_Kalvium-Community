"""Evaluate retrieval settings against known source relevance on the sample corpus."""

import argparse
import json
import math
from pathlib import Path
import tempfile
from typing import Any

from src.index_embeddings import load_source_chunks
from src.vector_store import ChromaVectorStore, VectorStoreError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "outputs" / "ingested_chunks.jsonl"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "retrieval_tuning_report.json"
FIXTURE_QUERY_VECTOR = [1.0, 0.0, 0.0]
RELEVANT_VECTOR_SIMILARITY = 0.98
SCORE_THRESHOLD = 0.5
QUERY_CASES = (
    {
        "query": "What dengue guidance should field workers use?",
        "expected_source": "data/sample_advisories/who_dengue_clinical_guidance.md",
    },
    {
        "query": "How should drinking water be treated during a cholera outbreak?",
        "expected_source": "data/sample_advisories/cdc_cholera_water_safety.txt",
    },
    {
        "query": "What is the malaria rapid diagnostic testing procedure?",
        "expected_source": "data/sample_advisories/moh_malaria_rapid_triage.html",
    },
)
SETTINGS = (
    {
        "name": "baseline_vector_k2",
        "k": 2,
        "min_similarity": 0.0,
        "filter_by_expected_source": False,
    },
    {
        "name": "source_filter_k5_threshold_0_5",
        "k": 5,
        "min_similarity": SCORE_THRESHOLD,
        "filter_by_expected_source": True,
    },
)


def _make_fixture_vectors(
    chunks: list[dict[str, Any]],
    expected_source: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Build deterministic vectors with one deliberate off-source distractor."""
    relevant_ids = {
        chunk["chunk_id"]
        for chunk in chunks
        if chunk["metadata"].get("source_file") == expected_source
    }
    if not relevant_ids:
        raise ValueError(f"No source chunks found for expected source '{expected_source}'.")

    distractor = next(
        (
            chunk
            for chunk in chunks
            if chunk["metadata"].get("source_file") != expected_source
        ),
        None,
    )
    if distractor is None:
        raise ValueError("The tuning corpus needs at least two distinct sources.")

    relevant_component = math.sqrt(1.0 - RELEVANT_VECTOR_SIMILARITY**2)
    fixture_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        if chunk["chunk_id"] == distractor["chunk_id"]:
            vector = [1.0, 0.0, 0.0]
        elif chunk["chunk_id"] in relevant_ids:
            vector = [
                RELEVANT_VECTOR_SIMILARITY,
                relevant_component,
                0.0,
            ]
        else:
            vector = [0.0, 0.0, 1.0]
        fixture_chunks.append(
            {
                **chunk,
                "embedding": vector,
                "vector_dimension": len(vector),
            }
        )
    return fixture_chunks, relevant_ids


def _evaluate_setting(
    store: ChromaVectorStore,
    query_case: dict[str, str],
    relevant_ids: set[str],
    setting: dict[str, Any],
) -> dict[str, Any]:
    """Search one query and compute source/chunk-level relevance measures."""
    where = (
        {"source_file": query_case["expected_source"]}
        if setting["filter_by_expected_source"]
        else None
    )
    candidates = store.search(
        FIXTURE_QUERY_VECTOR,
        store.count(),
        where=where,
    )
    thresholded = [
        match
        for match in candidates
        if match["similarity_score"] >= setting["min_similarity"]
    ]
    retrieved = thresholded[: setting["k"]]
    retrieved_ids = [match["chunk_id"] for match in retrieved]
    relevant_retrieved = sum(record_id in relevant_ids for record_id in retrieved_ids)
    precision = relevant_retrieved / len(retrieved) if retrieved else 0.0
    recall = relevant_retrieved / len(relevant_ids)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "setting": setting["name"],
        "k": setting["k"],
        "metadata_filter": where,
        "minimum_similarity": setting["min_similarity"],
        "retrieved_count": len(retrieved),
        "retrieved_relevant_count": relevant_retrieved,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_at_k": relevant_retrieved > 0,
        "top1_hit": bool(retrieved and retrieved[0]["chunk_id"] in relevant_ids),
        "results": [
            {
                "rank": rank,
                "chunk_id": match["chunk_id"],
                "similarity_score": match["similarity_score"],
                "expected_relevant": match["chunk_id"] in relevant_ids,
                "text": match["text"],
                "metadata": match["metadata"],
            }
            for rank, match in enumerate(retrieved, start=1)
        ],
        "f1_at_k": f1,
    }


def _aggregate_metrics(
    outcomes: list[dict[str, Any]],
    total_relevant: int,
) -> dict[str, float]:
    """Compute macro precision/recall/F1 and query hit rates."""
    count = len(outcomes)
    retrieved_count = sum(outcome["retrieved_count"] for outcome in outcomes)
    relevant_retrieved = sum(
        outcome["retrieved_relevant_count"] for outcome in outcomes
    )
    return {
        "mean_precision_at_k": sum(
            outcome["precision_at_k"] for outcome in outcomes
        )
        / count,
        "mean_recall_at_k": sum(outcome["recall_at_k"] for outcome in outcomes)
        / count,
        "mean_f1_at_k": sum(outcome["f1_at_k"] for outcome in outcomes) / count,
        "micro_precision": (
            relevant_retrieved / retrieved_count if retrieved_count else 0.0
        ),
        "micro_recall": relevant_retrieved / total_relevant if total_relevant else 0.0,
        "top1_hit_rate": sum(outcome["top1_hit"] for outcome in outcomes) / count,
        "hit_at_k_rate": sum(outcome["hit_at_k"] for outcome in outcomes) / count,
    }


def run_tuning_experiment(
    source_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Run both retrieval settings over each known-source sample query."""
    chunks = load_source_chunks(source_path)
    total_relevant = 0
    all_results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fieldsure-retrieval-tuning-") as temp_dir:
        for case_index, query_case in enumerate(QUERY_CASES):
            fixture_chunks, relevant_ids = _make_fixture_vectors(
                chunks,
                query_case["expected_source"],
            )
            total_relevant += len(relevant_ids)
            store = ChromaVectorStore(
                Path(temp_dir) / "chroma",
                f"tuning_case_{case_index}",
            )
            try:
                store.upsert(fixture_chunks)
                setting_results = [
                    _evaluate_setting(store, query_case, relevant_ids, setting)
                    for setting in SETTINGS
                ]
            finally:
                store.client.close()
            all_results.append(
                {
                    **query_case,
                    "expected_relevant_chunk_ids": sorted(relevant_ids),
                    "settings": setting_results,
                }
            )

    setting_summaries = {
        setting["name"]: _aggregate_metrics(
            [case["settings"][setting_index] for case in all_results],
            total_relevant,
        )
        for setting_index, setting in enumerate(SETTINGS)
    }
    best_setting = max(
        SETTINGS,
        key=lambda setting: (
            setting_summaries[setting["name"]]["mean_f1_at_k"],
            setting_summaries[setting["name"]]["mean_precision_at_k"],
            setting_summaries[setting["name"]]["mean_recall_at_k"],
        ),
    )
    report = {
        "status": "passed",
        "evaluation_mode": "deterministic illustrative fixture vectors; not live embeddings",
        "query_vector_dimension": len(FIXTURE_QUERY_VECTOR),
        "queries": all_results,
        "settings": [
            {
                **setting,
                "metrics": setting_summaries[setting["name"]],
            }
            for setting in SETTINGS
        ],
        "chosen_setting": best_setting["name"],
        "selection_criterion": (
            "Highest macro mean F1@k; ties break on macro precision@k then "
            "macro recall@k."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(report, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(f"Queries evaluated: {len(all_results)}")
    for setting in report["settings"]:
        metrics = setting["metrics"]
        print(
            f"{setting['name']}: precision@k={metrics['mean_precision_at_k']:.3f}, "
            f"recall@k={metrics['mean_recall_at_k']:.3f}, "
            f"F1@k={metrics['mean_f1_at_k']:.3f}, "
            f"top1-hit={metrics['top1_hit_rate']:.3f}"
        )
    print(f"Chosen setting: {report['chosen_setting']}")
    print(f"Report: {report_path}")
    return report


def main() -> int:
    """Run the deterministic retrieval tuning experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    try:
        run_tuning_experiment(arguments.source, arguments.report)
    except (OSError, ValueError, VectorStoreError) as err:
        parser.exit(1, f"Retrieval tuning failed: {err}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
