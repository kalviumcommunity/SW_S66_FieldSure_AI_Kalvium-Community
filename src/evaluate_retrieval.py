"""Measure precision and recall for a labeled query set against ChromaDB."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

from dotenv import load_dotenv

from src.config import LLMConfigError, load_llm_config
from src.embedding_fundamentals import (
    EmbeddingAPIError,
    cosine_similarity,
    generate_embeddings,
)
from src.index_embeddings import load_source_chunks
from src.vector_store import ChromaVectorStore, VectorStoreError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS = REPO_ROOT / "outputs" / "retrieval_evaluation_queries.json"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "retrieval_evaluation_report.json"
DEFAULT_SOURCE = REPO_ROOT / "outputs" / "ingested_chunks.jsonl"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "vectordb"
DEFAULT_COLLECTION = "fieldsure_chunks"
DEFAULT_K = 3
FIXTURE_RELEVANT_SIMILARITY = 0.95


def load_labeled_queries(path: Path) -> list[dict[str, Any]]:
    """Load and validate query labels with one or more relevant chunk IDs."""
    try:
        raw_labels: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON in labeled query file '{path}'.") from err
    if not isinstance(raw_labels, list) or not raw_labels:
        raise ValueError("Labeled query file must contain a non-empty JSON array.")

    labels: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    for index, label in enumerate(raw_labels):
        if not isinstance(label, dict):
            raise ValueError(f"Labeled query {index} must be a JSON object.")
        query_id = label.get("query_id")
        query = label.get("query")
        relevant_ids = label.get("relevant_chunk_ids")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"Labeled query {index} is missing query_id.")
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query_id '{query_id}'.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Labeled query '{query_id}' has empty query text.")
        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or any(not isinstance(item, str) or not item.strip() for item in relevant_ids)
        ):
            raise ValueError(
                f"Labeled query '{query_id}' needs non-empty relevant_chunk_ids."
            )
        if len(set(relevant_ids)) != len(relevant_ids):
            raise ValueError(
                f"Labeled query '{query_id}' has duplicate relevant chunk IDs."
            )
        seen_query_ids.add(query_id)
        labels.append({**label, "relevant_chunk_ids": relevant_ids})
    return labels


def _inspect_expected_chunks(
    store: ChromaVectorStore,
    relevant_ids: Sequence[str],
    query_embedding: Sequence[float],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Classify expected IDs as missing or indexed with their query similarity."""
    missing_ids: list[str] = []
    indexed: list[dict[str, Any]] = []
    for chunk_id in relevant_ids:
        try:
            record = store.get(chunk_id)
        except VectorStoreError as err:
            if "was not found" not in str(err):
                raise
            missing_ids.append(chunk_id)
            continue
        if len(record["embedding"]) != len(query_embedding):
            raise VectorStoreError(
                f"Expected relevant chunk '{chunk_id}' has vector dimension "
                f"{record['vector_dimension']}; query dimension is "
                f"{len(query_embedding)}."
            )
        indexed.append(
            {
                "chunk_id": chunk_id,
                "similarity_score": cosine_similarity(
                    query_embedding,
                    record["embedding"],
                ),
            }
        )
    return missing_ids, indexed


def evaluate_labeled_query(
    store: ChromaVectorStore,
    label: dict[str, Any],
    query_embedding: Sequence[float],
    k: int,
    min_similarity: float = -1.0,
) -> dict[str, Any]:
    """Compute precision@k and recall@k with diagnostics for missed labels."""
    if k < 1:
        raise ValueError("Evaluation k must be at least 1.")
    if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
        raise ValueError("Minimum similarity must be finite and between -1 and 1.")

    relevant_ids = set(label["relevant_chunk_ids"])
    missing_ids, indexed_expected = _inspect_expected_chunks(
        store,
        label["relevant_chunk_ids"],
        query_embedding,
    )
    candidates = store.search(query_embedding, k)
    retrieved = [
        item for item in candidates if item["similarity_score"] >= min_similarity
    ][:k]
    retrieved_ids = {item["chunk_id"] for item in retrieved}
    relevant_retrieved = sum(item["chunk_id"] in relevant_ids for item in retrieved)
    recall = relevant_retrieved / len(relevant_ids)
    precision = relevant_retrieved / k
    missed_ids = sorted(relevant_ids - retrieved_ids)

    failures: list[dict[str, str]] = []
    if missing_ids:
        failures.append(
            {
                "type": "relevant_chunk_not_indexed",
                "chunk_ids": ", ".join(sorted(missing_ids)),
                "likely_cause": (
                    "The labeled relevant chunks are absent from the collection; "
                    "check that the current corpus was completely indexed."
                ),
            }
        )
    still_indexed_misses = [
        item for item in indexed_expected if item["chunk_id"] in missed_ids
    ]
    if still_indexed_misses:
        missed_scores = ", ".join(
            f"{item['chunk_id']}={item['similarity_score']:.4f}"
            for item in still_indexed_misses
        )
        failures.append(
            {
                "type": (
                    "relevant_chunk_below_score_threshold"
                    if any(
                        item["similarity_score"] < min_similarity
                        for item in still_indexed_misses
                    )
                    else "relevant_chunk_ranked_below_k"
                ),
                "chunk_ids": ", ".join(item["chunk_id"] for item in still_indexed_misses),
                "similarity_scores": missed_scores,
                "likely_cause": (
                    "Relevant chunks exist but were filtered by the score threshold "
                    "or ranked below k. Review query wording, chunk boundaries, "
                    "embedding-model consistency, threshold, and k."
                ),
            }
        )
    if retrieved and retrieved[0]["chunk_id"] not in relevant_ids:
        failures.append(
            {
                "type": "top1_relevant_miss",
                "chunk_ids": retrieved[0]["chunk_id"],
                "similarity_scores": f"{retrieved[0]['similarity_score']:.4f}",
                "likely_cause": (
                    "An irrelevant chunk outranked every labeled relevant chunk. "
                    "Inspect the query-to-chunk match, distractor content, and "
                    "embedding-model consistency."
                ),
            }
        )

    return {
        "query_id": label["query_id"],
        "query": label["query"],
        "relevance_note": label.get("relevance_note", ""),
        "relevant_chunk_ids": sorted(relevant_ids),
        "k": k,
        "minimum_similarity": min_similarity,
        "retrieved_count": len(retrieved),
        "relevant_retrieved_count": relevant_retrieved,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_at_k": relevant_retrieved > 0,
        "top1_hit": bool(
            retrieved and retrieved[0]["chunk_id"] in relevant_ids
        ),
        "missing_relevant_chunk_ids": sorted(missing_ids),
        "expected_chunk_similarities": indexed_expected,
        "retrieved": [
            {
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "similarity_score": item["similarity_score"],
                "relevant": item["chunk_id"] in relevant_ids,
                "text": item["text"],
                "metadata": item["metadata"],
            }
            for rank, item in enumerate(retrieved, start=1)
        ],
        "failures": failures,
    }


def evaluate_query_set(
    store: ChromaVectorStore,
    labels: list[dict[str, Any]],
    query_embeddings: list[list[float]],
    k: int,
    min_similarity: float = -1.0,
) -> dict[str, Any]:
    """Evaluate all labels and aggregate precision, recall, and hit rates."""
    if len(labels) != len(query_embeddings):
        raise ValueError(
            f"Expected {len(labels)} query vectors, received {len(query_embeddings)}."
        )
    if not labels:
        raise ValueError("Provide at least one labeled query.")
    results = [
        evaluate_labeled_query(store, label, embedding, k, min_similarity)
        for label, embedding in zip(labels, query_embeddings, strict=True)
    ]
    total_relevant = sum(len(result["relevant_chunk_ids"]) for result in results)
    total_retrieved = sum(result["retrieved_count"] for result in results)
    total_relevant_retrieved = sum(
        result["relevant_retrieved_count"] for result in results
    )
    return {
        "query_count": len(results),
        "k": k,
        "minimum_similarity": min_similarity,
        "macro_precision_at_k": sum(
            result["precision_at_k"] for result in results
        )
        / len(results),
        "macro_recall_at_k": sum(
            result["recall_at_k"] for result in results
        )
        / len(results),
        "micro_precision_at_k": (
            total_relevant_retrieved / total_retrieved if total_retrieved else 0.0
        ),
        "micro_recall_at_k": (
            total_relevant_retrieved / total_relevant if total_relevant else 0.0
        ),
        "hit_at_k_rate": sum(result["hit_at_k"] for result in results) / len(results),
        "top1_hit_rate": sum(result["top1_hit"] for result in results) / len(results),
        "failure_count": sum(len(result["failures"]) for result in results),
        "queries": results,
    }


def _fixture_embeddings(
    chunks: list[dict[str, Any]],
    relevant_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Create deterministic two-dimensional vectors with an explicit top-1 miss."""
    relevant_set = set(relevant_ids)
    if not relevant_set.issubset({chunk["chunk_id"] for chunk in chunks}):
        raise ValueError("Every labeled relevant chunk must exist in the source corpus.")
    distractor = next(
        (chunk for chunk in chunks if chunk["chunk_id"] not in relevant_set),
        None,
    )
    if distractor is None:
        raise ValueError("Fixture evaluation requires at least one non-relevant chunk.")
    fixture: list[dict[str, Any]] = []
    relevant_component = math.sqrt(1.0 - FIXTURE_RELEVANT_SIMILARITY**2)
    for chunk in chunks:
        if chunk["chunk_id"] == distractor["chunk_id"]:
            embedding = [1.0, 0.0]
        elif chunk["chunk_id"] in relevant_set:
            embedding = [FIXTURE_RELEVANT_SIMILARITY, relevant_component]
        else:
            embedding = [0.0, 1.0]
        fixture.append(
            {
                **chunk,
                "embedding": embedding,
                "vector_dimension": len(embedding),
            }
        )
    return fixture


def run_fixture_evaluation(
    labels_path: Path,
    source_path: Path,
    report_path: Path,
    k: int,
    min_similarity: float = -1.0,
) -> dict[str, Any]:
    """Run a deterministic local ChromaDB evaluation without API credentials."""
    labels = load_labeled_queries(labels_path)
    source_chunks = load_source_chunks(source_path)
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fieldsure-retrieval-eval-") as temp_dir:
        for index, label in enumerate(labels):
            fixture_chunks = _fixture_embeddings(
                source_chunks,
                label["relevant_chunk_ids"],
            )
            store = ChromaVectorStore(
                Path(temp_dir) / "chroma",
                f"evaluation_{index}",
            )
            try:
                store.upsert(fixture_chunks)
                runs.append(
                    evaluate_labeled_query(
                        store,
                        label,
                        [1.0, 0.0],
                        k,
                        min_similarity,
                    )
                )
            finally:
                store.client.close()

    report = _aggregate_query_results(
        runs,
        k,
        evaluation_mode="illustrative fixtures",
        min_similarity=min_similarity,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report_summary(report, report_path)
    return report


def _aggregate_query_results(
    runs: list[dict[str, Any]],
    k: int,
    evaluation_mode: str,
    model: str | None = None,
    min_similarity: float = -1.0,
) -> dict[str, Any]:
    """Create a serializable evaluation report from per-query measurements."""
    total_relevant = sum(len(run["relevant_chunk_ids"]) for run in runs)
    total_retrieved = sum(run["retrieved_count"] for run in runs)
    total_relevant_retrieved = sum(run["relevant_retrieved_count"] for run in runs)
    return {
        "status": "passed" if not any(run["failures"] for run in runs) else "failed",
        "evaluation_mode": evaluation_mode,
        "embedding_model": model,
        "query_count": len(runs),
        "k": k,
        "minimum_similarity": min_similarity,
        "macro_precision_at_k": sum(run["precision_at_k"] for run in runs)
        / len(runs),
        "macro_recall_at_k": sum(run["recall_at_k"] for run in runs) / len(runs),
        "micro_precision_at_k": (
            total_relevant_retrieved / total_retrieved if total_retrieved else 0.0
        ),
        "micro_recall_at_k": (
            total_relevant_retrieved / total_relevant if total_relevant else 0.0
        ),
        "hit_at_k_rate": sum(run["hit_at_k"] for run in runs) / len(runs),
        "top1_hit_rate": sum(run["top1_hit"] for run in runs) / len(runs),
        "failure_count": sum(len(run["failures"]) for run in runs),
        "queries": runs,
    }


def run_live_evaluation(
    labels_path: Path,
    report_path: Path,
    database_path: Path,
    collection_name: str,
    model: str,
    k: int,
    min_similarity: float,
) -> dict[str, Any]:
    """Embed labeled queries and evaluate the configured persistent collection."""
    if not model.strip():
        raise ValueError("EMBEDDING_MODEL must match the indexed document model.")
    labels = load_labeled_queries(labels_path)
    config = load_llm_config()
    query_embeddings = generate_embeddings(
        [label["query"] for label in labels],
        model,
        config,
    )
    store = ChromaVectorStore(database_path, collection_name)
    try:
        evaluation = evaluate_query_set(
            store,
            labels,
            query_embeddings,
            k,
            min_similarity,
        )
    finally:
        store.client.close()
    report = _aggregate_query_results(
        evaluation["queries"],
        k,
        evaluation_mode="configured ChromaDB collection",
        model=model,
        min_similarity=min_similarity,
    )
    report["collection"] = collection_name
    report["database_path"] = str(database_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report_summary(report, report_path)
    return report


def print_report_summary(report: dict[str, Any], report_path: Path) -> None:
    """Print aggregate metrics and a concise note for every failed query."""
    print(f"Evaluation mode: {report['evaluation_mode']}")
    print(f"Queries: {report['query_count']} | k={report['k']}")
    print(f"Macro precision@k: {report['macro_precision_at_k']:.3f}")
    print(f"Macro recall@k: {report['macro_recall_at_k']:.3f}")
    print(f"Micro precision@k: {report['micro_precision_at_k']:.3f}")
    print(f"Micro recall@k: {report['micro_recall_at_k']:.3f}")
    print(f"Hit@k rate: {report['hit_at_k_rate']:.3f}")
    print(f"Top-1 hit rate: {report['top1_hit_rate']:.3f}")
    for query in report["queries"]:
        if query["failures"]:
            print(f"Failure analysis [{query['query_id']}]:")
            for failure in query["failures"]:
                print(f"- {failure['type']}: {failure['likely_cause']}")
    print(f"Evaluation report: {report_path}")


def main() -> int:
    """Run fixture evaluation or evaluate the configured live vector collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--min-similarity", type=float, default=-1.0)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Use deterministic fixture vectors and no API credentials.",
    )
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--collection")
    arguments = parser.parse_args()
    load_dotenv()

    if arguments.k < 1:
        parser.error("--k must be at least 1.")
    if not math.isfinite(arguments.min_similarity) or not -1 <= arguments.min_similarity <= 1:
        parser.error("--min-similarity must be between -1 and 1.")

    try:
        if arguments.self_test:
            run_fixture_evaluation(
                arguments.labels,
                DEFAULT_SOURCE,
                arguments.report,
                arguments.k,
                arguments.min_similarity,
            )
            return 0
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        database_path = arguments.db_path or Path(
            os.getenv("VECTOR_DB_PATH", str(DEFAULT_DB_PATH))
        )
        collection_name = (
            arguments.collection
            or os.getenv("VECTOR_DB_COLLECTION", DEFAULT_COLLECTION)
        )
        report = run_live_evaluation(
            arguments.labels,
            arguments.report,
            database_path,
            collection_name,
            model,
            arguments.k,
            arguments.min_similarity,
        )
    except (
        EmbeddingAPIError,
        LLMConfigError,
        OSError,
        ValueError,
        VectorStoreError,
    ) as err:
        print(f"Retrieval evaluation failed: {err}", file=sys.stderr)
        return 1
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
