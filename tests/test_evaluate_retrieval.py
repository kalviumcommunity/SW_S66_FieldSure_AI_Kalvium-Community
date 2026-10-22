"""Tests for labeled retrieval precision/recall evaluation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("chromadb")

from src.evaluate_retrieval import (
    DEFAULT_LABELS,
    DEFAULT_SOURCE,
    evaluate_labeled_query,
    evaluate_query_set,
    load_labeled_queries,
    main,
    run_fixture_evaluation,
)
from src.vector_store import ChromaVectorStore


def _labels() -> list[dict[str, object]]:
    return [
        {
            "query_id": "dengue",
            "query": "Which warning signs require urgent referral?",
            "relevant_chunk_ids": ["dengue-relevant"],
        },
        {
            "query_id": "cholera",
            "query": "What chlorine levels are required?",
            "relevant_chunk_ids": ["cholera-relevant"],
        },
    ]


def _records() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "irrelevant",
            "text": "Unrelated library instructions.",
            "metadata": {"source_file": "library.md", "chunk_index": 0},
            "embedding": [1.0, 0.0],
            "vector_dimension": 2,
        },
        {
            "chunk_id": "dengue-relevant",
            "text": "Dengue warning signs include persistent vomiting.",
            "metadata": {"source_file": "dengue.md", "chunk_index": 1},
            "embedding": [0.95, 0.3122498999],
            "vector_dimension": 2,
        },
        {
            "chunk_id": "cholera-relevant",
            "text": "Free residual chlorine targets for drinking water.",
            "metadata": {"source_file": "cholera.md", "chunk_index": 2},
            "embedding": [0.0, 1.0],
            "vector_dimension": 2,
        },
    ]


def test_load_labeled_queries_validates_and_preserves_known_ids(
    tmp_path: Path,
) -> None:
    labels_path = tmp_path / "labels.json"
    expected = _labels()
    labels_path.write_text(json.dumps(expected), encoding="utf-8")

    assert load_labeled_queries(labels_path) == expected


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"query_id": "one", "query": "query", "relevant_chunk_ids": []}],
        [
            {
                "query_id": "one",
                "query": "query",
                "relevant_chunk_ids": ["chunk"],
            },
            {
                "query_id": "one",
                "query": "second query",
                "relevant_chunk_ids": ["chunk"],
            },
        ],
    ],
)
def test_load_labeled_queries_rejects_invalid_labels(
    tmp_path: Path,
    payload: list[dict[str, object]],
) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_labeled_queries(labels_path)


def test_evaluation_computes_precision_recall_and_inspects_top1_miss(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "evaluation_test")
    store.upsert(_records())
    label = _labels()[0]

    result = evaluate_labeled_query(store, label, [1.0, 0.0], 2)

    assert result["relevant_retrieved_count"] == 1
    assert result["precision_at_k"] == pytest.approx(0.5)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["hit_at_k"] is True
    assert result["top1_hit"] is False
    assert result["retrieved"][0]["chunk_id"] == "irrelevant"
    assert result["retrieved"][1]["relevant"] is True
    assert any(item["type"] == "top1_relevant_miss" for item in result["failures"])


def test_evaluate_query_set_aggregates_query_metrics(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "aggregate_evaluation_test")
    store.upsert(_records())

    report = evaluate_query_set(
        store,
        _labels(),
        [[1.0, 0.0], [0.0, 1.0]],
        2,
    )

    assert report["query_count"] == 2
    assert report["macro_precision_at_k"] == pytest.approx(0.5)
    assert report["macro_recall_at_k"] == pytest.approx(1.0)
    assert report["micro_precision_at_k"] == pytest.approx(0.5)
    assert report["micro_recall_at_k"] == pytest.approx(1.0)
    assert report["hit_at_k_rate"] == 1.0
    assert report["top1_hit_rate"] == 0.5


def test_evaluation_reports_relevant_chunk_missing_from_index(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "missing_relevant_test")
    store.upsert([_records()[0]])

    result = evaluate_labeled_query(store, _labels()[0], [1.0, 0.0], 1)

    assert result["missing_relevant_chunk_ids"] == ["dengue-relevant"]
    assert result["recall_at_k"] == 0.0
    assert result["failures"][0]["type"] == "relevant_chunk_not_indexed"


def test_evaluation_reports_relevant_chunk_filtered_by_score_threshold(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "threshold_relevant_test")
    store.upsert(_records())

    result = evaluate_labeled_query(
        store,
        _labels()[0],
        [1.0, 0.0],
        3,
        min_similarity=0.97,
    )

    assert result["recall_at_k"] == 0.0
    assert result["failures"][0]["type"] == "relevant_chunk_below_score_threshold"


def test_fixture_evaluation_saves_report_and_inspects_top1_misses(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "evaluation.json"

    report = run_fixture_evaluation(
        DEFAULT_LABELS,
        DEFAULT_SOURCE,
        report_path,
        2,
    )

    assert report["query_count"] == 3
    assert report["evaluation_mode"] == "illustrative fixtures"
    assert report["macro_recall_at_k"] == pytest.approx(1.0)
    assert report["macro_precision_at_k"] == pytest.approx(0.5)
    assert report["top1_hit_rate"] == 0.0
    assert all(
        any(failure["type"] == "top1_relevant_miss" for failure in query["failures"])
        for query in report["queries"]
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_live_evaluation_embeds_query_set_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.evaluate_retrieval import run_live_evaluation

    labels_path = tmp_path / "labels.json"
    labels = [_labels()[0]]
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    monkeypatch.setattr(
        "src.evaluate_retrieval.load_llm_config",
        lambda: object(),
    )
    with patch(
        "src.evaluate_retrieval.generate_embeddings",
        return_value=[[1.0, 0.0]],
    ) as embed_queries:
        with patch("src.evaluate_retrieval.ChromaVectorStore") as store_class:
            store = store_class.return_value
            store.search.return_value = [
                {
                    "chunk_id": "dengue-relevant",
                    "text": "Dengue warning signs include persistent vomiting.",
                    "metadata": {"source_file": "dengue.md", "chunk_index": 1},
                    "similarity_score": 0.95,
                }
            ]
            store.get.return_value = {
                "chunk_id": "dengue-relevant",
                "text": "Dengue warning signs include persistent vomiting.",
                "metadata": {"source_file": "dengue.md", "chunk_index": 1},
                "embedding": [0.95, 0.3122498999],
                "vector_dimension": 2,
            }
            report = run_live_evaluation(
                labels_path,
                tmp_path / "report.json",
                tmp_path / "db",
                "evaluation",
                "same-model",
                1,
                -1.0,
            )

    embed_queries.assert_called_once()
    assert report["embedding_model"] == "same-model"
    assert report["macro_recall_at_k"] == 1.0
    store.client.close.assert_called_once()


def test_main_self_test_runs_without_embedding_api_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "fixture-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_retrieval",
            "--self-test",
            "--report",
            str(report_path),
        ],
    )

    assert main() == 0
    assert report_path.exists()
