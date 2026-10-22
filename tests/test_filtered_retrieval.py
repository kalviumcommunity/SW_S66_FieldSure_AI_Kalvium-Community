"""Tests for metadata filters and hybrid vector/keyword retrieval."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("chromadb")

from src.config import LLMConfig
from src.filtered_retrieval import (
    hybrid_rank,
    keyword_coverage,
    main,
    retrieve_filtered_comparison,
)
from src.vector_store import ChromaVectorStore


@pytest.fixture
def test_config() -> LLMConfig:
    return LLMConfig(
        api_base_url="https://example.test/v1",
        api_key="test-key",
        chat_model="unused",
    )


def _records() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "malaria-vector-control",
            "text": "Malaria prevention uses insecticide-treated nets.",
            "metadata": {
                "source_file": "moh_malaria.md",
                "chunk_index": 0,
                "section_heading": "Vector control",
            },
            "embedding": [1.0, 0.0, 0.0],
            "vector_dimension": 3,
        },
        {
            "chunk_id": "dengue-mosquito-prevention",
            "text": "Dengue prevention removes standing water and mosquito breeding sites.",
            "metadata": {
                "source_file": "who_dengue.md",
                "chunk_index": 1,
                "section_heading": "Dengue prevention",
            },
            "embedding": [0.9, 0.2, 0.0],
            "vector_dimension": 3,
        },
        {
            "chunk_id": "dengue-protection",
            "text": "Use repellents and window screens to reduce dengue mosquito bites.",
            "metadata": {
                "source_file": "who_dengue.md",
                "chunk_index": 2,
                "section_heading": "Personal protection",
            },
            "embedding": [0.75, 0.4, 0.1],
            "vector_dimension": 3,
        },
        {
            "chunk_id": "cholera-water",
            "text": "Treat drinking water and maintain chlorinated handwashing stations.",
            "metadata": {
                "source_file": "cdc_cholera.txt",
                "chunk_index": 0,
                "section_heading": "Water treatment",
            },
            "embedding": [0.6, 0.0, 0.8],
            "vector_dimension": 3,
        },
    ]


def test_metadata_filter_restricts_results_to_selected_source(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "source_filter_test")
    store.upsert(_records())

    unfiltered = store.search([1.0, 0.0, 0.0], 4)
    filtered = store.search(
        [1.0, 0.0, 0.0],
        4,
        where={"source_file": "who_dengue.md"},
    )

    assert len(unfiltered) == 4
    assert len(filtered) == 2
    assert {item["metadata"]["source_file"] for item in filtered} == {
        "who_dengue.md"
    }
    assert not {"malaria-vector-control", "cholera-water"} & {
        item["chunk_id"] for item in filtered
    }


def test_hybrid_ranking_promotes_exact_query_terms() -> None:
    candidates = [
        {
            "chunk_id": "loosely-related",
            "text": "Mosquito nets can reduce malaria risk.",
            "metadata": {"source_file": "malaria.md"},
            "similarity_score": 0.99,
            "cosine_distance": 0.01,
        },
        {
            "chunk_id": "exact-dengue-match",
            "text": "Dengue prevention reduces mosquito breeding.",
            "metadata": {"source_file": "dengue.md"},
            "similarity_score": 0.85,
            "cosine_distance": 0.15,
        },
    ]

    ranked = hybrid_rank("dengue mosquito prevention", candidates)

    assert ranked[0]["chunk_id"] == "exact-dengue-match"
    assert ranked[0]["keyword_score"] == 1.0
    assert ranked[0]["similarity_score"] > ranked[1]["similarity_score"]


def test_keyword_coverage_is_case_insensitive_and_distinct() -> None:
    match = {
        "text": "Dengue prevention protects against dengue.",
        "metadata": {"section_heading": "Mosquito control"},
    }

    assert keyword_coverage("DENGUE mosquito prevention", match) == pytest.approx(1.0)


def test_retrieve_comparison_embeds_query_once_and_returns_filtered_and_hybrid(
    tmp_path: Path,
    test_config: LLMConfig,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "comparison_test")
    store.upsert(_records())
    with patch(
        "src.filtered_retrieval.generate_embeddings",
        return_value=[[1.0, 0.0, 0.0]],
    ) as embed_query:
        results = retrieve_filtered_comparison(
            "dengue mosquito prevention",
            "test-embedding-model",
            test_config,
            store,
            3,
            {"source_file": "who_dengue.md"},
        )

    embed_query.assert_called_once_with(
        ["dengue mosquito prevention"],
        "test-embedding-model",
        test_config,
    )
    assert len(results["unfiltered"]) == 3
    assert len(results["filtered"]) == 2
    assert len(results["hybrid"]) == 2
    assert all(
        item["metadata"]["source_file"] == "who_dengue.md"
        for item in results["filtered"]
    )
    unfiltered_relevant = sum(
        item["metadata"]["source_file"] == "who_dengue.md"
        for item in results["unfiltered"][:2]
    )
    filtered_relevant = sum(
        item["metadata"]["source_file"] == "who_dengue.md"
        for item in results["filtered"][:2]
    )
    assert unfiltered_relevant / 2 == 0.5
    assert filtered_relevant / 2 == 1.0
    assert all("keyword_score" in item for item in results["hybrid"])


def test_retrieve_comparison_can_disable_hybrid(
    tmp_path: Path,
    test_config: LLMConfig,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "vector_only_test")
    store.upsert(_records())
    with patch(
        "src.filtered_retrieval.generate_embeddings",
        return_value=[[1.0, 0.0, 0.0]],
    ):
        results = retrieve_filtered_comparison(
            "dengue prevention",
            "test-embedding-model",
            test_config,
            store,
            2,
            {"source_file": "who_dengue.md"},
            use_hybrid=False,
        )

    assert set(results) == {"unfiltered", "filtered"}


@pytest.mark.parametrize(
    ("query", "limit", "where", "message"),
    [
        (" ", 2, {"source_file": "who_dengue.md"}, "query must not be empty"),
        ("dengue", 0, {"source_file": "who_dengue.md"}, "at least 1"),
        ("dengue", 2, {}, "non-empty metadata filter"),
    ],
)
def test_retrieve_comparison_rejects_invalid_parameters(
    tmp_path: Path,
    test_config: LLMConfig,
    query: str,
    limit: int,
    where: dict[str, str],
    message: str,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "invalid_compare_test")
    store.upsert(_records())

    with pytest.raises(ValueError, match=message):
        retrieve_filtered_comparison(
            query,
            "model",
            test_config,
            store,
            limit,
            where,
        )


def test_hybrid_rank_rejects_invalid_keyword_weight() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        hybrid_rank("dengue", [], keyword_weight=1.1)


def test_main_reports_missing_embedding_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["filtered_retrieval", "--db-path", str(tmp_path / "chroma")],
    )
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    assert main() == 1
    assert "EMBEDDING_MODEL is not configured" in capsys.readouterr().err


def test_main_prints_filtered_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_config: LLMConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database_path = tmp_path / "chroma"
    store = ChromaVectorStore(database_path, "cli_filter_test")
    store.upsert(_records())
    monkeypatch.setattr(
        "sys.argv",
        [
            "filtered_retrieval",
            "--query",
            "dengue prevention",
            "--where-field",
            "source_file",
            "--where-value",
            "who_dengue.md",
            "--k",
            "2",
            "--vector-only",
            "--db-path",
            str(database_path),
            "--collection",
            "cli_filter_test",
        ],
    )
    monkeypatch.setenv("EMBEDDING_MODEL", "configured-test-model")
    monkeypatch.setattr(
        "src.filtered_retrieval.load_llm_config",
        lambda: test_config,
    )
    monkeypatch.setattr(
        "src.filtered_retrieval.generate_embeddings",
        lambda texts, model, config: [[1.0, 0.0, 0.0]],
    )

    assert main() == 0

    output = capsys.readouterr().out
    assert "Unfiltered vector search" in output
    assert "Metadata-filtered vector search" in output
    assert "who_dengue.md" in output
    assert "Filtered hybrid search" not in output
