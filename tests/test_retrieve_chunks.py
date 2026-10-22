"""Tests for ChromaDB top-k similarity retrieval."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("chromadb")

from src.config import LLMConfig
from src.retrieve_chunks import main, parse_k_values, retrieve_for_k_values
from src.vector_store import ChromaVectorStore, VectorStoreError


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
            "chunk_id": "dengue-001",
            "text": "Remove standing water to reduce mosquito breeding.",
            "metadata": {
                "source_file": "who_dengue.md",
                "chunk_index": 1,
                "section_heading": "Dengue prevention",
            },
            "embedding": [0.98, 0.2, 0.0, 0.0],
            "vector_dimension": 4,
        },
        {
            "chunk_id": "dengue-002",
            "text": "Use window screens and repellents to prevent dengue.",
            "metadata": {
                "source_file": "who_dengue.md",
                "chunk_index": 2,
                "section_heading": "Personal protection",
            },
            "embedding": [0.8, 0.4, 0.1, 0.0],
            "vector_dimension": 4,
        },
        {
            "chunk_id": "malaria-001",
            "text": "Use insecticide-treated nets to reduce malaria transmission.",
            "metadata": {
                "source_file": "moh_malaria.md",
                "chunk_index": 0,
                "section_heading": "Vector control",
            },
            "embedding": [0.2, 0.9, 0.1, 0.0],
            "vector_dimension": 4,
        },
        {
            "chunk_id": "library-001",
            "text": "Return borrowed library books before their due date.",
            "metadata": {
                "source_file": "library_guide.md",
                "chunk_index": 4,
                "section_heading": "Borrowing",
            },
            "embedding": [0.0, 0.0, 0.0, 1.0],
            "vector_dimension": 4,
        },
    ]


def test_chroma_search_returns_top_k_with_similarity_text_and_metadata(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "retrieval_test")
    store.upsert(_records())

    matches = store.search([1.0, 0.0, 0.0, 0.0], 3)

    assert [match["chunk_id"] for match in matches] == [
        "dengue-001",
        "dengue-002",
        "malaria-001",
    ]
    assert all(
        matches[index]["similarity_score"]
        >= matches[index + 1]["similarity_score"]
        for index in range(len(matches) - 1)
    )
    assert matches[0]["similarity_score"] == pytest.approx(0.979804)
    assert matches[0]["cosine_distance"] == pytest.approx(
        1 - matches[0]["similarity_score"]
    )
    assert matches[0]["text"] == _records()[0]["text"]
    assert matches[0]["metadata"] == _records()[0]["metadata"]


def test_chroma_search_caps_results_at_available_record_count(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "top_k_limit_test")
    store.upsert(_records()[:2])

    assert len(store.search([1.0, 0.0, 0.0, 0.0], 10)) == 2


@pytest.mark.parametrize(
    ("query_embedding", "limit", "message"),
    [
        ([1.0], 0, "at least 1"),
        ([], 1, "finite numeric values"),
        ([float("nan")], 1, "finite numeric values"),
    ],
)
def test_chroma_search_rejects_invalid_inputs(
    tmp_path: Path,
    query_embedding: list[float],
    limit: int,
    message: str,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "invalid_search_test")
    store.upsert(_records())

    with pytest.raises(ValueError, match=message):
        store.search(query_embedding, limit)


def test_chroma_search_rejects_empty_collection(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "empty_search_test")

    with pytest.raises(VectorStoreError, match="empty ChromaDB collection"):
        store.search([1.0, 0.0], 2)


def test_retrieve_for_k_values_embeds_once_and_runs_each_search(
    tmp_path: Path,
    test_config: LLMConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "query_retrieval_test")
    store.upsert(_records())
    with patch(
        "src.retrieve_chunks.generate_embeddings",
        return_value=[[1.0, 0.0, 0.0, 0.0]],
    ) as embed_query:
        results = retrieve_for_k_values(
            "How can workers prevent dengue?",
            "test-embedding-model",
            test_config,
            store,
            (1, 3),
        )

    embed_query.assert_called_once_with(
        ["How can workers prevent dengue?"],
        "test-embedding-model",
        test_config,
    )
    assert [item["chunk_id"] for item in results[1]] == ["dengue-001"]
    assert [item["chunk_id"] for item in results[3]] == [
        "dengue-001",
        "dengue-002",
        "malaria-001",
    ]
    output = capsys.readouterr().out
    assert "Top 1 results" in output
    assert "Top 3 results" in output
    assert "similarity=" in output
    assert "who_dengue.md" in output
    assert "Dengue prevention" in output


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2,5", (2, 5)), ("1", (1,))],
)
def test_parse_k_values(value: str, expected: tuple[int, ...]) -> None:
    assert parse_k_values(value) == expected


@pytest.mark.parametrize("value", ["", "0,2", "-1", "2,2", "1,word"])
def test_parse_k_values_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_k_values(value)


def test_main_reports_missing_embedding_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["retrieve_chunks", "--db-path", str(tmp_path / "chroma")],
    )
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    assert main() == 1
    assert "EMBEDDING_MODEL is not configured" in capsys.readouterr().err
