"""Tests for query-to-chunk cosine similarity ranking."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import LLMConfig
from src.rank_chunks import (
    load_embedded_chunks,
    main,
    rank_chunks,
    run_ranking_job,
)


@pytest.fixture
def test_config() -> LLMConfig:
    return LLMConfig(
        api_base_url="https://example.test/v1",
        api_key="test-key",
        chat_model="unused",
    )


@pytest.fixture
def embedded_chunks() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "dengue-001",
            "text": "Remove standing water to reduce mosquito breeding.",
            "metadata": {
                "source_file": "who_dengue.md",
                "chunk_index": 1,
                "section_heading": "Dengue prevention",
            },
            "embedding": [0.9, 0.1, 0.0, 0.0],
            "vector_dimension": 4,
        },
        {
            "chunk_id": "malaria-002",
            "text": "Use insecticide-treated bed nets to prevent malaria.",
            "metadata": {
                "source_file": "who_malaria.md",
                "chunk_index": 2,
                "section_heading": "Vector control",
            },
            "embedding": [0.8, 0.2, 0.05, 0.0],
            "vector_dimension": 4,
        },
        {
            "chunk_id": "library-003",
            "text": "Return borrowed library books before their due date.",
            "metadata": {
                "source_file": "library_guide.md",
                "chunk_index": 3,
                "section_heading": "Borrowing",
            },
            "embedding": [0.0, 0.0, 1.0, 0.0],
            "vector_dimension": 4,
        },
    ]


def test_rank_chunks_orders_similar_to_unrelated_and_preserves_fields(
    embedded_chunks: list[dict[str, object]],
) -> None:
    ranked = rank_chunks([1.0, 0.0, 0.0, 0.0], embedded_chunks)

    assert [chunk["chunk_id"] for chunk in ranked] == [
        "dengue-001",
        "malaria-002",
        "library-003",
    ]
    assert ranked[0]["similarity_score"] == pytest.approx(0.993884)
    assert ranked[-1]["similarity_score"] == pytest.approx(0.0)
    assert ranked[0]["text"] == embedded_chunks[0]["text"]
    assert ranked[0]["metadata"] == embedded_chunks[0]["metadata"]


def test_rank_chunks_rejects_empty_query_or_corpus(
    embedded_chunks: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="Query embedding"):
        rank_chunks([], embedded_chunks)
    with pytest.raises(ValueError, match="chunks to rank"):
        rank_chunks([1.0], [])


def test_rank_chunks_rejects_incompatible_dimensions(
    embedded_chunks: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="equal-length"):
        rank_chunks([1.0], embedded_chunks)


def test_load_embedded_chunks_reads_valid_records(
    tmp_path: Path,
    embedded_chunks: list[dict[str, object]],
) -> None:
    source = tmp_path / "embeddings.jsonl"
    source.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in embedded_chunks),
        encoding="utf-8",
    )

    assert load_embedded_chunks(source) == embedded_chunks


@pytest.mark.parametrize(
    "records",
    [
        [{"chunk_id": "one", "text": "one", "metadata": {}, "embedding": [1.0]}],
        [
            {
                "chunk_id": "one",
                "text": "one",
                "metadata": {},
                "embedding": [1.0],
            },
            {
                "chunk_id": "two",
                "text": "two",
                "metadata": {},
                "embedding": [1.0, 2.0],
            },
        ],
        [
            {
                "chunk_id": "one",
                "text": "one",
                "metadata": {},
                "embedding": [float("nan")],
            },
            {
                "chunk_id": "two",
                "text": "two",
                "metadata": {},
                "embedding": [1.0],
            },
        ],
    ],
)
def test_load_embedded_chunks_rejects_invalid_vectors(
    tmp_path: Path,
    records: list[dict[str, object]],
) -> None:
    source = tmp_path / "embeddings.jsonl"
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_embedded_chunks(source)


def test_run_ranking_job_embeds_query_and_prints_ranked_results(
    tmp_path: Path,
    test_config: LLMConfig,
    embedded_chunks: list[dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "embeddings.jsonl"
    source.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in embedded_chunks),
        encoding="utf-8",
    )
    with patch(
        "src.rank_chunks.generate_embeddings",
        return_value=[[1.0, 0.0, 0.0, 0.0]],
    ) as generate:
        ranked = run_ranking_job(
            source,
            "How can workers reduce dengue risk?",
            "test-model",
            test_config,
        )

    generate.assert_called_once_with(
        ["How can workers reduce dengue risk?"],
        "test-model",
        test_config,
    )
    assert ranked[0]["chunk_id"] == "dengue-001"
    output = capsys.readouterr().out
    assert "cosine similarity" in output
    assert "Most similar: dengue-001" in output
    assert "Least similar: library-003" in output
    assert "who_dengue.md" in output
    assert "Dengue prevention" in output


def test_main_uses_embedding_model_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    test_config: LLMConfig,
) -> None:
    monkeypatch.setattr("sys.argv", ["rank_chunks"])
    monkeypatch.setenv("EMBEDDING_MODEL", "configured-test-model")
    monkeypatch.setattr("src.rank_chunks.load_llm_config", lambda: test_config)

    with patch("src.rank_chunks.run_ranking_job") as run_job:
        assert main() == 0

    assert run_job.call_args.args[0].name == "embedded_chunks.jsonl"
    assert run_job.call_args.args[1] == "How can field workers reduce dengue risk?"
    assert run_job.call_args.args[2:] == ("configured-test-model", test_config)
