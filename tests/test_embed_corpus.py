"""Tests for embedding prepared corpus chunks with retrieval metadata."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import LLMConfig
from src.embed_corpus import (
    load_corpus_chunks,
    main,
    run_embedding_job,
    write_embedded_chunks,
)
from src.embedding_fundamentals import embed_corpus_chunks


@pytest.fixture
def test_config() -> LLMConfig:
    return LLMConfig(
        api_base_url="https://example.test/v1",
        api_key="test-key",
        chat_model="unused",
    )


@pytest.fixture
def sample_chunks() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "dengue-000",
            "text": "Prevent dengue by reducing standing water.",
            "metadata": {
                "source_file": "who_dengue_guidance.md",
                "chunk_index": 0,
                "section_heading": "Prevention",
                "page_number": 2,
            },
        },
        {
            "chunk_id": "cholera-001",
            "text": "Treat drinking water before use.",
            "metadata": {
                "source_file": "cdc_cholera_advisory.txt",
                "chunk_index": 1,
                "section_heading": "Water safety",
                "page_number": None,
            },
        },
    ]


def test_load_corpus_chunks_reads_limit(
    tmp_path: Path,
    sample_chunks: list[dict[str, object]],
) -> None:
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in sample_chunks),
        encoding="utf-8",
    )

    assert load_corpus_chunks(source, limit=1) == sample_chunks[:1]


@pytest.mark.parametrize("limit", [0, -1])
def test_load_corpus_chunks_rejects_nonpositive_limit(
    tmp_path: Path,
    limit: int,
) -> None:
    source = tmp_path / "chunks.jsonl"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 1"):
        load_corpus_chunks(source, limit)


def test_load_corpus_chunks_rejects_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "chunks.jsonl"
    source.write_text("{invalid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_corpus_chunks(source)


def test_load_corpus_chunks_rejects_missing_metadata(tmp_path: Path) -> None:
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        json.dumps({"chunk_id": "chunk-1", "text": "some text"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metadata"):
        load_corpus_chunks(source)


def test_load_corpus_chunks_rejects_missing_chunk_id(tmp_path: Path) -> None:
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        json.dumps({"text": "some text", "metadata": {}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="chunk ID"):
        load_corpus_chunks(source)


def test_load_corpus_chunks_rejects_empty_corpus(tmp_path: Path) -> None:
    source = tmp_path / "chunks.jsonl"
    source.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No valid corpus chunks"):
        load_corpus_chunks(source)


def test_embed_corpus_chunks_preserves_source_and_metadata(
    test_config: LLMConfig,
    sample_chunks: list[dict[str, object]],
) -> None:
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    with patch(
        "src.embedding_fundamentals.generate_embeddings",
        return_value=vectors,
    ) as generate:
        embedded = embed_corpus_chunks(sample_chunks, "test-model", test_config)

    generate.assert_called_once_with(
        [chunk["text"] for chunk in sample_chunks],
        "test-model",
        test_config,
    )
    assert len(embedded) == len(sample_chunks)
    for original, stored, vector in zip(sample_chunks, embedded, vectors):
        assert stored["chunk_id"] == original["chunk_id"]
        assert stored["text"] == original["text"]
        assert stored["metadata"] == original["metadata"]
        assert stored["embedding"] == vector
        assert stored["vector_dimension"] == 3


def test_write_embedded_chunks_serializes_retrieval_record(
    tmp_path: Path,
    sample_chunks: list[dict[str, object]],
) -> None:
    output = tmp_path / "nested" / "embedded.jsonl"
    record = {
        **sample_chunks[0],
        "embedding": [0.1, 0.2, 0.3],
        "vector_dimension": 3,
    }

    write_embedded_chunks(output, [record])

    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_run_embedding_job_saves_vectors_and_reports_success(
    tmp_path: Path,
    test_config: LLMConfig,
    sample_chunks: list[dict[str, object]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "chunks.jsonl"
    source.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in sample_chunks),
        encoding="utf-8",
    )
    output = tmp_path / "embedded.jsonl"
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    with patch(
        "src.embedding_fundamentals.generate_embeddings",
        return_value=vectors,
    ):
        stored = run_embedding_job(source, output, "test-model", test_config)

    output_records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert output_records == stored
    assert len(output_records) == 2
    assert all(record["vector_dimension"] == 3 for record in output_records)
    report = capsys.readouterr().out
    assert "Chunks embedded: 2" in report
    assert "Vector length: 3" in report
    assert "who_dengue_guidance.md" in report
    assert "Prevention" in report
    assert "Text preview:" in report
    assert "[0.100000, 0.200000, 0.300000]" in report


def test_main_requires_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["embed_corpus"])
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setattr(
        "src.embed_corpus.load_llm_config",
        lambda: LLMConfig("https://example.test/v1", "test-key", "unused"),
    )

    assert main() == 1
    assert "EMBEDDING_MODEL is not configured" in capsys.readouterr().err


def test_main_uses_environment_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
    test_config: LLMConfig,
) -> None:
    monkeypatch.setattr("sys.argv", ["embed_corpus", "--limit", "2"])
    monkeypatch.setenv("EMBEDDING_MODEL", "configured-test-model")
    monkeypatch.setattr("src.embed_corpus.load_llm_config", lambda: test_config)

    with patch("src.embed_corpus.run_embedding_job") as run_job:
        assert main() == 0

    arguments = run_job.call_args.args
    assert arguments[0].name == "ingested_chunks.jsonl"
    assert arguments[1].name == "embedded_chunks.jsonl"
    assert arguments[2:] == ("configured-test-model", test_config, 2)
