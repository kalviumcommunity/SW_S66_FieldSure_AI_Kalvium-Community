"""Tests for corpus embedding indexing and stored-record verification."""

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("chromadb")

from src.index_embeddings import main, run_indexing_job


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def _corpus_records(count: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_chunks = []
    embedded_chunks = []
    for index in range(count):
        chunk_id = f"source-chunk-{index}"
        source = {
            "chunk_id": chunk_id,
            "text": f"Source text for prepared corpus chunk {index}.",
            "metadata": {
                "source_file": f"guidance-{index}.md",
                "chunk_index": index,
                "section_heading": f"Section {index}",
                "page_number": index + 1,
            },
        }
        source_chunks.append(source)
        embedded_chunks.append(
            {
                **source,
                "embedding": [
                    round((index + 1) * (dimension + 1) / 100, 4)
                    for dimension in range(8)
                ],
                "vector_dimension": 8,
            }
        )
    return source_chunks, embedded_chunks


def test_indexing_stores_entire_corpus_and_verifies_sample_records(
    tmp_path: Path,
) -> None:
    source, embedded = _corpus_records()
    embedded_path = tmp_path / "embedded.jsonl"
    source_path = tmp_path / "source.jsonl"
    report_path = tmp_path / "summary.json"
    _write_jsonl(embedded_path, embedded)
    _write_jsonl(source_path, source)

    report = run_indexing_job(
        embedded_path,
        source_path,
        tmp_path / "chroma",
        "index_test",
        report_path,
    )

    assert report["status"] == "passed"
    assert report["source_chunk_count"] == 3
    assert report["embedded_chunk_count"] == 3
    assert report["indexed_record_count"] == 3
    assert report["indexed_count_matches_source"] is True
    assert report["vector_dimension"] == 8
    assert all(item["matches_source"] for item in report["spot_checks"])
    assert report["spot_checks"][0]["text"] == source[0]["text"]
    assert report["spot_checks"][0]["metadata"] == source[0]["metadata"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_indexing_reports_missing_embedding_and_count_mismatch(
    tmp_path: Path,
) -> None:
    source, embedded = _corpus_records()
    embedded_path = tmp_path / "embedded.jsonl"
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(embedded_path, embedded[:2])
    _write_jsonl(source_path, source)

    report = run_indexing_job(
        embedded_path,
        source_path,
        tmp_path / "chroma",
        "incomplete_index_test",
        tmp_path / "summary.json",
    )

    assert report["status"] == "failed"
    assert report["indexed_record_count"] == 2
    assert report["indexed_count_matches_source"] is False
    assert any("has no embedding" in item for item in report["failures"])
    assert any("does not match source chunk count" in item for item in report["failures"])


def test_indexing_is_idempotent_for_same_embedded_corpus(tmp_path: Path) -> None:
    source, embedded = _corpus_records()
    embedded_path = tmp_path / "embedded.jsonl"
    source_path = tmp_path / "source.jsonl"
    _write_jsonl(embedded_path, embedded)
    _write_jsonl(source_path, source)
    database_path = tmp_path / "chroma"

    first = run_indexing_job(
        embedded_path,
        source_path,
        database_path,
        "idempotent_index_test",
        tmp_path / "first.json",
    )
    second = run_indexing_job(
        embedded_path,
        source_path,
        database_path,
        "idempotent_index_test",
        tmp_path / "second.json",
    )

    assert first["indexed_record_count"] == 3
    assert second["indexed_record_count"] == 3
    assert second["status"] == "passed"


def test_main_returns_nonzero_when_input_files_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "index_embeddings",
            "--input",
            str(tmp_path / "missing-embedded.jsonl"),
            "--source",
            str(tmp_path / "missing-source.jsonl"),
            "--report",
            str(tmp_path / "summary.json"),
            "--db-path",
            str(tmp_path / "chroma"),
        ],
    )

    assert main() == 1
    assert "Embedding indexing failed" in capsys.readouterr().err
