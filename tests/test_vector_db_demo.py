"""Tests for the ChromaDB collection setup and readback command."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.vector_db_demo import SMOKE_TEST_RECORD, main, run_demo


def test_run_demo_reads_back_vector_source_and_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("chromadb")
    actual = run_demo(
        [SMOKE_TEST_RECORD],
        tmp_path / "chroma",
        "demo_collection",
    )

    assert actual["chunk_id"] == SMOKE_TEST_RECORD["chunk_id"]
    assert actual["vector_dimension"] == len(SMOKE_TEST_RECORD["embedding"])
    assert actual["text"] == SMOKE_TEST_RECORD["text"]
    assert actual["metadata"] == SMOKE_TEST_RECORD["metadata"]
    output = capsys.readouterr().out
    assert "ChromaDB reachable: yes" in output
    assert "Collection vector dimension: 8" in output
    assert "Readback ID: fieldsure-chroma-readback-smoke-test" in output
    assert "Collection reachability test" in output


def test_main_self_test_uses_local_configured_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("chromadb")
    database_path = tmp_path / "configured-chroma"
    monkeypatch.setattr("sys.argv", ["vector_db_demo", "--self-test"])
    monkeypatch.setenv("VECTOR_DB_PATH", str(database_path))
    monkeypatch.setenv("VECTOR_DB_COLLECTION", "configured_collection")

    assert main() == 0

    assert database_path.exists()


def test_main_reports_missing_embedded_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("chromadb")
    missing_input = tmp_path / "missing.jsonl"
    monkeypatch.setattr("sys.argv", ["vector_db_demo", "--input", str(missing_input)])

    assert main() == 1
    assert "Vector database setup failed" in capsys.readouterr().err
