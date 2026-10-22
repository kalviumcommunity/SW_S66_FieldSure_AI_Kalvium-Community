"""Tests for the deterministic retrieval relevance-tuning experiment."""

import json
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from src.tune_retrieval import (
    DEFAULT_SOURCE,
    QUERY_CASES,
    SETTINGS,
    _make_fixture_vectors,
    run_tuning_experiment,
)


def _write_source_chunks(path: Path) -> None:
    sources = [
        "data/sample_advisories/who_dengue_clinical_guidance.md",
        "data/sample_advisories/cdc_cholera_water_safety.txt",
        "data/sample_advisories/moh_malaria_rapid_triage.html",
    ]
    path.write_text(
        "".join(
            json.dumps(
                {
                    "chunk_id": f"chunk-{index}",
                    "text": f"Prepared source chunk {index}",
                    "metadata": {
                        "source_file": source,
                        "section_heading": f"Section {index}",
                        "chunk_index": index,
                    },
                }
            )
            + "\n"
            for index, source in enumerate(sources)
        ),
        encoding="utf-8",
    )


def test_fixture_vectors_mark_expected_source_and_an_off_source_distractor() -> None:
    chunks = [
        {
            "chunk_id": "expected-1",
            "text": "Relevant source content.",
            "metadata": {"source_file": "expected.md"},
        },
        {
            "chunk_id": "other-1",
            "text": "Distractor content.",
            "metadata": {"source_file": "other.md"},
        },
    ]

    fixture_chunks, relevant_ids = _make_fixture_vectors(chunks, "expected.md")

    assert relevant_ids == {"expected-1"}
    assert fixture_chunks[1]["embedding"] == [1.0, 0.0, 0.0]
    assert fixture_chunks[0]["embedding"][0] == pytest.approx(0.98)
    assert all(chunk["vector_dimension"] == 3 for chunk in fixture_chunks)


def test_tuning_experiment_selects_filter_and_threshold_setting(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.jsonl"
    report_path = tmp_path / "retrieval_tuning.json"
    _write_source_chunks(source_path)

    report = run_tuning_experiment(source_path, report_path)

    assert report["status"] == "passed"
    assert report["evaluation_mode"].startswith("deterministic illustrative")
    assert len(report["queries"]) == 3
    assert len(report["settings"]) == 2
    assert report["chosen_setting"] == "source_filter_k5_threshold_0_5"
    baseline = report["settings"][0]["metrics"]
    tuned = report["settings"][1]["metrics"]
    assert baseline["mean_precision_at_k"] == pytest.approx(0.5)
    assert baseline["top1_hit_rate"] == 0.0
    assert tuned["mean_precision_at_k"] == pytest.approx(1.0)
    assert tuned["mean_recall_at_k"] == pytest.approx(1.0)
    assert tuned["mean_f1_at_k"] == pytest.approx(1.0)
    for case in report["queries"]:
        assert len(case["expected_relevant_chunk_ids"]) == 1
        assert (
            case["settings"][0]["results"][0]["metadata"]["source_file"]
            != case["expected_source"]
        )
        assert case["settings"][1]["precision_at_k"] == 1.0
        assert case["settings"][1]["recall_at_k"] == 1.0
        assert case["settings"][1]["results"][0]["expected_relevant"] is True
    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_tuning_case_expected_sources_exist_in_sample_corpus() -> None:
    source_records = [
        json.loads(line)
        for line in DEFAULT_SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    available_sources = {
        record["metadata"]["source_file"] for record in source_records
    }

    assert all(query["expected_source"] in available_sources for query in QUERY_CASES)
    assert {setting["name"] for setting in SETTINGS} == {
        "baseline_vector_k2",
        "source_filter_k5_threshold_0_5",
    }


def test_tuning_rejects_query_when_expected_source_is_missing(
    tmp_path: Path,
) -> None:
    chunks = [
        {
            "chunk_id": "chunk-1",
            "text": "One source chunk.",
            "metadata": {"source_file": "only.md"},
        }
    ]

    with pytest.raises(ValueError, match="No source chunks found"):
        _make_fixture_vectors(chunks, "missing.md")


def test_main_reports_missing_source_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_retrieval",
            "--source",
            str(tmp_path / "missing.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ],
    )

    with pytest.raises(SystemExit) as error:
        from src.tune_retrieval import main

        main()

    assert error.value.code == 1
    assert "Retrieval tuning failed" in capsys.readouterr().err
