"""Tests for known-relevance embedding quality checks."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import LLMConfig
from src.embedding_quality import (
    KNOWN_RELEVANCE_CASES,
    evaluate_quality_cases,
    main,
    run_quality_check,
    write_sanity_report,
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
    chunks = [
        {
            "chunk_id": case["expected_chunk_id"],
            "text": f"Prepared sample text for {case['name']}.",
            "metadata": {
                "source_file": f"{case['name'].split()[0]}_guidance.md",
                "chunk_index": index,
                "section_heading": case["name"],
            },
            "embedding": vector,
            "vector_dimension": 4,
        }
        for index, (case, vector) in enumerate(
            zip(
                KNOWN_RELEVANCE_CASES,
                (
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.02, 0.9998, 0.0],
                ),
            )
        )
    ]
    chunks.append(
        {
            "chunk_id": "doc_who_dengue_guidance_2024_c0001",
            "text": "Dengue is transmitted by Aedes mosquitoes.",
            "metadata": {
                "source_file": "who_dengue_guidance.md",
                "chunk_index": 1,
                "section_heading": "Dengue transmission",
            },
            "embedding": [0.0, 0.021, 0.9997, 0.0],
            "vector_dimension": 4,
        }
    )
    return chunks


def test_evaluate_quality_cases_reports_passes_and_borderline(
    embedded_chunks: list[dict[str, object]],
) -> None:
    query_embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.02, 0.9998, 0.0],
    ]

    report = evaluate_quality_cases(embedded_chunks, query_embeddings)

    assert report["test_count"] == 3
    assert report["pass_count"] == 2
    assert report["failure_count"] == 0
    assert report["borderline_count"] == 1
    assert [result["status"] for result in report["results"]] == [
        "passed",
        "passed",
        "borderline",
    ]
    assert report["results"][0]["top_source"] == "dengue_guidance.md"
    assert report["results"][2]["expected_rank"] == 1
    assert report["results"][2]["margin_over_runner_up"] < 0.05
    assert (
        report["results"][2]["runner_up_chunk_id"]
        == "doc_who_dengue_guidance_2024_c0001"
    )
    assert "borderline" in report["overall_note"].lower()


def test_evaluate_quality_cases_reports_misranked_expected_chunk(
    embedded_chunks: list[dict[str, object]],
) -> None:
    cases = [KNOWN_RELEVANCE_CASES[0]]
    report = evaluate_quality_cases(
        embedded_chunks,
        [[0.0, 1.0, 0.0, 0.0]],
        cases,
    )

    assert report["pass_count"] == 0
    assert report["failure_count"] == 1
    assert report["results"][0]["expected_rank"] != 1
    assert report["results"][0]["status"] == "failed"


def test_evaluate_quality_cases_rejects_missing_relevant_chunk(
    embedded_chunks: list[dict[str, object]],
) -> None:
    cases = [
        {
            "name": "missing expected",
            "query": "sample query",
            "expected_chunk_id": "unknown-chunk",
        }
    ]

    with pytest.raises(ValueError, match="missing from embedded corpus"):
        evaluate_quality_cases(embedded_chunks, [[1.0, 0.0, 0.0, 0.0]], cases)


@pytest.mark.parametrize(
    "query_embeddings, cases, threshold",
    [
        ([], KNOWN_RELEVANCE_CASES, 0.05),
        ([[1.0]], [KNOWN_RELEVANCE_CASES[0]], 0.05),
        ([[1.0, 0.0, 0.0, 0.0]], [KNOWN_RELEVANCE_CASES[0]], -0.01),
    ],
)
def test_evaluate_quality_cases_rejects_invalid_inputs(
    embedded_chunks: list[dict[str, object]],
    query_embeddings: list[list[float]],
    cases: tuple[dict[str, str], ...] | list[dict[str, str]],
    threshold: float,
) -> None:
    with pytest.raises(ValueError):
        evaluate_quality_cases(
            embedded_chunks,
            query_embeddings,
            cases,
            threshold,
        )


def test_write_sanity_report_persists_complete_json(
    tmp_path: Path,
) -> None:
    report = {
        "test_count": 1,
        "pass_count": 1,
        "failure_count": 0,
        "results": [{"top_source": "dengue_guidance.md", "top_score": 0.99}],
    }
    report_path = tmp_path / "reports" / "sanity.json"

    write_sanity_report(report_path, report)

    assert json.loads(report_path.read_text(encoding="utf-8")) == report


def test_run_quality_check_embeds_queries_and_saves_report(
    tmp_path: Path,
    embedded_chunks: list[dict[str, object]],
    test_config: LLMConfig,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus_path = tmp_path / "embedded.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in embedded_chunks),
        encoding="utf-8",
    )
    report_path = tmp_path / "sanity.json"
    query_embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.02, 0.9998, 0.0],
    ]

    with patch(
        "src.embedding_quality.generate_embeddings",
        return_value=query_embeddings,
    ) as generate:
        report = run_quality_check(
            corpus_path,
            report_path,
            "test-model",
            test_config,
        )

    generate.assert_called_once_with(
        [case["query"] for case in KNOWN_RELEVANCE_CASES],
        "test-model",
        test_config,
    )
    assert report["model"] == "test-model"
    assert report["corpus_path"] == str(corpus_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    output = capsys.readouterr().out
    assert "Tests: 3" in output
    assert "Passes: 2" in output
    assert "Failures: 0" in output
    assert "BORDERLINE: malaria urgent treatment" in output


def test_main_uses_embedding_model_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    test_config: LLMConfig,
) -> None:
    monkeypatch.setattr("sys.argv", ["embedding_quality"])
    monkeypatch.setenv("EMBEDDING_MODEL", "configured-test-model")
    monkeypatch.setattr(
        "src.embedding_quality.load_llm_config",
        lambda: test_config,
    )
    passing_report = {"failure_count": 0}

    with patch(
        "src.embedding_quality.run_quality_check",
        return_value=passing_report,
    ) as run_check:
        assert main() == 0

    assert run_check.call_args.args[0].name == "embedded_chunks.jsonl"
    assert run_check.call_args.args[1].name == "embedding_quality_sanity_report.json"
    assert run_check.call_args.args[2:] == ("configured-test-model", test_config)


def test_main_returns_failure_when_expected_relevance_is_misranked(
    monkeypatch: pytest.MonkeyPatch,
    test_config: LLMConfig,
) -> None:
    monkeypatch.setattr("sys.argv", ["embedding_quality"])
    monkeypatch.setenv("EMBEDDING_MODEL", "configured-test-model")
    monkeypatch.setattr(
        "src.embedding_quality.load_llm_config",
        lambda: test_config,
    )

    with patch(
        "src.embedding_quality.run_quality_check",
        return_value={"failure_count": 1},
    ):
        assert main() == 1
