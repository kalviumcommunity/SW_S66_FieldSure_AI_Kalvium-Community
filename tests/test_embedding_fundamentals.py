"""Tests for the embedding demonstration and API response validation."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.config import LLMConfig
from src.embedding_fundamentals import (
    EmbeddingAPIError,
    cosine_similarity,
    generate_embeddings,
    run_demo,
)


@pytest.fixture
def test_config() -> LLMConfig:
    return LLMConfig(
        api_base_url="https://example.test/v1/",
        api_key="test-key",
        chat_model="unused",
        request_timeout=10.0,
    )


def test_generate_embeddings_sends_texts_and_orders_vectors(
    test_config: LLMConfig,
) -> None:
    response_body = {
        "data": [
            {"index": 1, "embedding": [0.8, 0.6]},
            {"index": 0, "embedding": [1, 0]},
        ]
    }
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = response_body

    with patch(
        "src.embedding_fundamentals.httpx.Client.post",
        return_value=response,
    ) as post:
        vectors = generate_embeddings(["first", "second"], "example-model", test_config)

    request = post.call_args
    assert request.args[0] == "https://example.test/v1/embeddings"
    assert request.kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert request.kwargs["json"] == {
        "model": "example-model",
        "input": ["first", "second"],
    }
    assert vectors == [[1.0, 0.0], [0.8, 0.6]]
    assert len({len(vector) for vector in vectors}) == 1


def test_cosine_similarity_scores_similar_pair_higher() -> None:
    dengue_prevention = [0.8, 0.2, 0.1, 0.0]
    avoid_dengue = [0.75, 0.25, 0.12, 0.05]
    library_book = [0.05, 0.1, 0.95, 0.2]

    assert cosine_similarity(dengue_prevention, avoid_dengue) > cosine_similarity(
        dengue_prevention, library_book
    )


def test_run_demo_reports_dimensions_vectors_and_similarity(capsys) -> None:
    vectors = [
        [0.8, 0.2, 0.1, 0.0],
        [0.75, 0.25, 0.12, 0.05],
        [0.05, 0.1, 0.95, 0.2],
    ]
    config = LLMConfig("https://example.test/v1", "test-key", "unused")

    with patch("src.embedding_fundamentals.generate_embeddings", return_value=vectors):
        run_demo("example-model", config)

    output = capsys.readouterr().out
    assert "Vector dimension: 4" in output
    assert "All 3 vectors have the same length: yes" in output
    assert "Vector (first 4 values): [0.800000, 0.200000, 0.100000, 0.000000]" in output
    assert "Similar pair cosine similarity: 0.994717" in output
    assert "Unrelated pair cosine similarity: 0.190944" in output


@pytest.mark.parametrize(
    "first, second",
    [([], []), ([1.0], [1.0, 2.0]), ([0.0, 0.0], [1.0, 2.0])],
)
def test_cosine_similarity_rejects_invalid_vectors(
    first: list[float], second: list[float]
) -> None:
    with pytest.raises(ValueError):
        cosine_similarity(first, second)


@pytest.mark.parametrize(
    "records",
    [
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [0.0]}],
        [{"index": 0, "embedding": [1.0]}],
        [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}],
        [{"index": 0, "embedding": [float("nan")]}, {"index": 1, "embedding": [1.0]}],
    ],
)
def test_generate_embeddings_rejects_invalid_response(
    test_config: LLMConfig, records: list[dict[str, object]]
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.return_value = {"data": records}

    with patch("src.embedding_fundamentals.httpx.Client.post", return_value=response):
        with pytest.raises(EmbeddingAPIError):
            generate_embeddings(["first", "second"], "example-model", test_config)


def test_generate_embeddings_rejects_http_error(test_config: LLMConfig) -> None:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 401

    with patch("src.embedding_fundamentals.httpx.Client.post", return_value=response):
        with pytest.raises(EmbeddingAPIError, match="HTTP 401"):
            generate_embeddings(["text"], "example-model", test_config)


def test_generate_embeddings_rejects_empty_input(test_config: LLMConfig) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        generate_embeddings([" ", "text"], "example-model", test_config)


def test_generate_embeddings_rejects_invalid_json(test_config: LLMConfig) -> None:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.json.side_effect = json.JSONDecodeError("invalid", "x", 0)

    with patch("src.embedding_fundamentals.httpx.Client.post", return_value=response):
        with pytest.raises(EmbeddingAPIError, match="invalid JSON"):
            generate_embeddings(["text"], "example-model", test_config)
