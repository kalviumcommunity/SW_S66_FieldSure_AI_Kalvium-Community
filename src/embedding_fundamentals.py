"""Generate text embeddings and compare their semantic similarity."""

import argparse
import math
import os
import sys
from typing import Any, Sequence

import httpx

from src.config import LLMConfig, LLMConfigError, load_llm_config

SAMPLE_TEXTS = (
    "How can I prevent dengue fever?",
    "What steps help avoid getting dengue?",
    "How do I renew a library book?",
)
PREVIEW_DIMENSIONS = 8


class EmbeddingAPIError(Exception):
    """Raised when the embeddings API fails or returns invalid data."""


def _build_endpoint_url(base_url: str) -> str:
    cleaned_url = base_url.rstrip("/")
    if cleaned_url.endswith("/embeddings"):
        return cleaned_url
    return f"{cleaned_url}/embeddings"


def generate_embeddings(
    texts: Sequence[str],
    model: str,
    config: LLMConfig,
) -> list[list[float]]:
    """Request and validate one embedding vector per input text."""
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Provide at least one non-empty text to embed.")

    try:
        with httpx.Client(timeout=config.request_timeout) as client:
            response = client.post(
                _build_endpoint_url(config.api_base_url),
                headers={
                    "Authorization": f"Bearer {config.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={"model": model, "input": list(texts)},
            )
    except httpx.HTTPError as err:
        raise EmbeddingAPIError(
            f"Could not connect to the embeddings API: {type(err).__name__}."
        ) from err

    if response.status_code != 200:
        raise EmbeddingAPIError(
            f"Embeddings API returned HTTP {response.status_code}."
        )

    try:
        body: Any = response.json()
    except ValueError as err:
        raise EmbeddingAPIError("Embeddings API returned invalid JSON.") from err

    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        raise EmbeddingAPIError("Embeddings API response must contain a data list.")

    records = body["data"]
    if len(records) != len(texts):
        raise EmbeddingAPIError(
            f"Expected {len(texts)} embedding vectors, received {len(records)}."
        )

    ordered_vectors: list[list[float] | None] = [None] * len(texts)
    for record in records:
        if not isinstance(record, dict):
            raise EmbeddingAPIError("Each embedding response item must be an object.")
        index = record.get("index")
        vector = record.get("embedding")
        if not isinstance(index, int) or isinstance(index, bool):
            raise EmbeddingAPIError(
                "Each embedding response item needs an integer index."
            )
        if index < 0 or index >= len(texts) or ordered_vectors[index] is not None:
            raise EmbeddingAPIError(
                "Embedding response contains an invalid or duplicate index."
            )
        if not isinstance(vector, list) or not vector:
            raise EmbeddingAPIError(
                "Each embedding must be a non-empty numeric vector."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            for value in vector
        ):
            raise EmbeddingAPIError("Embedding vectors must contain finite numbers.")
        try:
            converted_vector = [float(value) for value in vector]
        except OverflowError as err:
            raise EmbeddingAPIError(
                "Embedding vectors must contain finite numbers."
            ) from err
        if any(not math.isfinite(value) for value in converted_vector):
            raise EmbeddingAPIError("Embedding vectors must contain finite numbers.")
        ordered_vectors[index] = converted_vector

    if any(vector is None for vector in ordered_vectors):
        raise EmbeddingAPIError("Embeddings API response is missing an input vector.")

    vectors = [vector for vector in ordered_vectors if vector is not None]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise EmbeddingAPIError(
            "Embeddings API returned vectors with different dimensions."
        )
    return vectors


def embed_corpus_chunks(
    chunks: Sequence[dict[str, Any]],
    model: str,
    config: LLMConfig,
) -> list[dict[str, Any]]:
    """Embed prepared chunks while preserving source text and retrieval metadata."""
    if not chunks:
        raise ValueError("Provide at least one corpus chunk to embed.")

    texts: list[str] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Corpus chunk {index} must be an object.")
        if not isinstance(chunk.get("chunk_id"), str) or not chunk["chunk_id"]:
            raise ValueError(f"Corpus chunk {index} must contain a chunk ID.")
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError(f"Corpus chunk {index} must contain non-empty text.")
        if not isinstance(chunk.get("metadata"), dict):
            raise ValueError(f"Corpus chunk {index} must contain metadata.")
        texts.append(chunk["text"])

    vectors = generate_embeddings(texts, model, config)
    return [
        {
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": vector,
            "vector_dimension": len(vector),
        }
        for chunk, vector in zip(chunks, vectors)
    ]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return cosine similarity for two non-empty vectors of equal length."""
    if not first or len(first) != len(second):
        raise ValueError("Cosine similarity requires equal-length, non-empty vectors.")

    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    dot_product = sum(left * right for left, right in zip(first, second))
    return dot_product / (first_norm * second_norm)


def run_demo(model: str, config: LLMConfig) -> None:
    """Generate example vectors and print their dimensions and similarities."""
    vectors = generate_embeddings(SAMPLE_TEXTS, model, config)
    dimension = len(vectors[0])
    print(f"Model: {model}")
    print(f"Vector dimension: {dimension}")
    print(f"All {len(vectors)} vectors have the same length: yes")

    for text, vector in zip(SAMPLE_TEXTS, vectors):
        preview = ", ".join(f"{value:.6f}" for value in vector[:PREVIEW_DIMENSIONS])
        if len(vector) > PREVIEW_DIMENSIONS:
            preview += ", ..."
        print(f'\nText: "{text}"')
        preview_length = min(dimension, PREVIEW_DIMENSIONS)
        print(f"Vector (first {preview_length} values): [{preview}]")

    similar_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])
    print(f"\nSimilar pair cosine similarity: {similar_score:.6f}")
    print(f"Unrelated pair cosine similarity: {unrelated_score:.6f}")
    if similar_score <= unrelated_score:
        raise EmbeddingAPIError(
            "The configured embedding model did not score the similar pair higher."
        )


def main() -> int:
    """Run the command-line demonstration using the configured embeddings API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help=(
            "Embedding model name (default: EMBEDDING_MODEL or "
            "text-embedding-3-small)."
        ),
    )
    arguments = parser.parse_args()

    try:
        run_demo(arguments.model, load_llm_config())
    except (EmbeddingAPIError, LLMConfigError, ValueError) as err:
        print(f"Embedding demo failed: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
