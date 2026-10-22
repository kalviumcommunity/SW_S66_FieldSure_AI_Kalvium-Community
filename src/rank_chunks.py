"""Rank stored corpus embeddings against a query using cosine similarity."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from src.config import LLMConfig, LLMConfigError, load_llm_config
from src.embedding_fundamentals import (
    EmbeddingAPIError,
    cosine_similarity,
    generate_embeddings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "outputs" / "embedded_chunks.jsonl"
DEFAULT_QUERY = "How can field workers reduce dengue risk?"
PREVIEW_TEXT_CHARACTERS = 240


def load_embedded_chunks(input_path: Path) -> list[dict[str, Any]]:
    """Load stored chunk embeddings and validate their retrieval fields."""
    chunks: list[dict[str, Any]] = []
    dimensions: set[int] = set()
    with input_path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                chunk: Any = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}."
                ) from err
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Expected a JSON object in {input_path} at line {line_number}."
                )
            if not isinstance(chunk.get("chunk_id"), str) or not chunk["chunk_id"]:
                raise ValueError(
                    f"Missing chunk ID in {input_path} at line {line_number}."
                )
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(
                    f"Missing non-empty chunk text in {input_path} "
                    f"at line {line_number}."
                )
            if not isinstance(chunk.get("metadata"), dict):
                raise ValueError(
                    f"Missing chunk metadata in {input_path} at line {line_number}."
                )
            vector = chunk.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise ValueError(
                    f"Missing embedding vector in {input_path} at line {line_number}."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    f"Invalid embedding values in {input_path} at line {line_number}."
                )
            dimensions.add(len(vector))
            chunks.append(chunk)

    if len(chunks) < 2:
        raise ValueError("At least two embedded chunks are needed for ranking.")
    if len(dimensions) != 1:
        raise ValueError("Stored chunk embeddings must all have the same dimension.")
    return chunks


def rank_chunks(
    query_embedding: Sequence[float],
    chunks: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return chunks sorted from highest to lowest cosine similarity."""
    if not chunks:
        raise ValueError("Provide embedded chunks to rank.")
    if not query_embedding:
        raise ValueError("Query embedding must not be empty.")

    ranked_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        vector = chunk.get("embedding")
        if not isinstance(vector, list):
            raise ValueError("Every ranked chunk must contain an embedding vector.")
        score = cosine_similarity(query_embedding, vector)
        ranked_chunks.append({**chunk, "similarity_score": score})
    return sorted(
        ranked_chunks,
        key=lambda chunk: chunk["similarity_score"],
        reverse=True,
    )


def run_ranking_job(
    input_path: Path,
    query: str,
    model: str,
    config: LLMConfig,
) -> list[dict[str, Any]]:
    """Embed a query, rank stored chunks, and report the most and least related."""
    if not query.strip():
        raise ValueError("Search query must not be empty.")

    chunks = load_embedded_chunks(input_path)
    query_embedding = generate_embeddings([query], model, config)[0]
    ranked_chunks = rank_chunks(query_embedding, chunks)

    print(
        "Metric: cosine similarity (compares vector direction "
        "independent of magnitude)"
    )
    print(f'Model: {model} | Query: "{query}"')
    print(f"Chunks ranked: {len(ranked_chunks)}")
    for rank, chunk in enumerate(ranked_chunks, start=1):
        metadata = chunk["metadata"]
        text_preview = " ".join(chunk["text"].split())
        if len(text_preview) > PREVIEW_TEXT_CHARACTERS:
            text_preview = text_preview[: PREVIEW_TEXT_CHARACTERS - 3] + "..."
        print(
            f'\n#{rank} | cosine={chunk["similarity_score"]:.6f} | '
            f'Chunk: {chunk["chunk_id"]} | '
            f'Source: {metadata.get("source_file", "unknown")} | '
            f'Index: {metadata.get("chunk_index", "unknown")} | '
            f'Section: {metadata.get("section_heading", "unknown")}'
        )
        print(f'Text preview: "{text_preview}"')

    print(f'\nMost similar: {ranked_chunks[0]["chunk_id"]}')
    print(f'Least similar: {ranked_chunks[-1]["chunk_id"]}')
    return ranked_chunks


def main() -> int:
    """Run similarity ranking using the configured API and embeddings model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    arguments = parser.parse_args()

    try:
        config = load_llm_config()
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            raise ValueError(
                "EMBEDDING_MODEL is not configured. Set it in your .env file."
            )
        run_ranking_job(arguments.input, arguments.query, model, config)
    except (EmbeddingAPIError, LLMConfigError, OSError, ValueError) as err:
        print(f"Chunk ranking failed: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
