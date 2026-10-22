"""Embed a small prepared corpus and save vectors with their source metadata."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from src.config import LLMConfig, LLMConfigError, load_llm_config
from src.embedding_fundamentals import EmbeddingAPIError, embed_corpus_chunks

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "outputs" / "ingested_chunks.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "embedded_chunks.jsonl"
DEFAULT_CHUNK_LIMIT = 3
PREVIEW_DIMENSIONS = 8
PREVIEW_TEXT_CHARACTERS = 240


def load_corpus_chunks(
    input_path: Path,
    limit: int = DEFAULT_CHUNK_LIMIT,
) -> list[dict[str, Any]]:
    """Load and validate up to ``limit`` chunks from an ingestion JSONL file."""
    if limit < 1:
        raise ValueError("Chunk limit must be at least 1.")

    chunks: list[dict[str, Any]] = []
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
            chunks.append(chunk)
            if len(chunks) == limit:
                break

    if not chunks:
        raise ValueError(f"No valid corpus chunks found in {input_path}.")
    return chunks


def write_embedded_chunks(
    output_path: Path,
    embedded_chunks: list[dict[str, Any]],
) -> None:
    """Write embeddings and their source chunk data as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for chunk in embedded_chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def run_embedding_job(
    input_path: Path,
    output_path: Path,
    model: str,
    config: LLMConfig,
    limit: int = DEFAULT_CHUNK_LIMIT,
) -> list[dict[str, Any]]:
    """Embed prepared corpus chunks, save results, and print verification details."""
    chunks = load_corpus_chunks(input_path, limit)
    embedded_chunks = embed_corpus_chunks(chunks, model, config)
    write_embedded_chunks(output_path, embedded_chunks)

    dimension = embedded_chunks[0]["vector_dimension"]
    print(f"Model: {model}")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Vector length: {dimension}")
    print(f"Stored vectors: {output_path}")
    for chunk in embedded_chunks:
        preview = ", ".join(
            f"{value:.6f}" for value in chunk["embedding"][:PREVIEW_DIMENSIONS]
        )
        if dimension > PREVIEW_DIMENSIONS:
            preview += ", ..."
        metadata = chunk["metadata"]
        text_preview = " ".join(chunk["text"].split())
        if len(text_preview) > PREVIEW_TEXT_CHARACTERS:
            text_preview = text_preview[: PREVIEW_TEXT_CHARACTERS - 3] + "..."
        print(
            f'\nChunk: {chunk["chunk_id"]} | Source: '
            f'{metadata.get("source_file", "unknown")} | '
            f'Index: {metadata.get("chunk_index", "unknown")} | '
            f'Section: {metadata.get("section_heading", "unknown")}'
        )
        print(f'Text preview: "{text_preview}"')
        print(f"Vector preview: [{preview}]")
    return embedded_chunks


def main() -> int:
    """Run corpus embedding with API credentials and model from environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=DEFAULT_CHUNK_LIMIT)
    arguments = parser.parse_args()

    try:
        config = load_llm_config()
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            raise ValueError(
                "EMBEDDING_MODEL is not configured. Set it in your .env file."
            )
        run_embedding_job(
            arguments.input,
            arguments.output,
            model,
            config,
            arguments.limit,
        )
    except (EmbeddingAPIError, LLMConfigError, OSError, ValueError) as err:
        print(f"Corpus embedding failed: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
