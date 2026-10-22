"""Index all stored corpus embeddings into the configured ChromaDB collection."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

from src.rank_chunks import load_embedded_chunks
from src.vector_store import ChromaVectorStore, VectorStoreError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDED_INPUT = REPO_ROOT / "outputs" / "embedded_chunks.jsonl"
DEFAULT_SOURCE_INPUT = REPO_ROOT / "outputs" / "ingested_chunks.jsonl"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "indexing_summary.json"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "vectordb"
DEFAULT_COLLECTION = "fieldsure_chunks"


def load_source_chunks(input_path: Path) -> list[dict[str, Any]]:
    """Load source chunks and validate their IDs and retrieval content."""
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with input_path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
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
            chunk_id = chunk.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Missing chunk ID in {input_path} at line {line_number}."
                )
            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID '{chunk_id}' in {input_path}."
                )
            if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
                raise ValueError(
                    f"Missing source text for '{chunk_id}' in {input_path}."
                )
            if not isinstance(chunk.get("metadata"), dict):
                raise ValueError(
                    f"Missing metadata for '{chunk_id}' in {input_path}."
                )
            seen_ids.add(chunk_id)
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No source chunks found in {input_path}.")
    return chunks


def _check_source_alignment(
    embedded_chunks: list[dict[str, Any]],
    source_chunks: list[dict[str, Any]],
) -> list[str]:
    """Report absent, unexpected, or mismatched source chunk records."""
    embedded_by_id = {chunk["chunk_id"]: chunk for chunk in embedded_chunks}
    source_by_id = {chunk["chunk_id"]: chunk for chunk in source_chunks}
    failures: list[str] = []

    for chunk_id in sorted(source_by_id.keys() - embedded_by_id.keys()):
        failures.append(f"Source chunk '{chunk_id}' has no embedding.")
    for chunk_id in sorted(embedded_by_id.keys() - source_by_id.keys()):
        failures.append(f"Embedded chunk '{chunk_id}' is absent from the source corpus.")
    for chunk_id in sorted(source_by_id.keys() & embedded_by_id.keys()):
        embedded = embedded_by_id[chunk_id]
        source = source_by_id[chunk_id]
        if embedded["text"] != source["text"]:
            failures.append(f"Embedded text for '{chunk_id}' differs from its source.")
        if embedded["metadata"] != source["metadata"]:
            failures.append(
                f"Embedded metadata for '{chunk_id}' differs from its source."
            )
    return failures


def _record_matches(
    actual: dict[str, Any],
    expected: dict[str, Any],
    dimension: int,
) -> bool:
    """Compare a stored record to its source chunk and embedding."""
    return (
        actual["chunk_id"] == expected["chunk_id"]
        and actual["text"] == expected["text"]
        and actual["metadata"] == expected["metadata"]
        and actual["vector_dimension"] == dimension
        and len(actual["embedding"]) == dimension
        and all(
            math.isclose(stored, source, rel_tol=1e-6, abs_tol=1e-7)
            for stored, source in zip(
                actual["embedding"], expected["embedding"], strict=True
            )
        )
    )


def run_indexing_job(
    embedded_input: Path,
    source_input: Path,
    database_path: Path,
    collection_name: str,
    report_path: Path,
) -> dict[str, Any]:
    """Index all embedded chunks, compare counts, and spot-check records."""
    embedded_chunks = load_embedded_chunks(embedded_input)
    source_chunks = load_source_chunks(source_input)
    failures = _check_source_alignment(embedded_chunks, source_chunks)
    store = ChromaVectorStore(database_path, collection_name)
    batch_size = store.client.get_max_batch_size()
    if batch_size < 1:
        raise VectorStoreError("ChromaDB reported an invalid maximum batch size.")
    dimension = len(embedded_chunks[0]["embedding"])
    for start in range(0, len(embedded_chunks), batch_size):
        store.upsert(embedded_chunks[start : start + batch_size])
    indexed_count = store.count()
    count_matches = indexed_count == len(source_chunks)
    if not count_matches:
        failures.append(
            f"Indexed count {indexed_count} does not match source chunk count "
            f"{len(source_chunks)}."
        )

    spot_checks: list[dict[str, Any]] = []
    checked_ids: set[str] = set()
    for chunk in (embedded_chunks[0], embedded_chunks[-1]):
        chunk_id = chunk["chunk_id"]
        if chunk_id in checked_ids:
            continue
        checked_ids.add(chunk_id)
        actual = store.get(chunk_id)
        matches = _record_matches(actual, chunk, dimension)
        if not matches:
            failures.append(f"Stored record '{chunk_id}' failed its integrity check.")
        spot_checks.append(
            {
                "chunk_id": chunk_id,
                "matches_source": matches,
                "text": actual["text"],
                "metadata": actual["metadata"],
                "vector_length": actual["vector_dimension"],
                "vector_preview": actual["embedding"][:8],
            }
        )

    report = {
        "status": "passed" if not failures else "failed",
        "source_chunk_count": len(source_chunks),
        "embedded_chunk_count": len(embedded_chunks),
        "indexed_record_count": indexed_count,
        "indexed_count_matches_source": count_matches,
        "vector_dimension": dimension,
        "collection": collection_name,
        "database_path": str(database_path),
        "failures": failures,
        "spot_checks": spot_checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
        report_file.write("\n")

    print(f"Source chunks: {report['source_chunk_count']}")
    print(f"Embedded chunks: {report['embedded_chunk_count']}")
    print(f"Indexed records: {report['indexed_record_count']}")
    print(f"Indexed count matches source: {count_matches}")
    print(f"Vector dimension: {dimension}")
    print(f"Spot checks passed: {sum(item['matches_source'] for item in spot_checks)}")
    for item in spot_checks:
        source_file = item["metadata"].get("source_file", "unknown")
        section = item["metadata"].get("section_heading", "unknown")
        preview = ", ".join(f"{value:.4f}" for value in item["vector_preview"])
        print(
            f"Spot check: {item['chunk_id']} | matches source: "
            f"{item['matches_source']} | vector length: {item['vector_length']} | "
            f"source: {source_file} | section: {section}"
        )
        print(f"Text: {item['text']}")
        print(f"Vector preview: [{preview}]")
        print(f"Metadata: {json.dumps(item['metadata'], ensure_ascii=False, sort_keys=True)}")
    for failure in failures:
        print(f"Failure: {failure}", file=sys.stderr)
    print(f"Indexing report: {report_path}")
    return report


def main() -> int:
    """Index the entire configured embeddings JSONL corpus into ChromaDB."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_EMBEDDED_INPUT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--collection")
    arguments = parser.parse_args()
    load_dotenv()

    database_path = arguments.db_path or Path(
        os.getenv("VECTOR_DB_PATH", str(DEFAULT_DB_PATH))
    )
    collection_name = (
        arguments.collection or os.getenv("VECTOR_DB_COLLECTION", DEFAULT_COLLECTION)
    )
    try:
        report = run_indexing_job(
            arguments.input,
            arguments.source,
            database_path,
            collection_name,
            arguments.report,
        )
    except (OSError, ValueError, VectorStoreError) as err:
        print(f"Embedding indexing failed: {err}", file=sys.stderr)
        return 1
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
