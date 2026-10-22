"""Create a persistent ChromaDB collection and verify record readback."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

from src.rank_chunks import load_embedded_chunks
from src.vector_store import ChromaVectorStore, VectorStoreError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "outputs" / "embedded_chunks.jsonl"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "vectordb"
DEFAULT_COLLECTION = "fieldsure_chunks"
SMOKE_TEST_COLLECTION_SUFFIX = "_smoke_test"
SMOKE_TEST_RECORD = {
    "chunk_id": "fieldsure-chroma-readback-smoke-test",
    "text": "Smoke test record: vector, source text, and metadata persisted in ChromaDB.",
    "metadata": {
        "source_file": "vector_db_smoke_test",
        "chunk_index": 0,
        "section_heading": "Collection reachability test",
        "page_number": None,
    },
    "embedding": [0.12, -0.08, 0.34, 0.56, -0.23, 0.45, 0.67, -0.19],
    "vector_dimension": 8,
}


def run_demo(
    records: list[dict[str, Any]],
    database_path: Path,
    collection_name: str,
) -> dict[str, Any]:
    """Persist records and verify that the first source record reads back."""
    store = ChromaVectorStore(database_path, collection_name)
    dimension = store.upsert(records)
    expected = records[0]
    actual = store.get(expected["chunk_id"])

    if (
        actual["text"] != expected["text"]
        or actual["metadata"] != expected["metadata"]
        or actual["vector_dimension"] != dimension
        or len(actual["embedding"]) != dimension
    ):
        raise VectorStoreError(
            f"ChromaDB readback for '{expected['chunk_id']}' did not match "
            "the inserted source record."
        )

    print(f"ChromaDB reachable: yes ({database_path})")
    print(f"Collection: {collection_name}")
    print(f"Collection vector dimension: {dimension}")
    print(f"Records in collection: {store.count()}")
    print(f"Readback ID: {actual['chunk_id']}")
    print(f"Vector length: {actual['vector_dimension']}")
    vector_preview = ", ".join(f"{value:.4f}" for value in actual["embedding"][:8])
    print(f"Vector preview: [{vector_preview}]")
    print(f"Text: {actual['text']}")
    print(
        "Metadata: "
        + json.dumps(actual["metadata"], ensure_ascii=False, sort_keys=True)
    )
    return actual


def main() -> int:
    """Run the ChromaDB smoke test or import an embedded JSONL corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--collection")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Insert and read a clearly labeled 8-dimensional test vector.",
    )
    arguments = parser.parse_args()
    load_dotenv()

    database_path = arguments.db_path or Path(
        os.getenv("VECTOR_DB_PATH", str(DEFAULT_DB_PATH))
    )
    collection_name = (
        arguments.collection or os.getenv("VECTOR_DB_COLLECTION", DEFAULT_COLLECTION)
    )
    if arguments.self_test and not arguments.collection:
        collection_name = f"{collection_name}{SMOKE_TEST_COLLECTION_SUFFIX}"

    try:
        records = (
            [SMOKE_TEST_RECORD]
            if arguments.self_test
            else load_embedded_chunks(arguments.input)
        )
        readback = run_demo(records, database_path, collection_name)
        if arguments.self_test:
            print("Record type: illustrative smoke-test fixture (not a model embedding)")
        return 0 if readback["chunk_id"] == records[0]["chunk_id"] else 1
    except (OSError, ValueError, VectorStoreError) as err:
        print(f"Vector database setup failed: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
