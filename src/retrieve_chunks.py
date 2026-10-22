"""Embed a query and retrieve top-k source chunks from the ChromaDB collection."""

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from dotenv import load_dotenv

from src.config import LLMConfig, LLMConfigError, load_llm_config
from src.embedding_fundamentals import EmbeddingAPIError, generate_embeddings
from src.vector_store import ChromaVectorStore, VectorStoreError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "data" / "vectordb"
DEFAULT_COLLECTION = "fieldsure_chunks"
DEFAULT_QUERY = "How can field workers prevent dengue?"
DEFAULT_K_VALUES = (3, 5)
PREVIEW_TEXT_CHARACTERS = 280


def parse_k_values(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of unique positive top-k values."""
    try:
        k_values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as err:
        raise ValueError("Top-k values must be comma-separated positive integers.") from err
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("Top-k values must be positive integers.")
    if len(set(k_values)) != len(k_values):
        raise ValueError("Top-k values must not contain duplicates.")
    return k_values


def retrieve_for_k_values(
    query: str,
    model: str,
    config: LLMConfig,
    store: ChromaVectorStore,
    k_values: Sequence[int],
) -> dict[int, list[dict[str, Any]]]:
    """Embed one query with the corpus model and search for each requested k."""
    if not query.strip():
        raise ValueError("Search query must not be empty.")
    if not model.strip():
        raise ValueError("Embedding model name must not be empty.")
    if not k_values or any(k < 1 for k in k_values):
        raise ValueError("Provide one or more positive top-k values.")

    query_embedding = generate_embeddings([query], model, config)[0]
    results = {
        k: store.search(query_embedding, k)
        for k in k_values
    }

    print(f"Embedding model: {model}")
    print(f"Query: {query}")
    print(f"Query vector dimension: {len(query_embedding)}")
    for k, matches in results.items():
        print(f"\nTop {k} results ({len(matches)} returned):")
        for rank, match in enumerate(matches, start=1):
            metadata = match["metadata"]
            text = " ".join(match["text"].split())
            if len(text) > PREVIEW_TEXT_CHARACTERS:
                text = text[: PREVIEW_TEXT_CHARACTERS - 3] + "..."
            print(
                f"#{rank} | similarity={match['similarity_score']:.6f} | "
                f"distance={match['cosine_distance']:.6f} | "
                f"ID={match['chunk_id']} | "
                f"Source={metadata.get('source_file', 'unknown')} | "
                f"Chunk index={metadata.get('chunk_index', 'unknown')} | "
                f"Section={metadata.get('section_heading', 'unknown')}"
            )
            print(f"Text: {text}")
            print(f"Metadata: {metadata}")
    return results


def main() -> int:
    """Run query retrieval from the configured persistent collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument(
        "--k-values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="Comma-separated top-k values (default: 3,5).",
    )
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
        k_values = parse_k_values(arguments.k_values)
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            raise ValueError(
                "EMBEDDING_MODEL is not configured. Set it to the same model used "
                "to create the stored chunk embeddings."
            )
        config = load_llm_config()
        store = ChromaVectorStore(database_path, collection_name)
        retrieve_for_k_values(arguments.query, model, config, store, k_values)
    except (
        EmbeddingAPIError,
        LLMConfigError,
        OSError,
        ValueError,
        VectorStoreError,
    ) as err:
        print(f"Chunk retrieval failed: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
