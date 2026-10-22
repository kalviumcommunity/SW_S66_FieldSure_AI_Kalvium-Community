"""Compare metadata-filtered vector retrieval with optional keyword hybrid search."""

import argparse
import math
import os
import re
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
DEFAULT_QUERY = "dengue mosquito prevention"
DEFAULT_FILTER_FIELD = "source_file"
DEFAULT_FILTER_VALUE = "data/sample_advisories/who_dengue_clinical_guidance.md"
DEFAULT_LIMIT = 5
DEFAULT_KEYWORD_WEIGHT = 0.3
PREVIEW_TEXT_CHARACTERS = 280
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def keyword_coverage(query: str, match: dict[str, Any]) -> float:
    """Return the fraction of distinct query terms found in chunk text/metadata."""
    query_terms = set(TOKEN_PATTERN.findall(query.casefold()))
    if not query_terms:
        return 0.0
    searchable = " ".join(
        (
            match["text"],
            str(match["metadata"]),
        )
    ).casefold()
    matched_terms = {
        term for term in query_terms if re.search(rf"\b{re.escape(term)}\b", searchable)
    }
    return len(matched_terms) / len(query_terms)


def hybrid_rank(
    query: str,
    matches: Sequence[dict[str, Any]],
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[dict[str, Any]]:
    """Rerank vector candidates using normalized cosine plus exact-term coverage."""
    if not 0 <= keyword_weight <= 1:
        raise ValueError("Keyword weight must be between 0 and 1.")
    ranked: list[dict[str, Any]] = []
    for match in matches:
        vector_score = match["similarity_score"]
        if not math.isfinite(vector_score):
            raise ValueError("Vector similarity scores must be finite.")
        keyword_score = keyword_coverage(query, match)
        normalized_vector_score = min(1.0, max(0.0, (vector_score + 1.0) / 2.0))
        ranked.append(
            {
                **match,
                "vector_similarity_score": vector_score,
                "keyword_score": keyword_score,
                "similarity_score": (
                    (1.0 - keyword_weight) * normalized_vector_score
                    + keyword_weight * keyword_score
                ),
            }
        )
    return sorted(
        ranked,
        key=lambda item: item["similarity_score"],
        reverse=True,
    )


def retrieve_filtered_comparison(
    query: str,
    model: str,
    config: LLMConfig,
    store: ChromaVectorStore,
    limit: int,
    where: dict[str, Any],
    use_hybrid: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Compare unfiltered vector search with filtered and hybrid retrieval."""
    if not query.strip():
        raise ValueError("Search query must not be empty.")
    if not model.strip():
        raise ValueError("Embedding model name must not be empty.")
    if limit < 1:
        raise ValueError("Result limit must be at least 1.")
    if not where:
        raise ValueError("Provide a non-empty metadata filter.")

    query_embedding = generate_embeddings([query], model, config)[0]
    unfiltered = store.search(query_embedding, limit)
    filtered = store.search(query_embedding, limit, where=where)
    results = {
        "unfiltered": unfiltered,
        "filtered": filtered,
    }
    if use_hybrid:
        candidate_limit = min(store.count(), max(limit * 4, limit))
        filtered_candidates = store.search(
            query_embedding,
            candidate_limit,
            where=where,
        )
        results["hybrid"] = hybrid_rank(query, filtered_candidates)[:limit]
    return results


def print_results(
    label: str,
    matches: Sequence[dict[str, Any]],
) -> None:
    """Print scores, source text, and metadata for a result set."""
    print(f"\n{label} ({len(matches)} results):")
    for rank, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        text = " ".join(match["text"].split())
        if len(text) > PREVIEW_TEXT_CHARACTERS:
            text = text[: PREVIEW_TEXT_CHARACTERS - 3] + "..."
        score_details = f"score={match['similarity_score']:.6f}"
        if "keyword_score" in match:
            score_details += (
                f" | vector={match['vector_similarity_score']:.6f}"
                f" | keyword={match['keyword_score']:.6f}"
            )
        print(
            f"#{rank} | {score_details} | ID={match['chunk_id']} | "
            f"Source={metadata.get('source_file', 'unknown')} | "
            f"Chunk index={metadata.get('chunk_index', 'unknown')} | "
            f"Section={metadata.get('section_heading', 'unknown')}"
        )
        print(f"Text: {text}")
        print(f"Metadata: {metadata}")


def main() -> int:
    """Run filtered/unfiltered vector search with optional hybrid reranking."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--k", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--where-field", default=DEFAULT_FILTER_FIELD)
    parser.add_argument("--where-value", default=DEFAULT_FILTER_VALUE)
    parser.add_argument("--vector-only", action="store_true")
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
        if not arguments.where_field.strip():
            raise ValueError("Metadata filter field must not be empty.")
        if not arguments.where_value.strip():
            raise ValueError("Metadata filter value must not be empty.")
        model = os.getenv("EMBEDDING_MODEL", "").strip()
        if not model:
            raise ValueError(
                "EMBEDDING_MODEL is not configured. Set it to the same model used "
                "to create the stored chunk embeddings."
            )
        config = load_llm_config()
        store = ChromaVectorStore(database_path, collection_name)
        where = {arguments.where_field: arguments.where_value}
        results = retrieve_filtered_comparison(
            arguments.query,
            model,
            config,
            store,
            arguments.k,
            where,
            use_hybrid=not arguments.vector_only,
        )
        print(f"Embedding model: {model}")
        print(f"Query: {arguments.query}")
        print(f"Metadata filter: {where}")
        print(f"Keyword hybrid enabled: {not arguments.vector_only}")
        print_results("Unfiltered vector search", results["unfiltered"])
        print_results("Metadata-filtered vector search", results["filtered"])
        if "hybrid" in results:
            print_results("Filtered hybrid search", results["hybrid"])
    except (
        EmbeddingAPIError,
        LLMConfigError,
        OSError,
        ValueError,
        VectorStoreError,
    ) as err:
        print(f"Filtered retrieval failed: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
