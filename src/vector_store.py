"""Persistent ChromaDB collection for embedding-backed source chunks."""

import json
import math
from pathlib import Path
from typing import Any, Sequence

import chromadb


class VectorStoreError(Exception):
    """Raised when vector storage or readback validation fails."""


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Encode source metadata into Chroma's scalar metadata fields."""
    if "metadata_json" in metadata:
        raise VectorStoreError(
            "Source metadata key 'metadata_json' is reserved by the vector store."
        )
    stored: dict[str, str | int | float | bool] = {
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    }
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) and not isinstance(value, bool):
            stored[key] = value
        elif isinstance(value, bool):
            stored[key] = value
        elif value is None:
            continue
        else:
            stored[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return stored


def _validate_records(records: Sequence[dict[str, Any]]) -> int:
    """Validate stored chunk records and return their common vector dimension."""
    if not records:
        raise VectorStoreError("At least one embedded chunk is required.")

    dimensions: set[int] = set()
    ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise VectorStoreError(f"Record {index} must be an object.")
        record_id = record.get("chunk_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise VectorStoreError(f"Record {index} is missing a chunk ID.")
        if record_id in ids:
            raise VectorStoreError(f"Duplicate chunk ID '{record_id}'.")
        ids.add(record_id)
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise VectorStoreError(f"Record '{record_id}' is missing source text.")
        if not isinstance(record.get("metadata"), dict):
            raise VectorStoreError(f"Record '{record_id}' is missing metadata.")
        vector = record.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise VectorStoreError(f"Record '{record_id}' is missing its vector.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise VectorStoreError(f"Record '{record_id}' has invalid vector values.")
        dimension = len(vector)
        declared_dimension = record.get("vector_dimension")
        if declared_dimension is not None and declared_dimension != dimension:
            raise VectorStoreError(
                f"Record '{record_id}' declares vector dimension "
                f"{declared_dimension}, but contains {dimension} values."
            )
        dimensions.add(dimension)

    if len(dimensions) != 1:
        raise VectorStoreError("All vectors in a collection must have one dimension.")
    return dimensions.pop()


class ChromaVectorStore:
    """Persist chunk vectors in a local ChromaDB cosine-similarity collection."""

    def __init__(self, path: Path, collection_name: str) -> None:
        if not collection_name.strip():
            raise ValueError("Chroma collection name must not be empty.")
        self.path = path
        try:
            self.client = chromadb.PersistentClient(path=str(path))
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as err:
            raise VectorStoreError(
                f"Could not open local ChromaDB collection '{collection_name}' "
                f"at '{path}': {err}"
            ) from err

    def upsert(self, records: Sequence[dict[str, Any]]) -> int:
        """Store chunks and bind the collection to the first vector dimension."""
        dimension = _validate_records(records)
        existing = self.collection.get(limit=1, include=["embeddings"])
        existing_embeddings = existing.get("embeddings")
        if existing_embeddings is not None and len(existing_embeddings):
            stored_dimension = len(existing_embeddings[0])
            if stored_dimension != dimension:
                raise VectorStoreError(
                    f"Collection contains {stored_dimension}-dimensional vectors; "
                    f"received {dimension}."
                )

        try:
            self.collection.upsert(
                ids=[record["chunk_id"] for record in records],
                embeddings=[
                    [float(value) for value in record["embedding"]]
                    for record in records
                ],
                documents=[record["text"] for record in records],
                metadatas=[
                    _chroma_metadata(record["metadata"]) for record in records
                ],
            )
        except Exception as err:
            raise VectorStoreError(
                f"Could not write records to the ChromaDB collection: {err}"
            ) from err
        return dimension

    def get(self, record_id: str) -> dict[str, Any]:
        """Read one stored record and restore its original metadata object."""
        try:
            result = self.collection.get(
                ids=[record_id],
                include=["embeddings", "documents", "metadatas"],
            )
        except Exception as err:
            raise VectorStoreError(
                f"Could not read record '{record_id}' from ChromaDB: {err}"
            ) from err

        ids = result.get("ids", [])
        if not ids:
            raise VectorStoreError(f"Record '{record_id}' was not found in ChromaDB.")
        metadata = result["metadatas"][0]
        try:
            source_metadata = json.loads(metadata.pop("metadata_json"))
        except (KeyError, TypeError, json.JSONDecodeError) as err:
            raise VectorStoreError(
                f"Record '{record_id}' has invalid stored source metadata."
            ) from err
        embedding = result["embeddings"][0]
        return {
            "chunk_id": ids[0],
            "text": result["documents"][0],
            "metadata": source_metadata,
            "embedding": [float(value) for value in embedding],
            "vector_dimension": len(embedding),
        }

    def count(self) -> int:
        """Return the number of records in the collection."""
        try:
            return self.collection.count()
        except Exception as err:
            raise VectorStoreError(
                f"Could not count ChromaDB collection records: {err}"
            ) from err

    def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return the nearest chunks with cosine similarity and source metadata."""
        if limit < 1:
            raise ValueError("Search result limit must be at least 1.")
        if not query_embedding or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in query_embedding
        ):
            raise ValueError("Query embedding must contain finite numeric values.")
        if self.count() == 0:
            raise VectorStoreError("Cannot search an empty ChromaDB collection.")

        try:
            result = self.collection.query(
                query_embeddings=[[float(value) for value in query_embedding]],
                n_results=limit,
                include=["distances", "documents", "metadatas"],
            )
        except Exception as err:
            raise VectorStoreError(f"ChromaDB similarity search failed: {err}") from err

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        if not (len(ids) == len(distances) == len(documents) == len(metadatas)):
            raise VectorStoreError("ChromaDB returned incomplete similarity results.")

        matches: list[dict[str, Any]] = []
        for record_id, distance, document, stored_metadata in zip(
            ids, distances, documents, metadatas, strict=True
        ):
            if distance is None or document is None or stored_metadata is None:
                raise VectorStoreError(
                    f"ChromaDB returned incomplete data for record '{record_id}'."
                )
            try:
                metadata = json.loads(stored_metadata["metadata_json"])
            except (KeyError, TypeError, json.JSONDecodeError) as err:
                raise VectorStoreError(
                    f"Record '{record_id}' has invalid stored source metadata."
                ) from err
            matches.append(
                {
                    "chunk_id": record_id,
                    "text": document,
                    "metadata": metadata,
                    "cosine_distance": float(distance),
                    "similarity_score": 1.0 - float(distance),
                }
            )
        return matches
