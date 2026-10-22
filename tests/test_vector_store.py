"""Integration tests for persistent local ChromaDB storage."""

import pytest

chromadb = pytest.importorskip("chromadb")

from src.vector_store import ChromaVectorStore, VectorStoreError


@pytest.fixture
def vector_record() -> dict[str, object]:
    return {
        "chunk_id": "test-chunk-001",
        "text": "Example source text stored with its vector.",
        "metadata": {
            "source_file": "sample.md",
            "chunk_index": 1,
            "section_heading": "Example section",
            "page_number": None,
            "extra_metadata": {"file_type": "md", "tags": ["sample"]},
        },
        "embedding": [0.1, -0.2, 0.3, -0.4],
        "vector_dimension": 4,
    }


def test_persistent_collection_inserts_and_reads_source_record(
    tmp_path,
    vector_record: dict[str, object],
) -> None:
    path = tmp_path / "chroma"
    store = ChromaVectorStore(path, "test_chunks")

    dimension = store.upsert([vector_record])
    readback = store.get("test-chunk-001")

    assert dimension == 4
    assert store.count() == 1
    assert readback["chunk_id"] == vector_record["chunk_id"]
    assert readback["vector_dimension"] == 4
    assert readback["text"] == vector_record["text"]
    assert readback["metadata"] == vector_record["metadata"]
    assert readback["embedding"] == pytest.approx(vector_record["embedding"])
    assert store.collection.metadata["hnsw:space"] == "cosine"
    assert len(store.collection.get(include=["embeddings"])["embeddings"][0]) == 4

    reopened = ChromaVectorStore(path, "test_chunks")
    assert reopened.get("test-chunk-001") == readback


def test_upsert_rejects_vector_dimension_change(
    tmp_path,
    vector_record: dict[str, object],
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "dimension_test")
    store.upsert([vector_record])
    wrong_dimension = {
        **vector_record,
        "embedding": [0.1, 0.2],
        "vector_dimension": 2,
    }

    with pytest.raises(VectorStoreError, match="contains 4-dimensional"):
        store.upsert([wrong_dimension])


def test_upsert_rejects_duplicate_ids_in_batch(
    tmp_path,
    vector_record: dict[str, object],
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "duplicate_test")

    with pytest.raises(VectorStoreError, match="Duplicate chunk ID"):
        store.upsert([vector_record, vector_record])


def test_get_reports_missing_record(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "missing_test")

    with pytest.raises(VectorStoreError, match="was not found"):
        store.get("missing")
