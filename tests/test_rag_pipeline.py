"""Unit and integration tests for RAG pipeline architecture & flow design (Module #28)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.rag_pipeline import (
    DEFAULT_MIN_SIMILARITY,
    FALLBACK_NO_CONTEXT,
    answer_query,
    assemble_context,
    build_rag_prompt,
    deterministic_text_embedding,
    embed_query,
    generate_answer,
    main,
    populate_vector_store,
    retrieve_context,
    run_pipeline_demo,
)
from src.vector_store import ChromaVectorStore, VectorStoreError


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    """Fixture providing representative public health chunks."""
    return [
        {
            "chunk_id": "doc_who_dengue_c0001",
            "text": "Dengue is transmitted by Aedes mosquitoes. Warning signs include severe abdominal pain and vomiting.",
            "metadata": {
                "source_file": "data/sample_advisories/who_dengue_clinical_guidance.md",
                "document_title": "WHO Dengue Guidance",
                "section_heading": "Warning Signs",
                "doc_id": "doc_who_dengue",
                "chunk_index": 1,
            },
            "similarity_score": 0.88,
            "cosine_distance": 0.12,
        },
        {
            "chunk_id": "doc_cdc_cholera_c0001",
            "text": "Cholera is a waterborne infection. Water disinfection requires NaDCC or chlorine.",
            "metadata": {
                "source_file": "data/sample_advisories/cdc_cholera_water_safety.txt",
                "document_title": "CDC Cholera Safety",
                "section_heading": "Water Disinfection",
                "doc_id": "doc_cdc_cholera",
                "chunk_index": 1,
            },
            "similarity_score": 0.45,
            "cosine_distance": 0.55,
        },
    ]


class MockVectorStore:
    """Mock vector store for isolated pipeline stage testing."""

    def __init__(self, matches: Sequence[dict[str, Any]]) -> None:
        self.matches = list(matches)
        self.last_query_vector: list[float] | None = None
        self.last_limit: int | None = None

    def search(
        self,
        query_embedding: Sequence[float],
        limit: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.last_query_vector = list(query_embedding)
        self.last_limit = limit
        results = self.matches
        if where:
            results = [
                m for m in results
                if all(m.get("metadata", {}).get(k) == v for k, v in where.items())
            ]
        return results[:limit]


# ------------------------------------------------------------------------------
# Stage 1: Embed Query Tests
# ------------------------------------------------------------------------------

def test_embed_query_validation():
    """Verify empty query inputs raise ValueError."""
    with pytest.raises(ValueError, match="Query must not be empty"):
        embed_query("")
    with pytest.raises(ValueError, match="Query must not be empty"):
        embed_query("   ")


def test_embed_query_with_custom_fn():
    """Verify injected embedding callable is used cleanly."""
    mock_fn = MagicMock(return_value=[0.1, 0.2, 0.3])
    vector = embed_query("How to prevent dengue?", embed_fn=mock_fn)
    assert vector == [0.1, 0.2, 0.3]
    mock_fn.assert_called_once_with("How to prevent dengue?")


def test_deterministic_text_embedding():
    """Verify deterministic vector generation has unit length and stable values."""
    vec1 = deterministic_text_embedding("dengue warning signs", dimension=64)
    vec2 = deterministic_text_embedding("dengue warning signs", dimension=64)
    assert len(vec1) == 64
    assert vec1 == vec2

    # Verify unit normalization: sum of squares ≈ 1.0
    norm_sq = sum(v * v for v in vec1)
    assert pytest.approx(norm_sq, abs=1e-4) == 1.0


# ------------------------------------------------------------------------------
# Stage 2: Retrieve Context Tests
# ------------------------------------------------------------------------------

def test_retrieve_context_top_k(sample_chunks):
    """Verify retrieval respects top-k limit and returns chunk records."""
    store = MockVectorStore(sample_chunks)
    query_vector = [0.1, 0.2, 0.3]
    retrieved = retrieve_context(query_vector, store=store, k=1)
    assert len(retrieved) == 1
    assert retrieved[0]["chunk_id"] == "doc_who_dengue_c0001"
    assert store.last_limit == 1
    assert store.last_query_vector == query_vector


def test_retrieve_context_validation(sample_chunks):
    """Verify validation on k and query vector."""
    store = MockVectorStore(sample_chunks)
    with pytest.raises(ValueError, match="Top-k must be an integer >= 1"):
        retrieve_context([0.1], store=store, k=0)
    with pytest.raises(ValueError, match="Query vector must not be empty"):
        retrieve_context([], store=store, k=2)


def test_retrieve_context_similarity_threshold(sample_chunks):
    """Verify minimum similarity filters low-scoring chunks."""
    store = MockVectorStore(sample_chunks)
    # Score threshold 0.5 should keep only dengue chunk (0.88), filtering cholera (0.45)
    filtered = retrieve_context([0.1], store=store, k=5, min_similarity=0.5)
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "doc_who_dengue_c0001"


def test_retrieve_context_empty_store_graceful():
    """Verify vector store errors or empty state return empty list without crashing."""
    failing_store = MagicMock()
    failing_store.search.side_effect = VectorStoreError("Collection is empty")
    results = retrieve_context([0.1, 0.2], store=failing_store, k=2)
    assert results == []


# ------------------------------------------------------------------------------
# Stage 3: Assemble Context Tests
# ------------------------------------------------------------------------------

def test_assemble_context_formatting(sample_chunks):
    """Verify context formatting includes numbered index, source, section, and text."""
    context = assemble_context(sample_chunks)
    assert "[1] Source: data/sample_advisories/who_dengue_clinical_guidance.md (Section: Warning Signs)" in context
    assert "Dengue is transmitted by Aedes mosquitoes." in context
    assert "[2] Source: data/sample_advisories/cdc_cholera_water_safety.txt (Section: Water Disinfection)" in context
    assert "Cholera is a waterborne infection." in context


def test_assemble_context_empty():
    """Verify empty chunks list produces an empty string."""
    assert assemble_context([]) == ""


def test_assemble_context_fallback_headers():
    """Verify chunks with only 'source' or 'document_title' format cleanly."""
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Sample guidance text.",
            "metadata": {"source": "guidance.pdf"},
        }
    ]
    context = assemble_context(chunks)
    assert "[1] Source: guidance.pdf\nSample guidance text." in context


# ------------------------------------------------------------------------------
# Stage 4: Generate Answer Tests
# ------------------------------------------------------------------------------

def test_generate_answer_empty_context():
    """Verify empty context immediately returns fallback message without LLM call."""
    mock_llm = MagicMock()
    answer = generate_answer("What is dengue?", context="", llm_fn=mock_llm)
    assert answer == FALLBACK_NO_CONTEXT
    mock_llm.assert_not_called()


def test_generate_answer_custom_llm_fn():
    """Verify custom llm_fn is called with the framed prompt."""
    def dummy_llm(prompt: str) -> str:
        return f"Synthesized from prompt: {len(prompt)} chars"

    context = "[1] Source: doc.txt\nDengue symptoms include fever."
    answer = generate_answer("What are symptoms?", context, llm_fn=dummy_llm)
    assert "Synthesized from prompt:" in answer


def test_generate_answer_with_client_mock():
    """Verify OpenAICompatibleClient is called with system and user messages."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.content = "Grounded response citing [doc.txt, Section]."
    mock_client.create_chat_completion.return_value = mock_result

    context = "[1] Source: doc.txt (Section: Triage)\nFever guidelines."
    query = "What is the triage?"
    answer = generate_answer(query, context, client=mock_client)

    assert answer == "Grounded response citing [doc.txt, Section]."
    mock_client.create_chat_completion.assert_called_once()
    args, kwargs = mock_client.create_chat_completion.call_args
    messages = kwargs.get("messages") or args[0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Context:\n" in messages[1]["content"]
    assert query in messages[1]["content"]


# ------------------------------------------------------------------------------
# Stage 5: Orchestrator (answer_query) Tests
# ------------------------------------------------------------------------------

def test_answer_query_end_to_end_success(sample_chunks):
    """Verify full pipeline flow: embed -> retrieve -> assemble -> generate -> sources."""
    store = MockVectorStore(sample_chunks)
    embed_fn = MagicMock(return_value=[0.1, 0.2, 0.3])
    llm_fn = MagicMock(return_value="Patients with warning signs must be referred immediately.")

    query = "What are dengue warning signs?"
    result = answer_query(
        query=query,
        k=2,
        store=store,
        embed_fn=embed_fn,
        llm_fn=llm_fn,
    )

    assert result["query"] == query
    assert result["answer"] == "Patients with warning signs must be referred immediately."
    assert len(result["sources"]) == 2
    assert result["sources"][0]["doc_id"] == "doc_who_dengue"
    assert "[1] Source: data/sample_advisories/who_dengue_clinical_guidance.md" in result["context"]

    embed_fn.assert_called_once_with(query)
    llm_fn.assert_called_once()


def test_answer_query_empty_retrieval_fallback():
    """Verify when retrieval returns no chunks, pipeline returns fallback and empty sources."""
    empty_store = MockVectorStore([])
    llm_mock = MagicMock()

    result = answer_query(
        query="What evidence is required for project submission?",
        k=4,
        store=empty_store,
        llm_fn=llm_mock,
    )

    assert result["answer"] == FALLBACK_NO_CONTEXT
    assert result["sources"] == []
    assert result["context"] == ""
    # LLM must not be called when retrieval yields nothing
    llm_mock.assert_not_called()


def test_answer_query_similarity_threshold_filtering(sample_chunks):
    """Verify chunks below min_similarity are filtered; if none pass, returns fallback."""
    store = MockVectorStore(sample_chunks)

    # Require minimum similarity 0.99 (both chunks are <= 0.88)
    result = answer_query(
        query="Unmatched query",
        k=2,
        store=store,
        min_similarity=0.99,
    )

    assert result["answer"] == FALLBACK_NO_CONTEXT
    assert result["sources"] == []


# ------------------------------------------------------------------------------
# Stage Isolation & Separation of Responsibilities Tests (Task 4)
# ------------------------------------------------------------------------------

def test_stage_isolation_modular_usage(sample_chunks):
    """Verify that every stage can be executed independently without the others."""
    # 1. Embed stage alone
    vec = embed_query("Isolated query")
    assert isinstance(vec, list) and len(vec) > 0

    # 2. Retrieve stage alone (no generation needed)
    store = MockVectorStore(sample_chunks)
    retrieved = retrieve_context(vec, store=store, k=1)
    assert len(retrieved) == 1

    # 3. Assemble stage alone (no vector store or LLM needed)
    context = assemble_context(sample_chunks)
    assert "Dengue" in context

    # 4. Prompt construction alone
    prompt = build_rag_prompt("My Question", context)
    assert "Context:\n" in prompt
    assert "Question: My Question" in prompt

    # 5. Generate stage alone (no database needed)
    ans = generate_answer("My Question", context, llm_fn=lambda p: "Direct answer")
    assert ans == "Direct answer"


# ------------------------------------------------------------------------------
# Integration Runner & CLI Tests (Task 3 & 5)
# ------------------------------------------------------------------------------

def test_populate_vector_store_and_run_demo(tmp_path):
    """Verify sample vector store initialization and pipeline demo execution."""
    test_db = tmp_path / "test_vectordb"
    test_output = tmp_path / "test_rag_pipeline_sample.md"

    demo_result = run_pipeline_demo(
        query="How to manage dengue warning signs?",
        negative_query="What evidence is required for project submission?",
        k=2,
        db_path=test_db,
        collection_name="test_rag_collection",
        output_path=test_output,
    )

    assert "positive_query" in demo_result
    assert "negative_query" in demo_result
    assert demo_result["positive_query"]["sources"] != []
    assert demo_result["negative_query"]["sources"] == []
    assert demo_result["negative_query"]["answer"] == FALLBACK_NO_CONTEXT

    # Verify markdown report generation
    assert test_output.is_file()
    content = test_output.read_text(encoding="utf-8")
    assert "# RAG Pipeline Architecture & Flow Design" in content
    assert "```mermaid" in content
    assert "What evidence is required for project submission?" in content


def test_main_cli_execution(tmp_path):
    """Verify CLI main entry point exits cleanly with code 0."""
    test_db = tmp_path / "cli_vectordb"
    test_output = tmp_path / "cli_sample.md"

    with patch("sys.argv", [
        "rag_pipeline.py",
        "--query", "What is dengue fever?",
        "--db-path", str(test_db),
        "--collection", "cli_test_collection",
        "--output", str(test_output),
    ]):
        exit_code = main()
        assert exit_code == 0
        assert test_output.is_file()
