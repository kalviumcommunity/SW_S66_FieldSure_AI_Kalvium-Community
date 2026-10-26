"""Automated test suite for Streaming Responses & Citation Display (Concept 3.47).

Validates:
    - Task 1: Progressive answer streaming via POST /query/stream and SSE event structure.
    - Task 2: Citation display with visible markers, document names, chunk IDs, and sections.
    - Task 3: Verifiable source text inspection attributes within citation payloads.
    - Task 4: Error handling, refusals, input validation, and interruption simulation.
    - Task 5: Sample interaction artifacts generation and validity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

from src.backend_api import QueryRequest, create_app, stream_query
from src.streaming_demo import (
    DEFAULT_SAMPLE_JSON,
    DEFAULT_SAMPLE_MD,
    parse_sse_stream,
    run_streaming_demo,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI test client."""
    app = create_app()
    return TestClient(app)


class TestStreamingAPIProgressiveDelivery:
    """Task 1: Verifies progressive SSE streaming from POST /query/stream."""

    def test_stream_endpoint_returns_event_stream(self, client: TestClient) -> None:
        """Endpoint responds with text/event-stream content type and 200 OK."""
        response = client.post(
            "/query/stream",
            json={"question": "What evidence is required for project submission?"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_stream_citations_event_emitted_first(self, client: TestClient) -> None:
        """The stream emits citations before progressive answer tokens."""
        response = client.post(
            "/query/stream",
            json={"question": "What evidence is required for project submission?"},
        )
        assert response.status_code == 200
        events = parse_sse_stream(response.text)
        assert len(events) >= 2

        # First event must be citations
        first_event = events[0]
        assert first_event["type"] == "citations"
        assert "sources" in first_event
        assert isinstance(first_event["sources"], list)
        assert len(first_event["sources"]) > 0

    def test_stream_tokens_reconstruct_full_answer(self, client: TestClient) -> None:
        """Token events stream progressively and concatenate into the full answer."""
        response = client.post(
            "/query/stream",
            json={"question": "What evidence is required for project submission?"},
        )
        assert response.status_code == 200
        events = parse_sse_stream(response.text)

        tokens = [e["text"] for e in events if e.get("type") == "token"]
        assert len(tokens) > 1, "Answer should be delivered in multiple progressive tokens"

        full_answer = "".join(tokens)
        assert len(full_answer.strip()) > 0
        assert "evidence" in full_answer.lower() or "submission" in full_answer.lower()

    def test_stream_terminates_with_done_event(self, client: TestClient) -> None:
        """Stream ends with a definitive {"type": "done"} event."""
        response = client.post(
            "/query/stream",
            json={"question": "What evidence is required for project submission?"},
        )
        events = parse_sse_stream(response.text)
        assert len(events) > 0
        last_event = events[-1]
        assert last_event["type"] == "done"


class TestCitationMetadataAndSourceInspection:
    """Task 2 & Task 3: Verifies citations format, markers, chunk IDs, and inspectable source text."""

    def test_citations_include_label_document_chunk_and_text(
        self, client: TestClient
    ) -> None:
        """Every citation provides markers, document name, chunk_id, section, and full text."""
        response = client.post(
            "/query/stream",
            json={"question": "What evidence is required for project submission?"},
        )
        events = parse_sse_stream(response.text)
        citations_event = next(e for e in events if e.get("type") == "citations")
        sources = citations_event["sources"]

        assert len(sources) > 0
        for src in sources:
            # Task 2: Visible citation marker such as [1]
            assert "label" in src
            assert src["label"].startswith("[") and src["label"].endswith("]")

            # Document name or source title
            assert "document" in src
            assert isinstance(src["document"], str)
            assert len(src["document"]) > 0

            # Chunk ID or location marker
            assert "chunk_id" in src
            assert src["chunk_id"] is not None

            # Task 3: Source text for user inspection
            assert "text" in src
            assert isinstance(src["text"], str)
            assert len(src["text"]) > 0, "Source text must be present for user inspection"

            # Optional section and score
            assert "score" in src
            assert src["score"] is not None

    def test_citations_source_text_matches_grounding_content(
        self, client: TestClient
    ) -> None:
        """Inspectable source text contains the verified evidence passage."""
        response = client.post(
            "/query/stream",
            json={"question": "What evidence is required for project submission?"},
        )
        events = parse_sse_stream(response.text)
        citations_event = next(e for e in events if e.get("type") == "citations")
        source = citations_event["sources"][0]

        # The project submission policy chunk text
        assert "git commits" in source["text"] or "evidence" in source["text"]


class TestStreamingErrorsAndRefusalHandling:
    """Task 4: Verifies safe refusals, validation errors, and pipeline error recovery."""

    def test_refusal_query_streams_refusal_tokens(self, client: TestClient) -> None:
        """Out-of-domain questions emit safe refusal citations and stream refusal message."""
        response = client.post(
            "/query/stream",
            json={"question": "What is the return policy for retail equipment?"},
        )
        assert response.status_code == 200
        events = parse_sse_stream(response.text)

        citations_event = next(e for e in events if e.get("type") == "citations")
        assert citations_event["status"] == "refused_weak_context"
        assert len(citations_event["sources"]) == 0

        # Refusal tokens must still stream progressively
        tokens = [e["text"] for e in events if e.get("type") == "token"]
        assert len(tokens) > 0
        refusal_text = "".join(tokens)
        assert (
            "reliable context" in refusal_text.lower()
            or "don't have enough" in refusal_text.lower()
        )

    def test_short_query_validation_error_before_stream(
        self, client: TestClient
    ) -> None:
        """Question under 3 characters returns 422 Unprocessable Entity without starting stream."""
        response = client.post(
            "/query/stream",
            json={"question": "hi"},
        )
        assert response.status_code == 422
        assert "text/event-stream" not in response.headers.get("content-type", "")

    def test_whitespace_query_validation_error(self, client: TestClient) -> None:
        """Whitespace-only question returns 422 validation error."""
        response = client.post(
            "/query/stream",
            json={"question": "      "},
        )
        assert response.status_code == 422

    def test_streaming_error_event_on_pipeline_failure(self) -> None:
        """When pipeline raises an exception during streaming, stream emits error event."""

        async def failing_stream(question: str):
            raise RuntimeError("Database connection timed out during streaming")
            yield {}  # make it a generator

        app = create_app(stream_pipeline_fn=failing_stream)
        custom_client = TestClient(app)

        response = custom_client.post(
            "/query/stream",
            json={"question": "What evidence is required for project submission?"},
        )
        assert response.status_code == 200
        events = parse_sse_stream(response.text)
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) == 1
        assert "stopped streaming" in error_events[0]["message"].lower()

    @pytest.mark.anyio
    async def test_functional_stream_query_wrapper(self) -> None:
        """Functional stream_query wrapper matches prompt reference specification."""
        req = QueryRequest(question="What evidence is required for project submission?")
        response = await stream_query(req, delay=0.0)
        assert response.media_type == "text/event-stream"


class TestStreamingDemoAndArtifacts:
    """Task 5: Validates demonstration execution and committed artifacts."""

    def test_run_streaming_demo_generates_valid_files(self, tmp_path: Path) -> None:
        """Demo exports structured JSON and readable Markdown interaction artifacts."""
        tmp_json = tmp_path / "sample_streaming.json"
        tmp_md = tmp_path / "sample_streaming.md"

        payload = run_streaming_demo(output_json=tmp_json, output_md=tmp_md)

        assert tmp_json.is_file()
        assert tmp_md.is_file()

        # Validate JSON content
        data = json.loads(tmp_json.read_text(encoding="utf-8"))
        assert len(data["interactions"]) == 3
        assert data["interactions"][0]["scenario"] == "grounded_answer_streaming"
        assert data["interactions"][1]["scenario"] == "safe_refusal_streaming"
        assert data["interactions"][2]["scenario"] == "stream_interruption_simulation"

        # Validate Markdown content
        md_text = tmp_md.read_text(encoding="utf-8")
        assert "# FieldSure AI — Streaming Responses & Citation Display" in md_text
        assert "Scenario #1: Progressive Grounded Answer Streaming" in md_text
        assert "Scenario #2: Safe Refusal Streaming" in md_text
        assert "Scenario #3: Stream Interruption & Partial Output Preservation" in md_text
        assert "Server-Sent Events (SSE) Wire Format" in md_text
