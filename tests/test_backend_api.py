"""Automated test suite for Backend API for the RAG Service (Concept 3.44).

Validates:
    - Task 1: POST /query endpoint accepts questions and returns grounded answers.
    - Task 2: Structured JSON response shapes (answer, sources, status, metadata).
    - Task 3: Input validation and HTTP status codes (400, 422, 500).
    - Task 4: Dynamic environment-driven configuration and secret protection.
    - Task 5: Sample request and response verification and artifact integrity.
"""

from pathlib import Path
from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.backend_api import (
    APIConfig,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Source,
    app,
    create_app,
    load_environment_config,
    query_rag,
    run_demo,
)


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient configured with default application."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Task 4 Tests: Configuration Management & Environment Variables
# ---------------------------------------------------------------------------

class TestBackendAPIConfig:
    """Tests environment variable loading, default values, and credential protection."""

    def test_default_config_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify defaults are applied when optional environment variables are absent."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("VECTOR_DB_URL", raising=False)
        monkeypatch.delenv("COLLECTION_NAME", raising=False)

        cfg = load_environment_config(strict=False)
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.vector_db_url == "http://localhost:6333"
        assert cfg.collection_name == "rag_chunks"
        assert cfg.chat_model == "gpt-4o-mini"
        assert cfg.api_base_url == "https://api.openai.com/v1"
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000

    def test_custom_environment_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify explicit environment variables override defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-custom-openai-secret-key-123456")
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        monkeypatch.setenv("VECTOR_DB_URL", "https://qdrant.internal.health.net:6333")
        monkeypatch.setenv("COLLECTION_NAME", "clinical_protocols")
        monkeypatch.setenv("CHAT_MODEL", "gpt-4o")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("HOST", "127.0.0.1")

        cfg = load_environment_config(strict=False)
        assert cfg.api_key == "sk-custom-openai-secret-key-123456"
        assert cfg.embedding_model == "BAAI/bge-large-en-v1.5"
        assert cfg.vector_db_url == "https://qdrant.internal.health.net:6333"
        assert cfg.collection_name == "clinical_protocols"
        assert cfg.chat_model == "gpt-4o"
        assert cfg.port == 9000
        assert cfg.host == "127.0.0.1"

    def test_strict_mode_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify strict mode raises ValueError when required credentials are missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        with pytest.raises(ValueError, match="OPENAI_API_KEY or API_KEY is required"):
            load_environment_config(strict=True)

    def test_safe_dict_masks_api_key(self) -> None:
        """Verify to_safe_dict masks credentials and never leaks raw API keys."""
        cfg = APIConfig(
            api_key="sk-proj-testcredential1234567890",
            embedding_model="text-embedding-3-small",
            vector_db_url="http://localhost:6333",
            collection_name="rag_chunks",
        )
        safe = cfg.to_safe_dict()
        assert "sk-proj-testcredential1234567890" not in str(safe)
        assert safe["api_key_configured"] is True
        assert safe["api_key_masked"].startswith("sk-")
        assert safe["api_key_masked"].endswith("7890")


# ---------------------------------------------------------------------------
# Task 1 & Task 2 Tests: Pydantic Schema Validation & Response Shapes
# ---------------------------------------------------------------------------

class TestPydanticSchemas:
    """Validates request and response schemas according to API contracts."""

    def test_query_request_valid(self) -> None:
        """Valid question parses cleanly."""
        req = QueryRequest(question="What evidence is required for project submission?")
        assert req.question == "What evidence is required for project submission?"

    def test_query_request_too_short(self) -> None:
        """Questions under 3 characters raise ValidationError."""
        with pytest.raises(ValidationError):
            QueryRequest(question="ab")

    def test_query_request_too_long(self) -> None:
        """Questions exceeding 1000 characters raise ValidationError."""
        long_q = "what is " + "x" * 1001
        with pytest.raises(ValidationError):
            QueryRequest(question=long_q)

    def test_query_request_whitespace_only(self) -> None:
        """Whitespace-only strings raise ValidationError."""
        with pytest.raises(ValidationError):
            QueryRequest(question="     ")

    def test_source_model_optional_fields(self) -> None:
        """Source model handles optional chunk_id and score attributes."""
        s1 = Source(source="guideline.pdf")
        assert s1.source == "guideline.pdf"
        assert s1.chunk_id is None
        assert s1.score is None

        s2 = Source(source="guideline.pdf", chunk_id="chunk_01", score=0.88)
        assert s2.chunk_id == "chunk_01"
        assert s2.score == 0.88

    def test_query_response_serialization(self) -> None:
        """QueryResponse formats expected fields including answer, sources, status, metadata."""
        resp = QueryResponse(
            answer="Protocol instructions.",
            sources=[Source(source="test.pdf", chunk_id="c1", score=0.91)],
            status="answered",
            metadata={"top_score": 0.91, "chunks_used": 1},
        )
        data = resp.model_dump()
        assert data["answer"] == "Protocol instructions."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "test.pdf"
        assert data["sources"][0]["score"] == 0.91
        assert data["status"] == "answered"
        assert data["metadata"]["top_score"] == 0.91


# ---------------------------------------------------------------------------
# Task 1, 2, 3 Tests: Query Endpoint Execution & Guardrails
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Validates the POST /query endpoint across answered and refusal states."""

    def test_query_endpoint_grounded_answer(self, client: TestClient) -> None:
        """Submit question with strong context; verify grounded answer and sources returned."""
        payload = {"question": "What evidence is required for project submission?"}
        res = client.post("/query", json=payload)

        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) >= 1
        assert data["status"] == "answered"
        assert "metadata" in data
        assert data["metadata"]["total_sources"] >= 1

    def test_query_endpoint_refusal_guardrail(self, client: TestClient) -> None:
        """Submit out-of-domain question; verify safe refusal and status 'refused_weak_context'."""
        payload = {"question": "What is the refund policy for products?"}
        res = client.post("/query", json=payload)

        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "refused_weak_context"
        assert "context" in data["answer"].lower() or "enough" in data["answer"].lower()
        assert data["sources"] == []
        assert data["metadata"]["is_refused"] is True

    def test_query_endpoint_custom_pipeline_injection(self) -> None:
        """Verify create_app supports custom pipeline dependency injection."""
        def custom_pipeline(q: str) -> Dict[str, Any]:
            return {
                "answer": "The submission requires a PR link, sample output, and a video explanation.",
                "sources": [
                    {"source": "submission-rubric.md", "chunk_id": "submission-rubric.md:2", "score": 0.84}
                ],
                "status": "answered",
                "top_score": 0.84,
                "chunks_used": 1,
            }

        custom_app = create_app(pipeline_fn=custom_pipeline)
        custom_client = TestClient(custom_app)

        res = custom_client.post("/query", json={"question": "What evidence is required?"})
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "The submission requires a PR link, sample output, and a video explanation."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "submission-rubric.md"
        assert data["sources"][0]["chunk_id"] == "submission-rubric.md:2"
        assert data["sources"][0]["score"] == 0.84
        assert data["status"] == "answered"

    def test_standalone_functional_wrapper(self) -> None:
        """Verify query_rag functional wrapper returns QueryResponse matching contract."""
        req = QueryRequest(question="What evidence is required for project submission?")
        res = query_rag(req)
        assert isinstance(res, QueryResponse)
        assert res.status == "answered"
        assert len(res.sources) >= 1


# ---------------------------------------------------------------------------
# Task 3 Tests: Input Validation & Error Handling (400, 422, 500)
# ---------------------------------------------------------------------------

class TestInputValidationAndErrorHandling:
    """Validates HTTP status codes for invalid inputs and pipeline exceptions."""

    def test_missing_question_field_returns_422(self, client: TestClient) -> None:
        """Missing 'question' key in request payload triggers 422 Unprocessable Entity."""
        res = client.post("/query", json={})
        assert res.status_code == 422
        errors = res.json()["detail"]
        assert any("question" in err.get("loc", []) for err in errors)

    def test_too_short_question_returns_422(self, client: TestClient) -> None:
        """Question shorter than 3 characters triggers 422 Unprocessable Entity."""
        res = client.post("/query", json={"question": "hi"})
        assert res.status_code == 422
        assert "detail" in res.json()

    def test_too_long_question_returns_422(self, client: TestClient) -> None:
        """Question exceeding 1000 characters triggers 422 Unprocessable Entity."""
        res = client.post("/query", json={"question": "a" * 1001})
        assert res.status_code == 422

    def test_whitespace_question_returns_422(self, client: TestClient) -> None:
        """Whitespace-only question triggers 422 Unprocessable Entity."""
        res = client.post("/query", json={"question": "      "})
        assert res.status_code == 422

    def test_invalid_json_type_returns_422(self, client: TestClient) -> None:
        """Non-string question type triggers 422 Unprocessable Entity."""
        res = client.post("/query", json={"question": 12345})
        assert res.status_code == 422

    def test_pipeline_value_error_returns_400(self) -> None:
        """Pipeline ValueError triggers HTTP 400 Bad Request with informative detail."""
        def failing_pipeline(q: str) -> Dict[str, Any]:
            raise ValueError("Query contains unparseable medical syntax.")

        app_fail = create_app(pipeline_fn=failing_pipeline)
        fail_client = TestClient(app_fail)

        res = fail_client.post("/query", json={"question": "What is the diagnosis?"})
        assert res.status_code == 400
        assert "unparseable medical syntax" in res.json()["detail"]

    def test_pipeline_unexpected_exception_returns_500(self) -> None:
        """Unhandled pipeline exception triggers HTTP 500 with 'RAG service failed'."""
        def crashing_pipeline(q: str) -> Dict[str, Any]:
            raise RuntimeError("Database connection pool exhausted")

        app_crash = create_app(pipeline_fn=crashing_pipeline)
        crash_client = TestClient(app_crash)

        res = crash_client.post("/query", json={"question": "What are the dosage guidelines?"})
        assert res.status_code == 500
        assert res.json()["detail"] == "RAG service failed"


# ---------------------------------------------------------------------------
# Auxiliary Endpoints Tests: Root, Health, Config
# ---------------------------------------------------------------------------

class TestAuxiliaryEndpoints:
    """Validates /, /health, and /config auxiliary endpoints."""

    def test_root_endpoint(self, client: TestClient) -> None:
        """GET / returns online status and endpoint sitemap."""
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "online"
        assert "POST /query" in data["endpoints"]["query"]

    def test_health_endpoint(self, client: TestClient) -> None:
        """GET /health returns healthy status."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["service"] == "FieldSure AI RAG Service"

    def test_config_endpoint_masks_secrets(self, client: TestClient) -> None:
        """GET /config returns active config with masked credentials."""
        res = client.get("/config")
        assert res.status_code == 200
        data = res.json()
        assert "embedding_model" in data
        assert "vector_db_url" in data
        assert "collection_name" in data
        assert "api_key_masked" in data
        # Ensure raw secret is never present
        assert "sk-proj-testcredential" not in str(data)


# ---------------------------------------------------------------------------
# Task 5 Tests: Sample Request/Response Artifacts Verification
# ---------------------------------------------------------------------------

class TestSampleOutputsAndArtifacts:
    """Verifies that sample evidence is correctly generated and committed."""

    def test_run_demo_generates_valid_json_and_md(self, tmp_path: Path) -> None:
        """Execute run_demo into temporary directory and verify output schema."""
        out_json = tmp_path / "sample.json"
        out_md = tmp_path / "sample.md"

        evidence = run_demo(output_json=out_json, output_md=out_md)
        assert out_json.is_file()
        assert out_md.is_file()

        assert "demonstrations" in evidence
        assert len(evidence["demonstrations"]) == 3
        assert evidence["demonstrations"][0]["response"]["status_code"] == 200
        assert evidence["demonstrations"][1]["response"]["body"]["status"] == "refused_weak_context"
        assert evidence["demonstrations"][2]["response"]["status_code"] == 422

        md_text = out_md.read_text(encoding="utf-8")
        assert "POST /query" in md_text
        assert "What evidence is required for project submission?" in md_text

    def test_committed_sample_artifacts_exist(self) -> None:
        """Verify committed sample JSON and MD artifacts exist in outputs/ directory."""
        sample_json = Path(__file__).resolve().parent.parent / "outputs" / "backend_api_sample.json"
        sample_md = Path(__file__).resolve().parent.parent / "outputs" / "backend_api_sample.md"

        assert sample_json.is_file(), f"Missing artifact: {sample_json}"
        assert sample_md.is_file(), f"Missing artifact: {sample_md}"

        import json
        with open(sample_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "demonstrations" in data
        assert len(data["demonstrations"]) == 3
