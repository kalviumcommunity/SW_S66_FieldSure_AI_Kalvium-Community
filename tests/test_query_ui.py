"""Automated test suite for Chat Interface & Query UI (Concept 3.46).

Validates:
    - Task 1: Question and answer UI contract compliance (QueryRequest, QueryResponse).
    - Task 2: Backend RAG API connectivity via POST /query and GET /health.
    - Task 3: Retrieved sources structure, chunk IDs, and score attributes.
    - Task 4: Loading and error handling states (validation errors, empty sources on refusal).
    - Task 5: Sample interaction generation, JSON artifact validity, and Markdown report.
"""

import json
from pathlib import Path
from typing import Any, Dict
import pytest
from fastapi.testclient import TestClient

from src.backend_api import create_app
from src.query_ui_demo import (
    DEFAULT_SAMPLE_JSON,
    DEFAULT_SAMPLE_MD,
    format_sample_markdown,
    run_sample_interactions,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI test client."""
    app = create_app()
    return TestClient(app)


class TestQueryUIAPIContract:
    """Task 1 & Task 2: Validates the backend API conforms to the Next.js UI contract."""

    def test_health_check_contract(self, client: TestClient) -> None:
        """UI Header pings /health to show online/offline status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "service" in data
        assert "version" in data

    def test_query_grounded_answer_contract(self, client: TestClient) -> None:
        """UI sends POST /query and renders answer + sources."""
        query = {"question": "What evidence is required for project submission?"}
        response = client.post("/query", json=query)
        assert response.status_code == 200
        data = response.json()

        # Contract fields required by AnswerCard and SourcesList
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0

        assert "status" in data
        assert data["status"] in ("answered", "refused_weak_context")

    def test_sources_attributes_conform_to_ui(self, client: TestClient) -> None:
        """Task 3: Verify sources contain source name, chunk_id, and score."""
        query = {"question": "What evidence is required for project submission?"}
        response = client.post("/query", json=query)
        assert response.status_code == 200
        data = response.json()

        for source_item in data["sources"]:
            assert "source" in source_item
            assert isinstance(source_item["source"], str)
            # chunk_id and score are optional but defined in schema
            assert "chunk_id" in source_item
            assert "score" in source_item


class TestQueryUIErrorAndRefusalStates:
    """Task 4: Validates loading, validation errors, and refusal handling."""

    def test_short_query_validation_error(self, client: TestClient) -> None:
        """UI shows error message when question is under 3 characters."""
        response = client.post("/query", json={"question": "a"})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_whitespace_query_validation_error(self, client: TestClient) -> None:
        """UI rejects whitespace-only questions."""
        response = client.post("/query", json={"question": "     "})
        assert response.status_code == 422

    def test_oversized_query_validation_error(self, client: TestClient) -> None:
        """UI rejects questions over 1000 characters."""
        long_q = "what is " + "a" * 1005
        response = client.post("/query", json={"question": long_q})
        assert response.status_code == 422

    def test_refusal_state_returns_empty_sources(self, client: TestClient) -> None:
        """UI detects weak context and renders safe refusal badge without hallucinations."""
        query = {"question": "Can you prescribe medications outside the approved formulary?"}
        response = client.post("/query", json=query)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "refused_weak_context"
        assert len(data["sources"]) == 0
        assert "reliable context" in data["answer"].lower() or "don't have enough" in data["answer"].lower()


class TestFrontendProjectStructure:
    """Verifies that the Next.js frontend codebase in frontend/ is properly structured."""

    def test_package_json_exists_and_configured(self) -> None:
        pkg_path = REPO_ROOT / "frontend" / "package.json"
        assert pkg_path.is_file(), "frontend/package.json must exist"
        data = json.loads(pkg_path.read_text(encoding="utf-8"))
        assert "dependencies" in data
        assert "next" in data["dependencies"]
        assert "react" in data["dependencies"]
        assert "react-dom" in data["dependencies"]

    def test_frontend_components_exist(self) -> None:
        components_dir = REPO_ROOT / "frontend" / "src" / "components"
        assert (components_dir / "Header.tsx").is_file()
        assert (components_dir / "QueryInput.tsx").is_file()
        assert (components_dir / "AnswerCard.tsx").is_file()
        assert (components_dir / "SourcesList.tsx").is_file()
        assert (components_dir / "DocumentUploadWidget.tsx").is_file()
        assert (components_dir / "SettingsModal.tsx").is_file()

    def test_frontend_app_files_exist(self) -> None:
        app_dir = REPO_ROOT / "frontend" / "src" / "app"
        assert (app_dir / "page.tsx").is_file()
        assert (app_dir / "layout.tsx").is_file()
        assert (app_dir / "globals.css").is_file()

    def test_frontend_api_library_exists(self) -> None:
        api_lib = REPO_ROOT / "frontend" / "src" / "lib" / "api.ts"
        assert api_lib.is_file()
        content = api_lib.read_text(encoding="utf-8")
        assert "askQuestion" in content
        assert "checkHealth" in content
        assert "uploadDocument" in content


class TestQueryUIDemoAndArtifacts:
    """Task 5: Validates generation of sample interaction artifacts."""

    def test_run_sample_interactions_generates_valid_files(self, tmp_path: Path) -> None:
        tmp_json = tmp_path / "sample.json"
        tmp_md = tmp_path / "sample.md"

        payload = run_sample_interactions(output_json=tmp_json, output_md=tmp_md)

        assert tmp_json.is_file()
        assert tmp_md.is_file()

        # Validate JSON content
        data = json.loads(tmp_json.read_text(encoding="utf-8"))
        assert data["total_interactions"] >= 5
        assert len(data["interactions"]) >= 5

        # Validate Markdown content
        md_text = tmp_md.read_text(encoding="utf-8")
        assert "# FieldSure AI — Chat Interface & Query UI" in md_text
        assert "Scenario #1" in md_text
        assert "Trustworthy Interface Details" in md_text
