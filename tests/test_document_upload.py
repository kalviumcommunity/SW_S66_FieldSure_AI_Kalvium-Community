"""Automated test suite for Document Upload & Indexing Endpoint (Concept 3.45).

Validates:
    - Task 1: Upload validation, secure storage, and path traversal protection.
    - Task 2: Document loading, cleaning, chunking, embedding, and vector indexing.
    - Task 3: Runtime searchability (new content searchable via POST /query without restart).
    - Task 4: Error handling and HTTP status codes (400, 413, 415, 500).
    - Task 5: Sample upload execution and artifact verification.
"""

from io import BytesIO
import json
from pathlib import Path
from typing import Any, Dict
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from src.backend_api import create_app
from src.document_upload import (
    MAX_UPLOAD_SIZE,
    RUNTIME_UPLOADED_CHUNKS,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    UploadResponse,
    UploadSummary,
    clean,
    clear_runtime_uploaded_chunks,
    embed_chunks,
    index_chunks,
    load_text,
    process_uploaded_document,
    run_upload_demo,
    store_upload,
    tag_chunks,
    token_chunks,
    upload_document,
    validate_upload,
)


@pytest.fixture(autouse=True)
def reset_runtime_chunks() -> None:
    """Clear runtime uploaded chunks before and after every test."""
    clear_runtime_uploaded_chunks()
    yield
    clear_runtime_uploaded_chunks()


@pytest.fixture
def client() -> TestClient:
    """Test client configured with fresh FastAPI instance."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Task 1 & Task 4 Tests: Validation & Secure Storage
# ---------------------------------------------------------------------------

class TestUploadValidationAndStorage:
    """Tests file type validation, filename sanitization, and size limits."""

    def test_validate_upload_valid_extensions(self) -> None:
        """Valid extensions (.txt, .md, .pdf) are accepted regardless of case."""
        for ext in [".txt", ".md", ".pdf", ".TXT", ".MD", ".PDF"]:
            file = UploadFile(filename=f"advisory{ext}", file=BytesIO(b"content"))
            suffix = validate_upload(file)
            assert suffix == ext.lower()

    def test_validate_upload_unsupported_extension_raises_415(self) -> None:
        """Unsupported extensions raise HTTP 415 Unsupported Media Type."""
        for ext in [".exe", ".py", ".csv", ".json", ".bin", ".sh"]:
            file = UploadFile(filename=f"payload{ext}", file=BytesIO(b"content"))
            with pytest.raises(HTTPException) as exc_info:
                validate_upload(file)
            assert exc_info.value.status_code == 415
            assert "Unsupported file type" in exc_info.value.detail

    def test_validate_upload_missing_filename_raises_400(self) -> None:
        """Missing or whitespace-only filename raises HTTP 400 Bad Request."""
        file = UploadFile(filename="", file=BytesIO(b"content"))
        with pytest.raises(HTTPException) as exc_info:
            validate_upload(file)
        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_store_upload_sanitizes_path_traversal(self, tmp_path: Path) -> None:
        """Directory traversal attempts (../../evil.txt) are sanitized to basename only."""
        malicious_filename = "../../../evil_traversal.md"
        content = b"# Safe Heading\nSafe document content."
        file = UploadFile(filename=malicious_filename, file=BytesIO(content))

        stored_path = await store_upload(file, upload_dir=tmp_path)
        assert stored_path.parent == tmp_path
        assert stored_path.name == "evil_traversal.md"
        assert stored_path.is_file()
        assert stored_path.read_bytes() == content

    @pytest.mark.anyio
    async def test_store_upload_empty_file_raises_400(self, tmp_path: Path) -> None:
        """Empty uploaded files (0 bytes) raise HTTP 400."""
        file = UploadFile(filename="empty_guidance.txt", file=BytesIO(b""))
        with pytest.raises(HTTPException) as exc_info:
            await store_upload(file, upload_dir=tmp_path)
        assert exc_info.value.status_code == 400
        assert "empty" in exc_info.value.detail.lower()

    @pytest.mark.anyio
    async def test_store_upload_oversized_file_raises_413(self, tmp_path: Path) -> None:
        """Files exceeding size limit raise HTTP 413 Payload Too Large."""
        large_content = b"X" * 1024
        file = UploadFile(filename="oversized.txt", file=BytesIO(large_content))
        with pytest.raises(HTTPException) as exc_info:
            await store_upload(file, upload_dir=tmp_path, max_size_bytes=500)
        assert exc_info.value.status_code == 413
        assert "exceeds size limit" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Task 2 Tests: Document Ingestion, Chunking, Embedding & Indexing
# ---------------------------------------------------------------------------

class TestDocumentProcessingPipeline:
    """Tests loading, cleaning, chunking, embedding, and indexing."""

    def test_load_text_markdown_and_txt(self, tmp_path: Path) -> None:
        """Plain text and Markdown files load cleanly."""
        md_file = tmp_path / "test_doc.md"
        md_file.write_text("# Test Title\nParagraph text.", encoding="utf-8")
        assert "Test Title" in load_text(md_file)

        txt_file = tmp_path / "test_doc.txt"
        txt_file.write_text("Plain text advisory line.", encoding="utf-8")
        assert "Plain text advisory" in load_text(txt_file)

    def test_load_text_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Nonexistent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_text(tmp_path / "nonexistent.md")

    def test_load_text_empty_file_raises(self, tmp_path: Path) -> None:
        """Blank document content raises ValueError."""
        blank_file = tmp_path / "blank.txt"
        blank_file.write_text("   \n\n  ", encoding="utf-8")
        with pytest.raises(ValueError, match="no readable text"):
            load_text(blank_file)

    def test_clean_normalizes_text(self) -> None:
        """Clean normalizes Unicode, whitespace, and layout."""
        raw = "Line 1   with   spaces.\r\nLine 2.\n\n\nLine 3."
        cleaned = clean(raw)
        assert "Line 1" in cleaned
        assert "Line 2." in cleaned
        assert "Line 3." in cleaned

    def test_token_chunks_structural_split(self) -> None:
        """Text is partitioned into chunk strings."""
        sample = (
            "# Section A\n" + "Word " * 120 + "\n\n"
            "# Section B\n" + "Another " * 120
        )
        chunks = token_chunks(sample, chunk_size=80, overlap=10)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) and len(c.strip()) > 0 for c in chunks)

    def test_tag_chunks_metadata_schema(self) -> None:
        """Tagged chunks carry source, chunk_id, document_title, and token count."""
        chunks = ["First chunk content.", "Second chunk content."]
        tagged = tag_chunks(source="uploads/outbreak_policy.md", chunks=chunks)
        assert len(tagged) == 2
        assert tagged[0]["chunk_id"].startswith("upld_")
        assert tagged[0]["metadata"]["source"] == "outbreak_policy.md"
        assert tagged[0]["metadata"]["document_title"] == "Outbreak Policy"
        assert tagged[0]["metadata"]["token_count"] > 0

    def test_embed_chunks_attaches_vectors(self) -> None:
        """Embed chunks attaches deterministic unit vectors and vector dimension."""
        tagged = tag_chunks(source="test.md", chunks=["Testing embedding generator."])
        embedded = embed_chunks(tagged)
        assert len(embedded) == 1
        assert "embedding" in embedded[0]
        assert embedded[0]["vector_dimension"] == 1024
        assert len(embedded[0]["embedding"]) == 1024

    def test_index_chunks_updates_runtime_registry(self) -> None:
        """index_chunks stores records into RUNTIME_UPLOADED_CHUNKS."""
        tagged = tag_chunks(source="test.md", chunks=["Chunk A", "Chunk B"])
        embedded = embed_chunks(tagged)
        count = index_chunks(embedded)
        assert count == 2
        assert len(RUNTIME_UPLOADED_CHUNKS) == 2

    def test_process_uploaded_document_end_to_end(self, tmp_path: Path) -> None:
        """process_uploaded_document runs complete pipeline and returns summary dict."""
        doc_path = tmp_path / "emergency_protocol.md"
        doc_path.write_text(
            "# Emergency Response\nDeploy emergency rapid response teams within 2 hours.",
            encoding="utf-8",
        )
        summary = process_uploaded_document(doc_path)
        assert summary["document"] == str(doc_path)
        assert summary["chunks"] >= 1
        assert summary["indexed"] == summary["chunks"]
        assert len(summary["tagged_chunks"]) == summary["chunks"]


# ---------------------------------------------------------------------------
# Task 1 & Task 4 Endpoint Tests: POST /documents
# ---------------------------------------------------------------------------

class TestDocumentUploadEndpoint:
    """Tests POST /documents endpoint contract, responses, and error handling."""

    def test_upload_markdown_success(self, client: TestClient) -> None:
        """Uploading valid Markdown file returns 200 with status 'indexed'."""
        content = b"# Cold Chain Advisory\nMaintain temperature strictly between 2C and 8C."
        response = client.post(
            "/documents",
            files={"file": ("advisory.md", content, "text/markdown")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["filename"] == "advisory.md"
        assert data["summary"]["chunks"] >= 1
        assert data["summary"]["indexed"] >= 1
        assert "advisory.md" in data["summary"]["document"]

    def test_upload_txt_success(self, client: TestClient) -> None:
        """Uploading plain text advisory returns 200 with status 'indexed'."""
        content = b"Public health field guideline for epidemiological monitoring."
        response = client.post(
            "/documents",
            files={"file": ("guideline.txt", content, "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["filename"] == "guideline.txt"

    def test_upload_unsupported_file_returns_415(self, client: TestClient) -> None:
        """Uploading unsupported format (.py, .exe, .sh) returns HTTP 415."""
        response = client.post(
            "/documents",
            files={"file": ("script.py", b"print('hello')", "text/x-python")},
        )
        assert response.status_code == 415
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_empty_file_returns_400(self, client: TestClient) -> None:
        """Uploading empty file returns HTTP 400 Bad Request."""
        response = client.post(
            "/documents",
            files={"file": ("empty.md", b"", "text/markdown")},
        )
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_standalone_upload_document_function(self, tmp_path: Path) -> None:
        """Standalone async upload_document function matches prompt interface."""
        content = b"# Standalone Policy\nProtocol instructions for field staff."
        file = UploadFile(filename="standalone.md", file=BytesIO(content))
        res = await upload_document(file)
        assert res.status == "indexed"
        assert res.filename == "standalone.md"
        assert res.summary.chunks >= 1
        assert res.summary.indexed >= 1


# ---------------------------------------------------------------------------
# Task 3 Tests: Runtime Searchability (Zero Restart)
# ---------------------------------------------------------------------------

class TestRuntimeSearchability:
    """Verifies that uploaded documents become immediately searchable without restart."""

    def test_runtime_searchability_flow(self, client: TestClient) -> None:
        """Full runtime searchability demonstration:

        1. Submit question before upload -> expect safe refusal ('refused_weak_context').
        2. Upload new document with distinct protocol directives via POST /documents.
        3. Submit question after upload -> expect grounded answer citing the new document.
        4. Application was never restarted.
        """
        question = "What temperature range must cold chain units maintain according to the remote protocol?"

        # Step 1: Query before upload
        pre_res = client.post("/query", json={"question": question})
        assert pre_res.status_code == 200
        pre_data = pre_res.json()
        assert pre_data["status"] == "refused_weak_context"
        assert pre_data["sources"] == []
        assert "don't have enough reliable context" in pre_data["answer"].lower()

        # Step 2: Upload new document at runtime
        policy_content = (
            "# Field Worker Cold Chain Storage Policy 2026\n\n"
            "## Remote Field Work Protocol\n"
            "All field epidemiologists and community health staff deployed to remote surveillance\n"
            "zones must carry certified solar-powered cold chain monitoring units. Cold chain units\n"
            "must maintain the temperature range for vaccines and biological specimens strictly between\n"
            "2°C and 8°C at all times according to remote protocol directives.\n"
        ).encode("utf-8")

        upload_res = client.post(
            "/documents",
            files={"file": ("new-policy.md", policy_content, "text/markdown")},
        )
        assert upload_res.status_code == 200
        upload_data = upload_res.json()
        assert upload_data["status"] == "indexed"
        assert upload_data["summary"]["indexed"] >= 1

        # Step 3: Query after upload on same client/server instance
        post_res = client.post("/query", json={"question": question})
        assert post_res.status_code == 200
        post_data = post_res.json()
        assert post_data["status"] == "answered"
        assert len(post_data["sources"]) > 0
        assert any("new-policy.md" in s["source"] for s in post_data["sources"])
        assert "2°C" in post_data["answer"] or "8°C" in post_data["answer"] or "cold chain" in post_data["answer"].lower()


# ---------------------------------------------------------------------------
# Task 5 Tests: Demonstration Runner & Artifact Verification
# ---------------------------------------------------------------------------

class TestUploadDemoRunner:
    """Verifies Task 5 demonstration runner and artifact generation."""

    def test_run_upload_demo_generates_valid_artifacts(self, tmp_path: Path) -> None:
        """run_upload_demo executes cleanly and produces structured JSON and Markdown."""
        out_json = tmp_path / "test_sample.json"
        out_md = tmp_path / "test_sample.md"

        evidence = run_upload_demo(output_json=out_json, output_md=out_md)

        assert out_json.is_file()
        assert out_md.is_file()

        # Verify JSON evidence structure
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["metadata"]["concept"] == "3.45 Document Upload & Indexing Endpoint"
        assert data["sample_upload"]["filename"] == "new-policy.md"
        assert data["runtime_searchability_test"]["runtime_searchability_confirmed"] is True
        assert data["error_handling_tests"]["unsupported_media_type"]["status_code"] == 415
        assert data["error_handling_tests"]["empty_file"]["status_code"] == 400

        # Verify Markdown report structure
        md_text = out_md.read_text(encoding="utf-8")
        assert "FieldSure AI — Document Upload & Indexing Verification" in md_text
        assert "POST /documents" in md_text
        assert "Runtime Searchability Verification" in md_text
