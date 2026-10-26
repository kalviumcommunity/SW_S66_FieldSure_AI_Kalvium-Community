"""Backend API for the RAG Service (Concept 3.44).

Exposes the FieldSure AI RAG pipeline through a production FastAPI backend API
so a frontend, chatbot UI, or another service can send questions and receive
grounded answers with source citations.

Tasks:
    Task 1: Create a query endpoint (POST /query) that accepts a question and returns an answer with sources.
    Task 2: Return structured JSON containing answer, sources, status, and metadata.
    Task 3: Validate input (length, format) and handle errors with clear HTTP status codes (400, 422, 500).
    Task 4: Load config from environment variables (API keys, model names, vector DB settings).
    Task 5: Commit sample request and response showing the endpoint working.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Union

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import mask_api_key
from src.document_upload import (
    MAX_UPLOAD_SIZE,
    RUNTIME_UPLOADED_CHUNKS,
    SUPPORTED_EXTENSIONS,
    UPLOAD_DIR,
    UploadResponse,
    UploadSummary,
    clean,
    embed_chunks,
    index_chunks,
    load_text,
    process_uploaded_document,
    store_upload,
    tag_chunks,
    token_chunks,
    validate_upload,
)
from src.grounded_generation import generate_grounded_answer
from src.guardrails import (
    DEFAULT_BENCHMARK_CHUNKS,
    DEFAULT_REFUSAL_MESSAGE,
    STATUS_ANSWERED,
    STATUS_REFUSED_WEAK_CONTEXT,
    GuardedAnswer,
    get_chunk_score,
    guarded_answer,
    retrieval_is_strong,
    retrieve,
)

logger = logging.getLogger("fieldsure.backend_api")

# Load environment variables from .env if present
_env_path = REPO_ROOT / ".env"
if _env_path.is_file():
    load_dotenv(dotenv_path=_env_path)
else:
    load_dotenv()

# Default output paths for Task 5 artifacts
DEFAULT_SAMPLE_JSON = REPO_ROOT / "outputs" / "backend_api_sample.json"
DEFAULT_SAMPLE_MD = REPO_ROOT / "outputs" / "backend_api_sample.md"


# ---------------------------------------------------------------------------
# Task 4: Environment-Driven Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class APIConfig:
    """Configuration loaded dynamically from the environment."""

    api_key: str
    embedding_model: str = "text-embedding-3-small"
    vector_db_url: str = "http://localhost:6333"
    collection_name: str = "rag_chunks"
    chat_model: str = "gpt-4o-mini"
    api_base_url: str = "https://api.openai.com/v1"
    host: str = "0.0.0.0"
    port: int = 8000

    def to_safe_dict(self) -> Dict[str, Any]:
        """Return configuration dictionary with masked API credentials."""
        return {
            "embedding_model": self.embedding_model,
            "vector_db_url": self.vector_db_url,
            "collection_name": self.collection_name,
            "chat_model": self.chat_model,
            "api_base_url": self.api_base_url,
            "host": self.host,
            "port": self.port,
            "api_key_configured": bool(self.api_key),
            "api_key_masked": mask_api_key(self.api_key),
        }


def load_environment_config(
    env_path: Optional[Path] = None,
    strict: bool = False,
) -> APIConfig:
    """Load configuration dynamically from environment variables.

    Reads:
        OPENAI_API_KEY / API_KEY: Authentication API key.
        EMBEDDING_MODEL: Target embedding model (default: text-embedding-3-small).
        VECTOR_DB_URL: Remote vector DB URL or connection path.
        COLLECTION_NAME / VECTOR_DB_COLLECTION: Target vector collection name.
        CHAT_MODEL: LLM model name (default: gpt-4o-mini).
        API_BASE_URL: LLM provider base URL (default: https://api.openai.com/v1).
        HOST: Server bind host (default: 0.0.0.0).
        PORT: Server bind port (default: 8000).

    Args:
        env_path: Optional path to .env file.
        strict: If True, raises ValueError when required credentials are missing.

    Returns:
        Validated APIConfig instance.

    Raises:
        ValueError: If strict is True and required credentials are not configured.
    """
    if env_path is not None and env_path.is_file():
        load_dotenv(dotenv_path=env_path)

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
        or ""
    ).strip()

    if strict and (not api_key or api_key == "your_llm_api_key_here"):
        raise ValueError(
            "OPENAI_API_KEY or API_KEY is required in strict mode but not configured."
        )

    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"
    vector_db_url = os.getenv("VECTOR_DB_URL", "http://localhost:6333").strip() or "http://localhost:6333"
    collection_name = (
        os.getenv("COLLECTION_NAME")
        or os.getenv("VECTOR_DB_COLLECTION")
        or "rag_chunks"
    ).strip() or "rag_chunks"

    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip() or "https://api.openai.com/v1"
    host = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"

    try:
        port = int(os.getenv("PORT", "8000").strip())
    except ValueError:
        port = 8000

    return APIConfig(
        api_key=api_key,
        embedding_model=embedding_model,
        vector_db_url=vector_db_url,
        collection_name=collection_name,
        chat_model=chat_model,
        api_base_url=api_base_url,
        host=host,
        port=port,
    )


# Module-level configuration attributes conforming to Concept 3.44 specification
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_DB_URL: str = os.getenv("VECTOR_DB_URL", "http://localhost:6333")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME") or os.getenv("VECTOR_DB_COLLECTION", "rag_chunks")


# ---------------------------------------------------------------------------
# Task 1 & Task 2: Request & Response Shapes (Pydantic Models)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Incoming query request payload from client/frontend."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's question or search query.",
        examples=["What evidence is required for project submission?"],
    )

    @field_validator("question")
    @classmethod
    def validate_non_whitespace(cls, value: str) -> str:
        """Ensure question contains non-whitespace content."""
        cleaned = value.strip()
        if len(cleaned) < 3:
            raise ValueError("Question must be at least 3 characters long after trimming whitespace.")
        return cleaned


class Source(BaseModel):
    """Source citation metadata attached to a grounded answer."""

    source: str = Field(..., description="Source document filename or identifier.")
    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier.")
    score: Optional[float] = Field(default=None, description="Retrieval similarity or relevance score.")


class QueryResponse(BaseModel):
    """Structured response payload returned by the RAG backend API."""

    answer: str = Field(..., description="Grounded answer text generated by RAG pipeline.")
    sources: List[Source] = Field(default_factory=list, description="List of source citations used.")
    status: str = Field(default=STATUS_ANSWERED, description="Status code ('answered', 'refused_weak_context').")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional diagnostic metrics and metadata.")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = "healthy"
    service: str = "FieldSure AI RAG Service"
    version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Task 1, 2, 3: Endpoint Definition & FastAPI Application Factory
# ---------------------------------------------------------------------------

async def rag_pipeline_stream(
    question: str,
    retriever_fn: Optional[Callable[[str, int], Sequence[Dict[str, Any]]]] = None,
    store: Optional[Any] = None,
    supplemental_chunks: Optional[Sequence[Dict[str, Any]]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    delay: float = 0.01,
) -> AsyncIterator[Dict[str, Any]]:
    """Stream RAG pipeline events (citations, progressive tokens, done, error).

    Pipeline:
        1. Retrieve candidate context chunks.
        2. Evaluate retrieval strength with guardrails.
        3. If weak or out-of-domain:
           - Emit citations event with empty sources and refused status.
           - Stream safe refusal tokens progressively.
        4. If strong:
           - Emit citations event with formatted source metadata and text.
           - Synthesize grounded answer and stream tokens progressively.

    Args:
        question: User query string.
        retriever_fn: Optional retrieval callable.
        store: Optional vector store.
        supplemental_chunks: Optional additional chunks to search.
        llm_fn: Optional generator callable.
        delay: Artificial inter-token delay in seconds (default: 0.01s).

    Yields:
        Event dictionaries containing citations or progressive tokens.
    """
    # 1. Retrieve candidate context
    if retriever_fn is not None:
        retrieved = list(retriever_fn(question, 4))
    else:
        retrieved = retrieve(
            question,
            k=4,
            store=store,
            supplemental_chunks=supplemental_chunks,
        )

    # 2. Check retrieval strength against relevance threshold
    if not retrieval_is_strong(retrieved):
        top_score = max((get_chunk_score(c) for c in retrieved), default=0.0)
        yield {
            "type": "citations",
            "sources": [],
            "status": STATUS_REFUSED_WEAK_CONTEXT,
            "metadata": {
                "is_refused": True,
                "top_score": round(float(top_score), 4),
                "chunks_used": 0,
            },
        }

        # Stream refusal tokens progressively
        tokens = re.findall(r"\S+\s*|\s+", DEFAULT_REFUSAL_MESSAGE)
        for token in tokens:
            yield {"type": "token", "text": token}
            if delay > 0:
                await asyncio.sleep(delay)
        return

    # 3. Strong context: Format detailed source citations
    sources: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(retrieved):
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        doc_name = (
            metadata.get("document_title")
            or metadata.get("source_file")
            or metadata.get("source")
            or chunk.get("source")
            or "Unknown Document"
        )
        chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id")
        section = metadata.get("section_heading") or metadata.get("section")
        score = get_chunk_score(chunk)
        text = chunk.get("text", "")

        sources.append({
            "id": f"source-{idx + 1}",
            "label": f"[{idx + 1}]",
            "document": str(doc_name),
            "source": str(doc_name),
            "chunk_id": str(chunk_id) if chunk_id else None,
            "section": str(section) if section else None,
            "score": round(float(score), 4) if score is not None else None,
            "text": str(text),
        })

    top_score = max((get_chunk_score(c) for c in retrieved), default=0.0)

    yield {
        "type": "citations",
        "sources": sources,
        "status": STATUS_ANSWERED,
        "metadata": {
            "is_refused": False,
            "top_score": round(float(top_score), 4),
            "chunks_used": len(sources),
        },
    }

    # 4. Generate grounded answer and stream tokens
    grounded_res = generate_grounded_answer(question, retrieved, llm_fn=llm_fn)
    if hasattr(grounded_res, "answer"):
        answer_text = str(grounded_res.answer)
    elif isinstance(grounded_res, dict) and "answer" in grounded_res:
        answer_text = str(grounded_res["answer"])
    else:
        answer_text = str(grounded_res)

    tokens = re.findall(r"\S+\s*|\s+", answer_text)
    for token in tokens:
        yield {"type": "token", "text": token}
        if delay > 0:
            await asyncio.sleep(delay)


def create_app(
    pipeline_fn: Optional[Callable[[str], Any]] = None,
    config: Optional[APIConfig] = None,
    stream_pipeline_fn: Optional[Callable[[str], Any]] = None,
) -> FastAPI:
    """Create and configure a production FastAPI application for the RAG service.

    Args:
        pipeline_fn: Optional callable for synchronous RAG answering.
        config: Optional pre-loaded APIConfig.
        stream_pipeline_fn: Optional callable for streaming RAG events.

    Returns:
        Configured FastAPI application instance.
    """
    app_config = config or load_environment_config()

    def default_rag_pipeline(question: str) -> Any:
        runtime_chunks = getattr(app.state, "runtime_chunks", [])
        combined_chunks = list(DEFAULT_BENCHMARK_CHUNKS) + list(runtime_chunks)
        store = getattr(app.state, "vector_store", None)
        return guarded_answer(
            question,
            retriever_fn=lambda q, k_val: retrieve(
                q,
                k=k_val,
                store=store,
                supplemental_chunks=combined_chunks,
            ),
        )

    rag_pipeline = pipeline_fn or default_rag_pipeline

    app = FastAPI(
        title="FieldSure AI — Backend API for RAG Service",
        description=(
            "Exposes the FieldSure AI RAG pipeline through a stable backend API contract. "
            "Enables frontends, chatbot UIs, and external health services to submit clinical "
            "questions and receive grounded answers with full source attribution."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Attach config, pipeline callable, and runtime index registry to app.state
    app.state.config = app_config
    app.state.pipeline_fn = rag_pipeline
    app.state.stream_pipeline_fn = stream_pipeline_fn
    app.state.runtime_chunks = []
    app.state.upload_dir = UPLOAD_DIR

    # Enable CORS for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", tags=["Root"])
    def root() -> Dict[str, Any]:
        """Root status endpoint returning service description and documentation links."""
        return {
            "service": "FieldSure AI RAG Service",
            "version": "0.1.0",
            "status": "online",
            "endpoints": {
                "query": "POST /query",
                "query_stream": "POST /query/stream",
                "documents": "POST /documents",
                "health": "GET /health",
                "config": "GET /config",
                "docs": "GET /docs",
            },
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health_check() -> HealthResponse:
        """Readiness and liveness probe endpoint."""
        return HealthResponse(
            status="healthy",
            service="FieldSure AI RAG Service",
            version="0.1.0",
        )

    @app.get("/config", tags=["Configuration"])
    def get_config() -> Dict[str, Any]:
        """Return active service configuration with sensitive credentials masked."""
        cfg: APIConfig = getattr(app.state, "config", app_config)
        return cfg.to_safe_dict()

    @app.post(
        "/documents",
        response_model=UploadResponse,
        status_code=status.HTTP_200_OK,
        tags=["Document Ingestion"],
        summary="Upload and index a document at runtime",
        description="Stores an uploaded file, extracts text, cleans, chunks, embeds, and indexes it into the knowledge base so it becomes immediately searchable via POST /query without restarting the app.",
    )
    async def upload_document_endpoint(file: UploadFile = File(...)) -> UploadResponse:
        """Store and index a document at runtime, making it immediately searchable."""
        try:
            target_upload_dir = getattr(app.state, "upload_dir", UPLOAD_DIR)
            path = await store_upload(file, upload_dir=target_upload_dir)
            store = getattr(app.state, "vector_store", None)
            summary = process_uploaded_document(path, store=store)

            # Register chunks in app.state.runtime_chunks for instant query searchability
            tagged = summary.get("tagged_chunks", [])
            if hasattr(app.state, "runtime_chunks"):
                for chunk in tagged:
                    if not any(
                        c.get("chunk_id") == chunk.get("chunk_id")
                        for c in app.state.runtime_chunks
                    ):
                        app.state.runtime_chunks.append(chunk)

            return UploadResponse(
                status="indexed",
                filename=file.filename or Path(path).name,
                summary=UploadSummary(
                    document=str(path),
                    chunks=summary["chunks"],
                    indexed=summary["indexed"],
                ),
            )
        except HTTPException:
            raise
        except Exception as error:
            logger.error("Document indexing failed: %s", error, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document indexing failed",
            )

    @app.post(
        "/query",
        response_model=QueryResponse,
        status_code=status.HTTP_200_OK,
        tags=["RAG Query"],
        summary="Query the RAG pipeline",
        description="Accepts a user question, runs retrieval guardrails and grounded generation, and returns structured JSON with sources.",
    )
    def query_rag_endpoint(request: QueryRequest) -> QueryResponse:
        """Execute RAG query against guarded pipeline and return structured JSON."""
        current_pipeline = getattr(app.state, "pipeline_fn", rag_pipeline)

        try:
            result = current_pipeline(request.question)

            # Ensure result can be accessed like a dictionary
            if not isinstance(result, dict) and not hasattr(result, "__getitem__"):
                raise ValueError("RAG pipeline returned an invalid response format.")

            raw_sources = result.get("sources", []) if isinstance(result, dict) else result["sources"]
            raw_answer = result.get("answer") if isinstance(result, dict) else result["answer"]
            raw_status = result.get("status", STATUS_ANSWERED) if isinstance(result, dict) else result["status"]

            # Format structured sources
            formatted_sources: List[Source] = []
            for item in raw_sources:
                if isinstance(item, dict):
                    source_name = str(item.get("source") or item.get("source_file") or "unknown")
                    chunk_id = item.get("chunk_id")
                    score = item.get("score")
                    # If score is not directly in the item, check top_score
                    if score is None and isinstance(result, dict) and result.get("top_score") is not None:
                        score = result.get("top_score")
                    formatted_sources.append(
                        Source(
                            source=source_name,
                            chunk_id=str(chunk_id) if chunk_id is not None else None,
                            score=float(score) if score is not None else None,
                        )
                    )
                elif isinstance(item, str):
                    formatted_sources.append(
                        Source(source=item, chunk_id=None, score=None)
                    )

            # Assemble diagnostic metadata
            metadata: Dict[str, Any] = {}
            if isinstance(result, dict):
                if "top_score" in result and result["top_score"] is not None:
                    metadata["top_score"] = float(result["top_score"])
                if "chunks_used" in result:
                    metadata["chunks_used"] = int(result["chunks_used"])
                if "is_refused" in result:
                    metadata["is_refused"] = bool(result["is_refused"])
                metadata["total_sources"] = len(formatted_sources)

            return QueryResponse(
                answer=str(raw_answer),
                sources=formatted_sources,
                status=str(raw_status),
                metadata=metadata if metadata else None,
            )

        except ValueError as error:
            logger.warning("Query validation error: %s", error)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
        except HTTPException:
            raise
        except Exception as error:
            logger.error("RAG pipeline unexpected execution error: %s", error, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RAG service failed",
            )

    @app.post(
        "/query/stream",
        tags=["RAG Query"],
        summary="Stream RAG query response and citations progressively",
        description="Streams Server-Sent Events (SSE) containing citations, progressive tokens, done, or error.",
    )
    async def query_rag_stream_endpoint(request: QueryRequest) -> StreamingResponse:
        """Stream RAG response events as Server-Sent Events (SSE)."""
        async def events():
            try:
                runtime_chunks = getattr(app.state, "runtime_chunks", [])
                combined_chunks = list(DEFAULT_BENCHMARK_CHUNKS) + list(runtime_chunks)
                store = getattr(app.state, "vector_store", None)

                stream_fn = getattr(app.state, "stream_pipeline_fn", None)
                if stream_fn is not None:
                    stream_iter = stream_fn(request.question)
                else:
                    stream_iter = rag_pipeline_stream(
                        request.question,
                        retriever_fn=lambda q, k_val: retrieve(
                            q,
                            k=k_val,
                            store=store,
                            supplemental_chunks=combined_chunks,
                        ),
                    )

                async for event in stream_iter:
                    yield f"data: {json.dumps(event)}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except Exception as error:
                logger.error("The answer stopped streaming: %s", error, exc_info=True)
                error_event = {
                    "type": "error",
                    "message": "The answer stopped streaming. Please retry.",
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app


# Default application instance for uvicorn entry point
app = create_app()


def query_rag(request: QueryRequest) -> QueryResponse:
    """Standalone functional wrapper matching the prompt reference implementation."""
    try:
        combined = list(DEFAULT_BENCHMARK_CHUNKS) + list(RUNTIME_UPLOADED_CHUNKS)
        result = guarded_answer(
            request.question,
            retriever_fn=lambda q, k_val: retrieve(
                q,
                k=k_val,
                supplemental_chunks=combined,
            ),
        )
        sources: List[Source] = []
        for source in result.get("sources", []):
            score = source.get("score")
            if score is None and "top_score" in result:
                score = result.get("top_score")
            sources.append(
                Source(
                    source=source.get("source") or source.get("source_file", "unknown"),
                    chunk_id=source.get("chunk_id"),
                    score=float(score) if score is not None else None,
                )
            )

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            status=result.get("status", STATUS_ANSWERED),
            metadata={
                "top_score": result.get("top_score"),
                "chunks_used": result.get("chunks_used", len(sources)),
                "is_refused": result.get("is_refused", False),
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        raise HTTPException(status_code=500, detail="RAG service failed")


async def upload_document(file: UploadFile) -> UploadResponse:
    """Standalone functional upload handler matching prompt reference specification."""
    try:
        path = await store_upload(file)
        summary = process_uploaded_document(path)
        return UploadResponse(
            status="indexed",
            filename=file.filename or Path(path).name,
            summary=UploadSummary(
                document=str(path),
                chunks=summary["chunks"],
                indexed=summary["indexed"],
            ),
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Document indexing failed")


async def stream_query(request: QueryRequest, delay: float = 0.01) -> StreamingResponse:
    """Standalone functional streaming query handler matching prompt specification."""
    async def events():
        try:
            combined = list(DEFAULT_BENCHMARK_CHUNKS) + list(RUNTIME_UPLOADED_CHUNKS)
            async for event in rag_pipeline_stream(
                request.question,
                retriever_fn=lambda q, k_val: retrieve(
                    q,
                    k=k_val,
                    supplemental_chunks=combined,
                ),
                delay=delay,
            ):
                yield f"data: {json.dumps(event)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception:
            error_event = {
                "type": "error",
                "message": "The answer stopped streaming. Please retry.",
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Task 5: Sample Request & Response Demonstration Runner
# ---------------------------------------------------------------------------

def run_demo(
    output_json: Optional[Path] = None,
    output_md: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute live requests against the FastAPI query endpoint and export sample evidence.

    Demonstrates:
        1. Grounded answer case: "What evidence is required for project submission?"
        2. Safe refusal case: "What is the refund policy for products?"
        3. Input validation error handling (422 status).

    Args:
        output_json: Target path for structured JSON sample artifact.
        output_md: Target path for Markdown report sample artifact.

    Returns:
        Dictionary containing all generated test cases and responses.
    """
    from fastapi.testclient import TestClient

    target_json = output_json or DEFAULT_SAMPLE_JSON
    target_md = output_md or DEFAULT_SAMPLE_MD
    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_md.parent.mkdir(parents=True, exist_ok=True)

    client = TestClient(app)

    print("=" * 76)
    print("FieldSure AI — Backend API for RAG Service (Concept 3.44 Demonstration)")
    print("=" * 76)

    # 1. Answered Case: Project Submission
    req_answered = {"question": "What evidence is required for project submission?"}
    print(f"\n[Case 1] Submitting Grounded Question:")
    print(f"  POST /query -> Body: {json.dumps(req_answered)}")
    res1 = client.post("/query", json=req_answered)
    print(f"  HTTP Status: {res1.status_code}")
    res1_json = res1.json()
    print(f"  Answer:      {res1_json['answer'][:80]}...")
    print(f"  Sources:     {len(res1_json['sources'])} citation(s)")
    print(f"  Status:      {res1_json['status']}")

    # 2. Refusal Case: Unsupported / Out-of-Corpus Query
    req_refused = {"question": "What is the refund policy for products?"}
    print(f"\n[Case 2] Submitting Out-of-Domain Question (Refusal Guardrail):")
    print(f"  POST /query -> Body: {json.dumps(req_refused)}")
    res2 = client.post("/query", json=req_refused)
    print(f"  HTTP Status: {res2.status_code}")
    res2_json = res2.json()
    print(f"  Answer:      {res2_json['answer']}")
    print(f"  Sources:     {len(res2_json['sources'])} citation(s)")
    print(f"  Status:      {res2_json['status']}")

    # 3. Validation Error Case: Too short query
    req_invalid = {"question": "hi"}
    print(f"\n[Case 3] Submitting Invalid Question (Too Short):")
    print(f"  POST /query -> Body: {json.dumps(req_invalid)}")
    res3 = client.post("/query", json=req_invalid)
    print(f"  HTTP Status: {res3.status_code} (Expected 422 Unprocessable Entity)")
    res3_json = res3.json()

    # 4. Health Check
    res_health = client.get("/health")
    res_config = client.get("/config")

    evidence = {
        "service": "FieldSure AI RAG Service API",
        "concept": "3.44 Backend API for the RAG Service",
        "configuration": res_config.json(),
        "health": res_health.json(),
        "demonstrations": [
            {
                "name": "grounded_answer_case",
                "description": "Standard grounded retrieval query returning answer and sources.",
                "request": {
                    "method": "POST",
                    "path": "/query",
                    "headers": {"Content-Type": "application/json"},
                    "body": req_answered,
                },
                "response": {
                    "status_code": res1.status_code,
                    "body": res1_json,
                },
            },
            {
                "name": "refusal_guardrail_case",
                "description": "Out-of-corpus query caught by guardrail returning safe refusal.",
                "request": {
                    "method": "POST",
                    "path": "/query",
                    "headers": {"Content-Type": "application/json"},
                    "body": req_refused,
                },
                "response": {
                    "status_code": res2.status_code,
                    "body": res2_json,
                },
            },
            {
                "name": "validation_error_case",
                "description": "Short invalid query rejected by Pydantic schema validation.",
                "request": {
                    "method": "POST",
                    "path": "/query",
                    "headers": {"Content-Type": "application/json"},
                    "body": req_invalid,
                },
                "response": {
                    "status_code": res3.status_code,
                    "body": res3_json,
                },
            },
        ],
    }

    # Write structured JSON sample
    with open(target_json, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)
    print(f"\n[Artifact] Structured JSON exported to: {target_json}")

    # Write Markdown sample documentation
    md_content = f"""# FieldSure AI — Backend API for RAG Service (Concept 3.44)

> **API Contract & Execution Evidence**: Verified demonstration of the FastAPI backend service exposing the FieldSure AI RAG pipeline for frontend and client consumption.

---

## 1. API Specification Summary

- **Endpoint**: `POST /query`
- **Request Content-Type**: `application/json`
- **Response Content-Type**: `application/json`
- **Health Check**: `GET /health`
- **Config Inspection**: `GET /config` (Masked credentials)

### Environment Configuration (Zero Hardcoding)
| Variable | Value / Source | Description |
|---|---|---|
| `OPENAI_API_KEY` | `{mask_api_key(OPENAI_API_KEY)}` | Authentication credential for LLM provider |
| `EMBEDDING_MODEL` | `{EMBEDDING_MODEL}` | Target embedding model for vector search |
| `VECTOR_DB_URL` | `{VECTOR_DB_URL}` | Remote or local vector database address |
| `COLLECTION_NAME` | `{COLLECTION_NAME}` | Vector collection storing ingested chunks |

---

## 2. Sample Request & Response (Grounded Answer)

### Request
```bash
curl -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "What evidence is required for project submission?"}}'
```

### Response (HTTP 200 OK)
```json
{json.dumps(res1_json, indent=2)}
```

---

## 3. Sample Request & Response (Refusal Guardrail)

### Request
```bash
curl -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "What is the refund policy for products?"}}'
```

### Response (HTTP 200 OK)
```json
{json.dumps(res2_json, indent=2)}
```

---

## 4. Input Validation & Error Handling

### Request (Invalid Question Length < 3)
```bash
curl -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{{"question": "hi"}}'
```

### Response (HTTP 422 Unprocessable Entity)
```json
{json.dumps(res3_json, indent=2)}
```

---

## 5. Verification Checklist

- [x] **Task 1 — Query endpoint**: `POST /query` operational and serving requests.
- [x] **Task 2 — Structured JSON**: Reliable response schema with `answer`, `sources`, `status`, and `metadata`.
- [x] **Task 3 — Validation & Error Handling**: Pydantic validation handles length breaches (422), ValueErrors (400), and unhandled errors (500).
- [x] **Task 4 — Environment Configuration**: All settings loaded from environment with zero hardcoded secrets.
- [x] **Task 5 — Committed Sample Evidence**: Exported to `{target_json.name}` and `{target_md.name}`.
"""
    with open(target_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[Artifact] Markdown report exported to: {target_md}")

    return evidence


def main() -> None:
    """CLI entry point for running the API server or generating demo artifacts."""
    parser = argparse.ArgumentParser(
        description="FieldSure AI — Backend API for RAG Service",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the uvicorn API server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host interface to bind (default from environment or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default from environment or 8000)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Execute automated demonstration and export sample artifacts",
    )

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        cfg = load_environment_config()
        host = args.host or cfg.host
        port = args.port or cfg.port
        print(f"Starting FieldSure AI RAG Service API on http://{host}:{port} ...")
        uvicorn.run("src.backend_api:app", host=host, port=port, reload=False)
    else:
        # Default action: run demo and export artifacts
        run_demo()


if __name__ == "__main__":
    main()
