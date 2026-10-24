"""API module entry point re-exporting backend API components (Concept 3.44)."""

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

__all__ = [
    "app",
    "create_app",
    "QueryRequest",
    "QueryResponse",
    "Source",
    "HealthResponse",
    "APIConfig",
    "load_environment_config",
    "query_rag",
    "run_demo",
]
