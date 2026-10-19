# FieldSure AI — Project Memory (MEMORY.md)

> **Role**: Member 1 — Document Processing & LLM Foundation  
> **Last Updated**: 2026-10-19  
> **Current Sprint Phase**: Documentation & Architectural Planning (Pre-Implementation)

---

## 1. Module Progress

- **Current Active Task**: Module #1 Completed; Ready for Module #2 (Repository & Team Workflow) or Module #3 (LLM API Access).
- **Completed Modules**: Module #1 (Development Environment & Project Workspace Setup: `pyproject.toml`, `.env.example`, `.gitignore`, directory scaffolding, `tests/test_workspace.py`).
- **In-Progress Modules**: Module #2 (GitHub Repository & Team Workflow Setup: Git initialized, docs established, scaffolding complete).
- **Next Planned Module**: Module #2 or Module #3 (LLM API Access & First Completion Call).

---

## 2. Key Technical Decisions

1. **Tooling & Environment**: Astral `uv` selected for fast, reproducible dependency resolution and virtual environment isolation (`pyproject.toml` + `uv.lock`).
2. **Text Extraction**:
   - `pypdf` proposed for pure-Python PDF extraction without system binary dependencies.
   - `python-docx` for `.docx` outline extraction.
   - Standard library UTF-8 reader for Markdown (`.md`) and plain text (`.txt`).
3. **Chunking & Token Budgeting**:
   - Recursive structural chunking splitting along semantic boundaries (headings, paragraphs, sentences).
   - Default budget: **500 tokens** chunk size with **50 tokens (10%) sliding overlap**.
   - Subword token accounting via `tiktoken` (`cl100k_base`).
4. **Metadata Traceability**:
   - Every chunk tagged with immutable provenance (`doc_id`, `chunk_id`, `source_file`, `document_title`, `document_version`, `section_heading`, `chunk_index`, `token_count`).
5. **LLM Client & Safety**:
   - Centralized `src/config.py` loading secrets from `.env`. Zero secrets logged or committed.
   - Explicit prompt constraints requiring citation of document and section for every claim, with deterministic refusal on unsupported queries.

---

## 3. Existing Reusable Functions & Components

*(To be populated as code modules are implemented)*
- *None implemented yet (pre-implementation phase).*

---

## 4. Current Issues & Open Decisions Requiring User Confirmation

1. **LLM Provider & Model**:
   - Target API provider (OpenAI, Anthropic, Google Gemini via OpenAI-compat or Gemini SDK) and default model name (`CHAT_MODEL`).
2. **Scanned PDF Handling**:
   - Confirmation whether public health documents will be purely digital text PDFs (handled by `pypdf`) or if scanned image OCR will be required.
3. **Downstream Embedding Window**:
   - Confirmation from Member 2 on target embedding model to lock in optimal chunk token size (default proposed: 500 tokens).
4. **Handoff Serialization**:
   - Default proposed: `outputs/ingested_chunks.jsonl` matching `DocumentChunk` dataclass.

---

## 5. Integration Contracts & Boundaries

- **With Member 2 (Embeddings & Vector DB)**:
  - Member 1 outputs validated chunks with clean text and complete metadata.
  - Member 2 takes these chunks to generate embeddings and populate Vector DB.
- **With Member 3 (RAG Orchestration & UI)**:
  - Member 1 provides tested `llm_client.py` and constrained system prompt templates (`prompts/system_prompt_constrained.txt`).
  - Member 3 builds dynamic context injection, citation validation at inference, FastAPI endpoints, and Chat UI.
