# FieldSure AI — Product Requirements Document (PRD)

> **Source of Truth Reference**: Synthesized from the official FieldSure AI project specifications and repository documentation (`Readme.md`).  
> **Status**: Active / Baseline  
> **Target Audience**: Public Health Field Workers (Field Epidemiologists, Community Health Workers, Outbreak Response Teams, Clinic Staff)

---

## 1. Project Overview & Problem Statement

### 1.1 Context & Problem Statement
Public health field workers operate in fast-paced, high-stakes environments responding to disease outbreaks, administering mass immunization campaigns, and applying regional public health advisories. During public health emergencies, field personnel must frequently consult complex, rapidly evolving guidance documents published by health authorities (e.g., CDC, WHO, state/national ministries of health).

**Key Pain Points:**
- **Information Overload & Search Latency**: Critical guidelines span hundreds of pages across disparate formats (PDFs, clinical protocols, policy briefs, bulletins). Finding specific, actionable directives takes valuable time.
- **Risk of Outdated or Conflicting Guidance**: Public health advisories are frequently superseded. Relying on outdated documents risks non-compliance and compromised patient/community safety.
- **Hallucination & Medical Misinformation Risk**: Standard consumer LLMs generate plausible-sounding but ungrounded medical guidance, which is unacceptable and dangerous in clinical and epidemiological operations.
- **Lack of Traceable Accountability**: Field workers cannot verify recommendations without explicit references to official chapter, section, and version numbers.

### 1.2 Proposed Solution: FieldSure AI
**FieldSure AI** is a specialized, production-grade Retrieval-Augmented Generation (RAG) assistant designed for public health field workers. It indexes official agency guidelines, protocols, and advisories to provide:
1. **100% Grounded Answers**: Responses are derived strictly from retrieved passages of official documents.
2. **Deterministic Source Citations**: Every factual claim is accompanied by verifiable document metadata (title, version, section, page/chunk).
3. **Deterministic Safe Refusals**: When available documentation does not contain adequate evidence to answer an inquiry, the system explicitly refuses rather than speculating or extrapolating.
4. **Version-Aware Guidance**: Deprecated, superseded, or obsolete advisories are flagged or filtered in favor of the latest active protocols.

---

## 2. Goals and Scope

### 2.1 Project Goals
- **Grounded Information Delivery**: Deliver verified, citation-backed answers to field workers' domain questions within seconds.
- **Safety First**: Eliminate ungrounded clinical extrapolation and enforce explicit fallbacks/refusals.
- **Auditability**: Ensure every answer can be traced back to the exact passage, document version, and publication date.
- **Robust Multi-Format Ingestion**: Ingest and process heterogeneous document formats (PDF, DOCX, Markdown, plain text) while preserving structural semantics.

### 2.2 System Scope
- **In Scope**:
  - Ingestion, text cleaning, structural chunking, and metadata tagging of official public health guidance documents.
  - Token-aware chunking with context-preserving overlaps.
  - Embedding generation and vector search with metadata filtering.
  - Prompt construction with strict grounding guardrails.
  - Retrieval-augmented generation with inline citations and safe refusals.
  - Backend API endpoints and an interactive Chat UI with streaming answers and citation panels.
- **Out of Scope**:
  - Autonomous patient diagnosis or prescribing medication (FieldSure is an informational reference tool, not a diagnostic medical device).
  - Direct integration with Electronic Health Records (EHR) or intake of Personal Health Information (PHI).
  - Ingestion of unverified public web pages, social media, or forum posts.

---

## 3. Team Ownership & Responsibilities

The FieldSure AI system is implemented across three specialized functional roles:

| Member / Role | Focus Area | Assigned Modules |
| :--- | :--- | :--- |
| **Member 1 (Current Role)** | **Document Processing & LLM Foundation** | **#1, #2, #3, #4, #10, #11, #12, #13, #14, #15** |
| **Member 2** | **Embeddings, Vector DB & Retrieval** | #16, #17, #18, #20, #21, #22, #23, #24, #25, #27 |
| **Member 3** | **RAG, Backend & Application** | #28, #29, #30, #31, #32, #34, #35, #36, #37, #38, #40 |

### 3.1 Member 1 Assigned Modules & Deliverables

1. **#1 Development Environment & Project Workspace Setup**:
   - Establish modern Python environment using `uv`.
   - Setup project configuration (`pyproject.toml`, `.env.example`, `.gitignore`).
2. **#2 GitHub Repository & Team Workflow Setup**:
   - Repository directory scaffolding (`data/`, `src/`, `prompts/`, `outputs/`, `frontend/`, `tests/`).
   - Coding standards, branch conventions, and pre-commit checks.
3. **#3 LLM API Access & First Completion Call**:
   - Secure environment configuration for LLM credentials.
   - Robust, non-leaking client initialization and baseline completion script (`src/chat_completion.py`).
4. **#4 Prompt Construction & System/User Roles**:
   - Prompt engineering for public health domain constraints.
   - Comparative evaluation between vague prompts and grounded, citation-mandating prompts (`src/prompt_comparison.py`).
5. **#10 Document Loading & Multi-Format Intake**:
   - Multi-format ingestion loaders supporting PDF, DOCX, Markdown, and TXT.
   - Validation of document integrity and readable streams.
6. **#11 Text Extraction & Cleaning Pipeline**:
   - Semantic text extraction preserving hierarchical headers and lists.
   - Sanitization pipeline: whitespace normalization, unicode cleaning (NFKC), artifact removal.
7. **#12 Document Chunking Strategies**:
   - Structural and recursive chunking preserving sentence and paragraph integrity.
8. **#13 Chunk Metadata & Source Tracking**:
   - Association of granular metadata (`chunk_id`, `doc_id`, `source_file`, `title`, `version`, `date`, `section`, `chunk_index`).
9. **#14 Token-Aware Chunk Sizing & Overlap**:
   - Strict token accounting respecting LLM embedding and context windows.
   - Configurable token budgets and sliding window overlap.
10. **#15 Corpus Preparation & Ingestion Validation**:
    - End-to-end ingestion pipeline execution on sample public health corpus.
    - Automated validation checks (leakage, empty chunks, metadata completeness) producing ingestion summary reports.

---

## 4. Functional Requirements

### 4.1 Ingestion & Document Processing (Member 1)
- **FR-1.1 Multi-Format Support**: System must ingest `.pdf`, `.docx`, `.md`, and `.txt` documents placed in `data/`.
- **FR-1.2 Text Normalization**: Extracted text must undergo whitespace standardization, unicode normalization, and header/footer cleanup without loss of clinical definitions.
- **FR-1.3 Structural Chunking**: Chunking must respect logical boundaries (sections, headings, paragraphs) rather than splitting mid-sentence or mid-definition.
- **FR-1.4 Token-Aware Boundaries**: Chunks must adhere to explicit token budgets (default target: 400–600 tokens) with configurable sliding overlap (default: 50–100 tokens).
- **FR-1.5 Metadata Traceability**: Each chunk must carry an immutable metadata block containing `doc_id`, `source_file`, `title`, `version`, `section_heading`, `publication_date`, `chunk_index`, and `token_count`.
- **FR-1.6 Ingestion Verification**: System must output validation statistics (total documents processed, valid chunks created, rejected files, average chunk size) and verify zero empty chunks.

### 4.2 LLM Foundation & Prompt Guardrails (Member 1)
- **FR-2.1 Secure Client Access**: Connect to specified LLM API via environment variables (`API_BASE_URL`, `API_KEY`, `CHAT_MODEL`) without exposing secrets in logs or terminal outputs.
- **FR-2.2 System Prompt Guardrails**: System prompt must instruct the model to:
  - Answer strictly using provided context.
  - Attribute every fact to its source document and section.
  - Deterministically refuse queries when context is missing or ambiguous.
  - Disclaim any diagnostic medical authority.
- **FR-2.3 Prompt Comparison Logging**: System must compare baseline vs. constrained prompts on standardized field health questions, logging payloads and token usage cleanly to `outputs/`.

### 4.3 Downstream Interfaces (Members 2 & 3 - Integration Contracts)
- **FR-3.1 Embeddings & Vector Storage (Member 2)**: Chunks produced by Member 1 must be consumable by Member 2's embedding generator and vector indexing pipeline.
- **FR-3.2 Query & RAG Orchestration (Member 3)**: LLM client interfaces and prompt templates produced by Member 1 must feed directly into Member 3's retrieval-augmented prompt builder, FastAPI backend, and Chat UI.

---

## 5. Non-Functional Requirements

- **NFR-1 Security & Privacy**:
  - Zero logging or persistence of API keys, credentials, or potential PHI/PII.
  - Strict exclusion of `.env` and local corpus raw caches from version control.
- **NFR-2 Reliability & Determinism**:
  - Graceful failure on corrupt documents or network/API interruptions, providing human-readable exit codes and error logs without raw stack trace leakage.
- **NFR-3 Performance & Efficiency**:
  - Ingestion pipeline must process typical guidance documents (50–100 pages) within seconds using memory-conscious streaming/iterative processing.
- **NFR-4 Maintainability & Clean Code**:
  - Strict PEP 8 adherence, 100% type annotations on public signatures, modular functional design without unnecessary object-oriented bloat.
- **NFR-5 Reproducibility**:
  - Fully locked and reproducible virtual environment using `uv` (`pyproject.toml` and `uv.lock`).

---

## 6. Safety, Privacy & Ethical Guidelines

1. **Strict Grounding Enforcement**: No extrapolation or general knowledge fallback is allowed when official documents are silent on an inquiry.
2. **Safe Refusal Behavior**: Refusals must be professional, transparent, and direct (e.g., *"The provided agency guidance does not contain information regarding [topic]. Please refer to your regional health officer or official health portal."*).
3. **Non-Diagnostic Disclaimer**: All system outputs are designated for administrative, operational, and protocol guidance only.
4. **Guidance Versioning**: The system must prioritize the latest publication date and flag superseded or withdrawn documents.
5. **No PHI/PII Handling**: The system is designed solely for public domain protocols; field workers must never input patient-identifiable data.

---

## 7. Acceptance Criteria (Member 1 Scope)

| Module | Acceptance Criteria |
| :--- | :--- |
| **#1 Dev Environment** | `uv sync` installs all dependencies without conflicts; environment activates cleanly on Windows/Linux/macOS; `.env.example` provides complete configuration keys. |
| **#2 Git & Workflow** | Repo directory layout matches standard; `.gitignore` blocks `.env`, `.venv`, `data/`, `__pycache__`; clean baseline ready for collaboration. |
| **#3 LLM API Access** | `src/chat_completion.py` connects to LLM, executes a test prompt, outputs valid completion, and logs execution without exposing API keys. |
| **#4 Prompt Construction** | `src/prompt_comparison.py` runs comparative queries, demonstrating that the constrained prompt successfully prevents hallucination and mandates citations compared to baseline. |
| **#10 Document Loading** | Document loader parses PDF, DOCX, MD, and TXT files from `data/`, correctly handling missing/invalid files with descriptive errors. |
| **#11 Text Extraction & Cleaning** | Raw text extracted with preserved headers/sections; excessive whitespace, control characters, and boilerplate artifacts cleaned cleanly. |
| **#12 Chunking Strategies** | Chunks preserve sentence/section boundaries; logical units remain intact without arbitrary splitting across semantic clauses. |
| **#13 Metadata Tracking** | Every chunk dictionary/object contains required metadata (`doc_id`, `source_file`, `title`, `version`, `section`, `chunk_index`, etc.). |
| **#14 Token-Aware Sizing** | Chunks conform to specified token window (e.g. 500 tokens max) with sliding overlap; verified via standard tokenizer. |
| **#15 Ingestion Validation** | Ingestion pipeline executes across a sample corpus, produces a validated chunk dataset ready for Member 2, and outputs an audit validation report. |

---

## 8. Proposed Implementation Decisions Requiring Confirmation

> *Note: These proposed decisions are documented for architectural review and require confirmation before implementation of downstream modules:*

1. **Primary PDF Parser**: Proposed `pypdf` for pure-Python lightweight execution without system binary dependencies (alternative: `pymupdf` or `pdfplumber`).
2. **DOCX Parser**: Proposed `python-docx` for structured paragraph and heading extraction.
3. **Tokenizer**: Proposed `tiktoken` (cl100k_base / o200k_base) for accurate token accounting.
4. **LLM Client Protocol**: Proposed standard OpenAI-compatible HTTP client library (`httpx` or `openai` SDK) compatible with endpoints specified by `API_BASE_URL`.
5. **Default Chunking Parameters**: Target chunk size of **500 tokens** with **50 tokens overlap** (10% ratio) as baseline for public health guidance passages.
