# FieldSure AI — System Architecture (Member 1 Scope)

> **Document Scope**: Architecture and design specifications for **Member 1 (Document Processing & LLM Foundation)**, detailing data flows, component boundaries, interfaces, and integration contracts with Member 2 (Embeddings & Retrieval) and Member 3 (RAG, Backend & Application).

---

## 1. System Overview & Module Responsibilities

Member 1 establishes the foundational data ingestion pipeline and LLM communication layer for FieldSure AI across 10 specific modules:

```mermaid
flowchart LR
    subgraph Member1_Foundation["👤 Member 1: Documents & LLM Foundation"]
        direction TB
        M1["#1 Dev Env (uv)"]
        M2["#2 Git & Workflow"]
        M3["#3 LLM API Access"]
        M4["#4 Prompt Construction"]
        M10["#10 Multi-Format Loader"]
        M11["#11 Text Extraction & Cleaning"]
        M12["#12 Structural Chunking"]
        M13["#13 Metadata Tagging"]
        M14["#14 Token-Aware Sizing & Overlap"]
        M15["#15 Corpus Validation"]
        
        M10 --> M11 --> M12 --> M14 --> M13 --> M15
        M1 --> M3 --> M4
    end

    subgraph Member2_Downstream["👤 Member 2: Vector & Retrieval"]
        M16["#16-#27 Embeddings & Vector DB Indexing"]
    end

    subgraph Member3_Downstream["👤 Member 3: RAG & UI"]
        M28["#28-#40 Context Augmentation, API & Chat UI"]
    end

    M15 -- "Validated Chunks (JSONL / Data Objects)" --> M16
    M4 -- "System Prompts & Client Contract" --> M28
```

### Module Responsibilities Breakdown

| Module | Identifier | Core Responsibility |
| :--- | :--- | :--- |
| **#1** | `workspace_setup` | Configure `pyproject.toml`, `.env.example`, `.gitignore`, and `uv` virtual environment. |
| **#2** | `repo_workflow` | Scaffolding directory tree, standardizing imports, linting/formatting rules, and team workflow. |
| **#3** | `llm_client` | Safe credential loading (`.env`), LLM client initialization, error-handled chat completion (`src/chat_completion.py`). |
| **#4** | `prompt_engineering` | Role-based prompt engineering (`prompts/`), baseline vs. constrained comparative evaluation (`src/prompt_comparison.py`). |
| **#10** | `doc_loader` | Multi-format reader dispatching `.pdf`, `.docx`, `.md`, and `.txt` files from `data/` safely. |
| **#11** | `text_cleaner` | Text extraction, unicode normalization (NFKC), excessive whitespace collapsing, header/footer cleaning. |
| **#12** | `chunker` | Semantic boundary chunking (splitting on section headers, markdown headings, paragraph breaks). |
| **#13** | `metadata_tracker` | Traceability tagging (`doc_id`, `source_file`, `title`, `version`, `date`, `section`, `chunk_index`). |
| **#14** | `token_budget` | Token-aware size enforcement (e.g. 500 max tokens) and sliding window overlap calculation. |
| **#15** | `corpus_validator` | End-to-end ingestion runner, corpus health audit, empty chunk filtering, and JSONL export for Member 2. |

---

## 2. Workspace & Folder Structure

The project workspace strictly isolates concerns. Member 1 creates and manages the ingestion pipeline, prompts, outputs, and LLM utilities without stepping into downstream directories:

```
SW_S66_FieldSure_AI_Kalvium-Community/
│
├── data/                         # Public health documents (PDF, DOCX, MD, TXT) [git-ignored]
│   └── sample_advisories/        # Sample verified guidance documents for testing/validation
│
├── src/                          # Application source code
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration & environment loader
│   ├── llm_client.py             # LLM API client wrapper with clean error handling
│   ├── chat_completion.py        # Module #3: Baseline completion entry point
│   ├── prompt_comparison.py      # Module #4: Comparative evaluation entry point
│   │
│   └── ingestion/                # Document processing pipeline (Modules #10-#15)
│       ├── __init__.py
│       ├── loader.py             # Module #10: Multi-format file intake
│       ├── cleaner.py            # Module #11: Text extraction & sanitization
│       ├── chunker.py            # Modules #12 & #14: Recursive & token-aware chunking
│       ├── metadata.py           # Module #13: Source & version metadata schema
│       └── pipeline.py           # Module #15: End-to-end ingestion & validation runner
│
├── prompts/                      # Prompt templates
│   ├── system_prompt_baseline.txt     # Baseline / unconstrained system prompt
│   └── system_prompt_constrained.txt  # Strict grounding & citation-enforced prompt
│
├── outputs/                      # Logs, execution artifacts, and sample outputs
│   ├── chat_completion_sample.log
│   ├── prompt_comparison_sample.log
│   └── ingestion_validation_report.json
│
├── tests/                        # Automated unit and integration tests
│   ├── test_llm_client.py
│   ├── test_loader.py
│   ├── test_cleaner.py
│   ├── test_chunker.py
│   └── test_ingestion_pipeline.py
│
├── .env.example                  # Environment secrets template
├── .gitignore                    # Git exclusion rules
├── pyproject.toml                # UV project configuration and dependency tree
├── uv.lock                       # Lockfile ensuring cross-platform reproducibility
├── Readme.md                     # Repository documentation
├── PRD.md                        # Product requirements document
├── ARCHITECTURE.md               # System architecture (this document)
├── DESIGN.md                     # Technical implementation decisions
├── RULES.md                      # Strict coding and architecture rules
├── TASKS.md                      # Module-by-module task tracking
├── MEMORY.md                     # Current state and project memory
└── AGENTS.md                     # AI agent operational guidelines
```

---

## 3. Data Flows & Processing Pipelines

### 3.1 Document Ingestion Flow (Modules #10–#15)

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Field Worker / Ingestion Script
    participant L as Loader (loader.py)
    participant C as Cleaner (cleaner.py)
    participant K as Chunker (chunker.py)
    participant M as Metadata (metadata.py)
    participant V as Validator (pipeline.py)
    participant M2 as Member 2 Vector DB

    Dev->>L: Ingest directory data/ (PDF/DOCX/MD/TXT)
    L->>L: Inspect file extensions & validate readability
    L->>C: Raw document stream + initial file metadata
    C->>C: Normalize unicode, strip headers/footers, normalize whitespace
    C->>K: Clean structured document text
    K->>K: Apply structural splitting (headings, paragraphs)
    K->>K: Enforce token budget (500 tokens) with sliding overlap (50 tokens)
    K->>M: Raw text chunks
    M->>M: Attach immutable metadata (chunk_id, doc_id, section, version, token_count)
    M->>V: List of Chunk objects
    V->>V: Validate zero empty chunks, schema completeness, token bounds
    V->>Dev: Ingestion audit summary (JSON report)
    V-->>M2: Export validated chunks dataset (JSONL / Data Objects)
```

### 3.2 LLM Client & Prompt Flow (Modules #3 & #4)

```mermaid
sequenceDiagram
    autonumber
    actor User as Field Worker Question / Script
    participant Cfg as Config (config.py)
    participant Client as LLM Client (llm_client.py)
    participant P as Prompts (prompts/)
    participant API as External LLM API Provider
    participant Out as Outputs / Logging

    User->>Cfg: Load API_BASE_URL, API_KEY, CHAT_MODEL from .env
    Cfg-->>Client: Safe configuration parameters
    User->>P: Load system_prompt_constrained.txt
    P-->>Client: Formatted role-based messages [System, User]
    Client->>API: POST /chat/completions (payload with temperature, messages)
    alt API Success
        API-->>Client: 200 OK + JSON completion response
        Client->>Out: Log token usage, response time, payload (masking secrets)
        Client-->>User: Grounded answer with inline citations
    else API / Config Error
        API-->>Client: 4xx / 5xx error or connection failure
        Client->>Out: Log sanitized error message without raw traceback
        Client-->>User: Graceful failure with descriptive status code
    end
```

### 3.3 End-to-End RAG Pipeline Architecture & Flow (Module #28)

```mermaid
sequenceDiagram
    autonumber
    actor User as Field Worker
    participant Orchestrator as Pipeline Orchestrator (answer_query)
    participant Embedder as Query Embedder (embed_query)
    participant Store as Vector Store (retrieve_context)
    participant Assembler as Context Assembler (assemble_context)
    participant Generator as Grounded Generator (generate_answer)

    User->>Orchestrator: 1. Submit Query
    Orchestrator->>Embedder: 2. embed_query(query)
    Embedder-->>Orchestrator: Dense Vector (e.g. 1024 or 1536 dim)
    Orchestrator->>Store: 3. retrieve_context(vector, k=4)
    alt Relevant Chunks Retrieved (similarity >= threshold)
        Store-->>Orchestrator: Top-k Ranked Chunks with Metadata
        Orchestrator->>Assembler: 4. assemble_context(chunks)
        Assembler-->>Orchestrator: Assembled Context with [1] Source: ...
        Orchestrator->>Generator: 5. generate_answer(query, context)
        Generator-->>Orchestrator: Grounded Response + Inline Citations
        Orchestrator-->>User: {"answer": ..., "sources": [...], "context": ...}
    else Empty Retrieval / Below Threshold
        Store-->>Orchestrator: Empty Chunks List []
        Orchestrator-->>User: Fallback: "I could not find relevant context for that question." (sources: [])
    end
```

---

## 4. Shared Interfaces & Data Contracts

To guarantee seamless handoffs across team members, Member 1 implements strict, typed data contracts:

### 4.1 Document & Chunk Data Contract (Handoff to Member 2)

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass(frozen=True)
class DocumentMetadata:
    doc_id: str                      # Unique hash of file content or name
    source_file: str                 # Relative path, e.g., "data/covid19_protocol_v2.pdf"
    title: str                       # Extracted or inferred document title
    document_version: str            # Version string, e.g., "v2.1", "2024-R1"
    publication_date: Optional[str]  # ISO date string (YYYY-MM-DD) if available
    agency: Optional[str]            # Issuing body, e.g., "WHO", "CDC", "MOH"
    file_type: str                   # "pdf", "docx", "md", "txt"

@dataclass(frozen=True)
class ChunkMetadata:
    chunk_id: str                    # Unique hash: f"{doc_id}_c{chunk_index:04d}"
    doc_id: str                      # Parent document reference
    source_file: str                 # File name / path reference for citations
    document_title: str              # Title for citation display
    document_version: str            # Version for citation verification
    section_heading: Optional[str]   # Nearest enclosing section/header
    chunk_index: int                 # 0-indexed position within parent document
    total_chunks: int                # Total number of chunks in parent document
    char_length: int                 # Character count of chunk text
    token_count: int                 # Accurate token count from tokenizer
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str                        # Cleaned text content ready for embedding
    metadata: ChunkMetadata          # Comprehensive traceable metadata
```

### 4.2 LLM Completion Interface (Handoff to Member 3)

```python
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass(frozen=True)
class LLMMessage:
    role: str                        # "system", "user", "assistant"
    content: str                     # Prompt or response text

@dataclass(frozen=True)
class CompletionResult:
    content: str                     # Generated response text
    model: str                       # Model identifier used
    prompt_tokens: int               # Prompt token consumption
    completion_tokens: int           # Generated token consumption
    total_tokens: int                # Aggregate token count
    finish_reason: str               # "stop", "length", etc.
    raw_response: Optional[Dict[str, Any]] = None
```

---

## 5. Integration Boundaries with Other Members

### 5.1 Member 1 ➔ Member 2 (Embeddings & Vector Database)
- **Handoff Artifact**: A serialized JSONL file (`outputs/ingested_chunks.jsonl`) or a Python generator yielding validated `DocumentChunk` instances.
- **Member 2 Expectation**: Member 2 consumes `chunk.text` to generate vector embeddings, and persists `chunk.metadata` directly into their Vector Database (e.g., ChromaDB, Qdrant, FAISS) for metadata filtering.
- **Contract Stability**: Member 1 guarantees that `chunk_id` is unique and deterministic, `text` is non-empty, and `token_count` does not exceed the model's embedding context length.

### 5.2 Member 1 ➔ Member 3 (RAG, Backend API & Application)
- **Handoff Artifact**:
  - Reusable `llm_client.py` providing a clean completion method.
  - Standardized system prompt template (`prompts/system_prompt_constrained.txt`) defining grounding rules, citation format, and refusal phrases.
- **Member 3 Expectation**: Member 3 imports the LLM client and system prompt template into their RAG orchestration pipeline, injecting retrieved context from Member 2 into the user prompt.

---

## 6. Error Handling & Validation Architecture

1. **Document Loading Errors**:
   - Non-existent files or unsupported extensions raise descriptive `UnsupportedFileError` or `CorruptDocumentError` rather than crashing the batch run.
   - Faulty documents are recorded in an error ledger; valid documents in the batch continue processing.
2. **Text Cleaning & Extraction Errors**:
   - Zero-length text after extraction triggers an empty document warning and skips chunking.
   - Malformed unicode characters are replaced safely via `errors="replace"`.
3. **Token Limit Breaches**:
   - The chunker checks each piece against the token threshold. If a single sentence exceeds the token budget (e.g., in unbroken code or long scientific tables), it falls back to character slicing at whitespace boundaries with warning logs.
4. **LLM API & Network Failures**:
   - Missing `.env` or missing API keys trigger an explicit `ConfigurationError` explaining how to configure `.env` before attempting any network requests.
   - HTTP timeouts, 401 unauthorized, or 429 rate limit errors are caught and logged with actionable hints (e.g., *"Check API key or rate quota"*), returning non-zero exit codes without leaking secrets or printing raw tracebacks.
