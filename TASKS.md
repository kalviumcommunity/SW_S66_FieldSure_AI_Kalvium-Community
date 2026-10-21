# FieldSure AI — Module Task Tracker (TASKS.md)

> **Owner**: Member 1 — Document Processing & LLM Foundation  
> **Assigned Modules**: #1, #2, #3, #4, #10, #11, #12, #13, #14, #15  
> **Status Conventions**: `Not Started` | `In Progress` | `Completed` | `Blocked`  
> **Baseline Evidence Assessment**: Based on inspection of the repository (only `.git` and `Readme.md` present), no application code, `pyproject.toml`, `.env.example`, or directory structure has been implemented yet. Tasks are marked accordingly.

---

## Summary Status Dashboard

| Module | Title | Status | Primary Output Files |
| :---: | :--- | :---: | :--- |
| **#1** | Development Environment & Project Workspace Setup | `Completed` | `pyproject.toml`, `.env.example`, `.gitignore`, workspace folders |
| **#2** | GitHub Repository & Team Workflow Setup | `In Progress` | Directory tree scaffolding, workflow & style guide |
| **#3** | LLM API Access & First Completion Call | `Completed` | `src/config.py`, `src/llm_client.py`, `src/chat_completion.py`, `outputs/chat_completion_sample.log` |
| **#4** | Prompt Construction & System/User Roles | `Not Started` | `prompts/`, `src/prompt_comparison.py`, `outputs/` |
| **#10** | Document Loading & Multi-Format Intake | `Not Started` | `src/ingestion/loader.py`, `tests/test_loader.py` |
| **#11** | Text Extraction & Cleaning Pipeline | `Not Started` | `src/ingestion/cleaner.py`, `tests/test_cleaner.py` |
| **#12** | Document Chunking Strategies | `Not Started` | `src/ingestion/chunker.py`, `tests/test_chunker.py` |
| **#13** | Chunk Metadata & Source Tracking | `Not Started` | `src/ingestion/metadata.py`, `tests/test_metadata.py` |
| **#14** | Token-Aware Chunk Sizing & Overlap | `Not Started` | `src/ingestion/chunker.py` (token logic), tests |
| **#15** | Corpus Preparation & Ingestion Validation | `Not Started` | `src/ingestion/pipeline.py`, `outputs/ingested_chunks.jsonl` |

---

## Detailed Module Specifications

### Module #1: Development Environment & Project Workspace Setup
- **Status**: `Completed`
- **Objective**: Establish a modern, reproducible Python development environment managed with `uv`, providing dependency management, isolated virtual environment, and environment configuration templates.
- **Dependencies**: None.
- **Expected Files**:
  - `pyproject.toml`
  - `.env.example`
  - `.gitignore`
  - `tests/test_workspace.py`
  - Workspace directories: `data/`, `src/`, `prompts/`, `outputs/`, `frontend/`, `tests/`
- **Implementation Subtasks**:
  1. [x] Create `.gitignore` ignoring `.env`, `.venv/`, `data/`, `outputs/*.log`, `__pycache__/`, `.pytest_cache/`.
  2. [x] Create `.env.example` defining all required configuration variables (`API_BASE_URL`, `API_KEY`, `CHAT_MODEL`, `EMBEDDING_MODEL`, `VECTOR_DB_URL`, `VECTOR_DB_PATH`).
  3. [x] Initialize `pyproject.toml` configured for Python >= 3.10 and declaring project metadata and dependencies (`pypdf`, `python-docx`, `tiktoken`, `python-dotenv`, `httpx`, `pytest`).
  4. [x] Scaffold structured project directories (`data/`, `src/`, `prompts/`, `outputs/`, `frontend/`, `tests/`) with `.gitkeep` and `README.md`.
  5. [x] Implement automated verification test in `tests/test_workspace.py`.
- **Acceptance Criteria**:
  - `pyproject.toml` contains all project dependencies and settings.
  - `.gitignore` prevents secret, cache, and artifact leakage.
  - Workspace directory tree aligns with `Readme.md` layout.
- **Required Tests**:
  - Verification test: `python tests/test_workspace.py`.

---

### Module #2: GitHub Repository & Team Workflow Setup
- **Status**: `In Progress` *(Git initialized; directory scaffolding and team workflow rules in development)*
- **Objective**: Set up directory structure, coding standards, branch conventions, and pre-commit checks for collaborative development across all three members.
- **Dependencies**: Module #1.
- **Expected Files**:
  - `src/__init__.py`
  - `src/ingestion/__init__.py`
  - `prompts/` (directory)
  - `outputs/` (directory with `.gitkeep`)
  - `data/` (directory with `.gitkeep` and README disclaimer)
  - `tests/__init__.py`
- **Implementation Subtasks**:
  1. [x] Git repository initialized with main branch.
  2. [x] Create core planning documents (`PRD.md`, `ARCHITECTURE.md`, `DESIGN.md`, `RULES.md`, `TASKS.md`, `MEMORY.md`, `AGENTS.md`).
  3. [ ] Scaffold empty source directories: `src/`, `src/ingestion/`, `prompts/`, `outputs/`, `data/`, `tests/`.
  4. [ ] Establish team workflow conventions (branch naming: `feature/member1-...`, PR reviews).
- **Acceptance Criteria**:
  - Directory tree strictly mirrors workspace layout in `Readme.md`.
  - Empty directories contain `.gitkeep` where required for git persistence.
- **Required Tests**:
  - Shell inspection verifying all directories exist and empty modules import cleanly.

---

### Module #3: LLM API Access & First Completion Call
- **Status**: `Completed`
- **Objective**: Implement secure LLM client initialization via environment variables and execute a verified baseline chat completion without credential leakage.
- **Dependencies**: Modules #1, #2.
- **Expected Files**:
  - `src/config.py`
  - `src/llm_client.py`
  - `src/chat_completion.py`
  - `outputs/chat_completion_sample.log`
  - `tests/test_llm_client.py`
- **Implementation Subtasks**:
  1. [x] Implement `src/config.py` to validate and load `API_BASE_URL`, `API_KEY`, `CHAT_MODEL`.
  2. [x] Implement `src/llm_client.py` wrapper wrapping HTTP completion requests with timeout and error handling.
  3. [x] Implement `src/chat_completion.py` executing a baseline public health query.
  4. [x] Sanitize logging to ensure API keys are never written to `outputs/chat_completion_sample.log` or stdout.
  5. [x] Handle API errors gracefully (exit code 1 for config error, exit code 2 for API error).
- **Acceptance Criteria**:
  - `uv run python src/chat_completion.py` runs and returns a valid completion.
  - Log file `outputs/chat_completion_sample.log` is generated without leaking credentials.
  - Missing API key produces actionable error message without raw traceback dump.
- **Required Tests**:
  - `tests/test_llm_client.py`: Unit test with mocked API response verifying token parsing and error handling.
- **Verification Evidence**:
  - Automated test suite `tests/test_llm_client.py` covers 200 OK completions, 401 auth errors, 429 rate limits, and network timeouts.
  - Sanitized baseline execution audit captured in `outputs/chat_completion_sample.log`.

---

### Module #4: Prompt Construction & System/User Roles
- **Status**: `Not Started`
- **Objective**: Design role-based prompt templates enforcing strict grounding, inline citations, and deterministic safe refusals for public health questions; build comparative test runner.
- **Dependencies**: Module #3.
- **Expected Files**:
  - `prompts/system_prompt_baseline.txt`
  - `prompts/system_prompt_constrained.txt`
  - `src/prompt_comparison.py`
  - `outputs/prompt_comparison_sample.log`
  - `tests/test_prompts.py`
- **Implementation Subtasks**:
  1. [ ] Author `system_prompt_baseline.txt` (standard, unconstrained assistant prompt).
  2. [ ] Author `system_prompt_constrained.txt` (mandating strict grounding, inline citations `[Doc, Section]`, safe refusal on ungrounded questions, non-diagnostic disclaimer).
  3. [ ] Implement `src/prompt_comparison.py` sending test questions across both prompts.
  4. [ ] Capture payloads, responses, token counts, and refusal triggers cleanly in `outputs/prompt_comparison_sample.log`.
- **Acceptance Criteria**:
  - Comparative execution clearly demonstrates baseline hallucination/extrapolation vs. constrained grounded refusal on out-of-domain queries.
  - Log records tokens and latency without exposing secret keys.
- **Required Tests**:
  - `tests/test_prompts.py`: Validate prompt template loading, variable formatting, and message structure.

---

### Module #10: Document Loading & Multi-Format Intake
- **Status**: `Not Started`
- **Objective**: Build a robust document loader supporting PDF, DOCX, Markdown, and TXT files, providing verified file streams and handling corrupted or missing files gracefully.
- **Dependencies**: Module #1.
- **Expected Files**:
  - `src/ingestion/loader.py`
  - `tests/test_loader.py`
- **Implementation Subtasks**:
  1. [ ] Implement file extension dispatcher (`.pdf`, `.docx`, `.md`, `.txt`).
  2. [ ] Integrate `pypdf` for page-wise text extraction.
  3. [ ] Integrate `python-docx` for structured paragraph extraction.
  4. [ ] Implement standard library readers for `.md` and `.txt`.
  5. [ ] Raise `UnsupportedFileError` on unknown extensions and catch file I/O exceptions cleanly.
- **Acceptance Criteria**:
  - Successfully loads sample files in all four target formats.
  - Returns raw text and initial file metadata (`source_file`, `file_type`, `file_size`).
  - Corrupt or invalid files raise descriptive exceptions without crashing the process.
- **Required Tests**:
  - `tests/test_loader.py`: Test reading valid fixtures across all 4 formats; test error handling for missing/unsupported files.

---

### Module #11: Text Extraction & Cleaning Pipeline
- **Status**: `Not Started`
- **Objective**: Develop a semantic cleaning pipeline to standardize extracted text, eliminate PDF/OCR noise, normalize unicode, and preserve clinical headers and list structures.
- **Dependencies**: Module #10.
- **Expected Files**:
  - `src/ingestion/cleaner.py`
  - `tests/test_cleaner.py`
- **Implementation Subtasks**:
  1. [ ] Implement `normalize_unicode(text)` using `unicodedata.normalize('NFKC', ...)`.
  2. [ ] Implement control character removal while preserving legitimate newlines and tabs.
  3. [ ] Implement whitespace collapsing (collapsing multi-spaces to single space; collapsing 3+ newlines to double newline).
  4. [ ] Implement header/footer and page-number stripping regex heuristics.
  5. [ ] Ensure clinical terms, numbers, abbreviations, and bullet structures remain intact.
- **Acceptance Criteria**:
  - Cleans messy raw text while maintaining semantic meaning and structural section markers.
  - Zero loss of medical terms, units (e.g. `mg/kg`), or protocol version numbers.
- **Required Tests**:
  - `tests/test_cleaner.py`: Test ligature normalization (`ﬁ` -> `fi`), whitespace collapsing, control character removal, and invariant preservation of clinical units.

---

### Module #12: Document Chunking Strategies
- **Status**: `Not Started`
- **Objective**: Implement structural and recursive document chunking that splits text at natural semantic boundaries (headings, paragraphs, sentences) rather than arbitrary character cuts.
- **Dependencies**: Module #11.
- **Expected Files**:
  - `src/ingestion/chunker.py`
  - `tests/test_chunker.py`
- **Implementation Subtasks**:
  1. [ ] Define recursive split hierarchy: `# ` headings -> `\n\n` paragraphs -> `\n` lines -> sentence boundaries.
  2. [ ] Implement section tracking to identify the nearest enclosing heading.
  3. [ ] Build recursive text splitter function.
- **Acceptance Criteria**:
  - Splits documents into logical units without breaking mid-sentence or mid-definition.
  - Section headers are preserved and prepended to chunks to retain context.
- **Required Tests**:
  - `tests/test_chunker.py`: Test chunking sample text with markdown headers; verify sentences are not split mid-word.

---

### Module #13: Chunk Metadata & Source Tracking
- **Status**: `Not Started`
- **Objective**: Attach immutable, traceable metadata to every chunk (`doc_id`, `chunk_id`, `source_file`, `title`, `version`, `date`, `section`, `chunk_index`, `token_count`).
- **Dependencies**: Modules #10, #12.
- **Expected Files**:
  - `src/ingestion/metadata.py`
  - `tests/test_metadata.py`
- **Implementation Subtasks**:
  1. [ ] Implement `DocumentChunk` and `ChunkMetadata` dataclasses.
  2. [ ] Generate deterministic hashes for `doc_id` (SHA-256) and `chunk_id` (`{doc_id}_c{index:04d}`).
  3. [ ] Extract or infer `document_title`, `version`, and `date` from filename or file headers.
  4. [ ] Attach section context to individual chunk metadata.
- **Acceptance Criteria**:
  - Every chunk contains 100% of required metadata attributes.
  - Chunk IDs are deterministic and reproducible across runs.
- **Required Tests**:
  - `tests/test_metadata.py`: Verify metadata creation, field types, and deterministic ID generation.

---

### Module #14: Token-Aware Chunk Sizing & Overlap
- **Status**: `Not Started`
- **Objective**: Enforce strict token budgets (e.g., target 500 tokens) using subword tokenization with configurable sliding overlap (e.g., 50 tokens) to prevent context boundary loss.
- **Dependencies**: Modules #12, #13.
- **Expected Files**:
  - Integrated into `src/ingestion/chunker.py`
  - `tests/test_token_chunker.py`
- **Implementation Subtasks**:
  1. [ ] Integrate `tiktoken` encoder (`cl100k_base`).
  2. [ ] Implement token counting utility `count_tokens(text: str) -> int`.
  3. [ ] Implement sliding window algorithm creating token-accurate chunks with defined overlap.
  4. [ ] Add safety fallback for monolithic single tokens/words exceeding budget.
- **Acceptance Criteria**:
  - 100% of produced chunks strictly fall within the token limit (e.g. <= 500 tokens).
  - Overlap tokens accurately bridge adjacent chunks without duplicating section headers.
- **Required Tests**:
  - `tests/test_token_chunker.py`: Assert all output chunks are under token ceiling; verify overlap content matches between chunk $N$ tail and chunk $N+1$ head.

---

### Module #15: Corpus Preparation & Ingestion Validation
- **Status**: `Not Started`
- **Objective**: Build the end-to-end ingestion pipeline runner, ingest a sample public health guidance corpus, perform comprehensive audit checks, and export validated chunks for Member 2.
- **Dependencies**: Modules #10, #11, #12, #13, #14.
- **Expected Files**:
  - `src/ingestion/pipeline.py`
  - `outputs/ingested_chunks.jsonl`
  - `outputs/ingestion_validation_report.json`
  - `tests/test_ingestion_pipeline.py`
- **Implementation Subtasks**:
  1. [ ] Curate 3–5 realistic sample public health advisories in `data/sample_advisories/` (e.g., WHO Rabies PEP, COVID-19 Vaccine Storage, Cholera Outbreak Guidelines).
  2. [ ] Implement `run_ingestion_pipeline(input_dir, output_file)` coordinating loader -> cleaner -> chunker -> metadata -> export.
  3. [ ] Implement validation audit checks:
     - Zero empty or whitespace-only chunks.
     - Zero missing metadata fields.
     - All chunks <= token budget.
     - Accurate total chunk counts per document.
  4. [ ] Export clean JSONL dataset ready for Member 2's vector ingestion.
  5. [ ] Write audit summary to `outputs/ingestion_validation_report.json`.
- **Acceptance Criteria**:
  - Full pipeline runs end-to-end on sample corpus with exit code 0.
  - Outputs valid `ingested_chunks.jsonl` matching data contract.
  - Validation report confirms 0 errors, 0 empty chunks, 100% metadata completeness.
- **Required Tests**:
  - `tests/test_ingestion_pipeline.py`: End-to-end integration test validating pipeline output and audit metrics.
