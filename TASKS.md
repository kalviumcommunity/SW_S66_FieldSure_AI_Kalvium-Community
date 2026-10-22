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
| **#4** | Prompt Construction & System/User Roles | `Completed` | `prompts/`, `src/prompt_comparison.py`, `outputs/prompt_comparison_sample.log` |
| **#5** | Tokens, Tokenization & Cost Estimation | `Completed` | `src/token_cost_estimator.py`, `outputs/token_cost_estimation_sample.log` |
| **#6** | Context Windows & Message History Management | `Completed` | `src/context_manager.py`, `outputs/context_window_management_sample.log` |
| **#10** | Document Loading & Multi-Format Intake | `Completed` | `src/ingestion/loader.py`, `tests/test_loader.py`, `outputs/document_loading_sample.log` |
| **#11** | Text Extraction & Cleaning Pipeline | `Completed` | `src/ingestion/cleaner.py`, `tests/test_cleaner.py`, `outputs/text_cleaning_sample.log` |
| **#12** | Document Chunking Strategies | `Completed` | `src/ingestion/chunker.py`, `tests/test_chunker.py`, `outputs/sample_chunks.json`, `outputs/chunking_strategy_comparison.log` |
| **#13** | Chunk Metadata & Source Tracking | `Completed` | `src/ingestion/metadata.py`, `tests/test_metadata.py`, `outputs/tagged_chunks_sample.json`, `outputs/source_tracing_sample.log` |
| **#14** | Token-Aware Chunk Sizing & Overlap | `Completed` | `src/ingestion/chunker.py`, `tests/test_token_chunker.py`, `outputs/token_chunks_sample.json`, `outputs/token_chunking_sample.log` |
| **#15** | Corpus Preparation & Ingestion Validation | `Completed` | `src/ingestion/pipeline.py`, `tests/test_ingestion_pipeline.py`, `outputs/ingested_chunks.jsonl`, `outputs/ingestion_validation_report.json`, `outputs/ingestion_summary.log` |

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
- **Status**: `Completed`
- **Objective**: Design role-based prompt templates enforcing strict grounding, inline citations, and deterministic safe refusals for public health questions; build comparative test runner.
- **Dependencies**: Module #3.
- **Expected Files**:
  - `prompts/system_prompt_baseline.txt`
  - `prompts/system_prompt_constrained.txt`
  - `prompts/README.md`
  - `src/prompt_comparison.py`
  - `outputs/prompt_comparison_sample.log`
  - `tests/test_prompts.py`
- **Implementation Subtasks**:
  1. [x] Author `system_prompt_baseline.txt` (standard, unconstrained assistant prompt).
  2. [x] Author `system_prompt_constrained.txt` (mandating strict grounding, inline citations `[Doc, Section]`, safe refusal on ungrounded questions, non-diagnostic disclaimer).
  3. [x] Implement `src/prompt_comparison.py` sending test questions across both prompts.
  4. [x] Capture payloads, responses, token counts, and refusal triggers cleanly in `outputs/prompt_comparison_sample.log`.
- **Acceptance Criteria**:
  - Comparative execution clearly demonstrates baseline hallucination/extrapolation vs. constrained grounded refusal on out-of-domain queries.
  - Log records tokens and latency without exposing secret keys.
- **Required Tests**:
  - `tests/test_prompts.py`: Validate prompt template loading, variable formatting, and message structure.
- **Verification Evidence**:
  - `prompts/README.md` documents side-by-side prompt analysis and architectural rationale for the chosen constrained prompt.
  - Automated tests in `tests/test_prompts.py` validate template presence, mandatory safety sections, and mocked comparative runner execution.
  - Sample comparative execution audit logged in `outputs/prompt_comparison_sample.log`.

---

### Module #5: Tokens, Tokenization & Cost Estimation
- **Status**: `Completed`
- **Objective**: Measure token consumption with subword tokenizer (tiktoken), establish differential input/output cost models, and demonstrate length-token relationships across diverse content types.
- **Dependencies**: Module #1.
- **Expected Files**:
  - `src/token_cost_estimator.py`
  - `outputs/token_cost_estimation_sample.log`
  - `tests/test_token_cost.py`
- **Implementation Subtasks**:
  1. [x] Implement subword token counting using `tiktoken` (`cl100k_base`).
  2. [x] Report token metrics across 3 project corpus samples (short question, clinical paragraph, full protocol document).
  3. [x] Calculate cost estimates across benchmark models (gpt-4o-mini, gpt-4o, claude-3-5-sonnet) with differential input/output rates and 10k monthly query scaling.
  4. [x] Profile text length vs token relationship across English prose, medical terminology, structured JSON, and multilingual Hindi text.
  5. [x] Capture structured analysis in `outputs/token_cost_estimation_sample.log`.
- **Acceptance Criteria**:
  - `python src/token_cost_estimator.py` runs and outputs clean terminal reports for all 5 tasks.
  - Audit record saved to `outputs/token_cost_estimation_sample.log`.
- **Required Tests**:
  - `tests/test_token_cost.py`: Unit tests validating counting, monotonic sample scaling, pricing arithmetic, and domain disproportionality.
- **Verification Evidence**:
  - Automated tests passing in `tests/test_token_cost.py`.
  - Sample analysis audit captured in `outputs/token_cost_estimation_sample.log`.

---

### Module #6: Context Windows & Message History Management
- **Status**: `Completed`
- **Objective**: Manage multi-turn conversation history, compute pre-request token budgets, and apply sliding-window trimming and turn summarization to enforce token budgets while perpetually preserving the system message.
- **Dependencies**: Modules #1, #3, #5.
- **Expected Files**:
  - `src/context_manager.py`
  - `outputs/context_window_management_sample.log`
  - `tests/test_context_manager.py`
- **Implementation Subtasks**:
  1. [x] Maintain multi-turn history structure tracking system prompt and alternating user/assistant messages.
  2. [x] Compute total message history tokens prior to each LLM request including chat framing overhead.
  3. [x] Implement token budget enforcement that trims older conversation turns while perpetually preserving the system prompt and most recent turn.
  4. [x] Demonstrate token budget management over an overflowing multi-turn clinical consultation (6 turns).
  5. [x] Record structured execution audit to `outputs/context_window_management_sample.log`.
- **Acceptance Criteria**:
  - `python src/context_manager.py` executes successfully and prints step-by-step token and trimming audit.
  - System prompt at index 0 is never pruned under any overflow scenario.
  - Final token count is strictly constrained within the configured token ceiling.
- **Required Tests**:
  - `tests/test_context_manager.py`: Unit tests validating multi-turn history, pre-request budgeting, system message preservation, recent turn retention, and overflowing trimming.
- **Verification Evidence**:
  - Automated tests passing in `tests/test_context_manager.py`.
  - Step-by-step trimming audit logged in `outputs/context_window_management_sample.log`.

---

### Module #10: Document Loading & Multi-Format Intake
- **Status**: `Completed`
- **Objective**: Build a robust document loader supporting PDF, DOCX, Markdown, and TXT files, providing verified file streams and handling corrupted or missing files gracefully.
- **Dependencies**: Module #1.
- **Expected Files**:
  - `src/ingestion/loader.py`
  - `data/sample_advisories/` (sample corpus across formats)
  - `outputs/document_loading_sample.log`
  - `tests/test_loader.py`
- **Implementation Subtasks**:
  1. [x] Implement file extension dispatcher (`.pdf`, `.docx`, `.md`, `.txt`, `.html`).
  2. [x] Integrate `pypdf` for page-wise text extraction.
  3. [x] Integrate `python-docx` for structured paragraph extraction.
  4. [x] Implement standard library readers for `.md`, `.txt`, and `.html`.
  5. [x] Raise `UnsupportedFileError` on unknown extensions and catch file I/O exceptions cleanly without crashing batch runs.
- **Acceptance Criteria**:
  - Successfully loads sample files in target formats into unified `LoadedDocument` representation.
  - Returns raw text and initial file metadata (`source_file`, `file_type`, `file_size`).
  - Corrupt, missing, or invalid files are skipped gracefully and recorded in an error ledger.
- **Required Tests**:
  - `tests/test_loader.py`: Test reading valid fixtures across formats; test error handling for missing/unsupported files.
- **Verification Evidence**:
  - Sample corpus established in `data/sample_advisories/` across Markdown, Plain Text, and HTML formats.
  - Automated tests passing in `tests/test_loader.py`.
  - Intake confirmation audit record saved in `outputs/document_loading_sample.log`.

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
- **Status**: `Completed`
- **Objective**: Implement structural and recursive document chunking that splits text at natural semantic boundaries (headings, paragraphs, sentences) rather than arbitrary character cuts.
- **Dependencies**: Module #11.
- **Expected Files**:
  - `src/ingestion/chunker.py`
  - `tests/test_chunker.py`
  - `outputs/sample_chunks.json`
  - `outputs/chunking_strategy_comparison.log`
- **Implementation Subtasks**:
  1. [x] Define recursive split hierarchy: `# ` headings -> `\n\n` paragraphs -> `\n` lines -> sentence boundaries.
  2. [x] Implement section tracking to identify the nearest enclosing heading.
  3. [x] Build recursive text splitter function.
  4. [x] Implement fixed-size with overlap and sentence-based chunking for comparative evaluation.
  5. [x] Calculate statistical metrics (chunk count, average characters and tokens, min/max).
  6. [x] Save sample chunks and comparison execution logs for reviewer inspection.
- **Acceptance Criteria**:
  - Splits documents into logical units without breaking mid-sentence or mid-definition.
  - Section headers are preserved and prepended to chunks to retain context.
- **Required Tests**:
  - `tests/test_chunker.py`: Test chunking sample text with markdown headers; verify sentences are not split mid-word.
- **Verification Evidence**:
  - Automated tests passing in `tests/test_chunker.py`.
  - Comparative strategy audit record saved in `outputs/chunking_strategy_comparison.log`.
  - Reviewer sample chunks with boundary inspection in `outputs/sample_chunks.json`.

---

### Module #13: Chunk Metadata & Source Tracking
- **Status**: `Completed`
- **Objective**: Attach immutable, traceable metadata to every chunk (`doc_id`, `chunk_id`, `source_file`, `title`, `version`, `date`, `section`, `chunk_index`, `token_count`).
- **Dependencies**: Modules #10, #12.
- **Expected Files**:
  - `src/ingestion/metadata.py`
  - `tests/test_metadata.py`
  - `outputs/tagged_chunks_sample.json`
  - `outputs/source_tracing_sample.log`
- **Implementation Subtasks**:
  1. [x] Implement `DocumentChunk` and `ChunkMetadata` dataclasses.
  2. [x] Generate deterministic hashes for `doc_id` (SHA-256) and `chunk_id` (`{doc_id}_c{index:04d}`).
  3. [x] Extract or infer `document_title`, `version`, and `date` from filename or file headers.
  4. [x] Attach section context to individual chunk metadata.
  5. [x] Implement source tracing engine (`trace_chunk_to_source`) linking retrieved chunks back to exact parent documents.
  6. [x] Standardize uniform 14-field schema contract across the entire public health corpus.
  7. [x] Implement metadata-based chunk filtering (`filter_chunks_by_metadata`) for targeted retrieval.
  8. [x] Implement automated completeness validation (`validate_chunk_metadata_completeness`).
- **Acceptance Criteria**:
  - Every chunk contains 100% of required metadata attributes.
  - Chunk IDs are deterministic and reproducible across runs.
- **Required Tests**:
  - `tests/test_metadata.py`: Verify metadata creation, field types, and deterministic ID generation.
- **Verification Evidence**:
  - Automated tests passing in `tests/test_metadata.py`.
  - Structured 14-field schema chunks sample saved in `outputs/tagged_chunks_sample.json`.
  - Lineage tracing audit log saved in `outputs/source_tracing_sample.log`.

---

### Module #14: Token-Aware Chunk Sizing & Overlap
- **Status**: `Completed`
- **Objective**: Enforce strict token budgets (e.g., target 500 tokens) using subword tokenization with configurable sliding overlap (e.g., 50 tokens) to prevent context boundary loss.
- **Dependencies**: Modules #12, #13.
- **Expected Files**:
  - `src/ingestion/chunker.py`
  - `tests/test_token_chunker.py`
  - `outputs/token_chunks_sample.json`
  - `outputs/token_chunking_sample.log`
- **Implementation Subtasks**:
  1. [x] Integrate `tiktoken` encoder (`cl100k_base`).
  2. [x] Implement token counting utility `count_tokens(text: str) -> int`.
  3. [x] Implement sliding window algorithm creating token-accurate chunks with defined overlap (`split_by_tokens`).
  4. [x] Add safety fallback for monolithic single tokens/words exceeding budget.
  5. [x] Implement boundary context preservation comparison (`compare_boundary_overlap`) showing zero-overlap vs controlled-overlap.
  6. [x] Implement mathematical overlap verification (`verify_chunk_overlap`).
- **Acceptance Criteria**:
  - 100% of produced chunks strictly fall within the token limit (e.g. <= 500 tokens).
  - Overlap tokens accurately bridge adjacent chunks without duplicating section headers.
- **Required Tests**:
  - `tests/test_token_chunker.py`: Assert all output chunks are under token ceiling; verify overlap content matches between chunk $N$ tail and chunk $N+1$ head.
- **Verification Evidence**:
  - Automated tests passing in `tests/test_token_chunker.py`.
  - Boundary context preservation demonstration saved in `outputs/token_chunks_sample.json`.
  - Token execution audit report logged in `outputs/token_chunking_sample.log`.

---

### Module #15: Corpus Preparation & Ingestion Validation
- **Status**: `Completed`
- **Objective**: Build the end-to-end ingestion pipeline runner, ingest a sample public health guidance corpus, perform comprehensive audit checks, and export validated chunks for Member 2.
- **Dependencies**: Modules #10, #11, #12, #13, #14.
- **Expected Files**:
  - `src/ingestion/pipeline.py`
  - `outputs/ingested_chunks.jsonl`
  - `outputs/ingestion_validation_report.json`
  - `outputs/ingestion_summary.log`
  - `tests/test_ingestion_pipeline.py`
- **Implementation Subtasks**:
  1. [x] Curate realistic sample public health advisories in `data/sample_advisories/` across Markdown, Plain Text, HTML, and unsupported formats.
  2. [x] Implement `run_ingestion_pipeline(input_dir, output_file)` coordinating loader -> cleaner -> chunker -> metadata -> export.
  3. [x] Implement validation audit checks:
     - Zero empty or whitespace-only chunks.
     - Zero missing metadata fields (100% 14-field contract compliance).
     - All chunks <= token budget.
     - Accurate total chunk counts per document.
     - Mathematical completeness reconciliation ($Discovered = Ingested + Failures$).
  4. [x] Export clean JSONL dataset ready for Member 2's vector ingestion (`outputs/ingested_chunks.jsonl`).
  5. [x] Write audit summary to `outputs/ingestion_validation_report.json` and `outputs/ingestion_summary.log`.
- **Acceptance Criteria**:
  - Full pipeline runs end-to-end on sample corpus with exit code 0.
  - Outputs valid `ingested_chunks.jsonl` matching data contract.
  - Validation report confirms 0 errors, 0 empty chunks, 100% metadata completeness.
- **Required Tests**:
  - `tests/test_ingestion_pipeline.py`: End-to-end integration test validating pipeline output, reconciliation math, and audit metrics.
- **Verification Evidence**:
  - Automated tests passing in `tests/test_ingestion_pipeline.py`.
  - Exported JSONL chunk dataset saved in `outputs/ingested_chunks.jsonl`.
  - Completeness validation report saved in `outputs/ingestion_validation_report.json`.
  - Ingestion run dashboard log recorded in `outputs/ingestion_summary.log`.
