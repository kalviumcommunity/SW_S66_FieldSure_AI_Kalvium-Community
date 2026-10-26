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
| **#21** | Vector Database Setup & Collection Design | `Completed` | `src/vector_store.py`, `src/vector_db_demo.py`, `tests/test_vector_store.py`, `outputs/vector_db_readback_sample.md` |
| **#22** | Indexing Embeddings & Metadata Storage | `Completed` | `src/index_embeddings.py`, `tests/test_index_embeddings.py`, `outputs/indexing_summary_sample.md` |
| **#23** | Similarity Search & Top-K Retrieval | `Completed` | `src/retrieve_chunks.py`, `tests/test_retrieve_chunks.py`, `outputs/top_k_retrieval_sample.md` |
| **#24** | Metadata Filtering & Hybrid Search | `Completed` | `src/filtered_retrieval.py`, `tests/test_filtered_retrieval.py`, `outputs/filtered_retrieval_sample.md` |
| **#25** | Retrieval Relevance Tuning | `Completed` | `src/tune_retrieval.py`, `tests/test_tune_retrieval.py`, `outputs/retrieval_tuning_summary.md`, `outputs/retrieval_tuning_report.json` |
| **#27** | Retrieval Evaluation & Recall Testing | `Completed` | `src/evaluate_retrieval.py`, `tests/test_evaluate_retrieval.py`, `outputs/retrieval_evaluation_report.json`, `outputs/retrieval_evaluation_summary.md` |
| **#28** | RAG Pipeline Architecture & Flow Design | `Completed` | `src/rag_pipeline.py`, `tests/test_rag_pipeline.py`, `outputs/rag_pipeline_sample.md` |
| **#38** | Context Injection & Prompt Augmentation | `Completed` | `src/context_injection.py`, `tests/test_context_injection.py`, `outputs/context_injection_sample.md` |
| **#39** | Grounded Answer Generation | `Completed` | `src/grounded_generation.py`, `tests/test_grounded_generation.py`, `outputs/grounded_generation_sample.md` |
| **#40** | Source Citation & Attribution | `Completed` | `src/source_citation.py`, `src/citation.py`, `tests/test_source_citation.py`, `outputs/source_citation_sample.md` |
| **#41** | Hallucination Guardrails & Refusal Handling | `Completed` | `src/guardrails.py`, `src/hallucination_guardrails.py`, `tests/test_guardrails.py`, `outputs/guardrails_sample.md` |
| **#42** | Conversational RAG & Follow-Up Context | `Completed` | `src/conversational_rag.py`, `src/conversation.py`, `tests/test_conversational_rag.py`, `outputs/conversational_rag_sample.md` |
| **#43** | RAG Evaluation & Answer Quality Scoring | `Completed` | `src/rag_evaluation.py`, `src/evaluate_rag.py`, `tests/test_rag_evaluation.py`, `outputs/rag_evaluation_test_set.json`, `outputs/rag_evaluation_report.json`, `outputs/rag_evaluation_summary.md`, `outputs/rag_evaluation_sample.md` |

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

---

### Module #16: Embeddings Fundamentals & Vector Representation
- **Status**: `Completed`
- **Objective**: Generate text embeddings for related and unrelated samples, verify vector dimensions, compare cosine similarity, and explain the meaning of embedding vectors.
- **Expected Files**:
  - `src/embedding_fundamentals.py`
  - `tests/test_embedding_fundamentals.py`
  - `outputs/embedding_fundamentals_sample.md`
- **Implementation Subtasks**:
  1. [x] Request embeddings for similar dengue-prevention texts and an unrelated library-book text.
  2. [x] Report vector dimensions and verify every returned vector has the same length.
  3. [x] Compare similar and unrelated pairs using cosine similarity and require the similar pair to score higher.
  4. [x] Explain that embeddings are numeric representations of meaning, not random IDs or keyword counts.
  5. [x] Save the demo output, vector preview, similarity scores, and explanation.
- **Acceptance Criteria**:
  - The demo uses the configured OpenAI-compatible embeddings API and handles malformed responses explicitly.
  - The output reports consistent dimensions and the similar pair scores higher than the unrelated pair.
- **Required Tests**:
  - `tests/test_embedding_fundamentals.py`: Verify API request and response handling, vector validation, cosine similarity, and demo output.
- **Verification Evidence**:
  - Automated tests passing in `tests/test_embedding_fundamentals.py`.
  - Captured output and vector representation explanation saved in `outputs/embedding_fundamentals_sample.md`.

---

### Module #17: Generating Embeddings via API
- **Status**: `Completed`
- **Objective**: Embed a small prepared corpus through the configured OpenAI-compatible API and store each vector with its source text and metadata.
- **Expected Files**:
  - `src/embed_corpus.py`
  - `tests/test_embed_corpus.py`
  - `outputs/embedding_api_sample.md`
- **Implementation Subtasks**:
  1. [x] Read prepared chunks from the ingestion JSONL output and request embeddings through the API.
  2. [x] Preserve source chunk text and metadata alongside each generated vector.
  3. [x] Read the API key, optional base URL, and embedding model from environment configuration.
  4. [x] Report embedded chunk count, vector length, and sample vector values.
  5. [x] Save a clearly labeled sample output showing chunk content, source metadata, vector length, and trimmed vector values.
- **Acceptance Criteria**:
  - Every stored vector corresponds to its source chunk and preserves retrieval metadata.
  - API failures and invalid corpus records are surfaced with actionable errors.
  - No API secrets or embedding model settings are hardcoded in the script.
- **Required Tests**:
  - `tests/test_embed_corpus.py`: Verify corpus validation, metadata preservation, output serialization, environment configuration, and verification output.
- **Verification Evidence**:
  - Focused automated tests pass.
  - Sample corpus output is saved in `outputs/embedding_api_sample.md` and identifies mocked vectors as illustrative.

---

### Module #18: Embedding Similarity & Distance Metrics
- **Status**: `Completed`
- **Objective**: Embed a query, compare it with stored corpus vectors using cosine similarity, and rank chunks from most to least related.
- **Expected Files**:
  - `src/rank_chunks.py`
  - `tests/test_rank_chunks.py`
  - `outputs/embedding_similarity_sample.md`
- **Implementation Subtasks**:
  1. [x] Compute cosine similarity using the shared embedding fundamentals utility.
  2. [x] Embed a user query and compare it against stored sample corpus chunk embeddings.
  3. [x] Rank matching chunks and report the most and least similar results with source text and metadata.
  4. [x] Explain that cosine similarity compares vector direction, independent of magnitude.
  5. [x] Save a sample ranking with scores and retrieval metadata.
- **Acceptance Criteria**:
  - All compared vectors have equal dimensions and valid finite values.
  - Ranking is in descending score order, with query/API and stored chunk data validated.
  - Sample output labels mocked scores clearly and identifies the most and least related chunks.
- **Required Tests**:
  - `tests/test_rank_chunks.py`: Verify ranking order, malformed corpus handling, query embedding use, and reported metadata.
- **Verification Evidence**:
  - Focused automated tests pass.
  - Sample ranking is saved in `outputs/embedding_similarity_sample.md`.

---

### Module #20: Embedding Quality Checks & Sanity Tests
- **Status**: `Completed`
- **Objective**: Check known query-to-source relevance using cosine ranking, report pass/fail and borderline cases, and highlight a surprising near-tie.
- **Expected Files**:
  - `src/embedding_quality.py`
  - `tests/test_embedding_quality.py`
  - `outputs/embedding_quality_sanity_report.json`
- **Implementation Subtasks**:
  1. [x] Define known dengue, cholera, and malaria query-to-chunk relevance pairs.
  2. [x] Confirm expected relevant chunks rank above unrelated chunks.
  3. [x] Identify and explain close-score borderline cases and report actual misrankings as failures.
  4. [x] Summarize counts, expected/top-ranked sources, scores, margins, and notes.
  5. [x] Save a reproducible sample sanity report with clearly identified illustrative vectors.
- **Acceptance Criteria**:
  - Each known target chunk is present and compared against the stored corpus vectors.
  - A target that is not ranked first is recorded as a failure; a small lead is flagged as borderline.
  - API-generated report is saved as JSON and failures produce a non-zero exit status.
- **Required Tests**:
  - `tests/test_embedding_quality.py`: Verify known relevance, failures, borderline detection, report output, and environment-configured API calls.
- **Verification Evidence**:
  - Focused automated tests pass.
  - Sample sanity results are saved in `outputs/embedding_quality_sanity_report.json`.

---

### Module #21: Vector Database Setup & Collection Design
- **Status**: `Completed`
- **Objective**: Configure a persistent local ChromaDB collection for embedded chunks, preserve source text and retrieval metadata, and verify an inserted record can be read back.
- **Expected Files**:
  - `src/vector_store.py`
  - `src/vector_db_demo.py`
  - `tests/test_vector_store.py`
  - `tests/test_vector_db_demo.py`
  - `outputs/vector_db_readback_sample.md`
- **Implementation Subtasks**:
  1. [x] Configure a local persistent ChromaDB client from `VECTOR_DB_PATH` and collection name from `VECTOR_DB_COLLECTION`.
  2. [x] Create a cosine-similarity collection; infer and record its vector dimension on first insertion, then reject dimension mismatches.
  3. [x] Store chunk IDs, embeddings, source text, and JSON-preserved retrieval metadata.
  4. [x] Insert and read back a test record, verifying its ID, vector length, source, and metadata.
  5. [x] Save the verified readback output and usage/configuration documentation.
- **Acceptance Criteria**:
  - Local persistence is reachable without a separately running database server.
  - All vectors in a collection have a consistent dimension; the first insertion establishes that dimension.
  - Retrieved test records preserve their IDs, source text, metadata, and vector length.
  - The smoke-test fixture is clearly distinguished from a live model embedding.
- **Required Tests**:
  - `tests/test_vector_store.py`: Verify persistence, readback, metadata preservation, and dimension validation.
  - `tests/test_vector_db_demo.py`: Verify user-visible readback output and configuration behavior.
- **Verification Evidence**:
  - Focused test run: 7 tests passed.
  - The local persistence smoke test reported a readable record ID, 8-value vector, source text, and original metadata. The vector is an illustrative fixture, not a model embedding.
  - Sample output is saved in `outputs/vector_db_readback_sample.md`.

---

### Module #22: Indexing Embeddings & Metadata Storage
- **Status**: `Completed`
- **Objective**: Insert every prepared corpus embedding into ChromaDB, compare the indexed count with the source corpus, and verify stored IDs, vectors, text, and metadata.
- **Expected Files**:
  - `src/index_embeddings.py`
  - `tests/test_index_embeddings.py`
  - `outputs/indexing_summary_sample.md`
- **Dependencies**: Module #21 (Vector Database Setup & Collection Design).
- **Implementation Subtasks**:
  1. [x] Load and index every record from the prepared embedded-corpus JSONL file.
  2. [x] Persist vectors, source text, and retrieval metadata in the configured collection.
  3. [x] Compare collection count and embedded chunk IDs against the original source corpus.
  4. [x] Read back first and last records and verify ID, vector, text, and metadata.
  5. [x] Save count, failures, and spot-check evidence in the committed summary.
- **Acceptance Criteria**:
  - The indexer processes the complete embedded corpus without a chunk limit.
  - Indexed record count matches the source corpus count for a complete input; failures and missing embeddings are reported.
  - Spot-checks verify stored vector values and original chunk content/metadata.
- **Required Tests**:
  - `tests/test_index_embeddings.py`: Verify all-record indexing, count comparison, spot-check integrity, idempotent upsert, and failure reporting.
- **Verification Evidence**:
  - Focused tests: 11 passed across indexing and vector-store coverage.
  - Full-corpus sample indexed 11 of 11 source chunks; count validation passed and both readback checks preserved vector values, IDs, source text, and metadata.
  - Vectors used for the sample were explicitly labeled illustrative fixtures, not live model embeddings.
  - Summary evidence is saved in `outputs/indexing_summary_sample.md`.

---

### Module #23: Similarity Search & Top-K Retrieval
- **Status**: `Completed`
- **Objective**: Embed a user query with the corpus embedding model, perform ChromaDB top-k similarity search, and return ranked chunks with scores, text, and metadata.
- **Dependencies**: Module #22 (Indexing Embeddings & Metadata Storage).
- **Expected Files**:
  - `src/retrieve_chunks.py`
  - `tests/test_retrieve_chunks.py`
  - `outputs/top_k_retrieval_sample.md`
- **Implementation Subtasks**:
  1. [x] Embed a query with the model configured by `EMBEDDING_MODEL`.
  2. [x] Search the configured ChromaDB collection using cosine distance.
  3. [x] Return ranked similarity scores, source text, and original metadata.
  4. [x] Run the sample query at two different k values and show how the result set changes.
  5. [x] Save sample results and document production usage.
- **Acceptance Criteria**:
  - Query vectors use the configured same model as indexed chunk embeddings.
  - Top-k results are ordered by similarity and contain scores, source text, and metadata.
  - Invalid k values, missing model configuration, empty collections, and embedding/search failures are explicit.
- **Required Tests**:
  - `tests/test_retrieve_chunks.py`: Verify score order, result metadata, k behavior, query embedding/model use, and invalid input handling.
- **Verification Evidence**:
  - Focused test run: 23 tests passed across retrieval and vector-store coverage.
  - Sample ChromaDB query ran with `k=2` and `k=5`; rankings, scores, and source metadata were printed and saved in `outputs/top_k_retrieval_sample.md`.
  - As no live embeddings file or API credentials were available, the committed sample clearly labels its query/document vectors as illustrative fixtures.

---

### Module #24: Metadata Filtering & Hybrid Search
- **Status**: `Completed`
- **Objective**: Compare filtered and unfiltered vector search and optionally rerank filtered results with exact keyword coverage.
- **Dependencies**: Module #23 (Similarity Search & Top-K Retrieval).
- **Expected Files**:
  - `src/filtered_retrieval.py`
  - `tests/test_filtered_retrieval.py`
  - `outputs/filtered_retrieval_sample.md`
- **Implementation Subtasks**:
  1. [x] Add ChromaDB metadata filtering for retrieval.
  2. [x] Compare unfiltered and source-filtered results for the same embedded query.
  3. [x] Add optional hybrid reranking with exact query-term matching.
  4. [x] Demonstrate improved precision on a dengue-source example.
  5. [x] Save filtered, unfiltered, and hybrid results with scores and metadata.
- **Acceptance Criteria**:
  - Filtered and unfiltered queries share the same query embedding and vector collection.
  - Filtered matches meet the requested metadata predicate and retain text and metadata.
  - Hybrid scoring reports vector and keyword components alongside the combined score.
  - Sample results explicitly distinguish illustrative test vectors from live embeddings.
- **Required Tests**:
  - `tests/test_filtered_retrieval.py`: Verify metadata filtering, query embedding reuse, hybrid keyword ranking, precision comparison, and validation.
- **Verification Evidence**:
  - Focused tests: 30 passed across filtering, query retrieval, and vector storage.
  - Illustrative sample ChromaDB run improved precision@2 from 0/2 unfiltered to 2/2 after applying the source filter; filtered hybrid results include vector and keyword component scores.
  - The sample uses explicit fixture vectors and a mocked query embedding, not live API output.

---

### Module #25: Retrieval Relevance Tuning
- **Status**: `Completed`
- **Objective**: Compare retrieval settings against known query-to-source relevance judgments and select the best measured configuration.
- **Dependencies**: Module #24 (Metadata Filtering & Hybrid Search).
- **Expected Files**:
  - `src/tune_retrieval.py`
  - `tests/test_tune_retrieval.py`
  - `outputs/retrieval_tuning_summary.md`
  - `outputs/retrieval_tuning_report.json`
- **Implementation Subtasks**:
  1. [x] Define dengue, cholera, and malaria queries with expected source documents.
  2. [x] Compare unfiltered vector `k=2` with expected-source filtering, `k=5`, and a `0.5` score threshold.
  3. [x] Report precision@k, recall@k, F1, top-1 hit, and hit@k.
  4. [x] Select the setting with the highest macro F1@k and explain the result.
  5. [x] Save query-level rankings, text, metadata, metrics, and the recommended setting.
- **Acceptance Criteria**:
  - Each evaluation query has a known expected source and relevant chunk set.
  - Settings are compared using consistent relevance measures.
  - Selected settings are justified by measured experiment results.
  - Illustrative-vector limitations are disclosed where live embeddings are unavailable.
- **Required Tests**:
  - `tests/test_tune_retrieval.py`: Verify expected-source labeling, ChromaDB experiment metrics, setting choice, and failure handling.
- **Verification Evidence**:
  - Focused tuning and retrieval regression suite: 39 tests passed.
  - On three sample queries, the baseline achieved mean precision@2 `0.500`, mean recall@2 `0.278`, mean F1@2 `0.356`, and top-1 hit `0.000`.
  - Expected-source filter + `k=5` + threshold `0.5` achieved precision, recall, F1, and top-1 hit `1.000`.
  - The reproducible sample uses deterministic fixtures, not live model embeddings; detailed results are saved in the JSON report and summary.

---

### Module #27: Retrieval Evaluation & Recall Testing
- **Status**: `Completed`
- **Objective**: Evaluate retrieval against labeled relevant chunk IDs and inspect misses using precision@k and recall@k.
- **Dependencies**: Module #25 (Retrieval Relevance Tuning).
- **Expected Files**:
  - `src/evaluate_retrieval.py`
  - `tests/test_evaluate_retrieval.py`
  - `outputs/retrieval_evaluation_queries.json`
  - `outputs/retrieval_evaluation_report.json`
  - `outputs/retrieval_evaluation_summary.md`
- **Implementation Subtasks**:
  1. [x] Label representative dengue, cholera, and malaria queries with relevant source chunk IDs.
  2. [x] Compute per-query and aggregate precision@k, recall@k, hit@k, and top-1 hit rate.
  3. [x] Report retrieved scores, source text, metadata, and missing or low-ranked relevant chunks.
  4. [x] Analyze a deliberate top-1 miss and document the limitations of illustrative fixture vectors.
  5. [x] Save a reproducible JSON report and readable interpretation.
- **Acceptance Criteria**:
  - Every evaluation query identifies one or more known relevant chunk IDs.
  - Query vectors use the configured embedding model for live evaluation.
  - The report distinguishes live evaluation from deterministic fixture output.
  - Evaluation failures include query-level evidence and likely causes.
- **Required Tests**:
  - `tests/test_evaluate_retrieval.py`: Verify label validation, precision/recall aggregation, missing-ID diagnostics, top-1 misses, live query embedding, and the offline fixture run.
- **Verification Evidence**:
  - Focused evaluation tests pass; the fixture report is generated without API credentials.
  - The committed fixture report has three labeled queries, full recall@3, precision@3 of `0.333`, and top-1 hit rate `0.000`.
  - Fixture scores validate the report and failure-analysis path only; they are not production embedding-quality results.

---

### Module #28: RAG Pipeline Architecture & Flow Design
- **Status**: `Completed`
- **Objective**: Design and implement the end-to-end RAG pipeline connecting user query, embedding generation, vector retrieval, context assembly, and grounded answer generation into clean, modular, testable stages.
- **Dependencies**: Module #23 (Similarity Search & Top-K Retrieval), Module #25 (Retrieval Relevance Tuning), Module #4 (Prompt Construction).
- **Expected Files**:
  - `src/rag_pipeline.py`
  - `tests/test_rag_pipeline.py`
  - `outputs/rag_pipeline_sample.md`
- **Implementation Subtasks**:
  1. [x] Document the full query-to-answer RAG flow including query embedding, retrieval, context assembly, grounded generation, and returned source metadata.
  2. [x] Represent each stage in code via dedicated, single-responsibility functions: `embed_query`, `retrieve_context`, `assemble_context`, and `generate_answer`.
  3. [x] Implement the pipeline orchestrator `answer_query` connecting the stages with graceful handling of empty retrieval states without hallucination.
  4. [x] Run the pipeline end to end on sample domain queries and demonstrate the generated answer plus retrieved sources.
  5. [x] Verify stage separation and testability through unit tests isolating each stage.
  6. [x] Commit pipeline structure, sequence diagram, stage responsibility table, and sample run output to `outputs/rag_pipeline_sample.md`.
- **Acceptance Criteria**:
  - Each RAG stage is represented as an independent, single-responsibility function.
  - Pipeline successfully executes end-to-end and returns `{"query": ..., "answer": ..., "sources": ..., "context": ..., "chunks": ...}`.
  - Empty retrieval deterministically returns fallback message (`"I could not find relevant context for that question."`) and empty sources list without hallucination or unnecessary LLM calls.
  - Functions can be tested in isolation (retrieval without generation, assembly without database).
- **Required Tests**:
  - `tests/test_rag_pipeline.py`: Comprehensive test suite verifying input validation, stage isolation, top-k retrieval, context formatting, LLM prompt framing, empty retrieval fallback, and CLI execution.
- **Verification Evidence**:
  - 19 automated tests passing in `tests/test_rag_pipeline.py`; 197 total test suite passing.
  - Sample query execution audit and sequence flow diagram captured in `outputs/rag_pipeline_sample.md`.

---

### Module #38: Context Injection & Prompt Augmentation
- **Status**: `Completed`
- **Objective**: Turn raw retrieved chunks into a grounded, citation-rich prompt that stays within the model's token budget and explicitly instructs the LLM to answer only from the provided context.
- **Dependencies**: Module #28 (RAG Pipeline Architecture & Flow Design), Module #5 (Tokens, Tokenization & Cost Estimation).
- **Expected Files**:
  - `src/context_injection.py`
  - `tests/test_context_injection.py`
  - `outputs/context_injection_sample.md`
- **Implementation Subtasks**:
  1. [x] Implement `format_chunk` adding clear source markers (`[1]`, `[2]`), source file, section, and document ID.
  2. [x] Implement `assemble_context` enforcing token budgeting with `count_tokens` to prevent context window overflow.
  3. [x] Implement `build_augmented_prompt` combining system grounding instructions, numbered context blocks, and user query.
  4. [x] Return structured prompt data including raw prompt, context block, token accounting, and source metadata.
  5. [x] Handle edge cases gracefully: empty chunks return fallback notice, oversized chunks are budget-capped.
  6. [x] Author comprehensive test suite in `tests/test_context_injection.py`.
  7. [x] Commit execution report and prompt inspection sample to `outputs/context_injection_sample.md`.
- **Acceptance Criteria**:
  - Formatted chunks include provenance header with source file, section, and chunk index.
  - Token budget is strictly enforced; chunks exceeding budget are omitted with warning log.
  - Augmented prompt clearly conveys grounding constraints and source attribution rules.
  - Returns `AugmentedPrompt` dataclass with prompt string, context text, token counts, and source list.
- **Required Tests**:
  - `tests/test_context_injection.py`: 44 automated tests covering chunk formatting, budget enforcement, prompt assembly, and edge cases.
- **Verification Evidence**:
  - 44 tests passing in `tests/test_context_injection.py`.
  - Sample report verified at `outputs/context_injection_sample.md`.

---

### Module #39: Grounded Answer Generation
- **Status**: `Completed`
- **Objective**: Generate answers strictly grounded in retrieved context, check that answers accurately reflect source material, return fallback when context is missing, and compare grounded vs ungrounded answers.
- **Dependencies**: Module #38 (Context Injection & Prompt Augmentation), Module #28 (RAG Pipeline Architecture & Flow Design).
- **Expected Files**:
  - `src/grounded_generation.py`
  - `tests/test_grounded_generation.py`
  - `outputs/grounded_generation_sample.md`
- **Implementation Subtasks**:
  1. [x] Implement `generate_grounded_answer(question, retrieved_chunks)` using `build_augmented_prompt` with explicit grounding rules.
  2. [x] Implement `verify_grounding(answer, context)` checking keyword overlap and sentence-level context support to detect unsupported claims.
  3. [x] Implement `answer_query(question, retrieved_chunks)` returning explicit fallback (`"I don't have enough information in the provided context."`) when chunks are missing or empty.
  4. [x] Implement `compare_grounded_vs_ungrounded(question, retrieved_chunks)` comparing answers generated with vs without retrieval.
  5. [x] Implement `print_grounding_check(result)` formatting grounding metrics, source citations, and supported sentence counts.
  6. [x] Support dictionary-style subscripting (`result["answer"]`) and `.to_dict()` on output dataclasses.
  7. [x] Implement CLI runner generating verification evidence and sample report at `outputs/grounded_generation_sample.md`.
  8. [x] Author comprehensive unit and integration test suite in `tests/test_grounded_generation.py`.
- **Acceptance Criteria**:
  - Answers are strictly generated from injected context; unsupported questions trigger clean fallback without hallucination.
  - Grounding verification accurately computes keyword overlap score and flags unsupported sentences.
  - Side-by-side comparison quantitatively demonstrates higher overlap with grounded RAG context vs parametric ungrounded responses.
  - All output structures support both dataclass attribute access and dictionary subscripting.
- **Required Tests**:
  - `tests/test_grounded_generation.py`: 52 automated tests covering context generation, grounding verification, fallback behaviour, ungrounded comparison, dictionary subscripting, and end-to-end integration.
- **Verification Evidence**:
  - 52 tests passing in `tests/test_grounded_generation.py`; 293 total test suite passing without regression.
  - Execution report and verified sample answers generated in `outputs/grounded_generation_sample.md`.

---

### Module #40: Source Citation & Attribution
- **Status**: `Completed`
- **Objective**: Connect answer claims back to retrieved source chunks using traceable metadata (document name, page, section, chunk ID), verify citations against original chunk text, avoid fabricated citations with deterministic fallback, and commit verification evidence.
- **Dependencies**: Module #39 (Grounded Answer Generation), Module #38 (Context Injection & Prompt Augmentation).
- **Expected Files**:
  - `src/source_citation.py`
  - `src/citation.py`
  - `tests/test_source_citation.py`
  - `outputs/source_citation_sample.md`
- **Implementation Subtasks**:
  1. [x] Implement `build_citation_map(chunks)` mapping citations `[1]`, `[2]` back to real documents and chunk locations (`source`, `chunk_id`, `chunk_index`, `section`, `text`, `page`).
  2. [x] Implement `assemble_context(chunks)` with dual string and tuple unpacking support (`context, _ = assemble_context(chunks)`).
  3. [x] Implement `build_cited_prompt(question, chunks)` instructing the model to cite only provided sources with markers and refuse when context is insufficient.
  4. [x] Implement `answer_with_citations(question)` returning user-facing response with answer and citation details dict.
  5. [x] Implement citation verification routines (`verify_citation`, `verify_all_citations`) confirming claims are supported by mapped source chunk text.
  6. [x] Implement fabricated citation detection (`detect_fabricated_citations`, `sanitize_answer_citations`) and no-source fallback returning `"I don't have enough information in the provided context."` with empty citations.
  7. [x] Support dictionary-style subscripting (`result["answer"]`, `result["citations"]`), attribute access, and plain dict conversion.
  8. [x] Author comprehensive unit and integration test suite in `tests/test_source_citation.py` (27 tests).
  9. [x] Generate execution verification report in `outputs/source_citation_sample.md`.
- **Acceptance Criteria**:
  - Generated answers include references to source chunks (`[1]`, `[2]`).
  - Each citation maps back to real document and location metadata.
  - Users can inspect and verify cited sources against original chunk text (`result["citations"]["[1]"]["text"]`).
  - Unsupported questions return safe fallback without fabricated citations.
  - Zero hardcoded secrets, zero credential leakage.
- **Required Tests**:
  - `tests/test_source_citation.py`: 27 automated tests covering citation mapping, prompt construction, answer generation, verification, fabrication detection, and dictionary compatibility.
- **Verification Evidence**:
  - 27 tests passing in `tests/test_source_citation.py`; 320 total test suite passing without regression.
  - Execution report and verified sample answers generated in `outputs/source_citation_sample.md`.

---

### Module #41: Hallucination Guardrails & Refusal Handling
- **Status**: `Completed`
- **Objective**: Detect weak, empty, or unrelated retrieval context before generation, return a safe refusal ("I don't have enough reliable context to answer that.") with status `refused_weak_context` when evidence is insufficient, apply relevance thresholds (`MIN_TOP_SCORE = 0.72`, `MIN_SUPPORTING_CHUNKS = 1`), preserve confident grounded answers (`answered`) when strong context exists, and commit sample refusal and answer outputs with execution evidence.
- **Dependencies**: Module #40 (Source Citation & Attribution), Module #39 (Grounded Answer Generation), Module #27 (Retrieval Evaluation & Recall Testing).
- **Expected Files**:
  - `src/guardrails.py`
  - `src/hallucination_guardrails.py`
  - `tests/test_guardrails.py`
  - `outputs/guardrails_sample.md`
- **Implementation Subtasks**:
  1. [x] Implement `retrieval_is_strong(chunks)` and `get_chunk_score(chunk)` detecting weak context using signals (empty chunks, low similarity score < 0.72, too few chunks).
  2. [x] Implement `evaluate_retrieval_quality(chunks)` providing comprehensive diagnostics (`is_strong`, `top_score`, `strong_chunks_count`, `reason`).
  3. [x] Implement `guarded_answer(question)` returning safe refusal (`"I don't have enough reliable context to answer that."`, status `refused_weak_context`, sources `[]`) when retrieval is weak.
  4. [x] Establish relevance thresholds (`MIN_TOP_SCORE = 0.72`, `MIN_SUPPORTING_CHUNKS = 1`) derived from retrieval evaluation tuning.
  5. [x] Preserve confident grounded answers (status `answered`, sources populated) when strong supporting context exists.
  6. [x] Support dictionary-style subscripting (`result["answer"]`, `result["status"]`, `result["sources"]`), attribute access, and dict unpacking (`{**result}`).
  7. [x] Author comprehensive unit and integration test suite in `tests/test_guardrails.py` (30 tests).
  8. [x] Generate execution verification report and sample outputs in `outputs/guardrails_sample.md`.
- **Acceptance Criteria**:
  - Detects weak or missing retrieval context using empty results, low similarity score (< 0.72), or too few chunks.
  - Returns safe refusal with empty sources and status `refused_weak_context` when context is insufficient.
  - Configurable relevance thresholds control refusal behavior.
  - Produces confident grounded answers with status `answered` when strong supporting context exists.
  - Zero hardcoded secrets, zero credential leakage.
- **Required Tests**:
  - `tests/test_guardrails.py`: 30 automated tests covering weak retrieval signals, threshold checks, safe refusals, confident answers, dict unpacking, and end-to-end report generation.
- **Verification Evidence**:
  - 30 tests passing in `tests/test_guardrails.py`; 350 total test suite passing without regression.
  - Execution report and verified sample answers generated in `outputs/guardrails_sample.md`.

---

### Module #42: Conversational RAG & Follow-Up Context
- **Status**: `Completed`
- **Objective**: Track conversation history across multiple turns, rewrite follow-up questions into standalone retrieval queries resolving conversational context and coreferences, retrieve relevant context using rewritten queries, demonstrate multi-turn dialogue with safe refusal guardrails on weak context, and commit sample dialogue evidence.
- **Dependencies**: Module #41 (Hallucination Guardrails & Refusal Handling), Module #40 (Source Citation & Attribution), Module #39 (Grounded Answer Generation).
- **Expected Files**:
  - `src/conversational_rag.py`
  - `src/conversation.py`
  - `tests/test_conversational_rag.py`
  - `outputs/conversational_rag_sample.md`
- **Implementation Subtasks**:
  1. [x] Implement conversation history tracking (`format_history_text`, `trim_history`) balancing turn and token limits across user and assistant turns.
  2. [x] Implement `rewrite_followup(history, question)` transforming dependent follow-ups into standalone retrieval queries while preserving search intent.
  3. [x] Implement `retrieve(question)` and `compare_retrieval(history, question)` showing naive retrieval disadvantages vs. rewritten query retrieval precision.
  4. [x] Implement `conversational_answer(history, user_question)` orchestrating query rewriting, retrieval, guardrail strength validation, grounded generation, and in-place history appending.
  5. [x] Implement `ConversationalAnswer` dict subclass supporting dictionary subscripting (`res["answer"]`, `res["rewritten_query"]`), attribute access, and dict unpacking.
  6. [x] Author comprehensive unit and integration test suite in `tests/test_conversational_rag.py` (27 tests).
  7. [x] Generate execution verification report and sample dialogues in `outputs/conversational_rag_sample.md`.
- **Acceptance Criteria**:
  - Conversation history tracks user and assistant turns without exceeding token ceilings.
  - Ambiguous follow-ups ("What about the video?", "Does it apply to Sprint 2?", "Can you explain that?") are resolved into standalone retrieval queries.
  - Rewritten queries retrieve relevant context chunks while naive retrieval returns unrelated chunks or weak relevance.
  - Safe refusal is maintained when retrieved context is weak or out of corpus.
  - Zero hardcoded secrets, zero credential leakage.
- **Required Tests**:
  - `tests/test_conversational_rag.py`: 27 automated tests covering history tracking, follow-up rewriting, comparative retrieval, conversational answer orchestration, dictionary compatibility, and end-to-end report generation.
- **Verification Evidence**:
  - 27 tests passing in `tests/test_conversational_rag.py`; 377 total test suite passing without regression.
  - Execution report and verified sample dialogues generated in `outputs/conversational_rag_sample.md`.

---

### Module #43: RAG Evaluation & Answer Quality Scoring
- **Status**: `Completed`
- **Objective**: Evaluate the end-to-end RAG system across correctness, grounding, and citation accuracy. Build a labeled test set containing questions, expected answer points, and expected sources; score answers for point coverage and retrieved context support; verify citation precision; summarize overall quality scores; diagnose notable failures with root-cause analysis (`weak_retrieval`, `missing_metadata`, `bad_citations`, `unsupported_claims`, `poor_prompt_instructions`); and commit evaluation scripts, test sets, scored reports, and summaries.
- **Dependencies**: Module #42 (Conversational RAG & Follow-Up Context), Module #41 (Hallucination Guardrails & Refusal Handling), Module #40 (Source Citation & Attribution), Module #27 (Retrieval Evaluation & Recall Testing).
- **Expected Files**:
  - `src/rag_evaluation.py`
  - `src/evaluate_rag.py`
  - `tests/test_rag_evaluation.py`
  - `outputs/rag_evaluation_test_set.json`
  - `outputs/rag_evaluation_report.json`
  - `outputs/rag_evaluation_summary.md`
  - `outputs/rag_evaluation_sample.md`
- **Implementation Subtasks**:
  1. [x] Task 1 — Prepare test set (`DEFAULT_TEST_SET`, `test_set`) containing queries, expected answer points, expected sources, and refusal expectations.
  2. [x] Task 2 — Implement answer scoring dimensions: `judge_expected_points` for factual concept/refusal coverage and `judge_grounding` for claim support against retrieved context.
  3. [x] Task 3 — Implement `check_citations` validating citation precision against expected source documents and detecting unwanted/fabricated citations.
  4. [x] Task 4 — Implement `diagnose_failure` and `summarize_evaluations` computing average metrics across dimensions and classifying failure causes (`weak_retrieval`, `missing_metadata`, `bad_citations`, `unsupported_claims`, `poor_prompt_instructions`) with actionable recommendations.
  5. [x] Task 5 — Implement end-to-end pipeline runner `score_answer`, `evaluate_test_set`, `run_rag_evaluation`, and CLI runner with `--self-test` option.
  6. [x] Author comprehensive unit and integration test suite in `tests/test_rag_evaluation.py` (33 tests).
  7. [x] Export and commit evaluation artifacts in `outputs/rag_evaluation_test_set.json`, `outputs/rag_evaluation_report.json`, `outputs/rag_evaluation_summary.md`, and `outputs/rag_evaluation_sample.md`.
- **Acceptance Criteria**:
  - Test set includes questions with expected answer points and expected source references.
  - Scores answers across correctness, grounding, and citation accuracy.
  - Checks citation accuracy against retrieved source documents.
  - Summarizes overall quality and notable failures with automated root-cause diagnosis.
  - Evaluation results, test set, scored results, citation checks, and summaries committed.
  - Zero hardcoded secrets, zero credential leakage.
- **Required Tests**:
  - `tests/test_rag_evaluation.py`: 33 automated tests covering test set schema, correctness judging, grounding validation, citation accuracy, failure diagnostics, summary metrics, and end-to-end evaluation execution.
- **Verification Evidence**:
  - 33 tests passing in `tests/test_rag_evaluation.py`; 410 total test suite passing without regression.
  - Execution reports generated in `outputs/rag_evaluation_report.json`, `outputs/rag_evaluation_summary.md`, and `outputs/rag_evaluation_sample.md`.

---

### Module #44: Backend API for the RAG Service
- **Status**: `Completed`
- **Objective**: Expose the FieldSure AI RAG pipeline through a stable FastAPI backend API for frontend, chatbot UI, or service integration. Accept question queries via `POST /query`, execute the guarded RAG pipeline, return structured JSON responses with answers and source citations, validate request schemas and handle errors with appropriate HTTP status codes (400, 422, 500), load configuration dynamically from environment variables with zero hardcoded credentials, and commit sample requests and responses.
- **Dependencies**: Module #41 (Hallucination Guardrails & Refusal Handling), Module #40 (Source Citation & Attribution), Module #39 (Grounded Answer Generation).
- **Expected Files**:
  - `src/backend_api.py`
  - `src/api.py`
  - `tests/test_backend_api.py`
  - `outputs/backend_api_sample.json`
  - `outputs/backend_api_sample.md`
- **Implementation Subtasks**:
  1. [x] Task 1 — Create a query endpoint: Implement `POST /query` accepting user questions and returning answers with source citations, along with `GET /health`, `GET /config`, and root endpoints.
  2. [x] Task 2 — Return structured JSON: Implement Pydantic models (`QueryRequest`, `Source`, `QueryResponse`) returning structured JSON containing `answer`, `sources`, `status`, and `metadata`.
  3. [x] Task 3 — Validate input and handle errors: Enforce Pydantic schema constraints (3-1000 characters, whitespace validation), returning HTTP 422 for unprocessable entities, HTTP 400 for ValueErrors, and HTTP 500 for unexpected RAG pipeline failures.
  4. [x] Task 4 — Load config from environment: Implement `APIConfig` and `load_environment_config` loading `OPENAI_API_KEY`/`API_KEY`, `EMBEDDING_MODEL`, `VECTOR_DB_URL`, `COLLECTION_NAME`, `CHAT_MODEL`, `PORT`, and `HOST` from environment with masked credentials and zero secret leakage.
  5. [x] Task 5 — Commit sample request and response: Generate and commit verified request/response artifacts in `outputs/backend_api_sample.json` and `outputs/backend_api_sample.md`.
  6. [x] Author comprehensive automated unit and integration test suite in `tests/test_backend_api.py` (26 tests).
- **Acceptance Criteria**:
  - `POST /query` endpoint operational and conforms to the specified API contract.
  - Returns structured JSON containing `answer`, `sources` (`source`, `chunk_id`, `score`), `status`, and diagnostic `metadata`.
  - Input validation rejects too short (<3), too long (>1000), whitespace-only, and malformed requests with clear status codes.
  - Server catches pipeline exceptions and returns clean 400/500 responses without leaking internal stack traces.
  - Zero hardcoded secrets, model names, or database URLs; fully configurable via environment variables.
  - Sample request and response evidence committed.
- **Required Tests**:
  - `tests/test_backend_api.py`: 26 automated tests covering config loading, schema validation, query execution, refusal guardrails, dependency injection, HTTP status codes, and artifact integrity.
- **Verification Evidence**:
  - 26 tests passing in `tests/test_backend_api.py`; 436 total test suite passing without regression.
  - Verification artifacts generated in `outputs/backend_api_sample.json` and `outputs/backend_api_sample.md`.

---

### Module #45: Document Upload & Indexing Endpoint
- **Status**: `Completed`
- **Objective**: Expand the FieldSure AI RAG knowledge base at runtime without restarting the application. Expose a document upload endpoint (`POST /documents`) that accepts a document file (`.txt`, `.md`, `.pdf`), validates file formats, sanitizes file paths to prevent directory traversal, stores it safely, ingests and cleans the text, segments it into token-budgeted chunks with structural overlap, tags chunks with immutable citation metadata, embeds chunks into dense vectors, and indexes them into the vector database and runtime index. Immediately enable `POST /query` to retrieve and answer questions grounded in the newly uploaded document without application restart.
- **Dependencies**: Module #44 (Backend API for RAG Service), Module #41 (Hallucination Guardrails & Refusal Handling), Module #40 (Source Citation & Attribution), Module #15 (Corpus Preparation & Ingestion Pipeline).
- **Expected Files**:
  - `src/document_upload.py`
  - `src/backend_api.py`
  - `src/api.py`
  - `tests/test_document_upload.py`
  - `outputs/document_upload_sample.json`
  - `outputs/document_upload_sample.md`
- **Implementation Subtasks**:
  1. [x] Task 1 — Create upload endpoint: Implement `POST /documents` accepting document files (`.txt`, `.md`, `.pdf`) and storing them safely with path traversal protection.
  2. [x] Task 2 — Ingest, embed, and index: Process uploaded document through loading, semantic text cleaning, token-aware structural chunking, metadata tagging, embedding, and vector database/runtime indexing.
  3. [x] Task 3 — Confirm runtime searchability: Verify that newly uploaded content becomes immediately searchable through `POST /query` with full grounded citation attribution without restarting the server.
  4. [x] Task 4 — Handle upload errors: Validate formats and file sizes gracefully, returning HTTP 415 for unsupported media types, HTTP 400 for empty or invalid files, HTTP 413 for oversized uploads, and HTTP 500 for pipeline failures without leaking tracebacks.
  5. [x] Task 5 — Commit sample upload run: Export and verify sample upload request, indexing summary (`status: indexed`), and follow-up query verification in `outputs/document_upload_sample.json` and `outputs/document_upload_sample.md`.
  6. [x] Author comprehensive automated unit and integration test suite in `tests/test_document_upload.py` (22 tests).
- **Acceptance Criteria**:
  - `POST /documents` endpoint operational, accepts `.txt`, `.md`, `.pdf` and returns structured summary `{status: 'indexed', filename: '...', summary: {document: '...', chunks: N, indexed: N}}`.
  - Directory traversal attacks (`../../`) sanitized to safe basenames.
  - Runtime searchability confirmed: pre-upload query refuses safely (`refused_weak_context`); post-upload query answers with verified citation (`answered`) without app restart.
  - Unsupported file formats rejected with HTTP 415; empty files rejected with HTTP 400; oversized files rejected with HTTP 413.
  - Zero hardcoded secrets, zero credential leakage.
- **Required Tests**:
  - `tests/test_document_upload.py`: 22 automated tests covering file format validation, traversal sanitization, size limits, text extraction, cleaning, chunking, metadata tagging, embedding, runtime searchability, error handling, and artifact generation.
- **Verification Evidence**:
  - 22 tests passing in `tests/test_document_upload.py`; 458 total test suite passing without regression.
  - Verification artifacts generated in `outputs/document_upload_sample.json` and `outputs/document_upload_sample.md`.

