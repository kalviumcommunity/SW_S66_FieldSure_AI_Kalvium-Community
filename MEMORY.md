# FieldSure AI — Project Memory (MEMORY.md)

> **Role**: Member 3 — RAG, Backend & Application (Module #46)  
> **Last Updated**: 2026-10-26
> **Current Sprint Phase**: Chat Interface & Query UI (Module #46)

---

## 1. Module Progress

- **Current Active Task**: Module #46 (Chat Interface & Query UI) completed.
- **Completed Modules**:
  - Module #1 (Development Environment & Project Workspace Setup: `pyproject.toml`, `.env.example`, `.gitignore`, directory scaffolding, `tests/test_workspace.py`).
  - Module #3 (LLM API Access & First Completion Call: `src/config.py`, `src/llm_client.py`, `src/chat_completion.py`, `outputs/chat_completion_sample.log`, `tests/test_llm_client.py`).
  - Module #4 (Prompt Construction & System/User Roles: `prompts/system_prompt_baseline.txt`, `prompts/system_prompt_constrained.txt`, `prompts/README.md`, `src/prompt_comparison.py`, `outputs/prompt_comparison_sample.log`, `tests/test_prompts.py`).
  - Module #5 (Tokens, Tokenization & Cost Estimation: `src/token_cost_estimator.py`, `outputs/token_cost_estimation_sample.log`, `tests/test_token_cost.py`).
  - Module #6 (Context Windows & Message History Management: `src/context_manager.py`, `outputs/context_window_management_sample.log`, `tests/test_context_manager.py`).
  - Module #10 (Document Loading & Multi-Format Intake: `src/ingestion/loader.py`, `data/sample_advisories/`, `outputs/document_loading_sample.log`, `tests/test_loader.py`).
  - Module #16 (Embeddings Fundamentals & Vector Representation: `src/embedding_fundamentals.py`, `outputs/embedding_fundamentals_sample.md`, `tests/test_embedding_fundamentals.py`).
  - Module #17 (Generating Embeddings via API: `src/embed_corpus.py`, `outputs/embedding_api_sample.md`, `tests/test_embed_corpus.py`).
  - Module #18 (Embedding Similarity & Distance Metrics: `src/rank_chunks.py`, `outputs/embedding_similarity_sample.md`, `tests/test_rank_chunks.py`).
  - Module #20 (Embedding Quality Checks & Sanity Tests: `src/embedding_quality.py`, `outputs/embedding_quality_sanity_report.json`, `tests/test_embedding_quality.py`).
  - Module #21 (Vector Database Setup & Collection Design: `src/vector_store.py`, `src/vector_db_demo.py`, `tests/test_vector_store.py`, `outputs/vector_db_readback_sample.md`).
  - Module #22 (Indexing Embeddings & Metadata Storage: `src/index_embeddings.py`, `tests/test_index_embeddings.py`, `outputs/indexing_summary_sample.md`).
  - Module #23 (Similarity Search & Top-K Retrieval: `src/retrieve_chunks.py`, `tests/test_retrieve_chunks.py`, `outputs/top_k_retrieval_sample.md`).
  - Module #24 (Metadata Filtering & Hybrid Search: `src/filtered_retrieval.py`, `tests/test_filtered_retrieval.py`, `outputs/filtered_retrieval_sample.md`).
  - Module #25 (Retrieval Relevance Tuning: `src/tune_retrieval.py`, `tests/test_tune_retrieval.py`, `outputs/retrieval_tuning_summary.md`).
  - Module #27 (Retrieval Evaluation & Recall Testing: `src/evaluate_retrieval.py`, `tests/test_evaluate_retrieval.py`, `outputs/retrieval_evaluation_queries.json`, `outputs/retrieval_evaluation_report.json`, `outputs/retrieval_evaluation_summary.md`).
  - Module #28 (RAG Pipeline Architecture & Flow Design: `src/rag_pipeline.py`, `tests/test_rag_pipeline.py`, `outputs/rag_pipeline_sample.md`).
  - Module #38 (Context Injection & Prompt Augmentation: `src/context_injection.py`, `tests/test_context_injection.py`, `outputs/context_injection_sample.md`).
  - Module #39 (Grounded Answer Generation: `src/grounded_generation.py`, `tests/test_grounded_generation.py`, `outputs/grounded_generation_sample.md`).
  - Module #40 (Source Citation & Attribution: `src/source_citation.py`, `src/citation.py`, `tests/test_source_citation.py`, `outputs/source_citation_sample.md`).
  - Module #41 (Hallucination Guardrails & Refusal Handling: `src/guardrails.py`, `src/hallucination_guardrails.py`, `tests/test_guardrails.py`, `outputs/guardrails_sample.md`).
  - Module #42 (Conversational RAG & Follow-Up Context: `src/conversational_rag.py`, `src/conversation.py`, `tests/test_conversational_rag.py`, `outputs/conversational_rag_sample.md`).
  - Module #43 (RAG Evaluation & Answer Quality Scoring: `src/rag_evaluation.py`, `src/evaluate_rag.py`, `tests/test_rag_evaluation.py`, `outputs/rag_evaluation_test_set.json`, `outputs/rag_evaluation_report.json`, `outputs/rag_evaluation_summary.md`, `outputs/rag_evaluation_sample.md`).
  - Module #44 (Backend API for the RAG Service: `src/backend_api.py`, `src/api.py`, `tests/test_backend_api.py`, `outputs/backend_api_sample.json`, `outputs/backend_api_sample.md`).
  - Module #45 (Document Upload & Indexing Endpoint: `src/document_upload.py`, `src/backend_api.py`, `tests/test_document_upload.py`, `outputs/document_upload_sample.json`, `outputs/document_upload_sample.md`).
  - Module #46 (Chat Interface & Query UI: `frontend/`, `src/query_ui_demo.py`, `tests/test_query_ui.py`).
- **In-Progress Modules**: Module #2 (GitHub Repository & Team Workflow Setup: Git initialized, docs established, scaffolding complete).

---

## 2. Key Technical Decisions

1. **Tooling & Environment**: Astral `uv` selected for fast, reproducible dependency resolution and virtual environment isolation (`pyproject.toml` + `uv.lock`).
2. **Text Extraction & Intake**:
   - `pypdf` for pure-Python PDF page-wise extraction without external binaries.
   - `python-docx` for structured Word document extraction.
   - Standard library UTF-8 reader with fallback encodings (`utf-8-sig`, `latin-1`) for Markdown (`.md`), plain text (`.txt`), and HTML (`.html`).
   - Batch intake gracefully skips corrupt, missing, or unsupported formats, cataloging failures into an audit error ledger without terminating pipeline runs.
3. **Chunking & Token Budgeting**:
   - Recursive structural chunking splitting along semantic boundaries (headings, paragraphs, sentences).
   - Default budget: **500 tokens** chunk size with **50 tokens (10%) sliding overlap**.
   - Subword token accounting via `tiktoken` (`cl100k_base`).
4. **Metadata Traceability**:
   - Every document maintains immutable provenance (`doc_id`, `source_file`, `file_path`, `file_type`, `file_size_bytes`).
5. **LLM Client & Safety**:
   - Centralized `src/config.py` loading secrets dynamically from `.env`. Zero secrets logged or committed.
   - Standardized `OpenAICompatibleClient` using `httpx` with timeout configuration and explicit error translation (401 auth, 429 rate limit).
   - Structured audit logging to `outputs/chat_completion_sample.log` and `outputs/prompt_comparison_sample.log` with masked credentials.
6. **Prompt Governance & Refusal Guardrails**:
   - Explicit prompt constraints requiring citation of document and section for every claim (`[Document Title, Section Name]`).
   - Deterministic safe refusal sentence for diagnostic or out-of-scope inquiries.
   - Mandatory non-diagnostic legal disclaimer on all guidance outputs.
7. **Token Economics & Pricing Models**:
   - Separate pricing calculation for input vs. output tokens (e.g. gpt-4o-mini at $0.15/1M input and $0.60/1M output).
   - Character-to-token ratio profiling demonstrates that medical terminology and non-Latin scripts (e.g. Devanagari) consume substantially more tokens per character than standard English.
8. **Context Window & History Management**:
   - System prompt at index 0 is strictly immutable and preserved across all trimming actions.
   - Pre-request token measurement accounts for subword content and message framing overhead (~3 tokens/message).
   - Sliding-window eviction prunes oldest turns while appending a concise summary tag to retain continuity and keep history within context ceilings.
9. **Grounded Generation & Source Accuracy Verification**:
   - Dual verification: content keyword overlap against injected context plus sentence-level support validation.
   - Informational disclaimer excluded from sentence-level checks to prevent false-positive hallucination flags.
   - Explicit fallback guarantee returning `"I don't have enough information in the provided context."` with empty sources when context is absent or unsupported.
10. **Source Citation & Traceable Attribution**:
   - `build_citation_map(chunks)` binds citation markers (`[1]`, `[2]`) to metadata (`source`, `chunk_id`, `chunk_index`, `section`, `text`, `page`).
   - `build_cited_prompt(question, chunks)` instructs the LLM to cite provided markers exclusively and refuse when unsupported.
   - Dual interface response: `CitedAnswer` dict subclass enabling both dictionary subscripting (`result["answer"]`, `result["citations"]`) and attribute access (`result.answer`).
   - Verifiability: `verify_citation` and `verify_all_citations` confirm claim-level source chunk lexical support.
   - Anti-hallucination guardrail: `detect_fabricated_citations` flags illegitimate markers; zero-chunk queries yield clean fallback (`"I don't have enough information in the provided context."`) with zero citations.
11. **Hallucination Guardrails & Refusal Protocol**:
   - Safe refusal standard: returns `"I don't have enough reliable context to answer that."` with `status="refused_weak_context"` and `sources=[]`.
   - Dual interface response: `GuardedAnswer` dict subclass supporting subscripting (`result["status"]`), attribute access (`result.status`), and dict unpacking (`{**result}`).
12. **Conversational RAG & Coreference Resolution**:
    - Pre-retrieval query rewriting: transforms conversational follow-ups (e.g. "What about the video?", "Does it apply to Sprint 2?", "Can you explain that?") into self-contained standalone search queries using dialogue history before embedding and retrieval.
    - Naive vs. rewritten disambiguation: prevents ambiguous or fragmentary queries from matching unrelated chunks (e.g., commercial camera setup) or missing relevant context.
    - Rolling dialogue history management: tracks both user questions and assistant replies in `[{"role": role, "content": text}]` format, enforcing max turn (6) and token (600) budgets via `trim_history` without losing coreference context.
    - Guardrail integration: evaluates retrieval quality on the rewritten query; returns safe refusal (`"I don't have enough reliable context to answer that."`, `refused_weak_context`) on weak or out-of-corpus inquiries, preserving confident grounded answers (`answered`) when strong evidence exists.
    - Dual interface response: `ConversationalAnswer` dict subclass supporting dictionary subscripting (`res["rewritten_query"]`), attribute access (`res.rewritten_query`), and dict unpacking (`{**res}`).
13. **RAG Evaluation & Multi-Dimensional Answer Quality Scoring**:
    - Three-dimensional evaluation rubric: **Correctness** (point/concept coverage), **Grounding** (claim support in retrieved context), and **Citation Accuracy** (precision of cited source documents vs. expected references).
    - Objective scoring: replaces subjective "vibes" with deterministic scoring (`judge_expected_points`, `judge_grounding`, `check_citations`).
    - Grounding contract: safe refusals on missing or insufficient context are 100% grounded (score 1.0) because no unsupported factual claims are made.
    - Automated failure root-cause taxonomy: automatically categorizes failures into `weak_retrieval`, `missing_metadata`, `bad_citations`, `unsupported_claims`, and `poor_prompt_instructions`, generating targeted remediation advice.

---

## 3. Existing Reusable Functions & Components

- `src.config.load_llm_config()`: Safely parses and validates `API_BASE_URL`, `API_KEY`, and `CHAT_MODEL` from environment or `.env`.
- `src.config.mask_api_key(key)`: Obfuscates API keys (e.g. `sk-...1234`) for zero-leakage logging.
- `src.llm_client.OpenAICompatibleClient`: Reusable client connecting to any OpenAI-compatible chat API with standard payload formatting and error handling.
- `src.llm_client.UsageInfo` & `ChatCompletionResult`: Standardized data classes for completion results and token usage.
- `src.chat_completion.run_chat_completion()`: Baseline execution runner with exit code contracts (0=success, 1=config error, 2=API error).
- `prompts/system_prompt_constrained.txt`: Production-ready prompt enforcing grounding, citations, and safe refusals.
- `src.prompt_comparison.run_prompt_comparison()`: Comparative evaluation harness testing prompt variations across triage and refusal scenarios.
- `src.token_cost_estimator.count_tokens(text)`: Accurately counts subword tokens using `tiktoken` with fallback heuristic.
- `src.token_cost_estimator.ModelPricing`: Dataclass calculating differential input/output costs for any model.
- `src.token_cost_estimator.profile_text(...)`: Computes characters-per-token and tokens-per-word metrics for length-token relationship analysis.
- `src.context_manager.ContextWindowManager`: Multi-turn dialogue manager with pre-request token measurement, system prompt preservation, and sliding-window trimming.
- `src.ingestion.loader.load_document(path)`: Robust multi-format document reader (`.pdf`, `.docx`, `.md`, `.txt`, `.html`) with source metadata preservation.
- `src.ingestion.loader.load_corpus(dir)`: Batch intake runner surviving bad files with detailed error ledger reporting.
- `src.embedding_fundamentals.generate_embeddings(...)`: Requests and validates embedding vectors from an OpenAI-compatible API; `cosine_similarity(...)` compares vectors.
- `src.embedding_fundamentals.embed_corpus_chunks(...)`: Keeps embedding vectors paired with original chunk text and metadata.
- `src.rank_chunks.rank_chunks(...)`: Ranks stored corpus embeddings against a query vector by cosine similarity.
- `src.embedding_quality.evaluate_quality_cases(...)`: Checks known query/chunk relevance, detects failures and borderline score margins, and prepares a sanity report.
- `src.vector_store.ChromaVectorStore`: Persists embedded source chunks and retrieval metadata in a local ChromaDB cosine collection, enforcing one vector dimension.
- `src.vector_db_demo.run_demo(...)`: Inserts embedded chunks and verifies source record readback from the configured persistent collection.
- `src.index_embeddings.run_indexing_job(...)`: Indexes the complete embeddings file, compares it with the source corpus, validates the indexed count, and spot-checks stored records.
- `src.vector_store.ChromaVectorStore.search(...)`: Retrieves nearest stored chunks with cosine similarity, source text, and restored metadata.
- `src.retrieve_chunks.retrieve_for_k_values(...)`: Embeds a query with the configured model and runs searches for each requested top-k value.
- `src.tune_retrieval.run_tuning_experiment(...)`: Evaluates retrieval settings against known source relevance and selects the best macro F1@k configuration.
- `src.rag_pipeline.embed_query(query, ...)`: Encodes query into dense vector via live API or unit-normalized deterministic hash vector.
- `src.rag_pipeline.retrieve_context(query_vector, store, k, ...)`: Performs vector lookup in ChromaDB with optional metadata filtering and score thresholding.
- `src.rag_pipeline.assemble_context(chunks)`: Assembles numbered, provenance-tagged context blocks (`[1] Source: {source} (Section: {section})`).
- `src.rag_pipeline.generate_answer(query, context, ...)`: Enforces strict grounding and context injection into prompt with safe refusal fallback.
- `src.rag_pipeline.answer_query(query, k, store, ...)`: Main pipeline orchestrator connecting embed -> retrieve -> assemble -> generate, handling empty retrieval deterministically.
- `src.context_injection.format_chunk(chunk, index)`: Attaches clear provenance header (`[N] Source: ... | Section: ...`) with metadata.
- `src.context_injection.assemble_context(chunks, max_tokens)`: Assembles numbered context blocks while enforcing subword token ceilings.
- `src.context_injection.build_augmented_prompt(question, chunks)`: Wraps context with explicit grounding instructions and fallback instructions into structured prompt.
- `src.grounded_generation.generate_grounded_answer(question, chunks)`: Synthesizes answers based strictly on injected context with explicit source attribution.
- `src.grounded_generation.verify_grounding(answer, context)`: Analyzes keyword overlap and sentence-level support, flagging unsupported claims.
- `src.grounded_generation.answer_query(question, chunks, retriever_fn)`: Main query answerer with missing-context fallback guarantee.
- `src.grounded_generation.direct_llm_call(question)`: Generates ungrounded answer directly from parametric memory for comparison.
- `src.grounded_generation.compare_grounded_vs_ungrounded(question, chunks)`: Runs side-by-side RAG grounded vs ungrounded comparison with overlap metrics.
- `src.grounded_generation.print_grounding_check(result)`: Renders human-readable grounding inspection report with citations and claim validation.
- `src.source_citation.build_citation_map(chunks)`: Maps 1-based citation markers (`[1]`, `[2]`) to source document, chunk ID, index, section, page, and text.
- `src.source_citation.assemble_context(chunks)`: Dual-interface context assembler supporting string operations and tuple unpacking (`context, _ = assemble_context(chunks)`).
- `src.source_citation.build_cited_prompt(question, chunks)`: Formats grounding instructions mandating inline source markers and refusal on missing context.
- `src.source_citation.retrieve(question, k)`: Retrieves candidate chunks with zero-relevance filtering to prevent fabricated citations.
- `src.source_citation.answer_with_citations(question, chunks, ...)`: Orchestrator returning `CitedAnswer` dict with answer, citations mapping, and fallback support.
- `src.source_citation.verify_citation(marker, answer, citation_map)`: Confirms citation marker validity, answer presence, and claim-level source chunk lexical support.
- `src.source_citation.verify_all_citations(answer, citation_map)`: Audits all markers in an answer, detecting supported claims, fabricated markers, and unreferenced sources.
- `src.source_citation.detect_fabricated_citations(answer, citation_map)`: Detects citation markers generated by model that do not exist in retrieved context.
- `src.source_citation.sanitize_answer_citations(answer, citation_map)`: Strips fabricated citation markers from answers.
- `src.source_citation.print_citation_verification(result)`: Renders human-readable console report for citation audit inspection.
- `src.guardrails.get_chunk_score(chunk)`: Extracts score across `"score"`, `"similarity_score"`, `"similarity"`, and `"cosine_distance"`.
- `src.guardrails.retrieval_is_strong(chunks, min_top_score, min_supporting_chunks)`: Assesses retrieval quality against relevance thresholds (`MIN_TOP_SCORE = 0.72`, `MIN_SUPPORTING_CHUNKS = 1`).
- `src.guardrails.evaluate_retrieval_quality(chunks)`: Produces comprehensive audit diagnostic report (`RetrievalQualityResult`) with failure reasoning.
- `src.guardrails.retrieve(question, k)`: Retrieves candidate chunks with guaranteed `"score"` and `"similarity_score"` attributes.
- `src.guardrails.guarded_answer(question, ...)`: Pipeline entry point refusing weak context (`"refused_weak_context"`) and returning confident grounded answers (`"answered"`).
- `src.guardrails.GuardedAnswer`: Dict subclass response supporting dictionary subscripting, attribute access, and dict unpacking.
- `src.guardrails.run_guardrails_demo(output_file)`: Execution verification runner generating sample refusal and answer reports.
- `src.conversational_rag.format_history_text(history)`: Formats user and assistant message history for prompt injection and diagnostic logging.
- `src.conversational_rag.trim_history(history, max_turns, max_tokens)`: In-place rolling history pruner preserving recent turns under token limits.
- `src.conversational_rag.rewrite_followup(history, question, llm_fn)`: Rewrites dependent follow-ups into standalone queries resolving references.
- `src.conversational_rag.retrieve(question, k, ...)`: Retrieves scored context chunks from ingested corpus and benchmark chunks.
- `src.conversational_rag.compare_retrieval(history, question, ...)`: Compares naive vs rewritten query retrieval metrics demonstrating precision gain.
- `src.conversational_rag.conversational_answer(history, user_question, ...)`: Multi-turn pipeline orchestrator rewriting queries, retrieving context, enforcing guardrails, and appending to history.
- `src.conversational_rag.ConversationalAnswer`: Dict subclass response supporting dictionary subscripting, attribute access, and dict unpacking.
- `src.conversational_rag.run_conversational_dialogue(questions, ...)`: Multi-turn sequential dialogue executor maintaining context across questions.
- `src.conversational_rag.run_conversational_rag_demo(output_file)`: Execution verification runner generating sample multi-turn dialogues and reports.
- `src.rag_evaluation.judge_expected_points(answer, expected_points)`: Measures factual coverage of expected concepts and refusal semantics.
- `src.rag_evaluation.judge_grounding(answer, retrieved_sources)`: Evaluates sentence-level claim support in retrieved chunk context, treating safe refusals as fully grounded.
- `src.rag_evaluation.check_citations(citations, expected_sources)`: Verifies citation accuracy against expected source document basenames, penalizing unwanted/fabricated citations.
- `src.rag_evaluation.diagnose_failure(scored_result, example)`: Identifies likely cause (`weak_retrieval`, `missing_metadata`, `bad_citations`, `unsupported_claims`, `poor_prompt_instructions`) with actionable recommendations.
- `src.rag_evaluation.score_answer(example, answer_fn, ...)`: Evaluates a single inquiry across all three dimensions, returning scores, citations, and failure diagnosis.
- `src.rag_evaluation.evaluate_test_set(test_set, ...)`: Batches evaluation across the full test set.
- `src.rag_evaluation.summarize_evaluations(rows)`: Computes macro average scores across dimensions and isolates notable failures (`min score < 1.0`).
- `src.rag_evaluation.format_evaluation_summary_md(summary, rows)`: Formats evaluation results, scores, and root-cause diagnoses into Markdown reports.
- `src.rag_evaluation.run_rag_evaluation(...)`: End-to-end evaluation runner exporting JSON reports, test sets, and Markdown summaries.
- `src.backend_api.create_app(pipeline_fn, config)`: FastAPI application factory exposing `POST /query`, `POST /documents`, `GET /health`, `GET /config`, `GET /`.
- `src.backend_api.QueryRequest`: Pydantic input schema enforcing 3–1000 character length and non-whitespace validation.
- `src.backend_api.Source`: Citation model with `source`, `chunk_id`, and retrieval `score`.
- `src.backend_api.QueryResponse`: Structured response schema with `answer`, `sources`, `status`, and `metadata`.
- `src.backend_api.load_environment_config(env_path, strict)`: Loads environment variables (`OPENAI_API_KEY`/`API_KEY`, `EMBEDDING_MODEL`, `VECTOR_DB_URL`, `COLLECTION_NAME`, `CHAT_MODEL`, `PORT`, `HOST`) with zero hardcoded credentials.
- `src.backend_api.query_rag(request)`: Functional endpoint handler managing RAG execution, ValueError mapping (HTTP 400), and internal failure safety (HTTP 500).
- `src.backend_api.run_demo(output_json, output_md)`: Test client execution runner generating sample request and response artifacts in `outputs/backend_api_sample.json` and `outputs/backend_api_sample.md`.
- `src.document_upload.validate_upload(file)`: Validates file existence and extension (.txt, .md, .pdf), returning lowercase suffix or raising HTTP 415/400.
- `src.document_upload.store_upload(file, upload_dir, max_size_bytes)`: Sanitizes filename to prevent directory traversal, rejects empty (HTTP 400) or oversized files (HTTP 413), and persists bytes safely to disk.
- `src.document_upload.load_text(path)`: Robust multi-format document text extractor supporting .txt, .md, and .pdf formats.
- `src.document_upload.clean(text)`: Semantic text cleaner and Unicode normalizer preserving clinical formatting.
- `src.document_upload.token_chunks(text, chunk_size, overlap)`: Structural token-budgeted chunker.
- `src.document_upload.tag_chunks(source, chunks)`: Enriches raw chunks with unique IDs, source document title, token count, and citation metadata.
- `src.document_upload.embed_chunks(chunks, model, config)`: Generates dense unit vector embeddings.
- `src.document_upload.index_chunks(chunks, store)`: Updates in-memory runtime registry and upserts into vector database.
- `src.document_upload.process_uploaded_document(path, store)`: Orchestrator running the complete 6-stage ingestion pipeline on uploaded documents.
- `src.document_upload.upload_document(file, upload_dir, store)`: Functional async handler returning `UploadResponse` with indexing summary.
- `src.document_upload.run_upload_demo(sample_file, output_json, output_md)`: Test runner demonstrating upload execution, runtime query searchability, and error handling.
- `src.api`: Consolidated API entry point re-exporting backend API and document upload endpoints and models.
- `frontend.src.lib.api.askQuestion(question, baseUrl)`: Async client dispatching query requests to `POST /query` and returning typed answers with citations and latency metadata.
- `frontend.src.lib.api.checkHealth(baseUrl)`: Health check client pinging `GET /health` with latency measurement.
- `frontend.src.lib.api.uploadDocument(file, baseUrl)`: Multi-format document upload client for runtime knowledge base ingestion via `POST /documents`.
- `frontend.src.components.AnswerCard`: React component displaying grounded answers with confidence badges, disclaimers, and clipboard actions.
- `frontend.src.components.SourcesList`: React component rendering retrieved document citations, chunk identifiers, and similarity score tags.
- `src.query_ui_demo.run_sample_interactions(output_json, output_md)`: Test runner generating JSON and Markdown interaction evidence across 6 query and refusal scenarios.

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
