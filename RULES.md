# FieldSure AI — Strict Engineering & Coding Rules (RULES.md)

> **Mandatory Guidelines**: All code generated, modified, or reviewed by AI agents or engineers for FieldSure AI must strictly adhere to the rules defined in this document. Violations of these rules are considered regressions.

---

## 1. Code Quality

1. **Write Clean, Readable, Maintainable Python**: Code should be self-documenting, idiomatic, and follow standard Python best practices.
2. **Follow PEP 8 & Meaningful Naming**: Use snake_case for functions and variables, PascalCase for classes, and UPPER_CASE for constants. Avoid ambiguous abbreviations (e.g., use `document_path` instead of `dp`, `chunk_index` instead of `ci`).
3. **Use Type Hints Where They Improve Clarity**: Add type annotations to all public function signatures, parameters, and return types. Use `typing` primitives (`Optional`, `Dict`, `List`, `Union`, `Tuple`, `Any`) or Python 3.10+ native pipe syntax where supported.
4. **Keep Functions Focused on One Responsibility**: Functions should perform a single logical task and stay under ~50 lines of code whenever possible. Extract helper functions for complex sub-operations.
5. **Prefer Straightforward Implementations Over Complicated Abstractions**: Avoid over-engineered object graphs, dynamic metaclasses, and multi-layered inheritance when simple functions or dataclasses suffice.
6. **Avoid Unnecessary Classes, Wrappers, Factories, and Design Patterns**: Do not introduce factory patterns, singletons, or custom wrapper classes unless there is an explicit architectural requirement. Prefer plain functions and standard library dataclasses.
7. **Do Not Duplicate Existing Functionality**: Before writing a utility or parsing routine, check existing codebase modules (e.g., `src/config.py`, `src/ingestion/cleaner.py`).
8. **Do Not Add Unused Imports, Variables, Functions, or Dependencies**: Maintain a zero-dead-code standard. Run clean import checks and prune all unused symbols.
9. **Do Not Create Extra Files When an Existing File is Appropriate**: Place related routines in their designated functional module rather than fragmenting the project into dozens of single-function files.
10. **Avoid Speculative Features and Premature Optimization**: Implement only the immediate requirements for the assigned module. Never add placeholder methods, "future-proofing" flags, or unused parameters.

---

## 2. Optimization

1. **Prefer Efficient Algorithms & Appropriate Data Structures**: Use sets and dictionaries for $O(1)$ lookups, de-duplication, and metadata indexing. Use generators and iterators for sequential stream processing.
2. **Avoid Repeated File Reads, Unnecessary API Calls, and Redundant Processing**: Load files once per ingestion pipeline run. Cache parsed structures in memory during batch execution rather than re-reading from disk.
3. **Process Large Documents in a Memory-Conscious Way**: Stream or iterate over large PDF/DOCX pages rather than loading multi-gigabyte files entirely into monolithic strings where practical.
4. **Avoid Unnecessary Intermediate Copies of Document Text**: Perform transformations in place or chain operations cleanly to avoid excessive memory allocation during text cleaning.
5. **Use Caching Only When It Provides a Demonstrated Benefit**: Do not add caching layers (e.g., `lru_cache`) unless profiling indicates a true computational or I/O bottleneck.
6. **Do Not Sacrifice Correctness or Readability for Minor Performance Gains**: Clear, verifiable clinical document processing always takes precedence over micro-optimizations.
7. **Explain Important Performance Tradeoffs**: When choosing a specific parsing or chunking approach with known performance implications, document the tradeoff directly in docstrings and design notes.

---

## 3. Architecture

1. **Follow the Existing Project Structure**: Adhere strictly to the workspace directory layout (`src/`, `prompts/`, `outputs/`, `data/`, `tests/`) specified in `Readme.md` and `ARCHITECTURE.md`.
2. **Keep Modules Loosely Coupled**: Ensure individual components (loader, cleaner, chunker, LLM client) can be imported, used, and tested independently.
3. **Use Reusable Functions When There is a Genuine Need**: Consolidate shared logic (e.g., environment variable fetching, safe token counting) into centralized modules (`config.py`, `chunker.py`).
4. **Avoid Circular Imports**: Structure imports hierarchically: models and schemas at the base, domain logic in the middle, and top-level entry points/CLI scripts at the top.
5. **Preserve Metadata and Source Traceability Throughout Ingestion**: Every text transformation must preserve document identity, version, and section context. Never discard or strip source metadata during chunking.
6. **Keep Interfaces Compatible with the Planned Downstream Embedding Pipeline**: Output chunk schemas must strictly conform to the `DocumentChunk` data contract agreed upon for Member 2's vector ingestion.
7. **Do Not Modify Another Member's Code Without Permission**: Confine all code development exclusively to Member 1's assigned modules (#1, #2, #3, #4, #10, #11, #12, #13, #14, #15). Do not create, modify, or refactor files belonging to Member 2 or Member 3.

---

## 4. Security & Safety

1. **Never Hardcode API Keys**: API keys, tokens, and sensitive URLs must never appear in source code, default arguments, comments, or commit messages.
2. **Never Expose Credentials in Logs**: Sanitize all logging output. Explicitly mask keys (e.g., `sk-...****`) and headers before writing to `outputs/` or terminal stdout.
3. **Use Environment Variables for Secrets**: Read all sensitive configurations dynamically from the environment via `src/config.py` using `python-dotenv`.
4. **Never Commit `.env` or Sensitive Data**: Enforce `.env` and `data/` exclusion in `.gitignore`. Provide a clean `.env.example` with placeholder values only.
5. **Validate Document Formats and Required Metadata**: Sanitize input file paths to prevent directory traversal (`../`). Validate MIME types and check required metadata attributes before processing.
6. **Never Fabricate Document Sources, Versions, Dates, or Citations**: If document metadata (e.g., publication date, section name) is missing from an input file, record it explicitly as `None` or `Unknown`. Never invent hallucinated dates or authoritative bodies.
7. **Do Not Generate Unsupported Medical Guidance**: System prompts and completions must strictly refuse ungrounded medical queries and enforce disclaimers stating that FieldSure AI is an informational tool, not a diagnostic system.

---

## 5. Testing

1. **Add Meaningful Tests for Implemented Functionality**: Every completed module must include corresponding automated unit tests in `tests/` validating core paths.
2. **Test Important Edge Cases and Failure Conditions**: Test handling of empty files, malformed documents, missing environment variables, network timeouts, and token overflow.
3. **Mock External API Calls in Unit Tests**: Never make real network requests to external LLM APIs during unit test execution. Always mock responses using `unittest.mock` or `pytest-mock`.
4. **Do Not Add Meaningless Tests Solely to Increase Test Counts**: Write purposeful tests that assert specific domain behaviors, data schemas, and error boundaries.
5. **Never Claim Tests Passed Unless They Were Actually Executed**: Every test reported as passing must be physically run via `uv run pytest` with verifiable terminal test reports.
6. **Fix the Root Cause of Errors Instead of Hiding Them**: Never suppress exceptions with bare `except:` clauses or empty error blocks. Identify, handle, and resolve the underlying failure.
