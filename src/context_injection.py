"""Context Injection & Prompt Augmentation (Module #29).

Turns raw retrieved chunks into a grounded, citation-rich prompt that stays
within the model's token budget and explicitly instructs the LLM to answer
only from the provided context.

Pipeline:
    Retrieved Chunks -> format_chunk (add source markers)
                     -> assemble_context (enforce token budget)
                     -> build_augmented_prompt (inject grounding instructions)

Reuses:
    - src.token_cost_estimator.count_tokens for accurate subword token counting.
    - Metadata schema from src.ingestion.metadata (doc_id, source_file, etc.).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.token_cost_estimator import count_tokens

logger = logging.getLogger("fieldsure.context_injection")

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

# Default token budget for the context block injected into the prompt.
# For an 8 000-token context window, reserve ~1 500 for the answer and
# ~800 for the system instructions + user question, leaving ~5 000 for chunks.
MAX_CONTEXT_TOKENS: int = 5000

# Token reserves that are subtracted from the full context window to compute
# the dynamic context budget when the caller supplies a total window size.
ANSWER_RESERVE_TOKENS: int = 1500
INSTRUCTION_RESERVE_TOKENS: int = 800

# Separator placed between formatted chunks in the assembled context block.
CHUNK_SEPARATOR: str = "\n\n---\n\n"

# Grounding instructions prepended to the user prompt.
GROUNDING_INSTRUCTION: str = (
    "You are a grounded assistant. Answer the question using ONLY the "
    "provided context below.\n"
    "If the answer is not in the context, say: "
    '"I don\'t have enough information in the provided context."\n'
    "When possible, cite sources using the markers like [1] or [2].\n"
    "Do not use external knowledge, do not speculate, and do not invent facts."
)

# Fallback refusal when there is no context at all.
FALLBACK_NO_CONTEXT: str = (
    "I don't have enough information in the provided context."
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AssembledContext:
    """Result of context assembly including the formatted text and bookkeeping."""

    text: str
    token_count: int
    chunks_included: int
    chunks_total: int
    sources_used: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AugmentedPrompt:
    """Complete augmented prompt ready to be sent to the LLM."""

    prompt: str
    context_text: str
    context_tokens: int
    total_prompt_tokens: int
    chunks_included: int
    chunks_total: int
    sources_used: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 1 & Task 3 — Format retrieved chunks with source markers
# ---------------------------------------------------------------------------

def format_chunk(index: int, chunk: Dict[str, Any]) -> str:
    """Format a single retrieved chunk with a numbered source marker.

    The marker format is:
        [<index>] <source_file>#<chunk_index>

    This enables the model to cite evidence back to the exact document and
    chunk position (e.g. ``[1]`` or ``[2]``), and lets reviewers audit
    which passage each fact was drawn from.

    Args:
        index: 1-based position of the chunk within the assembled context.
        chunk: Dictionary with at minimum ``'text'`` and ``'metadata'`` keys.
              Metadata should contain ``'source'`` or ``'source_file'`` and
              optionally ``'chunk_index'`` and ``'section_heading'``.

    Returns:
        A string containing the source marker header followed by the chunk text.

    Raises:
        ValueError: If the chunk dict is missing required keys.
    """
    if not isinstance(chunk, dict):
        raise ValueError("Chunk must be a dictionary.")
    if "text" not in chunk:
        raise ValueError("Chunk dictionary must contain a 'text' key.")

    metadata = chunk.get("metadata", {})

    # Resolve the source identifier, falling through several possible keys
    source = (
        metadata.get("source")
        or metadata.get("source_file")
        or metadata.get("document_title")
        or "Unknown Source"
    )

    chunk_index = metadata.get("chunk_index")
    section = metadata.get("section_heading")

    # Build the marker line
    marker_parts = [f"[{index}] {source}"]
    if chunk_index is not None:
        marker_parts.append(f"#chunk{chunk_index}")
    if section:
        marker_parts.append(f" (Section: {section})")

    marker = "".join(marker_parts)
    text = chunk.get("text", "").strip()

    return f"{marker}\n{text}"


# ---------------------------------------------------------------------------
# Task 2 — Enforce token budget during context assembly
# ---------------------------------------------------------------------------

def compute_context_budget(
    total_window_tokens: Optional[int] = None,
    answer_reserve: int = ANSWER_RESERVE_TOKENS,
    instruction_reserve: int = INSTRUCTION_RESERVE_TOKENS,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> int:
    """Compute the available token budget for the context block.

    If ``total_window_tokens`` is supplied (e.g. 8 000), the budget is::

        total_window_tokens - answer_reserve - instruction_reserve

    Otherwise the static ``max_context_tokens`` default is returned.

    Args:
        total_window_tokens: Full model context window size in tokens.
        answer_reserve: Tokens reserved for the model's generated answer.
        instruction_reserve: Tokens reserved for system/user instructions + question.
        max_context_tokens: Fallback static budget when window size is unknown.

    Returns:
        Maximum number of tokens available for injected context chunks.
    """
    if total_window_tokens is not None:
        budget = total_window_tokens - answer_reserve - instruction_reserve
        return max(budget, 0)
    return max_context_tokens


def assemble_context(
    chunks: Sequence[Dict[str, Any]],
    max_tokens: int = MAX_CONTEXT_TOKENS,
    total_window_tokens: Optional[int] = None,
) -> AssembledContext:
    """Assemble formatted, token-budgeted context from ranked retrieved chunks.

    Chunks are consumed in order (best-ranked first). Each chunk is formatted
    with a source marker via ``format_chunk``, and its token cost is measured.
    Once the next chunk would exceed the remaining budget, assembly stops.

    Args:
        chunks: Sequence of chunk dictionaries ordered by relevance (best first).
        max_tokens: Static maximum token allowance for context (default 5 000).
        total_window_tokens: If provided, overrides max_tokens with a dynamically
                             computed budget (see ``compute_context_budget``).

    Returns:
        An ``AssembledContext`` dataclass with the formatted text, token count,
        chunk counts, and source metadata list.
    """
    budget = compute_context_budget(
        total_window_tokens=total_window_tokens,
        max_context_tokens=max_tokens,
    )

    selected: List[str] = []
    used_tokens: int = 0
    sources_used: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        formatted = format_chunk(idx, chunk)
        token_cost = count_tokens(formatted)

        # Check whether adding this chunk would exceed the budget
        if used_tokens + token_cost > budget:
            logger.info(
                "Token budget reached after %d chunks (%d / %d tokens). "
                "Skipping remaining %d chunks.",
                len(selected),
                used_tokens,
                budget,
                len(chunks) - len(selected),
            )
            break

        selected.append(formatted)
        used_tokens += token_cost
        sources_used.append(chunk.get("metadata", {}))

    assembled_text = CHUNK_SEPARATOR.join(selected)

    return AssembledContext(
        text=assembled_text,
        token_count=used_tokens,
        chunks_included=len(selected),
        chunks_total=len(chunks),
        sources_used=sources_used,
    )


# ---------------------------------------------------------------------------
# Task 4 & Task 5 — Build the augmented prompt with grounding instructions
# ---------------------------------------------------------------------------

def build_augmented_prompt(
    question: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
    total_window_tokens: Optional[int] = None,
    grounding_instruction: Optional[str] = None,
) -> AugmentedPrompt:
    """Build a complete augmented prompt from a question and retrieved chunks.

    The prompt structure is::

        <grounding instruction>

        Context:
        <assembled chunks with source markers>

        Question: <user question>

    The grounding instruction explicitly tells the model:
    - Answer only from the provided context.
    - Cite sources using markers like [1], [2].
    - Refuse when the context is insufficient.

    Args:
        question: The user's question text.
        retrieved_chunks: Ranked retrieved chunk dicts with ``'text'`` and
                          ``'metadata'`` keys.
        max_context_tokens: Static token budget for the context block.
        total_window_tokens: Optional full context window to compute a dynamic
                             budget.
        grounding_instruction: Optional override for the default grounding
                               instruction text.

    Returns:
        An ``AugmentedPrompt`` dataclass containing the full prompt string,
        assembled context, token metrics, and source metadata.

    Raises:
        ValueError: If the question is empty or not a string.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    instruction = grounding_instruction or GROUNDING_INSTRUCTION

    # Assemble context within the token budget
    assembled = assemble_context(
        chunks=retrieved_chunks,
        max_tokens=max_context_tokens,
        total_window_tokens=total_window_tokens,
    )

    # Handle the case where no chunks could be included
    if assembled.chunks_included == 0:
        prompt = (
            f"{instruction}\n\n"
            f"Context:\n(No relevant context was retrieved.)\n\n"
            f"Question: {question.strip()}"
        )
    else:
        prompt = (
            f"{instruction}\n\n"
            f"Context:\n{assembled.text}\n\n"
            f"Question: {question.strip()}"
        )

    total_prompt_tokens = count_tokens(prompt)

    return AugmentedPrompt(
        prompt=prompt,
        context_text=assembled.text,
        context_tokens=assembled.token_count,
        total_prompt_tokens=total_prompt_tokens,
        chunks_included=assembled.chunks_included,
        chunks_total=assembled.chunks_total,
        sources_used=assembled.sources_used,
    )


# ---------------------------------------------------------------------------
# CLI / Demo Runner
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "context_injection_sample.md"
DEFAULT_CHUNKS_FILE = REPO_ROOT / "outputs" / "ingested_chunks.jsonl"


def _load_sample_chunks(path: Path, max_chunks: int = 6) -> List[Dict[str, Any]]:
    """Load ingested chunks from JSONL for the demo."""
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    chunks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            chunks.append(json.loads(line_str))
            if len(chunks) >= max_chunks:
                break
    return chunks


def run_context_injection_demo(
    question: str = "What are the warning signs and immediate field interventions for dengue?",
    chunks_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    max_context_tokens: int = MAX_CONTEXT_TOKENS,
) -> Dict[str, Any]:
    """Run the context injection demo, printing and saving a sample augmented prompt.

    Demonstrates:
      1. Formatting chunks with source markers.
      2. Assembling context within a token budget.
      3. Building the full augmented prompt with grounding instructions.
      4. A tight-budget scenario showing chunk truncation.
    """
    source_path = chunks_file or DEFAULT_CHUNKS_FILE
    out_path = output_file or DEFAULT_OUTPUT

    sample_chunks = _load_sample_chunks(source_path)
    if not sample_chunks:
        raise ValueError("No sample chunks available for demonstration.")

    # ----- Scenario 1: Normal budget (5 000 tokens) -----
    print("\n" + "=" * 80)
    print("FIELDSURE AI — CONTEXT INJECTION & PROMPT AUGMENTATION DEMO")
    print("=" * 80)

    result_normal = build_augmented_prompt(
        question=question,
        retrieved_chunks=sample_chunks,
        max_context_tokens=max_context_tokens,
    )

    print(f"\n[Scenario 1] Normal Budget ({max_context_tokens} tokens)")
    print(f"  Chunks included: {result_normal.chunks_included} / {result_normal.chunks_total}")
    print(f"  Context tokens:  {result_normal.context_tokens}")
    print(f"  Total prompt tokens: {result_normal.total_prompt_tokens}")
    print(f"  Sources used:    {len(result_normal.sources_used)}")
    print("-" * 80)
    print(result_normal.prompt[:600] + "\n  ... (truncated for display)")

    # ----- Scenario 2: Tight budget (200 tokens) to show truncation -----
    tight_budget = 200
    result_tight = build_augmented_prompt(
        question=question,
        retrieved_chunks=sample_chunks,
        max_context_tokens=tight_budget,
    )

    print(f"\n[Scenario 2] Tight Budget ({tight_budget} tokens)")
    print(f"  Chunks included: {result_tight.chunks_included} / {result_tight.chunks_total}")
    print(f"  Context tokens:  {result_tight.context_tokens}")
    print(f"  Total prompt tokens: {result_tight.total_prompt_tokens}")
    print("-" * 80)

    # ----- Scenario 3: Dynamic budget from total window -----
    window_size = 8000
    result_dynamic = build_augmented_prompt(
        question=question,
        retrieved_chunks=sample_chunks,
        total_window_tokens=window_size,
    )

    dynamic_budget = compute_context_budget(total_window_tokens=window_size)
    print(f"\n[Scenario 3] Dynamic Budget (window={window_size}, computed budget={dynamic_budget} tokens)")
    print(f"  Chunks included: {result_dynamic.chunks_included} / {result_dynamic.chunks_total}")
    print(f"  Context tokens:  {result_dynamic.context_tokens}")
    print(f"  Total prompt tokens: {result_dynamic.total_prompt_tokens}")
    print("=" * 80 + "\n")

    # ----- Write output report -----
    _write_sample_report(out_path, question, sample_chunks, result_normal, result_tight, result_dynamic, tight_budget, window_size)
    print(f"Sample report saved to: {out_path}\n")

    return {
        "normal": {
            "prompt": result_normal.prompt,
            "context_tokens": result_normal.context_tokens,
            "total_prompt_tokens": result_normal.total_prompt_tokens,
            "chunks_included": result_normal.chunks_included,
            "sources": result_normal.sources_used,
        },
        "tight": {
            "prompt": result_tight.prompt,
            "context_tokens": result_tight.context_tokens,
            "total_prompt_tokens": result_tight.total_prompt_tokens,
            "chunks_included": result_tight.chunks_included,
            "sources": result_tight.sources_used,
        },
        "dynamic": {
            "prompt": result_dynamic.prompt,
            "context_tokens": result_dynamic.context_tokens,
            "total_prompt_tokens": result_dynamic.total_prompt_tokens,
            "chunks_included": result_dynamic.chunks_included,
            "sources": result_dynamic.sources_used,
        },
    }


def _write_sample_report(
    path: Path,
    question: str,
    chunks: List[Dict[str, Any]],
    result_normal: AugmentedPrompt,
    result_tight: AugmentedPrompt,
    result_dynamic: AugmentedPrompt,
    tight_budget: int,
    window_size: int,
) -> None:
    """Write the demonstration output as a Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Format source table rows
    source_rows = ""
    for idx, src in enumerate(result_normal.sources_used, start=1):
        source_rows += (
            f"| {idx} | `{src.get('source_file', 'N/A')}` | "
            f"`{src.get('section_heading', 'N/A')}` | "
            f"{src.get('token_count', 'N/A')} |\n"
        )

    md = f"""# Context Injection & Prompt Augmentation — Execution Report

> **Module**: Module #29 — Context Injection & Prompt Augmentation
> **Status**: Verified Execution
> **Question**: {question}

---

## 1. Task Summary

| Task | Description | Status |
| :--- | :--- | :---: |
| **Task 1** | Inject retrieved chunks into the prompt as model context | ✅ |
| **Task 2** | Enforce token budget, leaving room for instructions + answer | ✅ |
| **Task 3** | Add source markers ([1], [2], ...) for citation auditing | ✅ |
| **Task 4** | Grounding instructions: answer only from context, refuse when insufficient | ✅ |
| **Task 5** | Sample augmented prompt with all features demonstrated | ✅ |

---

## 2. Source Markers & Chunk Formatting

Each chunk is formatted with a numbered marker containing the source file path,
chunk index, and section heading so the model can cite evidence precisely.

| # | Source File | Section | Tokens |
| :---: | :--- | :--- | :---: |
{source_rows}

---

## 3. Scenario 1 — Normal Budget ({MAX_CONTEXT_TOKENS} tokens)

- **Chunks Included**: {result_normal.chunks_included} / {result_normal.chunks_total}
- **Context Tokens**: {result_normal.context_tokens}
- **Total Prompt Tokens**: {result_normal.total_prompt_tokens}

### Full Augmented Prompt

```text
{result_normal.prompt}
```

---

## 4. Scenario 2 — Tight Budget ({tight_budget} tokens)

Demonstrates that when the token budget is restrictive, only the highest-ranked
chunks are included and lower-ranked ones are skipped.

- **Chunks Included**: {result_tight.chunks_included} / {result_tight.chunks_total}
- **Context Tokens**: {result_tight.context_tokens}
- **Total Prompt Tokens**: {result_tight.total_prompt_tokens}

### Truncated Augmented Prompt

```text
{result_tight.prompt}
```

---

## 5. Scenario 3 — Dynamic Budget (Window = {window_size} tokens)

Computed budget = {window_size} − {ANSWER_RESERVE_TOKENS} (answer) − {INSTRUCTION_RESERVE_TOKENS} (instructions) = {compute_context_budget(total_window_tokens=window_size)} tokens.

- **Chunks Included**: {result_dynamic.chunks_included} / {result_dynamic.chunks_total}
- **Context Tokens**: {result_dynamic.context_tokens}
- **Total Prompt Tokens**: {result_dynamic.total_prompt_tokens}

---

## 6. Grounding Instruction Text

The following instruction is injected at the top of every augmented prompt:

```text
{GROUNDING_INSTRUCTION}
```

This ensures the model:
1. Answers strictly from the provided context.
2. Cites source markers like [1], [2].
3. Refuses when the context is insufficient instead of hallucinating.

---

## 7. Returned Source Metadata

```json
{json.dumps(result_normal.sources_used, indent=2)}
```
"""
    path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the context injection demo from the command line."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="What are the warning signs and immediate field interventions for dengue?",
        help="Sample question to build the augmented prompt for.",
    )
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=DEFAULT_CHUNKS_FILE,
        help="Path to ingested_chunks.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to save the sample report.",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=MAX_CONTEXT_TOKENS,
        help="Maximum tokens for the context block.",
    )

    args = parser.parse_args()

    try:
        run_context_injection_demo(
            question=args.question,
            chunks_file=args.chunks_file,
            output_file=args.output,
            max_context_tokens=args.max_context_tokens,
        )
    except Exception as err:
        print(f"Context injection demo failed: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
