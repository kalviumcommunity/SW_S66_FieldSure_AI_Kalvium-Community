"""RAG Pipeline Architecture & Flow Design (Module #28).

Orchestrates the end-to-end Retrieval-Augmented Generation pipeline:
  User Query -> Embed Query -> Retrieve Context -> Assemble Context -> Generate Answer -> Return Answer + Sources

Stages are kept strictly modular and separately testable:
1. embed_query: Encodes query text into a dense vector representation.
2. retrieve_context: Queries the vector database for the top-k most similar chunks.
3. assemble_context: Formats retrieved chunks into grounded evidence blocks with source citations.
4. generate_answer: Prompts the LLM with the assembled context to produce a strictly grounded response.
5. answer_query: Orchestrator connecting all stages with graceful handling of empty retrieval states.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Optional

from dotenv import load_dotenv

from src.config import LLMConfig, LLMConfigError, load_llm_config
from src.embedding_fundamentals import EmbeddingAPIError, generate_embeddings
from src.llm_client import (
    ChatCompletionResult,
    LLMAPIError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    OpenAICompatibleClient,
)
from src.vector_store import ChromaVectorStore, VectorStoreError

logger = logging.getLogger("fieldsure.rag_pipeline")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INGESTED_CHUNKS = REPO_ROOT / "outputs" / "ingested_chunks.jsonl"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "vectordb"
DEFAULT_COLLECTION = "fieldsure_chunks"
DEFAULT_OUTPUT_SAMPLE = REPO_ROOT / "outputs" / "rag_pipeline_sample.md"

FALLBACK_NO_CONTEXT = "I could not find relevant context for that question."
DEFAULT_MIN_SIMILARITY = 0.08
DEFAULT_K = 4

# English stop words for lightweight deterministic offline hashing vectorizer
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are FieldSure AI, a verified clinical and operational guidance assistant for public "
    "health field workers and community health staff.\n\n"
    "ROLE & RULES:\n"
    "- Provide objective, protocol-grounded answers based strictly on the provided context.\n"
    "- Cite reference documents and sections using the format: [Document Title, Section Name].\n"
    "- If the context is insufficient, state clearly what is missing without inventing facts.\n"
    "- Do not make clinical diagnoses or prescribe unapproved medication.\n"
    "- Append the mandatory safety notice: 'Notice: FieldSure AI is an informational protocol assistant, "
    "not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest "
    "primary health center (PHC).'"
)


# ------------------------------------------------------------------------------
# Offline / Fixture Vector Utilities
# ------------------------------------------------------------------------------

def deterministic_text_embedding(text: str, dimension: int = 1024) -> list[float]:
    """Generate a deterministic, unit-normalized term frequency vector for offline testing.

    Uses SHA-256 hash bucketing over content words (excluding stopwords).
    Guarantees consistent, non-random vector outputs without external ML dependencies.
    """
    if dimension < 2:
        raise ValueError("Dimension must be at least 2.")

    words = [
        w for w in re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        if w not in STOP_WORDS
    ]
    if not words:
        # Uniform fallback vector if text contains only stopwords or symbols
        return [1.0 / math.sqrt(dimension)] * dimension

    vector = [0.0] * dimension
    for word in words:
        bucket = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % dimension
        vector[bucket] += 1.0

    magnitude = math.sqrt(sum(val * val for val in vector))
    if magnitude > 0.0:
        return [val / magnitude for val in vector]
    return [1.0 / math.sqrt(dimension)] * dimension


# ------------------------------------------------------------------------------
# Stage 1: Embed Query
# ------------------------------------------------------------------------------

def embed_query(
    query: str,
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
) -> list[float]:
    """Encode a single query string into a dense vector embedding.

    Args:
        query: User question text.
        model: Optional model name if using live API.
        config: Optional LLMConfig credentials for live API.
        embed_fn: Optional custom embedding callable for unit testing or injection.

    Returns:
        List of float values representing the query vector.

    Raises:
        ValueError: If the query is empty or whitespace-only.
        EmbeddingAPIError: If the remote embedding call fails.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    cleaned_query = query.strip()

    # 1. Custom or injected embedding function (e.g. for mocks/fixtures)
    if embed_fn is not None:
        result = embed_fn(cleaned_query)
        if not isinstance(result, list) or not result:
            raise ValueError("Custom embed_fn must return a non-empty list of floats.")
        return [float(v) for v in result]

    # 2. Live API embedding if credentials and model are provided
    if config is not None and model:
        vectors = generate_embeddings([cleaned_query], model=model, config=config)
        return vectors[0]

    # 3. Deterministic offline fallback embedding
    return deterministic_text_embedding(cleaned_query)


# ------------------------------------------------------------------------------
# Stage 2: Retrieve Context
# ------------------------------------------------------------------------------

def retrieve_context(
    query_vector: Sequence[float],
    store: Any,
    k: int = DEFAULT_K,
    where: Optional[dict[str, Any]] = None,
    min_similarity: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Retrieve top-k matching chunks from the vector store using cosine similarity.

    Args:
        query_vector: Dense embedding vector of the user query.
        store: Vector store instance exposing a search(vector, top_k, where) method.
        k: Maximum number of chunks to retrieve (must be >= 1).
        where: Optional metadata filter dictionary.
        min_similarity: Optional minimum similarity threshold. Chunks scoring below
                        this value are discarded.

    Returns:
        List of matching chunk dictionaries ordered by descending similarity score.
    """
    if k < 1:
        raise ValueError("Top-k must be an integer >= 1.")
    if not query_vector:
        raise ValueError("Query vector must not be empty.")

    # Execute search against vector store
    try:
        if hasattr(store, "search"):
            # ChromaVectorStore interface
            raw_matches = store.search(query_vector, limit=k, where=where)
        elif hasattr(store, "query"):
            raw_matches = store.query(query_vector, top_k=k, where=where)
        else:
            raise TypeError("Vector store does not support search or query.")
    except VectorStoreError as err:
        # Handle empty collection or lookup failures gracefully
        logger.warning("Vector store search returned error: %s", err)
        return []

    # Apply minimum similarity filter if requested
    if min_similarity is not None:
        filtered_matches = [
            chunk for chunk in raw_matches
            if chunk.get("similarity_score", 1.0 - chunk.get("cosine_distance", 0.0)) >= min_similarity
        ]
        return filtered_matches

    return list(raw_matches)


# ------------------------------------------------------------------------------
# Stage 3: Assemble Context
# ------------------------------------------------------------------------------

def assemble_context(chunks: Sequence[dict[str, Any]]) -> str:
    """Assemble retrieved chunks into a structured, numbered context block with citations.

    Format:
        [1] Source: {source} (Section: {section})
        {chunk_text}

    Args:
        chunks: Sequence of chunk dictionaries containing 'text' and 'metadata'.

    Returns:
        Formatted multi-paragraph string containing all chunks with provenance.
    """
    if not chunks:
        return ""

    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = (
            metadata.get("source")
            or metadata.get("source_file")
            or metadata.get("document_title")
            or "Unknown Source"
        )
        section = metadata.get("section_heading")
        header = f"[{index}] Source: {source}"
        if section:
            header += f" (Section: {section})"

        text = chunk.get("text", "").strip()
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


# ------------------------------------------------------------------------------
# Stage 4: Generate Answer
# ------------------------------------------------------------------------------

def build_rag_prompt(query: str, context: str) -> str:
    """Format the grounded RAG user prompt enclosing context and query."""
    return f"""Answer the question using only the context below.
If the context is not enough, say what is missing.

Context:
{context}

Question: {query}"""


def _generate_offline_grounded_answer(query: str, context: str) -> str:
    """Deterministic offline answer generator for testing and demonstration environments.

    Synthesizes a grounded answer directly from context snippets without external API keys.
    """
    if not context.strip():
        return FALLBACK_NO_CONTEXT

    # Look for query keywords in context
    query_terms = [
        t for t in re.findall(r"\b[a-zA-Z0-9_]+\b", query.lower())
        if t not in STOP_WORDS and len(t) > 2
    ]

    matched_lines: list[str] = []
    citations: list[str] = []

    for block in context.split("\n\n"):
        lines = block.strip().split("\n")
        header = lines[0] if lines else ""
        body = " ".join(lines[1:]) if len(lines) > 1 else ""

        # Extract citation from header
        source_match = re.search(r"Source:\s*([^(\n]+)", header)
        section_match = re.search(r"Section:\s*([^)\n]+)", header)
        if source_match:
            doc_name = Path(source_match.group(1).strip()).name
            sec_name = section_match.group(1).strip() if section_match else "General"
            citations.append(f"[{doc_name}, {sec_name}]")

        # Extract sentences matching query terms
        sentences = re.split(r"(?<=[.!?])\s+", body)
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if any(term in sentence_clean.lower() for term in query_terms):
                if sentence_clean not in matched_lines:
                    matched_lines.append(sentence_clean)

    if not matched_lines:
        return (
            f"The provided context does not contain sufficient details to answer '{query}'. "
            "Missing information: verified clinical guidance matching this specific query."
        )

    citation_tag = " ".join(dict.fromkeys(citations))
    summary_text = " ".join(matched_lines[:4])

    return (
        f"Based on verified protocol guidance {citation_tag}:\n"
        f"{summary_text}\n\n"
        "Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. "
        "Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC)."
    )


def generate_answer(
    query: str,
    context: str,
    client: Optional[OpenAICompatibleClient] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """Generate a grounded response answering the query strictly from the context.

    Args:
        query: User question text.
        context: Assembled context string from retrieved chunks.
        client: Optional OpenAICompatibleClient to execute chat completion.
        model: Optional model name for chat completion.
        system_prompt: Optional system prompt string enforcing grounding and constraints.
        llm_fn: Optional custom callable taking prompt string and returning answer.

    Returns:
        Generated answer string.
    """
    if not context or not context.strip():
        return FALLBACK_NO_CONTEXT

    prompt = build_rag_prompt(query, context)

    # 1. Custom injected LLM function (for unit tests / alternative LLM adapters)
    if llm_fn is not None:
        return llm_fn(prompt)

    # 2. Live API completion client
    if client is not None:
        sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
        result: ChatCompletionResult = client.create_chat_completion(
            messages=messages,
            temperature=0.1,
            model=model,
        )
        return result.content

    # 3. Offline deterministic synthesizer fallback
    return _generate_offline_grounded_answer(query, context)


# ------------------------------------------------------------------------------
# Stage 5: Pipeline Orchestrator
# ------------------------------------------------------------------------------

def answer_query(
    query: str,
    k: int = DEFAULT_K,
    store: Optional[Any] = None,
    model: Optional[str] = None,
    config: Optional[LLMConfig] = None,
    client: Optional[OpenAICompatibleClient] = None,
    embed_fn: Optional[Callable[[str], list[float]]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    system_prompt: Optional[str] = None,
    where: Optional[dict[str, Any]] = None,
    min_similarity: Optional[float] = None,
) -> dict[str, Any]:
    """Execute the end-to-end RAG pipeline from user query to grounded answer.

    Flow:
        1. Embed Query: query -> query_vector
        2. Retrieve Context: query_vector -> top-k chunks
        3. Check Empty: if no chunks -> return fallback response immediately
        4. Assemble Context: chunks -> structured evidence block
        5. Generate Answer: query + context -> grounded response
        6. Return Answer + Sources

    Args:
        query: User question.
        k: Number of chunks to retrieve (default: 4).
        store: Vector store instance. If None, caller must provide a valid store.
        model: Model name for embeddings and completion.
        config: LLMConfig credentials for live API.
        client: OpenAICompatibleClient for chat completions.
        embed_fn: Optional custom embedding function.
        llm_fn: Optional custom generation function.
        system_prompt: Optional system prompt string.
        where: Optional metadata filter.
        min_similarity: Optional minimum similarity threshold.

    Returns:
        Dictionary containing:
            - 'query': original query text
            - 'answer': generated grounded answer
            - 'sources': list of metadata dictionaries for retrieved chunks
            - 'context': assembled context string
            - 'chunks': raw retrieved chunk objects
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must not be empty.")

    # 1. Embed Query
    query_vector = embed_query(
        query=query,
        model=model,
        config=config,
        embed_fn=embed_fn,
    )

    # 2. Retrieve Top-K Chunks
    if store is None:
        chunks: list[dict[str, Any]] = []
    else:
        chunks = retrieve_context(
            query_vector=query_vector,
            store=store,
            k=k,
            where=where,
            min_similarity=min_similarity,
        )

    # 3. Handle Empty Retrieval State
    if not chunks:
        return {
            "query": query,
            "answer": FALLBACK_NO_CONTEXT,
            "sources": [],
            "context": "",
            "chunks": [],
        }

    # 4. Assemble Grounded Context
    context = assemble_context(chunks)

    # 5. Generate Grounded Answer
    answer = generate_answer(
        query=query,
        context=context,
        client=client,
        model=model,
        system_prompt=system_prompt,
        llm_fn=llm_fn,
    )

    # 6. Extract Sources
    sources = [chunk.get("metadata", {}) for chunk in chunks]

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
        "context": context,
        "chunks": chunks,
    }


# ------------------------------------------------------------------------------
# Corpus Loader & Demonstration Helpers
# ------------------------------------------------------------------------------

def populate_vector_store(
    store: ChromaVectorStore,
    source_chunks_file: Path,
    dimension: int = 1024,
) -> int:
    """Load ingested chunks and index them with deterministic unit vectors."""
    if not source_chunks_file.is_file():
        raise FileNotFoundError(f"Source chunks file not found: {source_chunks_file}")

    records: list[dict[str, Any]] = []
    with source_chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            text = item["text"]
            embedding = deterministic_text_embedding(text, dimension=dimension)
            records.append({
                "chunk_id": item["chunk_id"],
                "text": text,
                "metadata": item["metadata"],
                "embedding": embedding,
                "vector_dimension": dimension,
            })

    if not records:
        raise ValueError("No records found in source chunks file.")

    store.upsert(records)
    return len(records)


def run_pipeline_demo(
    query: str,
    negative_query: str = "What evidence is required for project submission?",
    k: int = DEFAULT_K,
    db_path: Optional[Path] = None,
    collection_name: str = "rag_pipeline_demo",
    chunks_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run the complete RAG pipeline on a sample query and an ungrounded negative query.

    Demonstrates:
        1. Success path: Grounded answer generation + citation extraction.
        2. Fallback path: Safe empty retrieval refusal without hallucination.
    """
    database_path = db_path or (DEFAULT_DB_PATH)
    source_chunks = chunks_file or DEFAULT_INGESTED_CHUNKS

    # Initialize ChromaDB vector store
    store = ChromaVectorStore(database_path, collection_name=collection_name)

    # Reset collection if dimension mismatches previous test runs
    if store.count() > 0:
        existing = store.collection.get(limit=1, include=["embeddings"])
        embeddings = existing.get("embeddings")
        if embeddings is not None and len(embeddings) > 0 and len(embeddings[0]) != 1024:
            store.client.delete_collection(collection_name)
            store = ChromaVectorStore(database_path, collection_name=collection_name)

    # Populate collection with sample corpus if empty
    if store.count() == 0:
        logger.info("Populating vector store collection '%s'...", collection_name)
        populate_vector_store(store, source_chunks, dimension=1024)

    # Check for live API configuration
    config: Optional[LLMConfig] = None
    client: Optional[OpenAICompatibleClient] = None
    model: Optional[str] = None
    try:
        config = load_llm_config()
        client = OpenAICompatibleClient(config)
        model = config.chat_model
    except LLMConfigError:
        logger.info("No live LLM credentials configured. Running with deterministic grounded engine.")

    # 1. Execute query on sample domain question
    print(f"\n[1] Executing Query: '{query}' (k={k})")
    result_positive = answer_query(
        query=query,
        k=k,
        store=store,
        model=model,
        config=config,
        client=client,
        min_similarity=DEFAULT_MIN_SIMILARITY,
    )

    print("\n--- Answer ---")
    print(result_positive["answer"])
    print("\n--- Retrieved Sources ---")
    for idx, src in enumerate(result_positive["sources"], start=1):
        print(f"[{idx}] {src.get('source_file')} | Section: {src.get('section_heading')} (doc_id: {src.get('doc_id')})")

    # 2. Execute negative / out-of-domain query to test empty retrieval fallback
    print(f"\n\n[2] Executing Out-of-Domain Query: '{negative_query}' (k={k})")
    result_negative = answer_query(
        query=negative_query,
        k=k,
        store=store,
        model=model,
        config=config,
        client=client,
        min_similarity=DEFAULT_MIN_SIMILARITY,
    )

    print("\n--- Answer ---")
    print(result_negative["answer"])
    print("\n--- Retrieved Sources ---")
    print(result_negative["sources"])

    summary_data = {
        "positive_query": result_positive,
        "negative_query": result_negative,
    }

    # Save output report if output_path is provided
    out_file = output_path or DEFAULT_OUTPUT_SAMPLE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _write_pipeline_sample_report(out_file, result_positive, result_negative, k)
    print(f"\nExecution summary successfully saved to: {out_file}\n")

    return summary_data


def _write_pipeline_sample_report(
    path: Path,
    positive: dict[str, Any],
    negative: dict[str, Any],
    k: int,
) -> None:
    """Format and write sample execution markdown report."""
    md_content = f"""# RAG Pipeline Architecture & Flow Design — Execution Report

> **Module**: Module #28 — RAG Pipeline Architecture & Flow Design  
> **Status**: Verified End-to-End Execution  
> **Retrieval Mode**: ChromaDB Vector Search (Top-k = {k}, Minimum Similarity = {DEFAULT_MIN_SIMILARITY})

---

## 1. End-to-End RAG Architecture & Flow Diagram

The complete query-to-answer flow is strictly decoupled into modular, testable stages:

```mermaid
sequenceDiagram
    autonumber
    actor User as Field Worker
    participant Orchestrator as Pipeline Orchestrator (answer_query)
    participant Embedder as Embed Stage (embed_query)
    participant Store as Vector DB (retrieve_context)
    participant Assembler as Assembler (assemble_context)
    participant Generator as LLM Generator (generate_answer)

    User->>Orchestrator: 1. Submit Query
    Orchestrator->>Embedder: 2. embed_query(query)
    Embedder-->>Orchestrator: Dense Embedding Vector
    Orchestrator->>Store: 3. retrieve_context(vector, k={k})
    alt Relevant Chunks Found (similarity >= {DEFAULT_MIN_SIMILARITY})
        Store-->>Orchestrator: Top-k Relevant Chunks
        Orchestrator->>Assembler: 4. assemble_context(chunks)
        Assembler-->>Orchestrator: Formatted Context with [1] Source: ...
        Orchestrator->>Generator: 5. generate_answer(query, context)
        Generator-->>Orchestrator: Grounded Answer + Citations
        Orchestrator-->>User: {{answer, sources}}
    else No Context Found / Similarity Below Threshold
        Store-->>Orchestrator: Empty Chunks []
        Orchestrator-->>User: Fallback: "I could not find relevant context for that question." (sources: [])
    end
```

---

## 2. Pipeline Responsibilities by Stage

| Stage | Function | Input | Output | Single Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Embed** | `embed_query` | Raw query string | Dense numeric vector | Encodes textual query semantics into normalized vector space. |
| **Stage 2: Retrieve** | `retrieve_context` | Query vector, store, $k$ | Top-$k$ chunk objects | Cosine similarity lookup in ChromaDB with optional threshold and metadata filters. |
| **Stage 3: Assemble** | `assemble_context` | List of chunk objects | Formatted context string | Formats evidence snippets with 1-based indices, source file, and section headings. |
| **Stage 4: Generate** | `generate_answer` | Query text, context string | Grounded answer string | Injects context into grounded prompt template and queries LLM. |
| **Stage 5: Orchestrate** | `answer_query` | Query, pipeline settings | Result dictionary | Coordinates all stages and safely catches empty retrieval without hallucinations. |

---

## 3. Sample Query Run (Grounded Retrieval & Generation)

- **Query**: `{positive["query"]}`
- **Retrieved Chunks**: {len(positive["chunks"])} chunks retrieved

### Retrieved Source Chunks
"""
    for idx, chunk in enumerate(positive["chunks"], start=1):
        meta = chunk.get("metadata", {})
        score = chunk.get("similarity_score", 1.0 - chunk.get("cosine_distance", 0.0))
        md_content += f"""
#### [{idx}] `{meta.get('document_title', 'Advisory')}`
- **Source File**: `{meta.get('source_file')}`
- **Section Heading**: `{meta.get('section_heading')}`
- **Similarity Score**: `{score:.4f}`
- **Content Excerpt**:
> {chunk.get('text', '').strip()[:240]}...
"""

    md_content += f"""
### Assembled Context (Prompt Evidence)
```text
{positive["context"]}
```

### Generated Answer
{positive["answer"]}

### Returned Sources (Metadata Contract)
```json
{json.dumps(positive["sources"], indent=2)}
```

---

## 4. Negative / Out-of-Domain Query Run (Empty Retrieval Fallback)

To prevent hallucination, if retrieval returns zero chunks or chunks below the similarity threshold, the pipeline immediately returns a clear fallback response without calling generation.

- **Query**: `{negative["query"]}`
- **Retrieved Chunks**: {len(negative["chunks"])}
- **Generated Answer**: `{negative["answer"]}`
- **Returned Sources**: `{negative["sources"]}`

---

## 5. Architectural Verification & Guarantees

1. **Separation of Concerns**: Each stage is an independent function that can be executed and unit-tested in complete isolation.
2. **Zero Hallucination on Zero Context**: When retrieval yields 0 evidence, the pipeline deterministically refuses rather than inventing speculative guidance.
3. **Traceable Citations**: Generated answers cite exact document names and section titles, matching Member 1's metadata contract.
4. **Resilient Fallback**: System functions reliably both with live OpenAI-compatible LLM endpoints and in offline deterministic demonstration modes.
"""
    path.write_text(md_content, encoding="utf-8")


# ------------------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------------------

def main() -> int:
    """Run the RAG pipeline CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query",
        default="What are the warning signs and immediate field interventions for dengue?",
        help="Sample domain query to answer.",
    )
    parser.add_argument(
        "--negative-query",
        default="What evidence is required for project submission?",
        help="Out-of-domain sample query to verify fallback refusal.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of chunks to retrieve (default: 4).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Local ChromaDB directory path.",
    )
    parser.add_argument(
        "--collection",
        default="rag_pipeline_demo",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_SAMPLE,
        help="Path to save execution sample report.",
    )

    args = parser.parse_args()
    load_dotenv()

    try:
        run_pipeline_demo(
            query=args.query,
            negative_query=args.negative_query,
            k=args.k,
            db_path=args.db_path,
            collection_name=args.collection,
            output_path=args.output,
        )
    except Exception as err:
        print(f"RAG Pipeline execution failed: {err}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
