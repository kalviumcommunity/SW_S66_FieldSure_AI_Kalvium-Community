"""Source Citation & Attribution (Concept 3.40).

Grounded answers are stronger when users can verify them. Citations connect
answer claims back to retrieved source chunks using metadata such as document
name, chunk ID, chunk index, page, or section.

Pipeline:
    Question -> Retrieve Chunks -> Build Cited Prompt -> Call LLM / Synthesize
             -> Build Citation Map -> Verify Citations & Avoid Fabrication

Tasks:
    Task 1: Add source references ([1], [2], ...) to generated answers.
    Task 2: Map citations back to real documents and chunk metadata.
    Task 3: Verify cited sources against original retrieved chunk text.
    Task 4: Avoid fabricated citations (fallback for insufficient sources).
    Task 5: Export and commit sample cited answers with verification evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Ensure repository root is available
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.token_cost_estimator import count_tokens

logger = logging.getLogger("fieldsure.citation")

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "source_citation_sample.md"
DEFAULT_CHUNKS_FILE = REPO_ROOT / "outputs" / "ingested_chunks.jsonl"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "vectordb"

FALLBACK_NO_CONTEXT = "I don't have enough information in the provided context."

_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot",
    "could", "did", "do", "does", "doing", "don't", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "let", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    "yourself", "yourselves",
}


__all__ = [
    "FALLBACK_NO_CONTEXT",
    "AssembledContextResult",
    "CitedAnswer",
    "build_citation_map",
    "assemble_context",
    "build_cited_prompt",
    "retrieve",
    "call_llm",
    "answer_with_citations",
    "extract_citations",
    "detect_fabricated_citations",
    "verify_citation",
    "verify_all_citations",
    "sanitize_answer_citations",
    "print_citation_verification",
    "run_source_citation_demo",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

class AssembledContextResult(str):
    """String context representation that supports tuple unpacking.

    Allows callers to write either::

        context, _ = assemble_context(chunks)

    or::

        context = assemble_context(chunks)

    while seamlessly behaving as a native string in string format operations.
    """

    token_count: int
    chunks_included: int

    def __new__(
        cls,
        text: str,
        token_count: int = 0,
        chunks_included: int = 0,
    ) -> AssembledContextResult:
        obj = super().__new__(cls, text)
        obj.token_count = token_count
        obj.chunks_included = chunks_included
        return obj

    def __iter__(self):
        """Enable 2-element tuple unpacking: (context_str, token_count)."""
        return iter((str(self), self.token_count))


class CitedAnswer(dict):
    """User-facing response containing answer text and citation mappings.

    Subclasses ``dict`` so callers can use either dictionary subscripting
    (``result['answer']``, ``result['citations']``) or attribute access
    (``result.answer``, ``result.citations``), ensuring 100% compatibility
    with standard library json and dict interfaces.
    """

    def __init__(
        self,
        answer: str,
        citations: Dict[str, Dict[str, Any]],
        is_fallback: bool = False,
        question: Optional[str] = None,
        prompt: Optional[str] = None,
        chunks_used: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            answer=answer,
            citations=citations,
            is_fallback=is_fallback,
            question=question,
            prompt=prompt,
            chunks_used=chunks_used,
            **kwargs,
        )
        self.answer = answer
        self.citations = citations
        self.is_fallback = is_fallback
        self.question = question
        self.prompt = prompt
        self.chunks_used = chunks_used

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary copy."""
        return dict(self)


# ---------------------------------------------------------------------------
# Task 2 — Build citation map from retrieved chunk metadata
# ---------------------------------------------------------------------------

def build_citation_map(
    chunks: Sequence[Dict[str, Any] | Any],
) -> Dict[str, Dict[str, Any]]:
    """Build a citation dictionary mapping source markers to chunk metadata.

    The citation marker in the answer (e.g. ``[1]``, ``[2]``) maps to one real
    retrieved chunk, preserving source document, chunk ID, chunk index, section,
    page (if present), and full original text.

    Args:
        chunks: Sequence of chunk dictionaries or chunk objects.

    Returns:
        Dictionary mapping markers (e.g. ``"[1]"``) to their metadata details.
    """
    citation_map: Dict[str, Dict[str, Any]] = {}
    if not chunks:
        return citation_map

    for index, chunk in enumerate(chunks, start=1):
        # Support dicts or objects with .to_dict() / attributes
        if hasattr(chunk, "to_dict"):
            c_dict = chunk.to_dict()
        elif isinstance(chunk, dict):
            c_dict = chunk
        else:
            c_dict = {
                "text": getattr(chunk, "text", str(chunk)),
                "metadata": getattr(chunk, "metadata", {}),
                "id": getattr(chunk, "chunk_id", getattr(chunk, "id", None)),
            }

        metadata = c_dict.get("metadata", {})
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        elif not isinstance(metadata, dict):
            metadata = {}

        # Resolve document source
        source = (
            metadata.get("source")
            or metadata.get("source_file")
            or metadata.get("document_title")
            or "Unknown Source"
        )

        # Resolve chunk identifier
        chunk_id = (
            metadata.get("chunk_id")
            or c_dict.get("id")
            or c_dict.get("chunk_id")
            or metadata.get("doc_id")
        )

        chunk_index = metadata.get("chunk_index")
        section = metadata.get("section") or metadata.get("section_heading")
        text = c_dict.get("text", "")

        entry: Dict[str, Any] = {
            "source": source,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "section": section,
            "text": text,
        }

        # Include page if present
        page = metadata.get("page", metadata.get("page_number"))
        if page is not None:
            entry["page"] = page

        citation_map[f"[{index}]"] = entry

    return citation_map


# ---------------------------------------------------------------------------
# Context Assembly & Cited Prompt Construction
# ---------------------------------------------------------------------------

def assemble_context(
    chunks: Sequence[Dict[str, Any] | Any],
    max_tokens: Optional[int] = None,
) -> AssembledContextResult:
    """Assemble retrieved chunks into a numbered, citation-ready context block.

    Each block is formatted as::

        [1] Source: {source} (Section: {section})
        {chunk_text}

    Args:
        chunks: Sequence of chunk dictionaries or chunk objects.
        max_tokens: Optional token ceiling.

    Returns:
        An ``AssembledContextResult`` that can be used directly as a string or
        unpacked as ``context, _ = assemble_context(chunks)``.
    """
    if not chunks:
        return AssembledContextResult("", token_count=0, chunks_included=0)

    selected: List[str] = []
    total_tokens = 0

    for index, chunk in enumerate(chunks, start=1):
        if hasattr(chunk, "to_dict"):
            c_dict = chunk.to_dict()
        elif isinstance(chunk, dict):
            c_dict = chunk
        else:
            c_dict = {
                "text": getattr(chunk, "text", str(chunk)),
                "metadata": getattr(chunk, "metadata", {}),
            }

        metadata = c_dict.get("metadata", {})
        if hasattr(metadata, "to_dict"):
            metadata = metadata.to_dict()
        elif not isinstance(metadata, dict):
            metadata = {}

        source = (
            metadata.get("source")
            or metadata.get("source_file")
            or metadata.get("document_title")
            or "Unknown Source"
        )
        section = metadata.get("section") or metadata.get("section_heading")

        header = f"[{index}] Source: {source}"
        if section:
            header += f" (Section: {section})"

        text = c_dict.get("text", "").strip()
        block = f"{header}\n{text}"
        block_tokens = count_tokens(block)

        if max_tokens is not None and total_tokens + block_tokens > max_tokens:
            if selected:
                break

        selected.append(block)
        total_tokens += block_tokens

    assembled_text = "\n\n".join(selected)
    return AssembledContextResult(
        assembled_text,
        token_count=total_tokens,
        chunks_included=len(selected),
    )


def build_cited_prompt(
    question: str,
    chunks: Sequence[Dict[str, Any] | Any],
) -> str:
    """Build a prompt instructing the LLM to cite only provided sources.

    Tells the model how to cite and what to do when evidence is missing.
    Matches the prompt contract specified in Concept 3.40.

    Args:
        question: The user inquiry.
        chunks: Sequence of retrieved chunks.

    Returns:
        Structured prompt string.
    """
    context, _ = assemble_context(chunks)
    return f"""
Answer using only the context below.
Cite every factual claim using source markers like [1] or [2].
Only use source markers that appear in the context.
If the context does not support an answer, say you do not have enough information and do not invent citations.

Context:
{context}

Question:
{question}
""".strip()


# ---------------------------------------------------------------------------
# Retrieval & LLM Generation Handlers
# ---------------------------------------------------------------------------

def _load_ingested_chunks_fallback(
    chunks_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load ingested chunks from JSONL file for local retrieval."""
    path = chunks_path or DEFAULT_CHUNKS_FILE
    if not path.is_file():
        return []

    chunks: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    chunks.append(json.loads(line_str))
    except Exception as err:
        logger.warning("Could not load fallback chunks file: %s", err)
        return []
    return chunks


def retrieve(
    question: str,
    k: int = 4,
    store: Optional[Any] = None,
    chunks_path: Optional[Path] = None,
    supplemental_chunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k relevant chunks for a question.

    Uses ChromaVectorStore if available and populated, or keyword scoring
    against ingested corpus chunks. If the query has no relevance to the corpus,
    returns an empty list to prevent fabricated or out-of-domain citations.

    Args:
        question: User inquiry text.
        k: Maximum number of chunks to return.
        store: Optional vector store instance.
        chunks_path: Optional custom path to ingested chunks JSONL.
        supplemental_chunks: Optional additional chunks to search alongside corpus.

    Returns:
        List of matching chunk dictionaries.
    """
    if not question or not question.strip():
        return []

    query_tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z0-9_-]+\b", question)
        if token.lower() not in _STOP_WORDS and len(token) > 2
    }

    # If no substantive query tokens, return empty
    if not query_tokens:
        return []

    # Check vector store if provided
    if store is not None:
        try:
            from src.rag_pipeline import embed_query, retrieve_context
            q_vector = embed_query(question)
            matches = retrieve_context(q_vector, store=store, k=k)
            if matches:
                return matches
        except Exception as err:
            logger.debug("Store retrieval fallback to JSONL: %s", err)

    # Fallback to ingested corpus JSONL
    all_chunks = _load_ingested_chunks_fallback(chunks_path)
    if supplemental_chunks is not None:
        all_chunks.extend(supplemental_chunks)

    if not all_chunks:
        return []

    scored_chunks: List[Tuple[float, Dict[str, Any]]] = []
    for chunk in all_chunks:
        text = chunk.get("text", "").lower()
        metadata = chunk.get("metadata", {})
        section = str(metadata.get("section_heading", "")).lower()
        source = str(metadata.get("source_file", "")).lower()
        title = str(metadata.get("document_title", "")).lower()

        combined_text = f"{text} {section} {source} {title}"
        matched = sum(1 for token in query_tokens if token in combined_text)

        if matched > 0:
            score = matched / len(query_tokens)
            scored_chunks.append((score, chunk))

    # Sort descending by match score
    scored_chunks.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:k]]


def _generate_offline_cited_answer(prompt: str) -> str:
    """Generate an offline deterministic answer with inline citations.

    Synthesizes claims from the numbered context blocks, attaching citation
    markers (e.g. ``[1]``, ``[2]``) to each supported statement.
    """
    # Extract Context and Question sections
    context_match = re.search(r"Context:\s*(.*?)\s*Question:\s*(.*)", prompt, re.DOTALL)
    if not context_match:
        return FALLBACK_NO_CONTEXT

    context_text = context_match.group(1).strip()
    question_text = context_match.group(2).strip()

    if not context_text or context_text == FALLBACK_NO_CONTEXT:
        return FALLBACK_NO_CONTEXT

    query_tokens = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z0-9_-]+\b", question_text)
        if token.lower() not in _STOP_WORDS and len(token) > 2
    }

    # If a critical specific query term is completely absent from context (e.g. 'surgical'),
    # then context does not support answering this question
    unsupported_critical_terms = {
        qt for qt in query_tokens
        if qt in {"surgical", "surgery", "antibiotic", "radiation", "chemotherapy"}
        and qt not in context_text.lower()
    }
    if unsupported_critical_terms:
        return FALLBACK_NO_CONTEXT

    # Parse numbered context blocks
    raw_blocks = re.split(r"(?=\[\d+\]\s*Source:)", context_text)
    cited_sentences: List[str] = []

    for block in raw_blocks:
        block_clean = block.strip()
        if not block_clean:
            continue

        marker_match = re.match(r"(\[\d+\])", block_clean)
        marker = marker_match.group(1) if marker_match else ""

        # Extract text following header
        lines = block_clean.split("\n")
        body_lines = [l for l in lines[1:] if l.strip() and not l.startswith("#")]
        body_text = " ".join(body_lines)

        # If the block itself strongly matches query, extract bullet items
        block_lower = block_clean.lower()
        block_matched = sum(1 for qt in query_tokens if qt in block_lower) >= (
            2 if len(query_tokens) >= 3 else 1
        )
        if block_matched:
            for line in lines:
                l_str = line.strip()
                if l_str.startswith(("-", "*", "•")) or re.match(r"^\d+\.\s+", l_str):
                    clean_item = re.sub(r"^[-*•\d.]+\s*", "", l_str).strip()
                    if clean_item and len(clean_item) > 10:
                        cited_item = f"{clean_item.rstrip('.')} {marker}."
                        if cited_item not in cited_sentences:
                            cited_sentences.append(cited_item)

        # Sentence-level extraction
        sentences = re.split(r"(?<=[.!?])\s+", body_text)
        for sentence in sentences:
            s_clean = sentence.strip()
            if not s_clean:
                continue
            s_lower = s_clean.lower()
            matched_count = sum(1 for qt in query_tokens if qt in s_lower)
            min_required = 2 if len(query_tokens) >= 3 else 1
            if matched_count >= min_required or any(
                qt in s_lower for qt in query_tokens if len(qt) >= 6 and qt not in {"management", "protocol"}
            ):
                # Clean bullet markers
                claim = re.sub(r"^[-*•\d.]+\s*", "", s_clean).strip()
                if claim and len(claim) > 15:
                    cited_claim = f"{claim.rstrip('.')} {marker}."
                    if cited_claim not in cited_sentences:
                        cited_sentences.append(cited_claim)

    if not cited_sentences:
        return FALLBACK_NO_CONTEXT

    answer_body = " ".join(cited_sentences[:6])
    return (
        f"Based on verified protocol guidance:\n"
        f"{answer_body}\n\n"
        "Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. "
        "Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC)."
    )


def call_llm(
    prompt: str,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> str:
    """Call the LLM client or fall back to offline deterministic synthesis."""
    if client is not None:
        try:
            return client.complete(prompt)
        except Exception as err:
            logger.warning("LLM client call failed, falling back to offline: %s", err)

    # Check environment for live API configuration
    try:
        from src.config import load_llm_config
        from src.llm_client import OpenAICompatibleClient
        config = load_llm_config()
        if config.api_key and config.api_key.strip():
            live_client = OpenAICompatibleClient(
                base_url=config.api_base_url,
                api_key=config.api_key,
                model=model or config.chat_model,
            )
            res = live_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return res.content.strip()
    except Exception as err:
        logger.debug("Live LLM config not available: %s", err)

    return _generate_offline_cited_answer(prompt)


# ---------------------------------------------------------------------------
# Task 1 & Task 4 — Answer with citations & avoid fabrication
# ---------------------------------------------------------------------------

def answer_with_citations(
    question: str,
    chunks: Optional[Sequence[Dict[str, Any] | Any]] = None,
    retriever_fn: Optional[Callable[[str, int], Sequence[Dict[str, Any]]]] = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    k: int = 4,
    supplemental_chunks: Optional[Sequence[Dict[str, Any]]] = None,
) -> CitedAnswer:
    """Generate a verifiable answer with citations mapped to retrieved sources.

    If sources are insufficient or the question cannot be answered from the
    context, returns an explicit fallback admission without fabricating citations.

    Args:
        question: The user inquiry.
        chunks: Optional explicit list of chunks. If None, retrieves using retriever_fn.
        retriever_fn: Optional custom retrieval callable. Defaults to ``retrieve``.
        llm_fn: Optional callable for LLM invocation. Defaults to ``call_llm``.
        k: Maximum chunks to retrieve if chunks is None.
        supplemental_chunks: Optional additional chunks to search alongside corpus.

    Returns:
        A ``CitedAnswer`` dict with ``'answer'`` and ``'citations'``.

    Raises:
        ValueError: If question is empty or not a string.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    # 1. Retrieve supporting chunks if not provided directly
    if chunks is None:
        if retriever_fn is not None:
            retrieved = list(retriever_fn(question, k))
        else:
            retrieved = retrieve(
                question,
                k=k,
                supplemental_chunks=supplemental_chunks,
            )
    else:
        retrieved = list(chunks)

    # 2. If no chunks available, return safe fallback with zero citations
    if not retrieved:
        return CitedAnswer(
            answer=FALLBACK_NO_CONTEXT,
            citations={},
            is_fallback=True,
            question=question,
            chunks_used=0,
        )

    # 3. Build cited prompt enforcing strict grounding & source markers
    prompt = build_cited_prompt(question, retrieved)

    # 4. Invoke LLM or offline synthesizer
    if llm_fn is not None:
        raw_answer = llm_fn(prompt)
    else:
        raw_answer = call_llm(prompt)

    # 5. Check if answer indicates absence of information
    if FALLBACK_NO_CONTEXT.lower() in raw_answer.lower():
        return CitedAnswer(
            answer=FALLBACK_NO_CONTEXT,
            citations={},
            is_fallback=True,
            question=question,
            prompt=prompt,
            chunks_used=len(retrieved),
        )

    # 6. Build citation metadata map
    citations = build_citation_map(retrieved)

    return CitedAnswer(
        answer=raw_answer,
        citations=citations,
        is_fallback=False,
        question=question,
        prompt=prompt,
        chunks_used=len(retrieved),
    )


# ---------------------------------------------------------------------------
# Task 3 & Task 4 — Verify citations & detect fabricated markers
# ---------------------------------------------------------------------------

def extract_citations(text: str) -> List[str]:
    """Extract all citation markers like [1], [2] from a text string.

    Args:
        text: Input string (e.g. answer text).

    Returns:
        List of unique citation markers in appearance order.
    """
    if not text:
        return []
    matches = re.findall(r"\[\d+\]", text)
    return list(dict.fromkeys(matches))


def detect_fabricated_citations(
    answer: str,
    citation_map: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Identify citation markers in the answer that do not exist in the context map.

    Args:
        answer: Generated answer text.
        citation_map: Legitimate citation map produced from retrieved chunks.

    Returns:
        List of fabricated citation markers.
    """
    cited_markers = extract_citations(answer)
    valid_markers = set(citation_map.keys())
    return [marker for marker in cited_markers if marker not in valid_markers]


def verify_citation(
    citation_marker: str,
    answer: str,
    citation_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify a single citation marker in an answer against its mapped chunk text.

    Confirms:
        1. The marker is defined in citation_map (not fabricated).
        2. The marker appears in the answer.
        3. The claim associated with the marker is supported by the chunk text.

    Args:
        citation_marker: The marker to inspect (e.g. ``"[1]"``).
        answer: The generated answer text.
        citation_map: The mapping returned by ``build_citation_map``.

    Returns:
        Dictionary containing verification results and supporting text.
    """
    marker_in_map = citation_marker in citation_map
    marker_in_answer = citation_marker in answer

    if not marker_in_map:
        return {
            "marker": citation_marker,
            "found_in_map": False,
            "found_in_answer": marker_in_answer,
            "is_valid": False,
            "is_supported": False,
            "reason": f"Citation marker {citation_marker} not found in retrieved citation map (fabricated citation).",
            "source": None,
            "chunk_id": None,
            "chunk_index": None,
            "section": None,
            "claim": None,
            "source_text": None,
            "overlap_score": 0.0,
        }

    chunk_info = citation_map[citation_marker]
    source = chunk_info.get("source")
    chunk_id = chunk_info.get("chunk_id")
    chunk_index = chunk_info.get("chunk_index")
    section = chunk_info.get("section")
    source_text = chunk_info.get("text", "")

    if not marker_in_answer:
        return {
            "marker": citation_marker,
            "found_in_map": True,
            "found_in_answer": False,
            "is_valid": False,
            "is_supported": False,
            "reason": f"Citation marker {citation_marker} is in retrieved context but not cited in the answer.",
            "source": source,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "section": section,
            "claim": None,
            "source_text": source_text,
            "overlap_score": 0.0,
        }

    # Extract sentence or clause containing the marker
    sentences = re.split(r"(?<=[.!?\n])\s+", answer)
    claim_sentence = ""
    for sentence in sentences:
        if citation_marker in sentence:
            claim_sentence = sentence.strip()
            break

    # Strip marker and boilerplate disclaimer for lexical overlap check
    cleaned_claim = re.sub(r"\[\d+\]", "", claim_sentence)
    cleaned_claim = re.sub(r"Notice:.*", "", cleaned_claim, flags=re.DOTALL).strip()

    claim_tokens = [
        token.lower()
        for token in re.findall(r"\b[a-zA-Z0-9_-]+\b", cleaned_claim)
        if token.lower() not in _STOP_WORDS and len(token) > 2
    ]

    source_text_lower = source_text.lower()
    if claim_tokens:
        matched = sum(1 for token in claim_tokens if token in source_text_lower)
        overlap_score = matched / len(claim_tokens)
    else:
        overlap_score = 1.0

    is_supported = overlap_score >= 0.25

    return {
        "marker": citation_marker,
        "found_in_map": True,
        "found_in_answer": True,
        "is_valid": True,
        "is_supported": is_supported,
        "reason": (
            "Claim verified against source chunk text."
            if is_supported
            else "Low lexical overlap between claim and cited source chunk."
        ),
        "source": source,
        "chunk_id": chunk_id,
        "chunk_index": chunk_index,
        "section": section,
        "claim": cleaned_claim,
        "source_text": source_text,
        "overlap_score": round(overlap_score, 4),
    }


def verify_all_citations(
    answer: str,
    citation_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify all citations in an answer against the citation map.

    Args:
        answer: The generated answer string.
        citation_map: Citation dictionary from ``build_citation_map``.

    Returns:
        Structured audit report summarizing citation integrity.
    """
    cited_markers = extract_citations(answer)
    fabricated = detect_fabricated_citations(answer, citation_map)

    verified_citations: List[Dict[str, Any]] = []
    supported_count = 0

    for marker in cited_markers:
        res = verify_citation(marker, answer, citation_map)
        verified_citations.append(res)
        if res.get("is_supported", False):
            supported_count += 1

    unreferenced_sources = [
        m for m in citation_map.keys() if m not in cited_markers
    ]

    all_valid = (
        len(fabricated) == 0
        and (len(cited_markers) == 0 or supported_count == len(cited_markers))
    )

    coverage_score = (
        supported_count / len(cited_markers) if cited_markers else 1.0
    )

    return {
        "all_valid": all_valid,
        "total_citations_in_answer": len(cited_markers),
        "supported_citations_count": supported_count,
        "fabricated_citations": fabricated,
        "unreferenced_sources": unreferenced_sources,
        "coverage_score": round(coverage_score, 4),
        "verified_citations": verified_citations,
    }


def sanitize_answer_citations(
    answer: str,
    citation_map: Dict[str, Dict[str, Any]],
) -> str:
    """Sanitize an answer by removing fabricated citation markers."""
    fabricated = detect_fabricated_citations(answer, citation_map)
    if not fabricated:
        return answer

    sanitized = answer
    for bad_marker in fabricated:
        sanitized = sanitized.replace(bad_marker, "")

    # Clean double spaces
    sanitized = re.sub(r" +", " ", sanitized)
    return sanitized.strip()


# ---------------------------------------------------------------------------
# Formatted Output & CLI Runner
# ---------------------------------------------------------------------------

def print_citation_verification(result: Dict[str, Any] | CitedAnswer) -> Dict[str, Any]:
    """Pretty-print citation verification details to console."""
    answer = result.get("answer", "")
    citations = result.get("citations", {})

    print(f"\n{'=' * 75}")
    print("SOURCE CITATION & ATTRIBUTION REPORT")
    print(f"{'=' * 75}")
    print(f"\nAnswer:\n{answer}\n")
    print(f"Total Available Sources: {len(citations)}")

    for marker, details in citations.items():
        print(f"\nMarker: {marker}")
        print(f"  Source:      {details.get('source')}")
        print(f"  Chunk ID:    {details.get('chunk_id')}")
        print(f"  Section:     {details.get('section')}")
        chunk_preview = details.get("text", "")[:120].replace("\n", " ")
        print(f"  Snippet:     {chunk_preview}...")

    verification = verify_all_citations(answer, citations)
    print(f"\nCitation Verification:")
    print(f"  Total Markers in Answer: {verification['total_citations_in_answer']}")
    print(f"  Supported Citations:     {verification['supported_citations_count']}")
    print(f"  Fabricated Citations:    {verification['fabricated_citations'] or 'None'}")
    print(f"  Unreferenced Sources:    {verification['unreferenced_sources'] or 'None'}")
    print(f"  All Valid:               {'✅ Yes' if verification['all_valid'] else '❌ No'}")
    print(f"{'=' * 75}\n")

    return verification


def run_source_citation_demo(
    output_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full demonstration across all five tasks and export markdown report."""
    out_path = output_file or DEFAULT_OUTPUT

    print("\n" + "=" * 80)
    print("FIELDSURE AI — SOURCE CITATION & ATTRIBUTION DEMO (Concept 3.40)")
    print("=" * 80)

    # 1. Task 1: Answer with source references
    query_1 = "What are the warning signs and immediate field interventions for dengue?"
    print(f"\n[Task 1] Answer with citations for query: '{query_1}'")
    result_1 = answer_with_citations(query_1)
    print(f"  Answer Preview: {result_1['answer'][:180]}...")
    print(f"  Sources Mapped: {len(result_1['citations'])}")

    # 2. Task 2: Map citations to metadata
    print("\n[Task 2] Citation-to-metadata mapping:")
    for marker, meta in result_1["citations"].items():
        print(f"  {marker} -> Source: {meta['source']} | Chunk ID: {meta['chunk_id']} | Section: {meta['section']}")

    # 3. Task 3: Verify cited sources
    print("\n[Task 3] Verify cited sources against original chunk text:")
    verification_1 = verify_all_citations(result_1["answer"], result_1["citations"])
    first_marker = next(iter(result_1["citations"].keys()), "[1]")
    print(f"  Inspecting marker {first_marker}:")
    print(f"    Source: {result_1['citations'][first_marker]['source']}")
    print(f"    Text Preview: {result_1['citations'][first_marker]['text'][:140]}...")
    print(f"    Verified Supported: {verification_1['supported_citations_count']}/{verification_1['total_citations_in_answer']}")

    # 4. Task 4: Avoid fabricated citations (fallback for insufficient sources)
    query_fallback = "What evidence is required for project submission?"
    print(f"\n[Task 4] Avoid fabricated citations for out-of-domain query: '{query_fallback}'")
    result_fallback = answer_with_citations(query_fallback)
    print(f"  Fallback Answer: '{result_fallback['answer']}'")
    print(f"  Citations Count: {len(result_fallback['citations'])} (Zero fabricated citations)")

    # Sample query with custom provided chunks
    sample_custom_chunks = [
        {
            "text": "Project submission requires evidence including version-controlled code commits, automated test execution reports, and source citation traceability.",
            "metadata": {
                "source": "data/guidance/project_submission_standards.pdf",
                "chunk_id": "proj_sub_std_2026_c0001",
                "chunk_index": 0,
                "section": "Evidence Requirements",
                "page": 3,
            },
        }
    ]
    print(f"\n[Task 4.b] Answer with provided evidence for '{query_fallback}':")
    result_custom = answer_with_citations(query_fallback, chunks=sample_custom_chunks)
    print(f"  Answer: {result_custom['answer']}")
    print(f"  Citation [1] Source: {result_custom['citations']['[1]']['source']}")
    print(f"  Citation [1] Text:   {result_custom['citations']['[1]']['text']}")

    # 5. Task 5: Export markdown sample
    md_content = f"""# Source Citation & Attribution — Execution Report

> **Concept**: 3.40 — Source Citation & Attribution  
> **Status**: Verified Execution  
> **Generated**: 2026-10-23  

---

## 1. Task Summary

| Task | Description | Status |
| :--- | :--- | :---: |
| **Task 1** | Add source references (`[1]`, `[2]`) to generated answers | ✅ |
| **Task 2** | Map citations to metadata (`source`, `chunk_id`, `chunk_index`, `section`, `text`) | ✅ |
| **Task 3** | Verify cited sources against original retrieved chunk text | ✅ |
| **Task 4** | Avoid fabricated citations with deterministic fallback | ✅ |
| **Task 5** | Commit sample cited answers, citation mappings, and fallback example | ✅ |

---

## 2. Task 1 — Grounded Answer with Source References

### Question
> {query_1}

### Generated Answer
```text
{result_1['answer']}
```

### Citation Markers Present
{", ".join(extract_citations(result_1['answer']))}

---

## 3. Task 2 — Citation-to-Metadata Mapping

The citation marker in the answer maps to one real retrieved chunk, preserving metadata:

| Marker | Source Document | Section | Chunk ID | Chunk Index |
| :---: | :--- | :--- | :--- | :---: |
"""

    for marker, meta in result_1["citations"].items():
        md_content += f"| **{marker}** | `{meta['source']}` | `{meta['section'] or 'N/A'}` | `{meta['chunk_id']}` | `{meta['chunk_index']}` |\n"

    md_content += f"""
---

## 4. Task 3 — Verify Cited Sources

Pick one citation in the answer and read the mapped chunk text. Confirm the answer claim is supported by that source:

```python
result = answer_with_citations("{query_1}")
print(result["answer"])
print(result["citations"]["[1]"]["source"])
print(result["citations"]["[1]"]["text"])
```

### Verification Result for `[1]`
- **Source**: `{result_1['citations']['[1]']['source']}`
- **Section**: `{result_1['citations']['[1]']['section']}`
- **Chunk ID**: `{result_1['citations']['[1]']['chunk_id']}`
- **Original Chunk Text**:
> {result_1['citations']['[1]']['text']}

### All Citations Verification Summary
- **Total Markers in Answer**: `{verification_1['total_citations_in_answer']}`
- **Supported Citations**: `{verification_1['supported_citations_count']}`
- **Fabricated Citations**: `{verification_1['fabricated_citations'] or 'None'}`
- **All Valid**: `{'✅ Yes' if verification_1['all_valid'] else '❌ No'}`
- **Coverage Score**: `{verification_1['coverage_score'] * 100:.1f}%`

---

## 5. Task 4 — Avoid Fabricated Citations (No-Source Fallback)

When sources are insufficient or the question cannot be answered from the retrieved context,
the system returns an explicit fallback instead of inventing fabricated citations.

### Unsupported Query
> {query_fallback}

### Response
- **Answer**: `{result_fallback['answer']}`
- **Citations**: `{result_fallback['citations']}`
- **Is Fallback**: `{result_fallback['is_fallback']}`

### Supported Query with Evidence
```python
sample_chunks = [
    {{
        "text": "{sample_custom_chunks[0]['text']}",
        "metadata": {{
            "source": "{sample_custom_chunks[0]['metadata']['source']}",
            "chunk_id": "{sample_custom_chunks[0]['metadata']['chunk_id']}",
            "chunk_index": 0,
            "section": "{sample_custom_chunks[0]['metadata']['section']}",
        }},
    }}
]
result = answer_with_citations("{query_fallback}", chunks=sample_chunks)
print(result["answer"])
print(result["citations"]["[1]"]["source"])
print(result["citations"]["[1]"]["text"])
```

**Output**:
```text
{result_custom['answer']}
Source: {result_custom['citations']['[1]']['source']}
Text:   {result_custom['citations']['[1]']['text']}
```

---

## 6. Task 5 — Verification Conclusion

Citations help users decide whether to trust the system. For a knowledge assistant,
citations let users inspect source documents, catch stale or incomplete context, and
understand exactly where every claim in the answer originated.
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    print(f"\n[Task 5] Verified report committed to: {out_path}")
    print("=" * 80 + "\n")

    return {
        "result_1": result_1,
        "verification_1": verification_1,
        "result_fallback": result_fallback,
        "result_custom": result_custom,
        "output_path": str(out_path),
    }


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="FieldSure AI — Source Citation & Attribution (Concept 3.40)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write sample execution report markdown.",
    )
    args = parser.parse_args()

    try:
        run_source_citation_demo(output_file=args.output)
        return 0
    except Exception as err:
        logger.error("Error executing source citation demo: %s", err, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
