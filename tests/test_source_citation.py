"""Automated test suite for Source Citation & Attribution (Concept 3.40)."""

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.source_citation import (
    FALLBACK_NO_CONTEXT,
    AssembledContextResult,
    CitedAnswer,
    answer_with_citations,
    assemble_context,
    build_citation_map,
    build_cited_prompt,
    call_llm,
    detect_fabricated_citations,
    extract_citations,
    print_citation_verification,
    retrieve,
    run_source_citation_demo,
    sanitize_answer_citations,
    verify_all_citations,
    verify_citation,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "text": "Dengue fever is an arboviral disease transmitted by Aedes mosquitoes.",
            "metadata": {
                "source": "who_dengue_guidance.pdf",
                "chunk_id": "chunk_dengue_001",
                "chunk_index": 0,
                "section": "1. Overview & Transmission",
                "page": 2,
            },
        },
        {
            "text": "Critical warning signs include persistent vomiting and mucosal bleeding.",
            "metadata": {
                "source": "who_dengue_guidance.pdf",
                "chunk_id": "chunk_dengue_002",
                "chunk_index": 1,
                "section": "2. Danger Signs",
                "page": 4,
            },
        },
        {
            "text": "Immediate field interventions require oral rehydration solution (ORS).",
            "metadata": {
                "source": "cdc_rehydration_protocol.txt",
                "chunk_id": "chunk_cdc_001",
                "chunk_index": 0,
                "section": "Field Rehydration Standards",
            },
        },
    ]


@pytest.fixture
def project_submission_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "text": "Project submission requires evidence such as git commits and automated test logs.",
            "metadata": {
                "source": "project_submission_policy.pdf",
                "chunk_id": "sub_chunk_001",
                "chunk_index": 0,
                "section": "Evidence Requirements",
                "page": 1,
            },
        }
    ]


# ---------------------------------------------------------------------------
# Task 2: Build Citation Map Tests
# ---------------------------------------------------------------------------

class TestBuildCitationMap:
    def test_build_citation_map_maps_all_chunks(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        assert len(citation_map) == 3
        assert "[1]" in citation_map
        assert "[2]" in citation_map
        assert "[3]" in citation_map

    def test_citation_map_preserves_metadata(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        entry1 = citation_map["[1]"]
        assert entry1["source"] == "who_dengue_guidance.pdf"
        assert entry1["chunk_id"] == "chunk_dengue_001"
        assert entry1["chunk_index"] == 0
        assert entry1["section"] == "1. Overview & Transmission"
        assert "arboviral disease" in entry1["text"]
        assert entry1["page"] == 2

    def test_citation_map_handles_empty_list(self):
        assert build_citation_map([]) == {}

    def test_citation_map_handles_fallback_field_names(self):
        chunks = [
            {
                "id": "alt_chunk_01",
                "text": "Sample text",
                "metadata": {
                    "source_file": "alt_file.md",
                    "section_heading": "Alt Section",
                    "chunk_index": 5,
                },
            }
        ]
        citation_map = build_citation_map(chunks)
        assert citation_map["[1]"]["source"] == "alt_file.md"
        assert citation_map["[1]"]["chunk_id"] == "alt_chunk_01"
        assert citation_map["[1]"]["section"] == "Alt Section"
        assert citation_map["[1]"]["chunk_index"] == 5

    def test_citation_map_handles_chunk_objects(self):
        class MockChunk:
            def __init__(self):
                self.text = "Object chunk text"
                self.metadata = {
                    "source": "object_source.pdf",
                    "chunk_id": "obj_001",
                    "section": "Object Sec",
                }

        citation_map = build_citation_map([MockChunk()])
        assert citation_map["[1]"]["source"] == "object_source.pdf"
        assert citation_map["[1]"]["chunk_id"] == "obj_001"
        assert citation_map["[1]"]["text"] == "Object chunk text"


# ---------------------------------------------------------------------------
# Context Assembly & Prompt Construction Tests
# ---------------------------------------------------------------------------

class TestAssembleContextAndPrompt:
    def test_assemble_context_supports_tuple_unpacking(self, sample_chunks):
        context, token_count = assemble_context(sample_chunks)
        assert isinstance(context, str)
        assert isinstance(token_count, int)
        assert "[1] Source: who_dengue_guidance.pdf" in context
        assert "[2] Source: who_dengue_guidance.pdf" in context
        assert "[3] Source: cdc_rehydration_protocol.txt" in context
        assert token_count > 0

    def test_assemble_context_supports_direct_string_use(self, sample_chunks):
        context = assemble_context(sample_chunks)
        assert isinstance(context, str)
        assert "Source: who_dengue_guidance.pdf" in str(context)
        assert f"{context}".startswith("[1]")

    def test_assemble_context_empty_list(self):
        context, token_count = assemble_context([])
        assert context == ""
        assert token_count == 0

    def test_build_cited_prompt_format_and_instructions(self, sample_chunks):
        question = "What are the warning signs of dengue?"
        prompt = build_cited_prompt(question, sample_chunks)

        # Mandatory grounding instructions
        assert "Answer using only the context below." in prompt
        assert "Cite every factual claim using source markers like [1] or [2]." in prompt
        assert "Only use source markers that appear in the context." in prompt
        assert "If the context does not support an answer, say you do not have enough information" in prompt
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert question in prompt
        assert "[1] Source: who_dengue_guidance.pdf" in prompt


# ---------------------------------------------------------------------------
# Task 1 & Task 4: Answer with Citations Tests
# ---------------------------------------------------------------------------

class TestAnswerWithCitations:
    def test_answer_with_citations_structure(self, sample_chunks):
        def mock_llm(p: str) -> str:
            return (
                "Dengue is transmitted by mosquitoes [1]. "
                "Warning signs include vomiting [2]. "
                "Treat immediately with ORS [3]."
            )

        result = answer_with_citations(
            question="What are the warning signs of dengue?",
            chunks=sample_chunks,
            llm_fn=mock_llm,
        )

        assert isinstance(result, dict)
        assert isinstance(result, CitedAnswer)
        assert "answer" in result
        assert "citations" in result
        assert result["citations"]["[1]"]["source"] == "who_dengue_guidance.pdf"
        assert result["citations"]["[2]"]["chunk_id"] == "chunk_dengue_002"
        assert "arboviral disease" in result["citations"]["[1]"]["text"]
        assert result.answer == result["answer"]
        assert result.citations == result["citations"]

    def test_fallback_when_chunks_empty(self):
        result = answer_with_citations(
            question="What evidence is required for project submission?",
            chunks=[],
        )
        assert result["answer"] == FALLBACK_NO_CONTEXT
        assert result["citations"] == {}
        assert result.is_fallback is True

    def test_fallback_when_llm_reports_insufficient_context(self, sample_chunks):
        def refusing_llm(p: str) -> str:
            return "I don't have enough information in the provided context."

        result = answer_with_citations(
            question="Unrelated question",
            chunks=sample_chunks,
            llm_fn=refusing_llm,
        )
        assert result["answer"] == FALLBACK_NO_CONTEXT
        assert result["citations"] == {}
        assert result.is_fallback is True

    def test_custom_retriever_fn(self, sample_chunks):
        called = False

        def mock_retriever(q: str, k: int):
            nonlocal called
            called = True
            return sample_chunks[:1]

        def mock_llm(p: str) -> str:
            return "Mosquito transmission documented [1]."

        result = answer_with_citations(
            question="Transmission mode?",
            retriever_fn=mock_retriever,
            llm_fn=mock_llm,
        )

        assert called is True
        assert len(result["citations"]) == 1
        assert "[1]" in result["citations"]

    def test_empty_question_raises_value_error(self):
        with pytest.raises(ValueError, match="Question must be a non-empty string"):
            answer_with_citations("")

        with pytest.raises(ValueError, match="Question must be a non-empty string"):
            answer_with_citations("   ")

    def test_project_submission_evidence_example(self, project_submission_chunks):
        result = answer_with_citations(
            question="What evidence is required for project submission?",
            chunks=project_submission_chunks,
        )
        assert "project submission" in result["answer"].lower()
        assert "[1]" in result["citations"]
        assert result["citations"]["[1]"]["source"] == "project_submission_policy.pdf"
        assert "git commits" in result["citations"]["[1]"]["text"]


# ---------------------------------------------------------------------------
# Task 3: Citation Verification Tests
# ---------------------------------------------------------------------------

class TestCitationVerification:
    def test_extract_citations(self):
        text = "Fact A [1], Fact B [2], and another mention of Fact A [1]."
        markers = extract_citations(text)
        assert markers == ["[1]", "[2]"]

    def test_detect_fabricated_citations(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        answer_with_hallucination = "Claim one [1], claim two [99], and claim three [5]."
        fabricated = detect_fabricated_citations(answer_with_hallucination, citation_map)
        assert "[99]" in fabricated
        assert "[5]" in fabricated
        assert "[1]" not in fabricated

    def test_verify_citation_valid(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        answer = "Dengue is an arboviral disease transmitted by mosquitoes [1]."
        res = verify_citation("[1]", answer, citation_map)

        assert res["is_valid"] is True
        assert res["is_supported"] is True
        assert res["source"] == "who_dengue_guidance.pdf"
        assert res["chunk_id"] == "chunk_dengue_001"
        assert res["overlap_score"] > 0.3

    def test_verify_citation_fabricated(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        answer = "Invented claim [99]."
        res = verify_citation("[99]", answer, citation_map)

        assert res["is_valid"] is False
        assert res["found_in_map"] is False
        assert "fabricated" in res["reason"]

    def test_verify_citation_not_in_answer(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        answer = "Only mentions marker [1]."
        res = verify_citation("[2]", answer, citation_map)

        assert res["is_valid"] is False
        assert res["found_in_map"] is True
        assert res["found_in_answer"] is False

    def test_verify_all_citations_comprehensive(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        answer = (
            "Dengue fever is an arboviral disease [1]. "
            "Warning signs include persistent vomiting [2]. "
            "Fabricated claim [99]."
        )
        report = verify_all_citations(answer, citation_map)

        assert report["all_valid"] is False
        assert report["total_citations_in_answer"] == 3
        assert report["supported_citations_count"] == 2
        assert "[99]" in report["fabricated_citations"]
        assert "[3]" in report["unreferenced_sources"]

    def test_sanitize_answer_citations(self, sample_chunks):
        citation_map = build_citation_map(sample_chunks)
        answer = "True claim [1] but false marker [99]."
        sanitized = sanitize_answer_citations(answer, citation_map)
        assert "[1]" in sanitized
        assert "[99]" not in sanitized


# ---------------------------------------------------------------------------
# Dataclass & Dictionary Compatibility Tests
# ---------------------------------------------------------------------------

class TestCitedAnswerCompatibility:
    def test_cited_answer_dict_and_attribute_access(self):
        ans = CitedAnswer(
            answer="Test answer [1].",
            citations={"[1]": {"source": "doc.txt"}},
            is_fallback=False,
            question="What is this?",
            chunks_used=1,
        )
        # Dictionary subscripting
        assert ans["answer"] == "Test answer [1]."
        assert ans["citations"]["[1]"]["source"] == "doc.txt"
        assert ans.get("chunks_used") == 1
        assert "answer" in ans

        # Attribute access
        assert ans.answer == "Test answer [1]."
        assert ans.citations == {"[1]": {"source": "doc.txt"}}
        assert ans.is_fallback is False

        # JSON serialization
        serialized = json.dumps(ans)
        deserialized = json.loads(serialized)
        assert deserialized["answer"] == "Test answer [1]."
        assert deserialized["citations"]["[1]"]["source"] == "doc.txt"


# ---------------------------------------------------------------------------
# Retrieval & Real Ingested Corpus Tests
# ---------------------------------------------------------------------------

class TestRetrievalAndCorpus:
    def test_retrieve_finds_dengue_chunks(self):
        chunks = retrieve("What are the warning signs and interventions for dengue?", k=3)
        assert len(chunks) > 0
        text_corpus = " ".join(c.get("text", "").lower() for c in chunks)
        assert "dengue" in text_corpus

    def test_retrieve_returns_empty_on_out_of_domain_query(self):
        chunks = retrieve("What evidence is required for project submission?")
        assert chunks == []

    def test_retrieve_empty_query_returns_empty(self):
        assert retrieve("") == []
        assert retrieve("   ") == []


# ---------------------------------------------------------------------------
# End-to-End Demo & Report Generation
# ---------------------------------------------------------------------------

class TestDemoAndSampleCommit:
    def test_run_source_citation_demo_creates_markdown_report(self, tmp_path):
        out_file = tmp_path / "test_citation_sample.md"
        demo_res = run_source_citation_demo(output_file=out_file)

        assert out_file.is_file()
        content = out_file.read_text(encoding="utf-8")
        assert "# Source Citation & Attribution — Execution Report" in content
        assert "Task 1 — Grounded Answer with Source References" in content
        assert "Task 2 — Citation-to-Metadata Mapping" in content
        assert "Task 3 — Verify Cited Sources" in content
        assert "Task 4 — Avoid Fabricated Citations" in content
        assert "I don't have enough information in the provided context." in content

        assert demo_res["result_1"]["citations"]
        assert demo_res["result_fallback"]["is_fallback"] is True
