"""Automated test suite for Conversational RAG & Follow-Up Context (Concept 3.42)."""

from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.conversational_rag import (
    DEFAULT_CONVERSATIONAL_CHUNKS,
    ConversationalAnswer,
    call_llm,
    compare_retrieval,
    conversational_answer,
    format_history_text,
    retrieve,
    rewrite_followup,
    run_conversational_dialogue,
    run_conversational_rag_demo,
    trim_history,
)
from src.guardrails import (
    DEFAULT_REFUSAL_MESSAGE,
    STATUS_ANSWERED,
    STATUS_REFUSED_WEAK_CONTEXT,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_submission_history() -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "What evidence is required for project submission?",
        },
        {
            "role": "assistant",
            "content": "The submission needs a PR link, sample output, and a video explanation with source evidence.",
        },
    ]


@pytest.fixture
def dengue_clinical_history() -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": "What are the critical warning signs for dengue fever?",
        },
        {
            "role": "assistant",
            "content": (
                "Critical warning signs include persistent vomiting (>3 episodes in 24 hours), "
                "severe abdominal pain, fluid accumulation, and mucosal bleeding."
            ),
        },
    ]


@pytest.fixture
def custom_test_chunks() -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": "test_proj_video_001",
            "text": "Project submission video explanation requires a concise demonstration of system execution and source evidence.",
            "metadata": {
                "source": "submission_guide.pdf",
                "section": "Video Walkthrough",
                "chunk_id": "test_proj_video_001",
            },
        },
        {
            "chunk_id": "test_media_camera_002",
            "text": "Studio video lighting and 4K camera setup guidelines for commercial broadcasting.",
            "metadata": {
                "source": "media_setup.pdf",
                "section": "Camera Setup",
                "chunk_id": "test_media_camera_002",
            },
        },
    ]


# ---------------------------------------------------------------------------
# Task 1: Conversation History Tracking & Limits Tests
# ---------------------------------------------------------------------------

class TestConversationHistoryTracking:
    def test_format_history_text_empty(self):
        text = format_history_text([])
        assert text == "No prior conversation."

    def test_format_history_text_multi_turn(self, project_submission_history):
        text = format_history_text(project_submission_history)
        assert "User: What evidence is required for project submission?" in text
        assert "Assistant: The submission needs a PR link" in text

    def test_trim_history_by_turn_count(self):
        history = [
            {"role": "user", "content": f"Question {i}"}
            for i in range(10)
        ]
        trimmed = trim_history(history, max_turns=4)
        assert len(trimmed) == 4
        assert trimmed[-1]["content"] == "Question 9"
        assert trimmed[0]["content"] == "Question 6"

    def test_trim_history_by_token_budget(self):
        # Create turns with known content length
        history = [
            {"role": "user", "content": "Short question 1"},
            {"role": "assistant", "content": "A very long detailed assistant response " * 20},
            {"role": "user", "content": "Short question 2"},
            {"role": "assistant", "content": "Another long response " * 20},
            {"role": "user", "content": "Latest brief question"},
        ]
        trimmed = trim_history(history, max_turns=10, max_tokens=100)
        # Should prune from head while preserving recent turns
        assert len(trimmed) < 5
        assert trimmed[-1]["content"] == "Latest brief question"

    def test_trim_history_empty_list(self):
        empty_list: List[Dict[str, str]] = []
        assert trim_history(empty_list) == []


# ---------------------------------------------------------------------------
# Task 2: Follow-Up Rewriting Tests
# ---------------------------------------------------------------------------

class TestFollowUpRewriting:
    def test_rewrite_without_history_returns_same_question(self):
        q = "What are the warning signs of dengue?"
        rewritten = rewrite_followup([], q)
        assert rewritten == q

    def test_rewrite_video_followup(self, project_submission_history):
        followup = "What about the video?"
        rewritten = rewrite_followup(project_submission_history, followup)
        assert "video explanation" in rewritten.lower()
        assert "project submission" in rewritten.lower()

    def test_rewrite_deadline_followup(self, project_submission_history):
        followup = "What about the deadline?"
        rewritten = rewrite_followup(project_submission_history, followup)
        assert "deadline" in rewritten.lower()
        assert "project submission" in rewritten.lower()

    def test_rewrite_explain_that_followup(self, project_submission_history):
        followup = "Can you explain that?"
        rewritten = rewrite_followup(project_submission_history, followup)
        assert "explain" in rewritten.lower()
        assert "project submission" in rewritten.lower()

    def test_rewrite_sprint_applicability_followup(self, project_submission_history):
        followup = "Does it apply to Sprint 2?"
        rewritten = rewrite_followup(project_submission_history, followup)
        assert "sprint 2" in rewritten.lower()
        assert "project submission" in rewritten.lower()

    def test_rewrite_clinical_management_followup(self, dengue_clinical_history):
        followup = "How should it be managed?"
        rewritten = rewrite_followup(dengue_clinical_history, followup)
        assert "dengue" in rewritten.lower()
        assert "managed" in rewritten.lower()

    def test_rewrite_clinical_fluids_followup(self, dengue_clinical_history):
        followup = "What fluids should be given?"
        rewritten = rewrite_followup(dengue_clinical_history, followup)
        assert "fluids" in rewritten.lower()
        assert "dengue" in rewritten.lower()

    def test_rewrite_clinical_nsaids_followup(self, dengue_clinical_history):
        followup = "Can NSAIDs like ibuprofen be used for it?"
        rewritten = rewrite_followup(dengue_clinical_history, followup)
        assert "nsaid" in rewritten.lower() or "ibuprofen" in rewritten.lower()
        assert "dengue" in rewritten.lower()

    def test_rewrite_with_custom_llm_fn(self, project_submission_history):
        def mock_llm_rewrite(prompt: str) -> str:
            assert "Latest question:\nWhat about the video?" in prompt
            return "Standalone query: What video explanation is required for project submission?"

        rewritten = rewrite_followup(
            project_submission_history,
            "What about the video?",
            llm_fn=mock_llm_rewrite,
        )
        assert rewritten == "What video explanation is required for project submission?"

    def test_rewrite_strips_quotes_and_preambles(self, project_submission_history):
        def mock_llm_rewrite(prompt: str) -> str:
            return '"What video requirements exist for project submission?"'

        rewritten = rewrite_followup(
            project_submission_history,
            "What about the video?",
            llm_fn=mock_llm_rewrite,
        )
        assert rewritten == "What video requirements exist for project submission?"

    def test_rewrite_invalid_question_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty string"):
            rewrite_followup([], "")

        with pytest.raises(ValueError, match="non-empty string"):
            rewrite_followup([], "   ")


# ---------------------------------------------------------------------------
# Task 3: Retrieval Using Rewritten Query Tests
# ---------------------------------------------------------------------------

class TestRetrievalUsingRewrittenQuery:
    def test_retrieve_empty_query_returns_empty_list(self):
        assert retrieve("") == []
        assert retrieve("   ") == []

    def test_retrieve_with_supplemental_chunks(self, custom_test_chunks):
        matches = retrieve(
            "video explanation project submission",
            k=2,
            supplemental_chunks=custom_test_chunks,
        )
        assert len(matches) > 0
        top_chunk = matches[0]
        assert top_chunk["chunk_id"] == "test_proj_video_001"
        assert top_chunk["score"] >= 0.72

    def test_compare_retrieval_demonstrates_rewrite_advantage(
        self,
        project_submission_history,
        custom_test_chunks,
    ):
        comp = compare_retrieval(
            project_submission_history,
            "What about the video?",
            k=2,
            supplemental_chunks=custom_test_chunks,
        )
        assert comp["original_question"] == "What about the video?"
        assert "video explanation is required for project submission" in comp["rewritten_query"].lower()
        # The rewritten query achieves strong quality
        assert comp["rewritten_is_strong"] is True
        assert comp["rewritten_top_score"] >= 0.72
        # Top chunk for rewritten query should be the project video explanation chunk
        assert comp["rewritten_chunks"][0]["chunk_id"] == "test_proj_video_001"


# ---------------------------------------------------------------------------
# Task 4: Conversational Answer Orchestrator Tests
# ---------------------------------------------------------------------------

class TestConversationalAnswer:
    def test_conversational_answer_first_turn_and_history_update(self):
        history: List[Dict[str, str]] = []
        question = "What evidence is required for project submission?"

        res = conversational_answer(history, question, k=2)

        assert res.status == STATUS_ANSWERED
        assert res["status"] == STATUS_ANSWERED
        assert res.is_refused is False
        assert len(res.sources) > 0
        assert "evidence" in res.answer.lower() or "submission" in res.answer.lower()

        # History should now contain 2 turns (user + assistant)
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": question}
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == res.answer

    def test_conversational_answer_followup_turn(self, project_submission_history):
        history = list(project_submission_history)
        followup = "What about the video?"

        res = conversational_answer(history, followup, k=2)

        assert res.status == STATUS_ANSWERED
        assert "video explanation" in res.rewritten_query.lower()
        assert "project submission" in res.rewritten_query.lower()
        assert len(res.sources) > 0

        # History updated to 4 turns total
        assert len(history) == 4
        assert history[2] == {"role": "user", "content": followup}
        assert history[3]["role"] == "assistant"

    def test_conversational_answer_weak_context_refusal(self, project_submission_history):
        history = list(project_submission_history)
        out_of_domain = "What is the return refund policy on retail store items?"

        res = conversational_answer(history, out_of_domain, k=2)

        assert res.status == STATUS_REFUSED_WEAK_CONTEXT
        assert res.is_refused is True
        assert res.answer == DEFAULT_REFUSAL_MESSAGE
        assert res.sources == []
        assert history[-1]["content"] == DEFAULT_REFUSAL_MESSAGE

    def test_conversational_answer_dict_interface_and_unpacking(self):
        history: List[Dict[str, str]] = []
        res = conversational_answer(history, "What evidence is required for project submission?")

        # Subscripting
        assert "answer" in res
        assert "rewritten_query" in res
        assert "sources" in res
        assert "status" in res

        # Attribute access
        assert res.answer is not None
        assert res.rewritten_query is not None
        assert isinstance(res.sources, list)

        # Unpacking
        unpacked = {**res, "custom_field": "test_val"}
        assert unpacked["custom_field"] == "test_val"
        assert unpacked["answer"] == res.answer

        # to_dict
        plain_dict = res.to_dict()
        assert isinstance(plain_dict, dict)
        assert plain_dict["status"] == res.status

    def test_conversational_answer_with_custom_retriever_and_llm(self):
        history: List[Dict[str, str]] = []

        def custom_retriever(q: str, k: int) -> List[Dict[str, Any]]:
            return [
                {
                    "chunk_id": "c_custom_001",
                    "text": "Special submission requirements for sprint review.",
                    "score": 0.95,
                    "similarity_score": 0.95,
                    "metadata": {"source": "custom.pdf", "chunk_id": "c_custom_001"},
                }
            ]

        def custom_llm(prompt: str) -> str:
            if "Rewrite the user's latest question" in prompt:
                return "Special sprint submission requirements"
            return "Custom grounded answer for sprint review."

        res = conversational_answer(
            history,
            "What about submission?",
            retriever_fn=custom_retriever,
            llm_fn=custom_llm,
        )

        assert res.status == STATUS_ANSWERED
        assert res.rewritten_query == "Special sprint submission requirements"
        assert res.answer == "Custom grounded answer for sprint review."
        assert len(res.sources) == 1

    def test_conversational_answer_invalid_question_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty string"):
            conversational_answer([], "")

    def test_run_conversational_dialogue(self):
        questions = [
            "What evidence is required for project submission?",
            "What about the video?",
            "Does it apply to Sprint 2?",
        ]
        history, results = run_conversational_dialogue(questions)

        assert len(results) == 3
        # 3 questions + 3 answers = 6 turns in history
        assert len(history) == 6
        assert results[1].rewritten_query != questions[1]
        assert "video explanation" in results[1].rewritten_query.lower()


# ---------------------------------------------------------------------------
# Task 5: Sample Dialogue Execution & Demo Tests
# ---------------------------------------------------------------------------

class TestConversationalRAGDemo:
    def test_run_conversational_rag_demo(self, tmp_path: Path):
        demo_output = tmp_path / "conversational_rag_sample.md"
        result = run_conversational_rag_demo(output_file=demo_output)

        assert demo_output.exists()
        content = demo_output.read_text(encoding="utf-8")

        # Verify Markdown structure
        assert "# Conversational RAG & Follow-Up Context — Execution Report" in content
        assert "## 1. Task Summary" in content
        assert "## 2. Conversational RAG Architecture & Flow" in content
        assert "## 3. Verified Multi-Turn Dialogue 1: Project Submission Standards" in content
        assert "## 4. Naive vs. Rewritten Retrieval Comparison" in content
        assert "## 5. Verified Multi-Turn Dialogue 2: Clinical Public Health Guidance (WHO Dengue)" in content
        assert "## 6. History Tracking & Token Limit Management" in content

        # Verify key entities and rewrites
        assert "What about the video?" in content
        assert "What video explanation is required for project submission?" in content
        assert "What are the critical warning signs for dengue fever?" in content
        assert "refused_weak_context" in content

        # Verify return metrics
        assert result["dialogue_1_turns"] == 4
        assert result["dialogue_2_turns"] == 3
        assert result["retrieval_comparison"]["rewritten_top_score"] >= 0.72
