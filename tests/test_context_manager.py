"""Unit and regression tests for context window and message history management (Module #6)."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from src.context_manager import ContextWindowManager, run_context_management_demo


def test_manager_initialization():
    """Verify context manager initializes with system prompt preserved at index 0."""
    system_text = "You are FieldSure AI, a clinical assistant."
    manager = ContextWindowManager(system_prompt=system_text, max_history_tokens=500)

    assert len(manager.messages) == 1
    assert manager.messages[0]["role"] == "system"
    assert manager.messages[0]["content"] == system_text
    assert manager.get_total_tokens() > 0


def test_maintain_multi_turn_history():
    """Verify alternating user and assistant messages are tracked cleanly (Task 1)."""
    manager = ContextWindowManager(system_prompt="System prompt", max_history_tokens=1000)

    manager.add_user_message("Query 1: What is malaria?")
    manager.add_assistant_message("Answer 1: Malaria is a mosquito-borne infectious disease.")
    manager.add_user_message("Query 2: How is it treated?")
    manager.add_assistant_message("Answer 2: Artemisinin-based combination therapy.")

    assert len(manager.messages) == 5
    assert manager.messages[1]["role"] == "user"
    assert manager.messages[2]["role"] == "assistant"
    assert manager.messages[3]["role"] == "user"
    assert manager.messages[4]["role"] == "assistant"


def test_pre_request_token_measurement():
    """Verify total tokens are calculated before and after messages (Task 2)."""
    manager = ContextWindowManager(system_prompt="You are a helpful assistant.", max_history_tokens=1000)

    initial_tokens = manager.get_total_tokens()
    manager.add_user_message("Please explain ORS preparation in community settings.")
    after_user_tokens = manager.get_total_tokens()

    assert after_user_tokens > initial_tokens, "Tokens must increase after adding user message"

    manager.add_assistant_message("Mix 1 packet of ORS into 1 liter of clean drinking water.")
    after_assistant_tokens = manager.get_total_tokens()

    assert after_assistant_tokens > after_user_tokens, "Tokens must increase after adding assistant reply"


def test_trimming_preserves_system_prompt():
    """Verify system message is NEVER evicted even during heavy overflow (Task 3)."""
    sys_prompt = "Permanent System Guardrail Prompt for Public Health."
    # Set tight budget
    manager = ContextWindowManager(system_prompt=sys_prompt, max_history_tokens=120)

    # Add multiple large turns to force trimming
    for i in range(1, 6):
        manager.process_turn(
            f"Long user question {i}: " + "symptoms " * 15,
            f"Long assistant answer {i}: " + "treatment " * 15,
        )

    # 1. System prompt MUST remain at index 0
    assert manager.messages[0]["role"] == "system"
    assert manager.messages[0]["content"] == sys_prompt


def test_trimming_preserves_most_recent_turn():
    """Verify the latest user question is preserved during trimming (Task 3)."""
    manager = ContextWindowManager(system_prompt="System prompt", max_history_tokens=150)

    manager.process_turn("Question 1", "Answer 1")
    manager.process_turn("Question 2", "Answer 2")
    manager.process_turn("Question 3", "Answer 3")
    final_turn = manager.process_turn("LATEST QUESTION: What about dengue?", "LATEST ANSWER: Monitor platelets.")

    # The latest assistant message and user question must be present in messages
    latest_user_msgs = [m for m in manager.messages if "LATEST QUESTION" in m["content"]]
    assert len(latest_user_msgs) == 1, "Latest user query must be retained in active context"


def test_overflowing_conversation_enforces_budget():
    """Verify long conversation is kept strictly within token budget (Task 4)."""
    budget = 180
    manager = ContextWindowManager(system_prompt="System prompt.", max_history_tokens=budget)

    trim_occurred = False
    for i in range(8):
        log_entry = manager.process_turn(
            f"Turn {i}: Describe triage for dehydration scenario with high clinical detail " * 2,
            f"Turn {i}: Recommended immediate response is oral rehydration therapy and referral " * 2,
        )
        if log_entry.was_trimmed:
            trim_occurred = True

    assert trim_occurred, "Trimming must have triggered on overflowing dialogue"
    # Final total tokens must not breach the ceiling significantly
    assert manager.get_total_tokens() <= budget + 25  # allow minimal buffer for final turn priming


def test_demo_runner_execution(tmp_path):
    """Verify run_context_management_demo executes cleanly with audit logging (Task 5)."""
    test_log = tmp_path / "test_context_management.log"

    with patch("src.context_manager.LOG_FILE_PATH", test_log):
        exit_code = run_context_management_demo()
        assert exit_code == 0
        assert test_log.is_file()

        log_content = test_log.read_text(encoding="utf-8")
        assert "Turn 1: Initial Presentation" in log_content
        assert "Turn 6: Discharge & Home Care Protocol" in log_content
        assert "system_message_preserved" in log_content
