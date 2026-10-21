"""Unit and behavioral tests for prompt templates and comparative prompt runner.

All network calls to LLMs are mocked in compliance with DESIGN.md and RULES.md.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.prompt_comparison import (
    BASELINE_PROMPT_PATH,
    CONSTRAINED_PROMPT_PATH,
    load_prompt_file,
    run_prompt_comparison,
)


def test_prompt_files_exist():
    """Verify both baseline and constrained prompt templates exist and are non-empty."""
    assert BASELINE_PROMPT_PATH.is_file(), f"Missing {BASELINE_PROMPT_PATH}"
    assert CONSTRAINED_PROMPT_PATH.is_file(), f"Missing {CONSTRAINED_PROMPT_PATH}"

    baseline_text = load_prompt_file(BASELINE_PROMPT_PATH)
    constrained_text = load_prompt_file(CONSTRAINED_PROMPT_PATH)

    assert len(baseline_text) > 0, "Baseline prompt template is empty"
    assert len(constrained_text) > len(baseline_text), "Constrained prompt must be more detailed"


def test_constrained_prompt_safety_and_scope_elements():
    """Verify system_prompt_constrained.txt contains all required safety mechanisms."""
    content = load_prompt_file(CONSTRAINED_PROMPT_PATH)

    # 1. Role identification
    assert "FieldSure AI" in content
    assert "public health" in content

    # 2. Scope constraints (positive & negative)
    assert "Do NOT make autonomous clinical diagnoses" in content
    assert "Do NOT prescribe" in content

    # 3. Citation instructions
    assert "[Document Title, Section Name]" in content

    # 4. Safe refusal fallback sentence
    assert (
        "I cannot provide guidance on this query because it is outside verified public health protocols"
        in content
    )

    # 5. Non-diagnostic disclaimer notice
    assert (
        "Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system"
        in content
    )


def test_load_prompt_file_missing_raises_error(tmp_path):
    """Verify load_prompt_file raises FileNotFoundError on invalid paths."""
    missing_file = tmp_path / "non_existent_prompt.txt"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prompt_file(missing_file)


def test_prompt_comparison_runner_success(monkeypatch, tmp_path):
    """Verify prompt comparison runner executes both queries across both prompts."""
    monkeypatch.setenv("API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("API_KEY", "sk-valid-test-key-99999")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o-mini")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Simulated comparative response for test verification.",
                }
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
    }
    mock_resp.text = json.dumps(mock_resp.json.return_value)

    test_log = tmp_path / "test_prompt_comparison.log"
    with patch("src.prompt_comparison.LOG_FILE_PATH", test_log):
        with patch("httpx.Client.post", return_value=mock_resp):
            exit_code = run_prompt_comparison()
            assert exit_code == 0
            assert test_log.is_file()

            log_text = test_log.read_text(encoding="utf-8")
            assert "Q1_IN_SCOPE_TRIAGE" in log_text
            assert "Q2_OUT_OF_SCOPE_DIAGNOSIS" in log_text
            assert "sk-valid-test-key-99999" not in log_text  # strictly masked


def test_prompt_comparison_config_error(monkeypatch):
    """Verify prompt comparison runner fails cleanly on missing configuration."""
    monkeypatch.delenv("API_KEY", raising=False)
    exit_code = run_prompt_comparison()
    assert exit_code == 1
