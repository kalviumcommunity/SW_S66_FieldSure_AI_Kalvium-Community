"""Unit and regression tests for LLM client configuration, completion, and error handling.

All external HTTP calls are strictly mocked in accordance with DESIGN.md and RULES.md.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest

from src.config import LLMConfig, LLMConfigError, load_llm_config, mask_api_key
from src.llm_client import (
    ChatCompletionResult,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMAPIError,
    OpenAICompatibleClient,
    UsageInfo,
)
from src.chat_completion import run_chat_completion


# ------------------------------------------------------------------------------
# Configuration & Masking Tests
# ------------------------------------------------------------------------------

def test_mask_api_key():
    """Verify API keys are masked properly for safe logging."""
    assert mask_api_key("") == "[NOT SET]"
    assert mask_api_key("short") == "***"
    assert mask_api_key("sk-1234567890abcdef") == "sk-...cdef"
    assert "1234567890" not in mask_api_key("sk-1234567890abcdef")


def test_load_llm_config_success(monkeypatch):
    """Verify loading valid configuration from environment variables."""
    monkeypatch.setenv("API_BASE_URL", "https://api.openai.com/v1/")
    monkeypatch.setenv("API_KEY", "sk-valid-test-key-12345")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("REQUEST_TIMEOUT", "45.0")

    config = load_llm_config()

    assert config.api_base_url == "https://api.openai.com/v1"  # trailing slash stripped
    assert config.api_key == "sk-valid-test-key-12345"
    assert config.chat_model == "gpt-4o-mini"
    assert config.request_timeout == 45.0


def test_load_llm_config_missing_key(monkeypatch):
    """Verify descriptive error when API_KEY is missing or placeholder."""
    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="API_KEY is not configured"):
        load_llm_config()

    monkeypatch.setenv("API_KEY", "your_llm_api_key_here")
    with pytest.raises(LLMConfigError, match="placeholder"):
        load_llm_config()


# ------------------------------------------------------------------------------
# LLM Client Mocked Tests
# ------------------------------------------------------------------------------

@pytest.fixture
def test_config():
    """Fixture providing standard test configuration."""
    return LLMConfig(
        api_base_url="https://api.openai.com/v1",
        api_key="sk-test-key-abcdef12345",
        chat_model="gpt-4o-mini",
        request_timeout=15.0,
    )


def test_client_endpoint_url_construction(test_config):
    """Verify chat completions endpoint URL normalization."""
    client = OpenAICompatibleClient(test_config)
    assert client._endpoint_url == "https://api.openai.com/v1/chat/completions"

    # Already containing /chat/completions
    cfg2 = LLMConfig(
        api_base_url="https://api.openai.com/v1/chat/completions",
        api_key="sk-test",
        chat_model="gpt-4o-mini",
    )
    client2 = OpenAICompatibleClient(cfg2)
    assert client2._endpoint_url == "https://api.openai.com/v1/chat/completions"


def test_successful_chat_completion(test_config):
    """Verify successful chat completion parses content and token usage."""
    mock_response_body = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1729000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Early signs of dehydration include sunken eyes and lethargy.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 15,
            "total_tokens": 57,
        },
    }

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_response_body
    mock_resp.text = json.dumps(mock_response_body)

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        client = OpenAICompatibleClient(test_config)
        messages = [
            {"role": "system", "content": "You are a health assistant."},
            {"role": "user", "content": "Signs of dehydration?"},
        ]
        result = client.create_chat_completion(messages=messages, temperature=0.2)

        assert mock_post.called
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "gpt-4o-mini"
        assert call_kwargs["json"]["temperature"] == 0.2
        assert len(call_kwargs["json"]["messages"]) == 2

        assert result.content == "Early signs of dehydration include sunken eyes and lethargy."
        assert result.role == "assistant"
        assert result.model == "gpt-4o-mini"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 42
        assert result.usage.completion_tokens == 15
        assert result.usage.total_tokens == 57


def test_401_authentication_error(test_config):
    """Verify HTTP 401 raises LLMAuthenticationError with provider message."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = '{"error": {"message": "Incorrect API key provided"}}'
    mock_resp.json.return_value = {"error": {"message": "Incorrect API key provided"}}

    with patch("httpx.Client.post", return_value=mock_resp):
        client = OpenAICompatibleClient(test_config)
        with pytest.raises(LLMAuthenticationError) as exc_info:
            client.create_chat_completion([{"role": "user", "content": "hello"}])

        assert "Authentication failed (HTTP 401)" in str(exc_info.value)
        assert "Incorrect API key provided" in str(exc_info.value)


def test_429_rate_limit_error(test_config):
    """Verify HTTP 429 raises LLMRateLimitError with provider message."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.text = '{"error": {"message": "You exceeded your current quota"}}'
    mock_resp.json.return_value = {"error": {"message": "You exceeded your current quota"}}

    with patch("httpx.Client.post", return_value=mock_resp):
        client = OpenAICompatibleClient(test_config)
        with pytest.raises(LLMRateLimitError) as exc_info:
            client.create_chat_completion([{"role": "user", "content": "hello"}])

        assert "Rate limit or quota exceeded (HTTP 429)" in str(exc_info.value)
        assert "exceeded your current quota" in str(exc_info.value)


def test_connection_timeout_error(test_config):
    """Verify timeout raises LLMConnectionError."""
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timed out")):
        client = OpenAICompatibleClient(test_config)
        with pytest.raises(LLMConnectionError) as exc_info:
            client.create_chat_completion([{"role": "user", "content": "hello"}])

        assert "timed out" in str(exc_info.value).lower()


def test_connection_network_error(test_config):
    """Verify network error raises LLMConnectionError."""
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
        client = OpenAICompatibleClient(test_config)
        with pytest.raises(LLMConnectionError) as exc_info:
            client.create_chat_completion([{"role": "user", "content": "hello"}])

        assert "Failed to connect to LLM server" in str(exc_info.value)


def test_malformed_response_error(test_config):
    """Verify malformed payload (missing choices) raises LLMAPIError."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = '{"id": "test", "choices": []}'
    mock_resp.json.return_value = {"id": "test", "choices": []}

    with patch("httpx.Client.post", return_value=mock_resp):
        client = OpenAICompatibleClient(test_config)
        with pytest.raises(LLMAPIError, match="Missing or malformed 'choices'"):
            client.create_chat_completion([{"role": "user", "content": "hello"}])


# ------------------------------------------------------------------------------
# Entry Point (chat_completion.py) Runner Tests
# ------------------------------------------------------------------------------

def test_run_chat_completion_success(monkeypatch, tmp_path):
    """Verify run_chat_completion executes successfully with code 0."""
    monkeypatch.setenv("API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("API_KEY", "sk-mock-valid-key-999")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o-mini")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "Administer ORS immediately."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp.text = json.dumps(mock_resp.json.return_value)

    test_log = tmp_path / "test_sample.log"
    with patch("src.chat_completion.LOG_FILE_PATH", test_log):
        with patch("httpx.Client.post", return_value=mock_resp):
            exit_code = run_chat_completion()
            assert exit_code == 0
            assert test_log.is_file()
            log_content = test_log.read_text(encoding="utf-8")
            assert "Administer ORS immediately." in log_content
            assert "sk-mock-valid-key-999" not in log_content  # strictly masked


def test_run_chat_completion_config_error(monkeypatch):
    """Verify run_chat_completion returns code 1 on missing configuration."""
    monkeypatch.delenv("API_KEY", raising=False)
    exit_code = run_chat_completion()
    assert exit_code == 1


def test_run_chat_completion_auth_error(monkeypatch):
    """Verify run_chat_completion returns code 2 on 401 authentication error."""
    monkeypatch.setenv("API_KEY", "sk-invalid-auth-key-000")

    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = '{"error": {"message": "Invalid API key"}}'
    mock_resp.json.return_value = {"error": {"message": "Invalid API key"}}

    with patch("httpx.Client.post", return_value=mock_resp):
        exit_code = run_chat_completion()
        assert exit_code == 2
