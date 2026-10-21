"""OpenAI-compatible LLM client wrapper for FieldSure AI.

Supports any OpenAI-compatible API endpoint (OpenAI, OpenRouter, Ollama, vLLM, etc.)
with robust error classification, timeouts, and sanitized logging.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import Any, Dict, List, Optional
import httpx

from src.config import LLMConfig, mask_api_key

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Domain Exceptions
# ------------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM client failures."""


class LLMAuthenticationError(LLMError):
    """Raised on HTTP 401 Unauthorized — invalid or expired credentials."""


class LLMRateLimitError(LLMError):
    """Raised on HTTP 429 Too Many Requests — rate limit or quota exceeded."""


class LLMConnectionError(LLMError):
    """Raised on network connectivity issues or request timeouts."""


class LLMAPIError(LLMError):
    """Raised on other HTTP error codes or unexpected API response payloads."""


# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class UsageInfo:
    """Token consumption metrics returned by the LLM API."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["UsageInfo"]:
        """Parse token usage if provided by provider."""
        if not data:
            return None
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
        )


@dataclass
class ChatCompletionResult:
    """Encapsulates the successful completion reply and exchange metadata."""

    content: str
    model: str
    role: str = "assistant"
    usage: Optional[UsageInfo] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------
# Client Implementation
# ------------------------------------------------------------------------------

class OpenAICompatibleClient:
    """Resilient client communicating with any OpenAI-compatible chat API."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._endpoint_url = self._build_endpoint_url(config.api_base_url)

    @staticmethod
    def _build_endpoint_url(base_url: str) -> str:
        """Construct normalized chat completions URL."""
        cleaned = base_url.strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        return f"{cleaned}/chat/completions"

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletionResult:
        """Send a chat completion request to the configured LLM endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature (default 0.2 for deterministic guidance).
            max_tokens: Optional limit on generated completion tokens.

        Returns:
            ChatCompletionResult containing text reply, model name, and token usage.

        Raises:
            LLMAuthenticationError: On HTTP 401 (invalid/expired API key).
            LLMRateLimitError: On HTTP 429 (quota or concurrency exceeded).
            LLMConnectionError: On connection failure or timeout.
            LLMAPIError: On other API error responses.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.config.chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.debug(
            "Dispatching LLM chat completion to %s (model: %s, key: %s)",
            self._endpoint_url,
            self.config.chat_model,
            mask_api_key(self.config.api_key),
        )

        try:
            with httpx.Client(timeout=self.config.request_timeout) as client:
                response = client.post(
                    self._endpoint_url,
                    headers=headers,
                    json=payload,
                )
        except (httpx.ConnectError, httpx.NetworkError) as err:
            raise LLMConnectionError(
                f"Failed to connect to LLM server at '{self._endpoint_url}'. "
                f"Verify network connectivity and endpoint status. Details: {err}"
            ) from err
        except httpx.TimeoutException as err:
            raise LLMConnectionError(
                f"Request to LLM server timed out after {self.config.request_timeout}s. Details: {err}"
            ) from err
        except Exception as err:
            raise LLMConnectionError(f"Unexpected connection failure: {err}") from err

        # Process HTTP status codes
        self._handle_status_errors(response)

        # Parse JSON body
        try:
            body = response.json()
        except json.JSONDecodeError as err:
            raise LLMAPIError(
                f"LLM API returned invalid JSON (HTTP {response.status_code}): {response.text[:200]}"
            ) from err

        choices = body.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMAPIError(f"Missing or malformed 'choices' in API response: {body}")

        first_choice = choices[0]
        message_obj = first_choice.get("message", {})
        content = message_obj.get("content", "")
        role = message_obj.get("role", "assistant")
        model = body.get("model", self.config.chat_model)
        usage = UsageInfo.from_dict(body.get("usage"))

        return ChatCompletionResult(
            content=content,
            model=model,
            role=role,
            usage=usage,
            raw_response=body,
        )

    def _handle_status_errors(self, response: httpx.Response) -> None:
        """Inspect HTTP response and raise domain-specific, actionable exceptions."""
        if response.status_code == 200:
            return

        # Attempt to extract error details if provided by OpenAI/provider schema
        error_detail = response.text.strip()
        try:
            err_json = response.json()
            if isinstance(err_json, dict) and "error" in err_json:
                raw_err = err_json["error"]
                if isinstance(raw_err, dict):
                    error_detail = raw_err.get("message", error_detail)
                elif isinstance(raw_err, str):
                    error_detail = raw_err
        except Exception:
            pass

        if response.status_code == 401:
            raise LLMAuthenticationError(
                f"Authentication failed (HTTP 401): Invalid or expired API key. "
                f"Please verify 'API_KEY' in your .env file. Provider message: {error_detail}"
            )

        if response.status_code == 429:
            raise LLMRateLimitError(
                f"Rate limit or quota exceeded (HTTP 429). "
                f"Check provider account balance, plan limits, or retry with backoff. "
                f"Provider message: {error_detail}"
            )

        if response.status_code == 404:
            raise LLMAPIError(
                f"Endpoint or model not found (HTTP 404): Check API_BASE_URL ('{self._endpoint_url}') "
                f"and CHAT_MODEL ('{self.config.chat_model}'). Provider message: {error_detail}"
            )

        raise LLMAPIError(
            f"LLM API returned HTTP {response.status_code}: {error_detail}"
        )
