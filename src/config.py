"""Configuration management for FieldSure AI LLM client.

Reads environment variables safely from .env and ensures zero credential leakage.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class LLMConfigError(ValueError):
    """Raised when mandatory LLM configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    """Immutable LLM client configuration values."""

    api_base_url: str
    api_key: str
    chat_model: str
    request_timeout: float = 30.0


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe logging and display without leaking credentials.

    Examples:
        'sk-proj-abcdef123456' -> 'sk-...3456'
        'short' -> '***'
    """
    if not api_key:
        return "[NOT SET]"
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:3]}...{api_key[-4:]}"


def load_llm_config(env_path: Optional[Path] = None) -> LLMConfig:
    """Load and validate LLM configuration from environment or .env file.

    Args:
        env_path: Optional explicit path to the .env file. If None, resolves from
                  the repository root or current working directory.

    Returns:
        LLMConfig instance with validated connection parameters.

    Raises:
        LLMConfigError: If mandatory keys (like API_KEY) are missing or set to placeholder defaults.
    """
    if env_path is not None:
        load_dotenv(dotenv_path=env_path)
    else:
        # Check standard locations (repo root or current directory)
        repo_root = Path(__file__).resolve().parent.parent
        root_env = repo_root / ".env"
        if root_env.is_file():
            load_dotenv(dotenv_path=root_env)
        else:
            load_dotenv()

    # Base URL (default to OpenAI standard endpoint)
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    if not api_base_url:
        api_base_url = "https://api.openai.com/v1"

    # API Key
    api_key = os.getenv("API_KEY", "").strip()
    if not api_key or api_key == "your_llm_api_key_here":
        raise LLMConfigError(
            "API_KEY is not configured or still set to the placeholder. "
            "Please configure 'API_KEY' in your .env file (copy from .env.example) "
            "with a valid OpenAI-compatible API key."
        )

    # Chat Model
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini").strip()
    if not chat_model:
        chat_model = "gpt-4o-mini"

    # Timeout
    timeout_raw = os.getenv("REQUEST_TIMEOUT", "30.0").strip()
    try:
        request_timeout = float(timeout_raw)
    except ValueError:
        request_timeout = 30.0

    return LLMConfig(
        api_base_url=api_base_url,
        api_key=api_key,
        chat_model=chat_model,
        request_timeout=request_timeout,
    )
