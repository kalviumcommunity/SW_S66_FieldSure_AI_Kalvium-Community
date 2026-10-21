"""Chat completion entry point for FieldSure AI (Module #3: LLM API Access & First Completion Call).

Executes a baseline public health query against an OpenAI-compatible endpoint,
safely logs request/response payloads without credential leakage, and handles failures cleanly.
Tasks: Config from .env, chat request/response, sanitized logging, and error handling.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

from src.config import LLMConfigError, load_llm_config, mask_api_key
from src.llm_client import (
    ChatCompletionResult,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMAPIError,
    OpenAICompatibleClient,
)

# Output log path specified in architecture
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
LOG_FILE_PATH = OUTPUTS_DIR / "chat_completion_sample.log"


def setup_logger(log_file: Path) -> logging.Logger:
    """Configure structured logging to both file and console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fieldsure.chat_completion")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-runs
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler (console)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def format_log_entry(
    endpoint: str,
    model: str,
    masked_key: str,
    messages: List[Dict[str, str]],
    result: ChatCompletionResult,
) -> str:
    """Format an audit record containing request, response, and token metrics."""
    timestamp = datetime.now(timezone.utc).isoformat()
    entry: Dict[str, Any] = {
        "timestamp": timestamp,
        "endpoint": endpoint,
        "model": model,
        "api_key_masked": masked_key,
        "request": {
            "messages": messages,
        },
        "response": {
            "role": result.role,
            "content": result.content,
        },
        "token_usage": {
            "prompt_tokens": result.usage.prompt_tokens if result.usage else None,
            "completion_tokens": result.usage.completion_tokens if result.usage else None,
            "total_tokens": result.usage.total_tokens if result.usage else None,
        },
    }
    return json.dumps(entry, indent=2, ensure_ascii=False)


def run_chat_completion() -> int:
    """Execute the chat completion workflow.

    Returns:
        0 on success,
        1 on configuration errors,
        2 on API/communication errors.
    """
    logger = setup_logger(LOG_FILE_PATH)
    logger.info("Initializing FieldSure AI LLM client...")

    # Task 1: Configure client from environment
    try:
        config = load_llm_config()
    except LLMConfigError as err:
        logger.error("Configuration Error: %s", err)
        print(f"\n[CONFIGURATION ERROR] {err}", file=sys.stderr)
        print("Hint: Check your .env file or copy from .env.example.\n", file=sys.stderr)
        return 1

    masked_key = mask_api_key(config.api_key)
    logger.info(
        "Configuration loaded: Base URL=%s | Model=%s | Key=%s",
        config.api_base_url,
        config.chat_model,
        masked_key,
    )

    client = OpenAICompatibleClient(config)

    # Task 2: Prepare system and user messages
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are FieldSure AI, an expert assistant for community health workers. "
                "Provide accurate, practical, and grounded public health guidance. "
                "Always state that guidance is for triage and reference, not diagnostic decisions."
            ),
        },
        {
            "role": "user",
            "content": (
                "What are the early warning signs of dehydration in pediatric diarrhea cases, "
                "and what immediate field actions are recommended before referral to a health center?"
            ),
        },
    ]

    # Task 3: Log outgoing request payload
    logger.info("Sending chat completion request with %d messages...", len(messages))
    for idx, msg in enumerate(messages, 1):
        logger.info("  [Message %d - %s]: %s", idx, msg["role"].upper(), msg["content"])

    # Task 4: Execute call and handle errors clearly without raw stack traces
    try:
        result = client.create_chat_completion(messages=messages, temperature=0.2)
    except LLMAuthenticationError as err:
        logger.error("Authentication Failure: %s", err)
        print(f"\n[AUTHENTICATION ERROR (401)] {err}\n", file=sys.stderr)
        return 2
    except LLMRateLimitError as err:
        logger.error("Rate Limit / Quota Exceeded: %s", err)
        print(f"\n[RATE LIMIT ERROR (429)] {err}\n", file=sys.stderr)
        return 2
    except LLMConnectionError as err:
        logger.error("Connection Failure: %s", err)
        print(f"\n[CONNECTION ERROR] {err}\n", file=sys.stderr)
        return 2
    except LLMAPIError as err:
        logger.error("LLM API Error: %s", err)
        print(f"\n[API ERROR] {err}\n", file=sys.stderr)
        return 2
    except Exception as err:
        logger.error("Unexpected error during completion call: %s", err)
        print(f"\n[UNEXPECTED ERROR] An unexpected error occurred: {err}\n", file=sys.stderr)
        return 2

    # Print model's text reply to stdout
    print("\n" + "=" * 70)
    print("FIELDSURE AI — CHAT COMPLETION RESPONSE")
    print("=" * 70)
    print(f"Model: {result.model}")
    print("-" * 70)
    print(result.content)
    print("-" * 70)

    # Token usage logging and display
    if result.usage:
        usage_str = (
            f"Prompt Tokens: {result.usage.prompt_tokens} | "
            f"Completion Tokens: {result.usage.completion_tokens} | "
            f"Total Tokens: {result.usage.total_tokens}"
        )
        print(f"Token Usage: {usage_str}")
        logger.info("Token metrics: %s", usage_str)
    else:
        print("Token Usage: [Usage metrics not supplied by provider]")
        logger.info("Token metrics: not supplied by provider")
    print("=" * 70 + "\n")

    # Write formatted audit exchange to log
    audit_record = format_log_entry(
        endpoint=config.api_base_url,
        model=result.model,
        masked_key=masked_key,
        messages=messages,
        result=result,
    )
    logger.info("Chat completion transaction successfully logged to %s", LOG_FILE_PATH)

    # Append complete JSON audit block to sample log for review
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("\n--- Structured Transaction Audit ---\n")
        f.write(audit_record + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_chat_completion())
