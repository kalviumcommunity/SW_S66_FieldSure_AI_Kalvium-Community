"""Unit tests for token counting, cost modeling, and token length analysis (Module #5)."""

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from src.token_cost_estimator import (
    BENCHMARK_MODELS,
    ModelPricing,
    SAMPLE_CLINICAL_PARAGRAPH,
    SAMPLE_FULL_DOCUMENT,
    SAMPLE_SHORT_QUESTION,
    count_tokens,
    profile_text,
    run_token_and_cost_estimation,
)


def test_count_tokens_empty():
    """Verify empty or None text returns zero tokens."""
    assert count_tokens("") == 0


def test_count_tokens_basic_sentence():
    """Verify basic English sentence token count."""
    text = "FieldSure AI provides reliable public health guidance."
    tokens = count_tokens(text)
    assert tokens > 0
    # Standard sentence of ~8 words should be roughly 8-12 tokens
    assert 6 <= tokens <= 15


def test_three_corpus_samples_monotonicity():
    """Verify that sample token counts scale strictly with content length (Task 2)."""
    tokens_short = count_tokens(SAMPLE_SHORT_QUESTION)
    tokens_para = count_tokens(SAMPLE_CLINICAL_PARAGRAPH)
    tokens_doc = count_tokens(SAMPLE_FULL_DOCUMENT)

    assert tokens_short > 0
    assert tokens_para > tokens_short, "Paragraph must contain more tokens than short question"
    assert tokens_doc > tokens_para, "Full document must contain more tokens than single paragraph"

    # Verify realistic domain bounds
    assert 10 <= tokens_short <= 30
    assert 50 <= tokens_para <= 120
    assert 200 <= tokens_doc <= 500


def test_cost_calculation_math():
    """Verify differential input vs output billing mathematics (Task 3)."""
    pricing = ModelPricing(
        model_name="test-model",
        input_price_per_1m=1.00,
        output_price_per_1m=4.00,
        context_window_tokens=100_000,
    )

    cost = pricing.calculate_cost(input_tokens=500_000, output_tokens=250_000)

    # 500k input @ $1.00/1M = $0.50
    assert cost["input_cost_usd"] == pytest.approx(0.50, rel=1e-4)
    # 250k output @ $4.00/1M = $1.00
    assert cost["output_cost_usd"] == pytest.approx(1.00, rel=1e-4)
    # Total = $1.50
    assert cost["total_cost_usd"] == pytest.approx(1.50, rel=1e-4)
    assert cost["total_tokens"] == 750_000


def test_length_token_disproportionality():
    """Verify length and token count track together but are not strictly proportional (Task 4)."""
    english_prose = "Please drink clean boiled water and wash hands thoroughly with soap before eating."
    hindi_text = "कृपया पीने के लिए केवल उबला हुआ पानी उपयोग करें और खाने से पहले हाथ अच्छी तरह धोएं।"

    profile_en = profile_text("English", "prose", english_prose)
    profile_hi = profile_text("Hindi", "non-latin", hindi_text)

    # Both express similar public health instructions, but character/token density differs
    assert profile_en.chars_per_token > 0
    assert profile_hi.chars_per_token > 0

    # Non-Latin scripts require more subwords per character (lower chars_per_token)
    assert profile_en.chars_per_token > profile_hi.chars_per_token, (
        f"English chars/token ({profile_en.chars_per_token}) should exceed Hindi chars/token ({profile_hi.chars_per_token})"
    )


def test_benchmark_models_configured():
    """Verify industry benchmark models are present and properly priced."""
    assert "gpt-4o-mini" in BENCHMARK_MODELS
    assert "gpt-4o" in BENCHMARK_MODELS
    assert "claude-3-5-sonnet" in BENCHMARK_MODELS

    mini = BENCHMARK_MODELS["gpt-4o-mini"]
    assert mini.output_price_per_1m > mini.input_price_per_1m


def test_run_token_and_cost_estimation_runner(tmp_path):
    """Verify runner executes successfully and writes audit record to log (Task 5)."""
    test_log = tmp_path / "test_token_cost.log"

    with patch("src.token_cost_estimator.LOG_FILE_PATH", test_log):
        exit_code = run_token_and_cost_estimation()
        assert exit_code == 0
        assert test_log.is_file()

        log_content = test_log.read_text(encoding="utf-8")
        assert "TASK 1 & 2" in log_content or "Structured Token & Cost Estimation Audit" in log_content
        assert "gpt-4o-mini" in log_content
