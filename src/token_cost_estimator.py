"""Tokenization and cost estimation module for FieldSure AI (Module #5).

Accurately measures token consumption using tiktoken, calculates provider costs
accounting for differential input/output rates, and analyzes text length vs token ratios.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("fieldsure.token_cost")

# Default outputs directory and log path
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
LOG_FILE_PATH = OUTPUTS_DIR / "token_cost_estimation_sample.log"


# ------------------------------------------------------------------------------
# Pricing Models (per 1 Million Tokens in USD)
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPricing:
    """Pricing configuration per 1M tokens for a specific model."""

    model_name: str
    input_price_per_1m: float
    output_price_per_1m: float
    context_window_tokens: int

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Compute separate input, output, and total costs in USD."""
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m
        return {
            "model": self.model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(input_cost + output_cost, 6),
        }


# Industry standard pricing benchmarks
BENCHMARK_MODELS: Dict[str, ModelPricing] = {
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        context_window_tokens=128_000,
    ),
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
        context_window_tokens=128_000,
    ),
    "claude-3-5-sonnet": ModelPricing(
        model_name="claude-3-5-sonnet",
        input_price_per_1m=3.00,
        output_price_per_1m=15.00,
        context_window_tokens=200_000,
    ),
}


# ------------------------------------------------------------------------------
# Tokenizer Engine
# ------------------------------------------------------------------------------

def get_tokenizer(encoding_name: str = "cl100k_base"):
    """Acquire tiktoken encoding or fallback cleanly if offline."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception as err:
        logger.warning("tiktoken unavailable (%s), using regex subword estimator.", err)
        return None


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Accurately count subword tokens for a given text string.

    Args:
        text: The string to tokenize.
        encoding_name: The tokenizer encoding to use (default: 'cl100k_base').

    Returns:
        Integer count of tokens.
    """
    if not text:
        return 0

    tokenizer = get_tokenizer(encoding_name)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass

    # Heuristic subword approximation (~4 characters per token for English text)
    # Splits on whitespace and punctuation to account for subword fragmentation
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    # Estimate extra subwords for words longer than 6 characters
    extra = sum(max(0, (len(t) - 4) // 4) for t in tokens if t.isalnum())
    return max(1, len(tokens) + extra)


# ------------------------------------------------------------------------------
# Text Samples for Corpus Analysis (Task 2)
# ------------------------------------------------------------------------------

SAMPLE_SHORT_QUESTION = (
    "What are the primary clinical danger signs of severe dehydration in children under five?"
)

SAMPLE_CLINICAL_PARAGRAPH = (
    "In community flood emergencies, cholera triage requires rapid categorization into Plan A, "
    "Plan B, or Plan C dehydration. Patients presenting with sunken eyes, poor skin turgor, "
    "and extreme lethargy must be stabilized immediately with low-osmolarity Oral Rehydration "
    "Salts (ORS) while preparing urgent transport to a Primary Health Center (PHC). Field health "
    "workers must maintain 0.05% chlorine handwashing points and immediately isolate patient vomitus "
    "to arrest localized fecal-oral outbreak transmission."
)

SAMPLE_FULL_DOCUMENT = (
    "# FieldSure AI Clinical Protocol Advisory: Pediatric Malaria & Febrile Illness Triage\n\n"
    "## 1. Executive Protocol Summary\n"
    "This guidance protocol provides community health workers (CHWs) and auxiliary nurse midwives "
    "(ANMs) with standardized clinical triage procedures for pediatric patients presenting with "
    "acute febrile illness in malaria-endemic surveillance zones.\n\n"
    "## 2. Immediate Initial Assessment\n"
    "All children presenting with axillary temperature exceeding 37.5°C or a caregiver-reported "
    "fever within the previous 48 hours must undergo rapid screening. Assess for the following "
    "general danger signs:\n"
    "- Inability to drink or breastfeed.\n"
    "- Persistent vomiting of all ingested fluids.\n"
    "- Generalized convulsions or abnormal motor spasms.\n"
    "- Lethargy, unconsciousness, or extreme prostration.\n\n"
    "## 3. Diagnostic Testing & Rapid Diagnostic Tests (mRDT)\n"
    "Perform a blood finger-prick malaria Rapid Diagnostic Test (mRDT) for Plasmodium falciparum "
    "and Plasmodium vivax in accordance with sterile single-use lancet protocols. Read test strips "
    "precisely between 15 and 20 minutes.\n\n"
    "## 4. Field Treatment Protocols\n"
    "1. Uncomplicated P. falciparum Malaria: Administer weight-based Artemisinin-based Combination "
    "Therapy (ACT), specifically Artemether-Lumefantrine, with food or milk to optimize gastrointestinal "
    "absorption.\n"
    "2. Severe Malaria Danger Signs: Administer a single pre-referral rectal artesunate capsule "
    "(100 mg for infants < 3 years; 200 mg for older children) and arrange immediate emergency "
    "ambulance transfer.\n\n"
    "## 5. Non-Diagnostic Notice\n"
    "FieldSure AI is an informational reference system designed for community healthcare workers. "
    "It does not supersede physician evaluation or national treatment guidelines."
)


# ------------------------------------------------------------------------------
# Length vs Token Relationship Analysis (Task 4)
# ------------------------------------------------------------------------------

TEST_DIVERSE_SAMPLES: List[Dict[str, str]] = [
    {
        "category": "Standard English Prose",
        "description": "Simple conversational English field advice",
        "text": "Please drink clean boiled water and wash hands thoroughly with soap before eating.",
    },
    {
        "category": "Medical Jargon & Multisyllabic Terms",
        "description": "Complex clinical terms that fragment into multiple BPE subwords",
        "text": "Meningoencephalitis, gastroenteritis, thrombocythemia, and hyperbilirubinemia.",
    },
    {
        "category": "Structured Data / JSON",
        "description": "Code and data schemas containing punctuation, keys, and formatting syntax",
        "text": '{"patient_id": "CHW-8491", "triage_grade": "Plan_C", "bp_systolic": 85, "active": true}',
    },
    {
        "category": "Multilingual Text (Hindi Health Advisory)",
        "description": "Non-Latin Devanagari script showing higher token density per character",
        "text": "कृपया पीने के लिए केवल उबला हुआ पानी उपयोग करें और खाने से पहले हाथ अच्छी तरह धोएं।",
    },
]


@dataclass
class TokenProfile:
    """Metrics profiling text length against token counts."""

    category: str
    description: str
    char_count: int
    word_count: int
    token_count: int
    chars_per_token: float
    tokens_per_word: float


def profile_text(category: str, description: str, text: str) -> TokenProfile:
    """Analyze character, word, and token relationships for a snippet."""
    char_count = len(text)
    word_count = len(text.split())
    tokens = count_tokens(text)
    chars_per_token = round(char_count / tokens, 2) if tokens else 0.0
    tokens_per_word = round(tokens / word_count, 2) if word_count else 0.0

    return TokenProfile(
        category=category,
        description=description,
        char_count=char_count,
        word_count=word_count,
        token_count=tokens,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
    )


# ------------------------------------------------------------------------------
# Runner & Sample Generation
# ------------------------------------------------------------------------------

def setup_logger(log_file: Path) -> logging.Logger:
    """Set up dual file and console logging."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fieldsure.token_cost_runner")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def run_token_and_cost_estimation() -> int:
    """Execute complete token analysis, cost estimation, and relationship reporting."""
    log = setup_logger(LOG_FILE_PATH)
    log.info("Starting FieldSure AI Token & Cost Estimation Analysis (Module #5)...")

    # --------------------------------------------------------------------------
    # Task 1 & 2: Report Counts for Three Varying Samples
    # --------------------------------------------------------------------------
    samples = [
        ("Short Question (Query)", "User input triage query", SAMPLE_SHORT_QUESTION),
        ("Clinical Paragraph (Chunk)", "Medium size text chunk from field guide", SAMPLE_CLINICAL_PARAGRAPH),
        ("Full Protocol Document", "Comprehensive advisory document", SAMPLE_FULL_DOCUMENT),
    ]

    sample_reports = []
    print("\n" + "=" * 80)
    print("TASK 1 & 2: TOKEN COUNTS FOR THREE CORPUS SAMPLES")
    print("=" * 80)

    for name, desc, text in samples:
        profile = profile_text(name, desc, text)
        sample_reports.append(profile)
        print(f"► Sample: {profile.category}")
        print(f"  Description:  {profile.description}")
        print(f"  Characters:   {profile.char_count:,}")
        print(f"  Words:        {profile.word_count:,}")
        print(f"  Tokens:       {profile.token_count:,}")
        print(f"  Chars/Token:  {profile.chars_per_token} chars/token")
        print(f"  Tokens/Word:  {profile.tokens_per_word} tokens/word")
        print("-" * 80)

    # --------------------------------------------------------------------------
    # Task 3: Cost Estimates Across Model Benchmarks (Input vs Output)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 3: COST ESTIMATION (DIFFERENTIAL INPUT VS OUTPUT PRICING)")
    print("=" * 80)

    # Typical RAG query scenario:
    # Input tokens = System prompt (350) + Retrieved context (3 chunks = 1,500) + User query (30) = 1,880 tokens
    # Output tokens = Model answer (250 tokens)
    input_tokens = 1_880
    output_tokens = 250

    cost_breakdowns = []
    for model_key, pricing in BENCHMARK_MODELS.items():
        cost = pricing.calculate_cost(input_tokens, output_tokens)
        cost_breakdowns.append(cost)

        # Scale to 10,000 monthly field queries
        monthly_cost = round(cost["total_cost_usd"] * 10_000, 2)

        print(f"► Model: {pricing.model_name}")
        print(f"  Rate:           ${pricing.input_price_per_1m:.2f}/1M input | ${pricing.output_price_per_1m:.2f}/1M output")
        print(f"  Per-Query Input Cost  ({input_tokens:,} tokens):  ${cost['input_cost_usd']:.6f}")
        print(f"  Per-Query Output Cost ({output_tokens:,} tokens):   ${cost['output_cost_usd']:.6f}")
        print(f"  Single Query Total:                        ${cost['total_cost_usd']:.6f}")
        print(f"  Scale Estimate (10,000 queries/month):     ${monthly_cost:,.2f} USD")
        print("-" * 80)

    # --------------------------------------------------------------------------
    # Task 4: Length vs Token Relationship Across Content Domains
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TASK 4: LENGTH–TOKEN RELATIONSHIP ACROSS DIVERSE CONTENT TYPES")
    print("=" * 80)

    diverse_profiles = []
    for item in TEST_DIVERSE_SAMPLES:
        p = profile_text(item["category"], item["description"], item["text"])
        diverse_profiles.append(p)
        print(f"► Domain: {p.category}")
        print(f"  Text:         \"{item['text'][:60]}...\"" if len(item['text']) > 60 else f"  Text:         \"{item['text']}\"")
        print(f"  Length:       {p.char_count} chars | {p.word_count} words | {p.token_count} tokens")
        print(f"  Ratio:        {p.chars_per_token} chars/token | {p.tokens_per_word} tokens/word")
        print("-" * 80)

    print("\nKey Takeaways:")
    print("1. Standard English averages ~4.0 to 4.5 characters per token.")
    print("2. Medical jargon fragments into subwords (~2.5 to 3.0 characters per token), raising token counts.")
    print("3. Structured syntax (JSON/code) incurs overhead from individual punctuation tokens.")
    print("4. Non-Latin scripts (Hindi/Devanagari) require significantly more tokens (~1.0 to 1.5 chars/token).")
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # Task 5: Save Structured Audit Record
    # --------------------------------------------------------------------------
    audit_data = {
        "module": "Module #5 — Tokens, Tokenization & Cost Estimation",
        "sample_counts": [
            {
                "category": p.category,
                "description": p.description,
                "characters": p.char_count,
                "words": p.word_count,
                "tokens": p.token_count,
                "chars_per_token": p.chars_per_token,
                "tokens_per_word": p.tokens_per_word,
            }
            for p in sample_reports
        ],
        "pricing_evaluations": cost_breakdowns,
        "length_token_relationships": [
            {
                "category": p.category,
                "description": p.description,
                "characters": p.char_count,
                "words": p.word_count,
                "tokens": p.token_count,
                "chars_per_token": p.chars_per_token,
                "tokens_per_word": p.tokens_per_word,
            }
            for p in diverse_profiles
        ],
    }

    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("\n--- Structured Token & Cost Estimation Audit ---\n")
        f.write(json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n")

    log.info("Token and cost estimation analysis successfully saved to %s", LOG_FILE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(run_token_and_cost_estimation())
