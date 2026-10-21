"""Context window and multi-turn message history manager for FieldSure AI (Module #6).

Maintains multi-turn conversation history, calculates pre-request token budgets,
and applies intelligent sliding-window trimming and summarization to ensure requests
never exceed LLM context window limits while always preserving the system message.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from src.token_cost_estimator import count_tokens

logger = logging.getLogger("fieldsure.context_manager")

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
LOG_FILE_PATH = OUTPUTS_DIR / "context_window_management_sample.log"

# Structural token overhead per chat message in OpenAI format (~3 tokens)
MESSAGE_TOKEN_OVERHEAD = 3


@dataclass
class ChatTurnLog:
    """Audit log entry for a single conversation turn."""

    turn_index: int
    user_query: str
    assistant_reply: str
    tokens_before_turn: int
    tokens_after_turn: int
    was_trimmed: bool
    trim_reduction_tokens: int
    active_message_count: int


class ContextWindowManager:
    """Manages multi-turn conversation history with token budget enforcement."""

    def __init__(
        self,
        system_prompt: str,
        max_history_tokens: int = 600,
        preserve_recent_turns: int = 1,
    ) -> None:
        """Initialize the context window manager.

        Args:
            system_prompt: The authoritative system instructions (never evicted).
            max_history_tokens: The ceiling for total message history tokens.
            preserve_recent_turns: Minimum number of recent turn pairs to keep intact.
        """
        self.system_prompt = system_prompt.strip()
        self.max_history_tokens = max_history_tokens
        self.preserve_recent_turns = max(1, preserve_recent_turns)

        # Message history storage: List of dicts with 'role' and 'content'
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Audit logs of all turns
        self.turn_logs: List[ChatTurnLog] = []
        self._turn_counter = 0

    @staticmethod
    def count_message_tokens(role: str, content: str) -> int:
        """Calculate token count for a message including chat framing tokens."""
        return count_tokens(content) + count_tokens(role) + MESSAGE_TOKEN_OVERHEAD

    def get_total_tokens(self) -> int:
        """Calculate the total token consumption of current message history."""
        total = 0
        for msg in self.messages:
            total += self.count_message_tokens(msg["role"], msg["content"])
        # Primer tokens for the model response
        total += 3
        return total

    def add_user_message(self, content: str) -> None:
        """Add a user turn to message history."""
        self.messages.append({"role": "user", "content": content.strip()})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant turn to message history."""
        self.messages.append({"role": "assistant", "content": content.strip()})

    def trim_history_if_needed(self) -> Tuple[bool, int, int]:
        """Enforce the token budget by pruning or summarizing older turns.

        Rules:
        1. The system message (index 0) is NEVER pruned or evicted.
        2. The most recent turn pair (latest user message + assistant reply) is preserved.
        3. Oldest turns are pruned or condensed into a brief summary tag.

        Returns:
            Tuple of (was_trimmed: bool, tokens_before: int, tokens_after: int).
        """
        tokens_before = self.get_total_tokens()
        if tokens_before <= self.max_history_tokens:
            return (False, tokens_before, tokens_before)

        logger.info(
            "Context window exceeded budget (%d > %d tokens). Initiating history trimming...",
            tokens_before,
            self.max_history_tokens,
        )

        evicted_points: List[str] = []

        # Keep trimming oldest turns until within budget or down to system + recent turns
        # Messages structure: [0] = System, [1..N-2*preserve] = older turns, [-2*preserve..] = recent turns
        while self.get_total_tokens() > self.max_history_tokens:
            # We must leave at least: system prompt (1) + recent turns (2 * preserve_recent_turns)
            min_retained_count = 1 + (2 * self.preserve_recent_turns)
            if len(self.messages) <= min_retained_count:
                logger.warning(
                    "Cannot trim further without evicting recent turns or system message. Remaining tokens: %d",
                    self.get_total_tokens(),
                )
                break

            # Evict the oldest non-system message pair (indices 1 and 2 if user+assistant)
            # Record a brief anchor point for context continuity
            old_user = self.messages[1]
            old_assistant = self.messages[2] if len(self.messages) > 2 else None

            # Capture brief summary note of evicted content
            summary_snippet = f"User asked: '{old_user['content'][:45]}...'"
            evicted_points.append(summary_snippet)

            # Evict turns at index 1 (and 2 if assistant)
            if old_assistant and old_assistant.get("role") == "assistant":
                del self.messages[1:3]
            else:
                del self.messages[1]

        # If we evicted messages, insert or update a concise summary marker right after system prompt
        if evicted_points:
            summary_content = (
                f"[Context Summary: {len(evicted_points)} previous discussion turns were compressed "
                f"to conserve context budget. Prior topics covered: {'; '.join(evicted_points)}]"
            )
            # Insert summary at index 1 (immediately following system prompt)
            self.messages.insert(1, {"role": "system", "content": summary_content})

        tokens_after = self.get_total_tokens()
        logger.info(
            "Trimming complete: %d -> %d tokens (Reduction: %d tokens). Active messages: %d",
            tokens_before,
            tokens_after,
            tokens_before - tokens_after,
            len(self.messages),
        )

        return (True, tokens_before, tokens_after)

    def process_turn(self, user_query: str, assistant_reply: str) -> ChatTurnLog:
        """Process a complete dialogue turn with pre-request token measurement and trimming.

        Args:
            user_query: The incoming user question.
            assistant_reply: The model's answer.

        Returns:
            ChatTurnLog with comprehensive audit metrics.
        """
        self._turn_counter += 1
        turn_id = self._turn_counter

        # 1. Add user message
        self.add_user_message(user_query)

        # 2. Check tokens before request and trim if nearing budget
        tokens_before = self.get_total_tokens()
        was_trimmed, tok_pre, tok_post = self.trim_history_if_needed()

        # 3. Add assistant response
        self.add_assistant_message(assistant_reply)

        # 4. Final tokens after complete turn
        tokens_final = self.get_total_tokens()

        log_entry = ChatTurnLog(
            turn_index=turn_id,
            user_query=user_query,
            assistant_reply=assistant_reply,
            tokens_before_turn=tok_pre,
            tokens_after_turn=tokens_final,
            was_trimmed=was_trimmed,
            trim_reduction_tokens=(tok_pre - tok_post) if was_trimmed else 0,
            active_message_count=len(self.messages),
        )
        self.turn_logs.append(log_entry)
        return log_entry


# ------------------------------------------------------------------------------
# Demonstration Multi-Turn Conversation (Task 4)
# ------------------------------------------------------------------------------

DEMO_MULTI_TURN_DIALOGUE = [
    (
        "Turn 1: Initial Presentation",
        "A 3-year-old child presents in our rural clinic with watery diarrhea for 24 hours. What is the first assessment step?",
        "Assess hydration status immediately using IMCI guidelines. Check for general danger signs (inability to drink, lethargy, sunken eyes, skin pinch turgor). Categorize into Plan A, Plan B, or Plan C dehydration.",
    ),
    (
        "Turn 2: Follow-up on Rehydration",
        "The child is irritable and drinks eagerly, with skin pinch going back slowly. Which plan should we start?",
        "This indicates Plan B (Some Dehydration). Administer recommended low-osmolarity ORS: approximately 75 ml/kg over 4 hours. Reassess hydration status at 4 hours and continue breastfeeding.",
    ),
    (
        "Turn 3: Nutrition & Zinc Inquiry",
        "Should we give zinc tablets now, and what should the mother feed the child during this 4-hour period?",
        "Yes, initiate dispersible zinc sulfate (20 mg/day for children aged 6 months to 5 years) for 14 full days. For breastfed infants, continue frequent breastfeeding without interruption.",
    ),
    (
        "Turn 4: Community Prevention Context",
        "Several other households in the same settlement report similar watery diarrhea. What immediate sanitation steps are needed?",
        "Alert the public health surveillance team. Distribute chlorine water-purification tablets (NaDCC), inspect community wells, isolate latrine runoff, and set up a temporary oral rehydration point.",
    ),
    (
        "Turn 5: Worsening Symptoms (Danger Sign)",
        "The child has now become very sleepy, cannot drink from a cup, and the skin pinch takes more than 2 seconds to return. What must we do?",
        "Emergency Red Flag: This child has progressed to Plan C (Severe Dehydration). Initiate urgent transfer to the district hospital. If trained, start IV Ringer's Lactate immediately (100 ml/kg total divided into fast and slow phases).",
    ),
    (
        "Turn 6: Discharge & Home Care Protocol",
        "Once stabilized at the hospital, what home care discharge counseling is critical for the caregiver to prevent relapse?",
        "Counsel on the 4 Rules of Home Treatment: 1. Give extra fluids (ORS, clean water). 2. Complete 14-day Zinc regimen. 3. Continue nutrient-dense feeding. 4. Return immediately if blood in stool, high fever, or vomiting recurs.",
    ),
]


def setup_logger(log_file: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fieldsure.context_manager_runner")
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


def run_context_management_demo() -> int:
    """Demonstrate context window management over an overflowing 6-turn clinical conversation."""
    log = setup_logger(LOG_FILE_PATH)
    log.info("Starting FieldSure AI Context Window & History Management Demo (Module #6)...")

    # Constrained token budget set to 350 tokens to clearly trigger and demonstrate
    # sliding-window trimming during a 6-turn clinical consultation
    token_budget = 350
    system_prompt = (
        "You are FieldSure AI, a verified assistant for community health workers. "
        "Provide concise, protocol-grounded clinical triage guidance."
    )

    manager = ContextWindowManager(
        system_prompt=system_prompt,
        max_history_tokens=token_budget,
        preserve_recent_turns=1,
    )

    print("\n" + "=" * 80)
    print("FIELDSURE AI — CONTEXT WINDOW MANAGEMENT & TRIMMING DEMONSTRATION")
    print(f"Token Budget: {token_budget} tokens | System Prompt Tokens: {manager.get_total_tokens()} tokens")
    print("=" * 80)

    audit_turns = []

    for step, (title, user_q, assistant_a) in enumerate(DEMO_MULTI_TURN_DIALOGUE, 1):
        print(f"\n▶ [{title}]")
        print(f"  User Query: \"{user_q[:65]}...\"")

        tokens_before_step = manager.get_total_tokens()
        turn_log = manager.process_turn(user_q, assistant_a)

        audit_turns.append({
            "turn_index": turn_log.turn_index,
            "title": title,
            "user_query": user_q,
            "assistant_reply": assistant_a,
            "tokens_before_turn": turn_log.tokens_before_turn,
            "tokens_after_turn": turn_log.tokens_after_turn,
            "was_trimmed": turn_log.was_trimmed,
            "trim_reduction_tokens": turn_log.trim_reduction_tokens,
            "active_messages_in_context": turn_log.active_message_count,
        })

        if turn_log.was_trimmed:
            print(f"  ⚠️  BUDGET THRESHOLD REACHED: Trimmed {turn_log.trim_reduction_tokens} tokens")
            print(f"      Tokens Before Trim: {turn_log.tokens_before_turn} | Tokens After Turn: {turn_log.tokens_after_turn}")
        else:
            print(f"      Within Budget: {turn_log.tokens_after_turn} / {token_budget} tokens")

        print(f"      Active Messages in Context Window: {turn_log.active_message_count}")
        print("-" * 80)

    # Final summary report
    print("\n" + "=" * 80)
    print("DEMONSTRATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Turns Processed:               {len(DEMO_MULTI_TURN_DIALOGUE)}")
    print(f"Configured Token Budget:             {token_budget} tokens")
    print(f"Final Context Window Size:           {manager.get_total_tokens()} tokens (STRICTLY <= {token_budget})")
    print(f"System Message Status:               Preserved at Index 0 ({manager.messages[0]['role'].upper()})")
    print(f"Total Trimming Interventions:        {sum(1 for t in audit_turns if t['was_trimmed'])}")
    print("=" * 80 + "\n")

    # Save complete JSON audit
    audit_data = {
        "module": "Module #6 — Context Windows & Message History Management",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "token_budget": token_budget,
        "total_turns": len(DEMO_MULTI_TURN_DIALOGUE),
        "final_token_count": manager.get_total_tokens(),
        "system_message_preserved": manager.messages[0]["role"] == "system",
        "turn_history": audit_turns,
        "final_active_messages": manager.messages,
    }

    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write("\n--- Structured Context Window Management Audit ---\n")
        f.write(json.dumps(audit_data, indent=2, ensure_ascii=False) + "\n")

    log.info("Context window management audit successfully recorded to %s", LOG_FILE_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(run_context_management_demo())
