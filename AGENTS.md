# FieldSure AI — AI Agent Guidelines (AGENTS.md)

> **Mandatory Operational Instructions**: Any AI coding agent working within the FieldSure AI repository must follow this operational protocol strictly.

---

## 1. Pre-Implementation Checklist (Before Making Changes)

1. **Read Core Planning Documents**:
   - Inspect [PRD.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/PRD.md) for product context, requirements, and constraints.
   - Inspect [ARCHITECTURE.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/ARCHITECTURE.md) for data flow, directory layout, and integration boundaries.
   - Inspect [DESIGN.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/DESIGN.md) for technical decisions, schemas, and library choices.
   - Inspect [RULES.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/RULES.md) for strict code quality, security, and optimization rules.
   - Inspect [TASKS.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/TASKS.md) to check module status, subtasks, and acceptance criteria.
   - Inspect [MEMORY.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/MEMORY.md) for current state, decisions, and reusable routines.
2. **Inspect Existing Code**:
   - Always check existing files and functions before creating new ones. Never duplicate logic.
3. **Strict Scope Control**:
   - Implement **ONLY** the single module explicitly requested by the user.
   - Never implement Member 2 or Member 3 modules.
   - Do not generate speculative code, placeholder stubs, or future-module files in advance.

---

## 2. Implementation Execution Protocol

1. **Adhere to [RULES.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/RULES.md)**:
   - Clean, readable, PEP-8 compliant Python with type hints.
   - Standard library first; only use dependencies agreed upon in [DESIGN.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/DESIGN.md).
   - Zero hardcoded secrets; zero credential leakage in logs or console.
   - Clean error handling with informative domain messages; no raw unhandled tracebacks.
2. **Maintain Single-Responsibility Functions**:
   - Write simple, focused functions instead of complex object-oriented hierarchies.

---

## 3. Verification & Reporting Protocol

1. **Execute Tests**:
   - Run relevant tests using `uv run pytest <test_path>`.
   - Never claim tests passed without executing them and verifying the output.
2. **Document What Changed & Why**:
   - Explain modifications clearly, citing architectural rationale and design choices.
3. **Update Tracking Files**:
   - Update [TASKS.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/TASKS.md) status (e.g., mark subtasks checked, set status to `Completed`) **only after** automated test verification.
   - Update [MEMORY.md](file:///c:/Users/Vansh/OneDrive/Desktop/Sprint%20-2/SW_S66_FieldSure_AI_Kalvium-Community/MEMORY.md) with any new reusable functions or completed milestones.
