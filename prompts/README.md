# FieldSure AI — System Prompt Design & Analysis

This directory contains system prompt templates and rationale for **Module #4 (Prompt Construction & System/User Roles)**.

---

## 1. Prompt Inventory

| File | Type | Description |
| :--- | :--- | :--- |
| [`system_prompt_baseline.txt`](./system_prompt_baseline.txt) | Baseline / Vague | Minimalist prompt (`You are a helpful assistant for healthcare workers...`) lacking role bounds, negative constraints, or safe refusals. |
| [`system_prompt_constrained.txt`](./system_prompt_constrained.txt) | Production / Constrained | Fully specified prompt defining FieldSure AI's role, operational boundaries, positive/negative scopes, format constraints, deterministic safe refusal, and legal/safety disclaimers. |

---

## 2. Chosen Prompt & Architectural Rationale

**Selected Prompt**: [`system_prompt_constrained.txt`](./system_prompt_constrained.txt)

### Why It Works Better:
1. **Explicit Role & Persona**:
   - Anchors the model as **FieldSure AI**, a verified guidance assistant specifically for public health and community health workers, eliminating generic, conversational filler.
2. **Defined Operational Scope (Positive & Negative)**:
   - Tells the model explicitly what to do (triage steps, danger signs, protocol citations) and what **never** to do (no clinical diagnoses, no unverified prescription dosages).
3. **Deterministic Safe Refusals**:
   - Equips the assistant with an exact fallback sentence when out-of-scope or dangerous diagnostic requests are received, preventing hallucinated medical advice.
4. **Structured Format & Field Usability**:
   - Enforces concise, numbered/bulleted action steps under ~250 words, optimized for mobile reading by field staff in rural/emergency conditions.
5. **Safety & Governance Guardrail**:
   - Enforces a mandatory non-diagnostic disclaimer at the conclusion of every guidance message, complying with public health liability requirements.

---

## 3. Comparison Summary Matrix

| Evaluation Dimension | Baseline Prompt (`system_prompt_baseline.txt`) | Constrained Prompt (`system_prompt_constrained.txt`) |
| :--- | :--- | :--- |
| **Role Definition** | Generic assistant | Public health protocol assistant |
| **Out-of-Scope Diagnostic Query** | Tends to speculate or give medication advice | Safely refuses with deterministic fallback |
| **Formatting & Length** | Unstructured, conversational, verbose | Action-oriented bullet points, prioritized, <250 words |
| **Inline Citations** | None | Enforces `[Document, Section]` format |
| **Safety Disclaimer** | Missing | Mandatory non-diagnostic notice appended |
