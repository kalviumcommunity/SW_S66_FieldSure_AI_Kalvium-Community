# Hallucination Guardrails & Refusal Handling — Execution Report

> **Concept**: 3.41 — Hallucination Guardrails & Refusal Handling  
> **Status**: Verified Execution  
> **Generated**: 2026-10-23  

---

## 1. Task Summary

| Task | Description | Status |
| :--- | :--- | :---: |
| **Task 1** | Detect weak retrieval using signals (empty results, score below threshold, too few chunks) | ✅ |
| **Task 2** | Return a safe refusal (`"I don't have enough reliable context to answer that."`) | ✅ |
| **Task 3** | Use relevance threshold (`MIN_TOP_SCORE = 0.72`, `MIN_SUPPORTING_CHUNKS = 1`) | ✅ |
| **Task 4** | Preserve confident grounded answers when strong supporting context exists | ✅ |
| **Task 5** | Commit sample refusal and answer cases with execution evidence | ✅ |

---

## 2. Guardrail Logic & Quality Thresholds

Guardrails run **before generation** so the model is not asked to synthesize answers from poor or missing evidence.

```python
MIN_TOP_SCORE = 0.72
MIN_SUPPORTING_CHUNKS = 1

def retrieval_is_strong(chunks):
    if not chunks:
        return False

    strong_chunks = [chunk for chunk in chunks if chunk["score"] >= MIN_TOP_SCORE]
    return len(strong_chunks) >= MIN_SUPPORTING_CHUNKS

def guarded_answer(question):
    chunks = retrieve(question, k=4)

    if not retrieval_is_strong(chunks):
        return {
            "answer": "I don't have enough reliable context to answer that.",
            "sources": [],
            "status": "refused_weak_context"
        }

    return {
        **generate_grounded_answer(question, chunks),
        "status": "answered"
    }
```

---

## 3. Verified Sample Outputs

### Case 1: Successfully Answered Case (Strong Context)

- **Question**: `What evidence is required for project submission?`
- **Status**: `answered`
- **Top Score**: `0.75` (Threshold: `>= 0.72`)
- **Answer**:
```text
Based on verified protocol guidance :
Project submission requires evidence such as git commits, automated test execution reports, and source citation traceability.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

#### Supporting Sources
| # | Source File | Section | Chunk ID |
| :-: | :--- | :--- | :--- |
| **[1]** | `project_submission_policy.pdf` | `Evidence Requirements` | `proj_sub_std_2026_c0001` |


---

### Case 2: Safe Refusal Case (Unsupported / Out-of-Corpus Query)

- **Question**: `What is the refund policy for a product not in this corpus?`
- **Status**: `refused_weak_context`
- **Top Score**: `0.25`
- **Sources**: `[]`
- **Answer**:
```text
I don't have enough reliable context to answer that.
```

---

### Case 3: Confident Clinical Answer Case (Dengue Protocols)

- **Question**: `What are the critical warning signs for dengue fever?`
- **Status**: `answered`
- **Top Score**: `0.8`
- **Answer**:
```text
Based on verified protocol guidance :
Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours). Case Definition & Diagnostic Phases] Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes. **Febrile Phase**: Abrupt onset of high fever (>38.5°C), severe retro-orbital headache, arthralgia, myalgia, and transient maculopapular rash. **Critical Phase**: Occurs around day 3–7 of illness at defervescence.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

---

### Case 4: Weak Retrieval Score Below Threshold

- **Question**: `What specific chemotherapy regimen is prescribed?`
- **Provided Score**: `0.55` (Below `0.72`)
- **Status**: `refused_weak_context`
- **Answer**:
```text
I don't have enough reliable context to answer that.
```

---

## 4. Console Verification Output

```text
answered: answered Based on verified protocol guidance :
Project submission requires evidence such as git commits, automated test execution...
refused: refused_weak_context I don't have enough reliable context to answer that.
```

---

## 5. Engineering Tradeoff: Refusing vs. Answering

- **Refusing too often**: Frustrates users and limits the system's utility.
- **Answering too freely**: Risks hallucinations and misinformation.
- **In High-Stakes Public Health**: Refusing is safer when evidence is weak because a confident unsupported answer can cause serious real-world harm.
- **Relevance Threshold Rationale**: The threshold `MIN_TOP_SCORE = 0.72` was derived empirically from retrieval relevance tuning (Module #25) and recall testing (Module #27), cleanly separating genuine protocol matches from noise.
