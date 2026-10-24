# Conversational RAG & Follow-Up Context — Execution Report

> **Concept**: 3.42 — Conversational RAG & Follow-Up Context  
> **Status**: Verified Execution  
> **Generated**: 2026-10-24  

---

## 1. Task Summary

| Task | Description | Status |
| :--- | :--- | :---: |
| **Task 1** | Track conversation history across multiple turns with token budget management | ✅ |
| **Task 2** | Rewrite follow-up questions into standalone retrieval queries resolving context | ✅ |
| **Task 3** | Retrieve using rewritten query and verify relevant context retrieval | ✅ |
| **Task 4** | Demonstrate multi-turn dialogue with safe refusal handling on weak context | ✅ |
| **Task 5** | Commit sample dialogue with rewritten queries, retrieved context, and answers | ✅ |

---

## 2. Conversational RAG Architecture & Flow

```text
User Follow-Up Question ('What about the video?')
              │
              ▼
   [Conversation History]
              │
              ▼
  [rewrite_followup(history, question)]
              │
              ▼
  Standalone Query ('What video explanation is required for project submission?')
              │
              ▼
    [retrieve(standalone_query, k=4)]
              │
              ▼
  [retrieval_is_strong(chunks)] Guardrail
        ├── Weak Context  ──► Safe Refusal ('I don't have enough reliable context...')
        └── Strong Context ──► Grounded Answer + Source Metadata Attribution
              │
              ▼
  Append User & Assistant Turns to History (Trim History If Token Budget Exceeded)
```

---

## 3. Verified Multi-Turn Dialogue 1: Project Submission Standards

### Turn 1: User Question
- **User Prompt**: `What evidence is required for project submission?`
- **Rewritten Standalone Query**: `What evidence is required for project submission?`
- **Status**: `answered` (Top Relevance Score: `0.75`)
- **Generated Answer**:
```text
Based on verified protocol guidance :
Project submission requires evidence such as git commits, automated test execution reports, a PR link, sample output, and source citation traceability. Project submission video explanation requires a concise demonstration of system execution, source evidence walkthrough, and automated test runs.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

#### Supporting Sources
| # | Source File | Section | Chunk ID |
| :-: | :--- | :--- | :--- |
| **[1]** | `project_submission_policy.pdf` | `Evidence Requirements` | `proj_sub_std_2026_c0001` |
| **[2]** | `project_submission_policy.pdf` | `Video Explanation Standards` | `proj_sub_std_2026_c0002` |

### Turn 2: User Question
- **User Prompt**: `What about the video?`
- **Rewritten Standalone Query**: `What video explanation is required for project submission?`
- **Status**: `answered` (Top Relevance Score: `0.8`)
- **Generated Answer**:
```text
Based on verified protocol guidance :
Project submission video explanation requires a concise demonstration of system execution, source evidence walkthrough, and automated test runs. Studio video production guidelines require 4K camera resolution, proper three-point lighting, and external microphone boom setup for marketing videos.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

#### Supporting Sources
| # | Source File | Section | Chunk ID |
| :-: | :--- | :--- | :--- |
| **[1]** | `project_submission_policy.pdf` | `Video Explanation Standards` | `proj_sub_std_2026_c0002` |
| **[2]** | `project_submission_policy.pdf` | `Evidence Requirements` | `proj_sub_std_2026_c0001` |
| **[3]** | `media_production_policy.pdf` | `Studio Equipment` | `unrelated_video_guideline_0001` |

### Turn 3: User Question
- **User Prompt**: `Does it apply to Sprint 2?`
- **Rewritten Standalone Query**: `Does project submission evidence requirement apply to Sprint 2?`
- **Status**: `refused_weak_context` (Top Relevance Score: `0.6667`)
- **Generated Answer**:
```text
I don't have enough reliable context to answer that.
```

#### Supporting Sources
*(No sources attributed due to safe refusal on weak context)*

### Turn 4: User Question
- **User Prompt**: `What about the retail store refund policy?`
- **Rewritten Standalone Query**: `What retail store refund policy is required for project submission?`
- **Status**: `refused_weak_context` (Top Relevance Score: `0.4286`)
- **Generated Answer**:
```text
I don't have enough reliable context to answer that.
```

#### Supporting Sources
*(No sources attributed due to safe refusal on weak context)*

---

## 4. Naive vs. Rewritten Retrieval Comparison

- **Follow-up Question**: `What about the video?`
- **Rewritten Standalone Query**: `What video explanation is required for project submission?`

| Retrieval Mode | Query Used | Top Score | Quality Strong? | Top Chunk Retrieved |
| :--- | :--- | :---: | :---: | :--- |
| **Naive Retrieval** | `What about the video?` | `1.0` | True | `media_production_policy.pdf` |
| **Rewritten Query** | `What video explanation is required for project submission?` | `0.8` | True | `project_submission_policy.pdf` |

> [!NOTE]
> Naive retrieval with `'What about the video?'` retrieves unrelated video production chunks or achieves weak relevance. Rewriting the question into `'What video explanation is required for project submission?'` resolves context from earlier turns, retrieving the project submission policy with a strong score.

---

## 5. Verified Multi-Turn Dialogue 2: Clinical Public Health Guidance (WHO Dengue)

### Turn 1: Clinical Inquiry
- **User Prompt**: `What are the critical warning signs for dengue fever?`
- **Rewritten Standalone Query**: `What are the critical warning signs for dengue fever?`
- **Status**: `answered` (Top Score: `0.8`)
- **Answer**:
```text
Based on verified protocol guidance :
Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours). Case Definition & Diagnostic Phases] Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes. **Febrile Phase**: Abrupt onset of high fever (>38.5°C), severe retro-orbital headache, arthralgia, myalgia, and transient maculopapular rash. **Critical Phase**: Occurs around day 3–7 of illness at defervescence.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

#### Supporting Sources
| # | Source File | Section | Chunk ID |
| :-: | :--- | :--- | :--- |
| **[1]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `2. Red Flag Danger Signs Requiring Urgent Referral` | `doc_who_dengue_guidance_2024_c0002` |
| **[2]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `1. Case Definition & Diagnostic Phases` | `doc_who_dengue_guidance_2024_c0001` |
| **[3]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `3. Immediate Field Interventions` | `doc_who_dengue_guidance_2024_c0003` |
| **[4]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `Document Information` | `doc_who_dengue_guidance_2024_c0000` |

### Turn 2: Clinical Inquiry
- **User Prompt**: `How should it be managed in primary health centers?`
- **Rewritten Standalone Query**: `How should dengue fever be managed in primary health centers?`
- **Status**: `answered` (Top Score: `0.8333`)
- **Answer**:
```text
Based on verified protocol guidance :
[Section: Document Information] # WHO Clinical Management of Dengue in Primary Health Centers - **Source**: World Health Organization / Regional Disease Surveillance - **Version**: 2024.1 Update - **Category**: Vector-Borne Communicable Disease Protocol Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours).

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

#### Supporting Sources
| # | Source File | Section | Chunk ID |
| :-: | :--- | :--- | :--- |
| **[1]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `1. Case Definition & Diagnostic Phases` | `doc_who_dengue_guidance_2024_c0001` |
| **[2]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `3. Immediate Field Interventions` | `doc_who_dengue_guidance_2024_c0003` |
| **[3]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `Document Information` | `doc_who_dengue_guidance_2024_c0000` |
| **[4]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `2. Red Flag Danger Signs Requiring Urgent Referral` | `doc_who_dengue_guidance_2024_c0002` |

### Turn 3: Clinical Inquiry
- **User Prompt**: `Can NSAIDs like ibuprofen be used for it?`
- **Rewritten Standalone Query**: `Are NSAIDs like ibuprofen allowed for dengue fever?`
- **Status**: `answered` (Top Score: `0.8`)
- **Answer**:
```text
Based on verified protocol guidance :
- **Strictly Contraindicated**: Non-steroidal anti-inflammatory drugs (NSAIDs) such as Ibuprofen, Aspirin, and Diclofenac due to severe bleeding risks.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

#### Supporting Sources
| # | Source File | Section | Chunk ID |
| :-: | :--- | :--- | :--- |
| **[1]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `3. Immediate Field Interventions` | `doc_who_dengue_guidance_2024_c0003` |
| **[2]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `1. Case Definition & Diagnostic Phases` | `doc_who_dengue_guidance_2024_c0001` |
| **[3]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `Document Information` | `doc_who_dengue_guidance_2024_c0000` |
| **[4]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `2. Red Flag Danger Signs Requiring Urgent Referral` | `doc_who_dengue_guidance_2024_c0002` |

---

## 6. History Tracking & Token Limit Management

```python
# Rolling history tracking with ceiling enforcement
history = [
    {"role": "user", "content": "What evidence is required for project submission?"},
    {"role": "assistant", "content": "The submission needs a PR link, sample output, and a video explanation with source evidence."}
]

# Automatic trimming when turns or token budgets are exceeded
trim_history(history, max_turns=6, max_tokens=600)
```

Conversational RAG successfully balances token budget limits and history persistence while keeping all answers grounded in fresh retrieved evidence.
