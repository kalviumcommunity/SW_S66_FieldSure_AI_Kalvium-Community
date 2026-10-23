# Grounded Answer Generation — Execution Report

> **Concept**: 3.39 — Grounded Answer Generation
> **Status**: Verified Execution
> **Question**: What are the warning signs and immediate field interventions for dengue?

---

## 1. Task Summary

| Task | Description | Status |
| :--- | :--- | :---: |
| **Task 1** | Generate from injected context with explicit grounding | ✅ |
| **Task 2** | Check source accuracy — verify answer reflects chunks | ✅ |
| **Task 3** | Missing-context fallback for unsupported questions | ✅ |
| **Task 4** | Compare grounded vs ungrounded answers | ✅ |
| **Task 5** | Commit sample grounded answers with verification evidence | ✅ |

---

## 2. Task 1 — Grounded Answer from Injected Context

### Question
> What are the warning signs and immediate field interventions for dengue?

### Generated Answer
```text
Based on verified protocol guidance :
[Section: Document Information] # WHO Clinical Management of Dengue in Primary Health Centers Case Definition & Diagnostic Phases] Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes. Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours). Immediate Field Interventions] - Administer oral rehydration with ORS, coconut water, or fruit juices.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

### Supporting Chunks Used: 6

| # | Source File | Section | Doc ID |
| :---: | :--- | :--- | :--- |
| 1 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `Document Information` | `doc_who_dengue_guidance_2024` |
| 2 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `1. Case Definition & Diagnostic Phases` | `doc_who_dengue_guidance_2024` |
| 3 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `2. Red Flag Danger Signs Requiring Urgent Referral` | `doc_who_dengue_guidance_2024` |
| 4 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `3. Immediate Field Interventions` | `doc_who_dengue_guidance_2024` |
| 5 | `data/sample_advisories/cdc_cholera_water_safety.txt` | `1. GENERAL OUTBREAK CONTEXT` | `doc_cdc_cholera_protocol_2024` |
| 6 | `data/sample_advisories/cdc_cholera_water_safety.txt` | `2. WATER TREATMENT STANDARDS` | `doc_cdc_cholera_protocol_2024` |


---

## 3. Task 2 — Source Accuracy Verification

Verifies that the generated answer reflects the content of the retrieved chunks
and does not add unsupported claims.

| Metric | Value |
| :--- | :--- |
| **Keyword Overlap Score** | `0.7910` |
| **Answer Sentences Checked** | `3` |
| **Supported Sentences** | `3` |
| **Unsupported Claims** | `0` |
| **Is Grounded** | ✅ Yes |

### Unsupported Claims Detail

| Status | Claim |
| :---: | :--- |
| ✅ | No unsupported claims detected |


---

## 4. Task 3 — Missing-Context Fallback

Tested with a question that has no supporting context in the corpus.

- **Question**: `What evidence is required for project submission?`
- **Answer**: `I don't have enough information in the provided context.`
- **Is Fallback**: `True`
- **Sources**: `[]`

The system correctly refuses to answer when the context does not contain
supporting information, rather than generating an unsupported response.

---

## 5. Task 4 — Grounded vs Ungrounded Comparison

Ran the same question with and without retrieval to show how grounding
changes the answer.

### With Retrieval (Grounded)
```text
Based on verified protocol guidance :
[Section: Document Information] # WHO Clinical Management of Dengue in Primary Health Centers Case Definition & Diagnostic Phases] Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes. Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours). Immediate Field Interventions] - Administer oral rehydration with ORS, coconut water, or fruit juices.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

**Keyword overlap with context**: `0.7910`

### Without Retrieval (Ungrounded)
```text
Based on general knowledge: The answer to 'What are the warning signs and immediate field interventions for dengue?' would depend on the specific policies and requirements of the institution or organization. Generally, submissions require documented evidence such as completion reports, progress logs, supervisor evaluations, and relevant appendices. Specific requirements should be confirmed with the issuing authority.
```

**Keyword overlap with context**: `0.3125`

### Comparison Summary

| Metric | Grounded | Ungrounded |
| :--- | :---: | :---: |
| **Has Sources** | ✅ | ❌ |
| **Keyword Overlap** | `0.7910` | `0.3125` |
| **Is Fallback** | `False` | N/A |

The grounded answer should reflect the retrieved chunks and include source
markers or source metadata. The ungrounded answer may sound fluent, but it
is not tied to the corpus.

---

## 6. Grounding Reduces Hallucination

Grounding gives the model evidence and limits what it should use. It does not
make answers automatically perfect, so you still need citation checks,
fallback behavior, and evaluation.

Key guarantees of this module:
1. **Context-only generation**: Answers are derived strictly from retrieved passages.
2. **Source traceability**: Every fact can be traced to specific document chunks.
3. **Honest refusal**: Missing context triggers an explicit fallback, not fabrication.
4. **Measurable grounding**: Keyword overlap scores quantify how well answers match sources.
