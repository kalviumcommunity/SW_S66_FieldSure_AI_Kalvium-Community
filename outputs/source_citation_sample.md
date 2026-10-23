# Source Citation & Attribution — Execution Report

> **Concept**: 3.40 — Source Citation & Attribution  
> **Status**: Verified Execution  
> **Generated**: 2026-10-23  

---

## 1. Task Summary

| Task | Description | Status |
| :--- | :--- | :---: |
| **Task 1** | Add source references (`[1]`, `[2]`) to generated answers | ✅ |
| **Task 2** | Map citations to metadata (`source`, `chunk_id`, `chunk_index`, `section`, `text`) | ✅ |
| **Task 3** | Verify cited sources against original retrieved chunk text | ✅ |
| **Task 4** | Avoid fabricated citations with deterministic fallback | ✅ |
| **Task 5** | Commit sample cited answers, citation mappings, and fallback example | ✅ |

---

## 2. Task 1 — Grounded Answer with Source References

### Question
> What are the warning signs and immediate field interventions for dengue?

### Generated Answer
```text
Based on verified protocol guidance:
Immediate Field Interventions] - Administer oral rehydration with ORS, coconut water, or fruit juices [1]. Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours) [2]. [Section: Step-by-Step RDT Field Testing Procedure] - Verify expiry date and desiccant packet integrity inside the foil pouch [3]. Add 4 drops of assay buffer immediately into well B [3].

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

### Citation Markers Present
[1], [2], [3]

---

## 3. Task 2 — Citation-to-Metadata Mapping

The citation marker in the answer maps to one real retrieved chunk, preserving metadata:

| Marker | Source Document | Section | Chunk ID | Chunk Index |
| :---: | :--- | :--- | :--- | :---: |
| **[1]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `3. Immediate Field Interventions` | `doc_who_dengue_guidance_2024_c0003` | `3` |
| **[2]** | `data/sample_advisories/who_dengue_clinical_guidance.md` | `2. Red Flag Danger Signs Requiring Urgent Referral` | `doc_who_dengue_guidance_2024_c0002` | `2` |
| **[3]** | `data/sample_advisories/moh_malaria_rapid_triage.html` | `Step-by-Step RDT Field Testing Procedure` | `doc_moh_malaria_triage_2024_c0001` | `1` |
| **[4]** | `data/sample_advisories/moh_malaria_rapid_triage.html` | `Treatment & Urgent Referral Protocols` | `doc_moh_malaria_triage_2024_c0002` | `2` |

---

## 4. Task 3 — Verify Cited Sources

Pick one citation in the answer and read the mapped chunk text. Confirm the answer claim is supported by that source:

```python
result = answer_with_citations("What are the warning signs and immediate field interventions for dengue?")
print(result["answer"])
print(result["citations"]["[1]"]["source"])
print(result["citations"]["[1]"]["text"])
```

### Verification Result for `[1]`
- **Source**: `data/sample_advisories/who_dengue_clinical_guidance.md`
- **Section**: `3. Immediate Field Interventions`
- **Chunk ID**: `doc_who_dengue_guidance_2024_c0003`
- **Original Chunk Text**:
> [Section: 3. Immediate Field Interventions]
- Administer oral rehydration with ORS, coconut water, or fruit juices. Avoid plain water to prevent hyponatremia.
- Use Paracetamol (10-15 mg/kg per dose) for fever management.
- **Strictly Contraindicated**: Non-steroidal anti-inflammatory drugs (NSAIDs) such as Ibuprofen, Aspirin, and Diclofenac due to severe bleeding risks.

### All Citations Verification Summary
- **Total Markers in Answer**: `3`
- **Supported Citations**: `3`
- **Fabricated Citations**: `None`
- **All Valid**: `✅ Yes`
- **Coverage Score**: `100.0%`

---

## 5. Task 4 — Avoid Fabricated Citations (No-Source Fallback)

When sources are insufficient or the question cannot be answered from the retrieved context,
the system returns an explicit fallback instead of inventing fabricated citations.

### Unsupported Query
> What evidence is required for project submission?

### Response
- **Answer**: `I don't have enough information in the provided context.`
- **Citations**: `{}`
- **Is Fallback**: `True`

### Supported Query with Evidence
```python
sample_chunks = [
    {
        "text": "Project submission requires evidence including version-controlled code commits, automated test execution reports, and source citation traceability.",
        "metadata": {
            "source": "data/guidance/project_submission_standards.pdf",
            "chunk_id": "proj_sub_std_2026_c0001",
            "chunk_index": 0,
            "section": "Evidence Requirements",
        },
    }
]
result = answer_with_citations("What evidence is required for project submission?", chunks=sample_chunks)
print(result["answer"])
print(result["citations"]["[1]"]["source"])
print(result["citations"]["[1]"]["text"])
```

**Output**:
```text
Based on verified protocol guidance:
Project submission requires evidence including version-controlled code commits, automated test execution reports, and source citation traceability [1].

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
Source: data/guidance/project_submission_standards.pdf
Text:   Project submission requires evidence including version-controlled code commits, automated test execution reports, and source citation traceability.
```

---

## 6. Task 5 — Verification Conclusion

Citations help users decide whether to trust the system. For a knowledge assistant,
citations let users inspect source documents, catch stale or incomplete context, and
understand exactly where every claim in the answer originated.
