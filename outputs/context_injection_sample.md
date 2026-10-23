# Context Injection & Prompt Augmentation — Execution Report

> **Module**: Module #29 — Context Injection & Prompt Augmentation
> **Status**: Verified Execution
> **Question**: What are the warning signs and immediate field interventions for dengue?

---

## 1. Task Summary

| Task | Description | Status |
| :--- | :--- | :---: |
| **Task 1** | Inject retrieved chunks into the prompt as model context | ✅ |
| **Task 2** | Enforce token budget, leaving room for instructions + answer | ✅ |
| **Task 3** | Add source markers ([1], [2], ...) for citation auditing | ✅ |
| **Task 4** | Grounding instructions: answer only from context, refuse when insufficient | ✅ |
| **Task 5** | Sample augmented prompt with all features demonstrated | ✅ |

---

## 2. Source Markers & Chunk Formatting

Each chunk is formatted with a numbered marker containing the source file path,
chunk index, and section heading so the model can cite evidence precisely.

| # | Source File | Section | Tokens |
| :---: | :--- | :--- | :---: |
| 1 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `Document Information` | 48 |
| 2 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `1. Case Definition & Diagnostic Phases` | 96 |
| 3 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `2. Red Flag Danger Signs Requiring Urgent Referral` | 95 |
| 4 | `data/sample_advisories/who_dengue_clinical_guidance.md` | `3. Immediate Field Interventions` | 82 |
| 5 | `data/sample_advisories/cdc_cholera_water_safety.txt` | `1. GENERAL OUTBREAK CONTEXT` | 88 |
| 6 | `data/sample_advisories/cdc_cholera_water_safety.txt` | `2. WATER TREATMENT STANDARDS` | 92 |


---

## 3. Scenario 1 — Normal Budget (5000 tokens)

- **Chunks Included**: 6 / 6
- **Context Tokens**: 902
- **Total Prompt Tokens**: 997

### Full Augmented Prompt

```text
You are a grounded assistant. Answer the question using ONLY the provided context below.
If the answer is not in the context, say: "I don't have enough information in the provided context."
When possible, cite sources using the markers like [1] or [2].
Do not use external knowledge, do not speculate, and do not invent facts.

Context:
[1] data/sample_advisories/who_dengue_clinical_guidance.md#chunk0 (Section: Document Information)
[Section: Document Information]
# WHO Clinical Management of Dengue in Primary Health Centers

## Document Information
- **Source**: World Health Organization / Regional Disease Surveillance
- **Version**: 2024.1 Update
- **Category**: Vector-Borne Communicable Disease Protocol

---

[2] data/sample_advisories/who_dengue_clinical_guidance.md#chunk1 (Section: 1. Case Definition & Diagnostic Phases)
[Section: 1. Case Definition & Diagnostic Phases]
Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes. The clinical course consists of three distinct phases:
1. **Febrile Phase**: Abrupt onset of high fever (>38.5°C), severe retro-orbital headache, arthralgia, myalgia, and transient maculopapular rash.
2. **Critical Phase**: Occurs around day 3–7 of illness at defervescence. Characterized by plasma leakage, pleural effusion, ascites, hemoconcentration, and mucosal bleeding.
3. **Recovery Phase**: Gradual reabsorption of extravasated fluid, stabilization of blood pressure, and return of appetite.

---

[3] data/sample_advisories/who_dengue_clinical_guidance.md#chunk2 (Section: 2. Red Flag Danger Signs Requiring Urgent Referral)
[Section: 2. Red Flag Danger Signs Requiring Urgent Referral]
Community health workers must evaluate patients daily during the critical phase for warning signs:
- Persistent vomiting (> 3 episodes in 24 hours).
- Severe abdominal pain or tenderness.
- Fluid accumulation (ascites, pleural effusion).
- Mucosal bleeding (epistaxis, gingival bleeding).
- Lethargy, restlessness, or altered sensorium.
- Hepatomegaly (> 2 cm below costal margin).
- Progressive hematocrit elevation with rapid platelet count drop.

---

[4] data/sample_advisories/who_dengue_clinical_guidance.md#chunk3 (Section: 3. Immediate Field Interventions)
[Section: 3. Immediate Field Interventions]
- Administer oral rehydration with ORS, coconut water, or fruit juices. Avoid plain water to prevent hyponatremia.
- Use Paracetamol (10-15 mg/kg per dose) for fever management.
- **Strictly Contraindicated**: Non-steroidal anti-inflammatory drugs (NSAIDs) such as Ibuprofen, Aspirin, and Diclofenac due to severe bleeding risks.

---

[5] data/sample_advisories/cdc_cholera_water_safety.txt#chunk0 (Section: 1. GENERAL OUTBREAK CONTEXT)
[Section: 1. GENERAL OUTBREAK CONTEXT]
CDC EMERGENCY HEALTH BULLETIN: CHOLERA OUTBREAK WATER AND SANITATION PROTOCOL
DOCUMENT IDENTIFIER: CDC-WASH-CHOLERA-2024
TARGET AUDIENCE: Community Health Workers, Disaster Relief Teams, Camp Coordinators

1. GENERAL OUTBREAK CONTEXT
Vibrio cholerae transmission is predominantly waterborne through ingestion of water or food contaminated with feces from an infected person. In flood emergencies or displaced person camps, rapid establishment of safe water points is the single most critical intervention.

---

[6] data/sample_advisories/cdc_cholera_water_safety.txt#chunk1 (Section: 2. WATER TREATMENT STANDARDS)
[Section: 2. WATER TREATMENT STANDARDS]
2. WATER TREATMENT STANDARDS
- All drinking water distributed in settlement camps must be disinfected with sodium dichloroisocyanurate (NaDCC) or calcium hypochlorite.
- Target Free Residual Chlorine (FRC) levels:
  * At point of distribution: 0.5 to 1.0 mg/L (ppm).
  * At household storage containers after 24 hours: Minimum 0.2 mg/L (ppm).
  * Turbid water (NTU > 5) must undergo flocculation or filtration prior to chlorination.

Question: What are the warning signs and immediate field interventions for dengue?
```

---

## 4. Scenario 2 — Tight Budget (200 tokens)

Demonstrates that when the token budget is restrictive, only the highest-ranked
chunks are included and lower-ranked ones are skipped.

- **Chunks Included**: 1 / 6
- **Context Tokens**: 84
- **Total Prompt Tokens**: 174

### Truncated Augmented Prompt

```text
You are a grounded assistant. Answer the question using ONLY the provided context below.
If the answer is not in the context, say: "I don't have enough information in the provided context."
When possible, cite sources using the markers like [1] or [2].
Do not use external knowledge, do not speculate, and do not invent facts.

Context:
[1] data/sample_advisories/who_dengue_clinical_guidance.md#chunk0 (Section: Document Information)
[Section: Document Information]
# WHO Clinical Management of Dengue in Primary Health Centers

## Document Information
- **Source**: World Health Organization / Regional Disease Surveillance
- **Version**: 2024.1 Update
- **Category**: Vector-Borne Communicable Disease Protocol

Question: What are the warning signs and immediate field interventions for dengue?
```

---

## 5. Scenario 3 — Dynamic Budget (Window = 8000 tokens)

Computed budget = 8000 − 1500 (answer) − 800 (instructions) = 5700 tokens.

- **Chunks Included**: 6 / 6
- **Context Tokens**: 902
- **Total Prompt Tokens**: 997

---

## 6. Grounding Instruction Text

The following instruction is injected at the top of every augmented prompt:

```text
You are a grounded assistant. Answer the question using ONLY the provided context below.
If the answer is not in the context, say: "I don't have enough information in the provided context."
When possible, cite sources using the markers like [1] or [2].
Do not use external knowledge, do not speculate, and do not invent facts.
```

This ensures the model:
1. Answers strictly from the provided context.
2. Cites source markers like [1], [2].
3. Refuses when the context is insufficient instead of hallucinating.

---

## 7. Returned Source Metadata

```json
[
  {
    "chunk_id": "doc_who_dengue_guidance_2024_c0000",
    "doc_id": "doc_who_dengue_guidance_2024",
    "source_file": "data/sample_advisories/who_dengue_clinical_guidance.md",
    "document_title": "WHO Clinical Management of Dengue in Primary Health Centers",
    "document_version": "2024.1 Update",
    "agency": "WHO",
    "publication_date": "2024",
    "section_heading": "Document Information",
    "chunk_index": 0,
    "total_chunks": 4,
    "char_length": 258,
    "token_count": 48,
    "page_number": null,
    "extra_metadata": {
      "strategy": "token_structural",
      "file_type": "md"
    }
  },
  {
    "chunk_id": "doc_who_dengue_guidance_2024_c0001",
    "doc_id": "doc_who_dengue_guidance_2024",
    "source_file": "data/sample_advisories/who_dengue_clinical_guidance.md",
    "document_title": "WHO Clinical Management of Dengue in Primary Health Centers",
    "document_version": "2024.1 Update",
    "agency": "WHO",
    "publication_date": "2024",
    "section_heading": "1. Case Definition & Diagnostic Phases",
    "chunk_index": 1,
    "total_chunks": 4,
    "char_length": 472,
    "token_count": 96,
    "page_number": null,
    "extra_metadata": {
      "strategy": "token_structural",
      "file_type": "md"
    }
  },
  {
    "chunk_id": "doc_who_dengue_guidance_2024_c0002",
    "doc_id": "doc_who_dengue_guidance_2024",
    "source_file": "data/sample_advisories/who_dengue_clinical_guidance.md",
    "document_title": "WHO Clinical Management of Dengue in Primary Health Centers",
    "document_version": "2024.1 Update",
    "agency": "WHO",
    "publication_date": "2024",
    "section_heading": "2. Red Flag Danger Signs Requiring Urgent Referral",
    "chunk_index": 2,
    "total_chunks": 4,
    "char_length": 487,
    "token_count": 95,
    "page_number": null,
    "extra_metadata": {
      "strategy": "token_structural",
      "file_type": "md"
    }
  },
  {
    "chunk_id": "doc_who_dengue_guidance_2024_c0003",
    "doc_id": "doc_who_dengue_guidance_2024",
    "source_file": "data/sample_advisories/who_dengue_clinical_guidance.md",
    "document_title": "WHO Clinical Management of Dengue in Primary Health Centers",
    "document_version": "2024.1 Update",
    "agency": "WHO",
    "publication_date": "2024",
    "section_heading": "3. Immediate Field Interventions",
    "chunk_index": 3,
    "total_chunks": 4,
    "char_length": 412,
    "token_count": 82,
    "page_number": null,
    "extra_metadata": {
      "strategy": "token_structural",
      "file_type": "md"
    }
  },
  {
    "chunk_id": "doc_cdc_cholera_protocol_2024_c0000",
    "doc_id": "doc_cdc_cholera_protocol_2024",
    "source_file": "data/sample_advisories/cdc_cholera_water_safety.txt",
    "document_title": "CDC EMERGENCY HEALTH BULLETIN: CHOLERA OUTBREAK WATER AND SANITATION PROTOCOL",
    "document_version": "CDC-WASH-CHOLERA-2024",
    "agency": "CDC",
    "publication_date": "2024",
    "section_heading": "1. GENERAL OUTBREAK CONTEXT",
    "chunk_index": 0,
    "total_chunks": 4,
    "char_length": 498,
    "token_count": 88,
    "page_number": null,
    "extra_metadata": {
      "strategy": "token_structural",
      "file_type": "txt"
    }
  },
  {
    "chunk_id": "doc_cdc_cholera_protocol_2024_c0001",
    "doc_id": "doc_cdc_cholera_protocol_2024",
    "source_file": "data/sample_advisories/cdc_cholera_water_safety.txt",
    "document_title": "CDC EMERGENCY HEALTH BULLETIN: CHOLERA OUTBREAK WATER AND SANITATION PROTOCOL",
    "document_version": "CDC-WASH-CHOLERA-2024",
    "agency": "CDC",
    "publication_date": "2024",
    "section_heading": "2. WATER TREATMENT STANDARDS",
    "chunk_index": 1,
    "total_chunks": 4,
    "char_length": 470,
    "token_count": 92,
    "page_number": null,
    "extra_metadata": {
      "strategy": "token_structural",
      "file_type": "txt"
    }
  }
]
```
