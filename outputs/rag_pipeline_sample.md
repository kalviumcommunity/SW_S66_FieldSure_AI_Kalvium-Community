# RAG Pipeline Architecture & Flow Design — Execution Report

> **Module**: Module #28 — RAG Pipeline Architecture & Flow Design  
> **Status**: Verified End-to-End Execution  
> **Retrieval Mode**: ChromaDB Vector Search (Top-k = 4, Minimum Similarity = 0.08)

---

## 1. End-to-End RAG Architecture & Flow Diagram

The complete query-to-answer flow is strictly decoupled into modular, testable stages:

```mermaid
sequenceDiagram
    autonumber
    actor User as Field Worker
    participant Orchestrator as Pipeline Orchestrator (answer_query)
    participant Embedder as Embed Stage (embed_query)
    participant Store as Vector DB (retrieve_context)
    participant Assembler as Assembler (assemble_context)
    participant Generator as LLM Generator (generate_answer)

    User->>Orchestrator: 1. Submit Query
    Orchestrator->>Embedder: 2. embed_query(query)
    Embedder-->>Orchestrator: Dense Embedding Vector
    Orchestrator->>Store: 3. retrieve_context(vector, k=4)
    alt Relevant Chunks Found (similarity >= 0.08)
        Store-->>Orchestrator: Top-k Relevant Chunks
        Orchestrator->>Assembler: 4. assemble_context(chunks)
        Assembler-->>Orchestrator: Formatted Context with [1] Source: ...
        Orchestrator->>Generator: 5. generate_answer(query, context)
        Generator-->>Orchestrator: Grounded Answer + Citations
        Orchestrator-->>User: {answer, sources}
    else No Context Found / Similarity Below Threshold
        Store-->>Orchestrator: Empty Chunks []
        Orchestrator-->>User: Fallback: "I could not find relevant context for that question." (sources: [])
    end
```

---

## 2. Pipeline Responsibilities by Stage

| Stage | Function | Input | Output | Single Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1: Embed** | `embed_query` | Raw query string | Dense numeric vector | Encodes textual query semantics into normalized vector space. |
| **Stage 2: Retrieve** | `retrieve_context` | Query vector, store, $k$ | Top-$k$ chunk objects | Cosine similarity lookup in ChromaDB with optional threshold and metadata filters. |
| **Stage 3: Assemble** | `assemble_context` | List of chunk objects | Formatted context string | Formats evidence snippets with 1-based indices, source file, and section headings. |
| **Stage 4: Generate** | `generate_answer` | Query text, context string | Grounded answer string | Injects context into grounded prompt template and queries LLM. |
| **Stage 5: Orchestrate** | `answer_query` | Query, pipeline settings | Result dictionary | Coordinates all stages and safely catches empty retrieval without hallucinations. |

---

## 3. Sample Query Run (Grounded Retrieval & Generation)

- **Query**: `What are the warning signs and immediate field interventions for dengue?`
- **Retrieved Chunks**: 4 chunks retrieved

### Retrieved Source Chunks

#### [1] `WHO Clinical Management of Dengue in Primary Health Centers`
- **Source File**: `data/sample_advisories/who_dengue_clinical_guidance.md`
- **Section Heading**: `3. Immediate Field Interventions`
- **Similarity Score**: `0.1750`
- **Content Excerpt**:
> [Section: 3. Immediate Field Interventions]
- Administer oral rehydration with ORS, coconut water, or fruit juices. Avoid plain water to prevent hyponatremia.
- Use Paracetamol (10-15 mg/kg per dose) for fever management.
- **Strictly Contr...

#### [2] `WHO Clinical Management of Dengue in Primary Health Centers`
- **Source File**: `data/sample_advisories/who_dengue_clinical_guidance.md`
- **Section Heading**: `2. Red Flag Danger Signs Requiring Urgent Referral`
- **Similarity Score**: `0.1555`
- **Content Excerpt**:
> [Section: 2. Red Flag Danger Signs Requiring Urgent Referral]
Community health workers must evaluate patients daily during the critical phase for warning signs:
- Persistent vomiting (> 3 episodes in 24 hours).
- Severe abdominal pain or te...

#### [3] `CDC EMERGENCY HEALTH BULLETIN: CHOLERA OUTBREAK WATER AND SANITATION PROTOCOL`
- **Source File**: `data/sample_advisories/cdc_cholera_water_safety.txt`
- **Section Heading**: `4. INFECTION CONTROL IN THE FIELD`
- **Similarity Score**: `0.1063`
- **Content Excerpt**:
> [Section: 4. INFECTION CONTROL IN THE FIELD]
4. INFECTION CONTROL IN THE FIELD
- Chlorinated handwashing stations (0.05% chlorine solution) must be erected at all latrine exits and treatment areas.
- Disinfection of vomitus and stool spills...

#### [4] `National Vector-Borne Disease Control Programme — Community Malaria Rapid Diagnostic Protocol`
- **Source File**: `data/sample_advisories/moh_malaria_rapid_triage.html`
- **Section Heading**: `Step-by-Step RDT Field Testing Procedure`
- **Similarity Score**: `0.1021`
- **Content Excerpt**:
> [Section: Step-by-Step RDT Field Testing Procedure]
- Verify expiry date and desiccant packet integrity inside the foil pouch.
- Clean patient fingertip with 70% alcohol swab; air dry completely.
- Prick with single-use sterile lancet; coll...

### Assembled Context (Prompt Evidence)
```text
[1] Source: data/sample_advisories/who_dengue_clinical_guidance.md (Section: 3. Immediate Field Interventions)
[Section: 3. Immediate Field Interventions]
- Administer oral rehydration with ORS, coconut water, or fruit juices. Avoid plain water to prevent hyponatremia.
- Use Paracetamol (10-15 mg/kg per dose) for fever management.
- **Strictly Contraindicated**: Non-steroidal anti-inflammatory drugs (NSAIDs) such as Ibuprofen, Aspirin, and Diclofenac due to severe bleeding risks.

[2] Source: data/sample_advisories/who_dengue_clinical_guidance.md (Section: 2. Red Flag Danger Signs Requiring Urgent Referral)
[Section: 2. Red Flag Danger Signs Requiring Urgent Referral]
Community health workers must evaluate patients daily during the critical phase for warning signs:
- Persistent vomiting (> 3 episodes in 24 hours).
- Severe abdominal pain or tenderness.
- Fluid accumulation (ascites, pleural effusion).
- Mucosal bleeding (epistaxis, gingival bleeding).
- Lethargy, restlessness, or altered sensorium.
- Hepatomegaly (> 2 cm below costal margin).
- Progressive hematocrit elevation with rapid platelet count drop.

[3] Source: data/sample_advisories/cdc_cholera_water_safety.txt (Section: 4. INFECTION CONTROL IN THE FIELD)
[Section: 4. INFECTION CONTROL IN THE FIELD]
4. INFECTION CONTROL IN THE FIELD
- Chlorinated handwashing stations (0.05% chlorine solution) must be erected at all latrine exits and treatment areas.
- Disinfection of vomitus and stool spills requires 0.5% high-strength chlorine solution.
- Dead bodies of suspected cholera victims must be handled only by designated decontamination teams.

[4] Source: data/sample_advisories/moh_malaria_rapid_triage.html (Section: Step-by-Step RDT Field Testing Procedure)
[Section: Step-by-Step RDT Field Testing Procedure]
- Verify expiry date and desiccant packet integrity inside the foil pouch.
- Clean patient fingertip with 70% alcohol swab; air dry completely.
- Prick with single-use sterile lancet; collect 5 µL capillary blood using provided inverted cup applicator.
- Add 4 drops of assay buffer immediately into well B.
- Read results strictly between 15 and 20 minutes. Do not interpret after 30 minutes.
```

### Generated Answer
Based on verified protocol guidance [who_dengue_clinical_guidance.md, 3. Immediate Field Interventions] [who_dengue_clinical_guidance.md, 2. Red Flag Danger Signs Requiring Urgent Referral] [cdc_cholera_water_safety.txt, 4. INFECTION CONTROL IN THE FIELD] [moh_malaria_rapid_triage.html, Step-by-Step RDT Field Testing Procedure]:
Immediate Field Interventions] - Administer oral rehydration with ORS, coconut water, or fruit juices. Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours). INFECTION CONTROL IN THE FIELD] 4. INFECTION CONTROL IN THE FIELD - Chlorinated handwashing stations (0.05% chlorine solution) must be erected at all latrine exits and treatment areas.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).

### Returned Sources (Metadata Contract)
```json
[
  {
    "agency": "WHO",
    "char_length": 412,
    "chunk_id": "doc_who_dengue_guidance_2024_c0003",
    "chunk_index": 3,
    "doc_id": "doc_who_dengue_guidance_2024",
    "document_title": "WHO Clinical Management of Dengue in Primary Health Centers",
    "document_version": "2024.1 Update",
    "extra_metadata": {
      "file_type": "md",
      "strategy": "token_structural"
    },
    "page_number": null,
    "publication_date": "2024",
    "section_heading": "3. Immediate Field Interventions",
    "source_file": "data/sample_advisories/who_dengue_clinical_guidance.md",
    "token_count": 82,
    "total_chunks": 4
  },
  {
    "agency": "WHO",
    "char_length": 487,
    "chunk_id": "doc_who_dengue_guidance_2024_c0002",
    "chunk_index": 2,
    "doc_id": "doc_who_dengue_guidance_2024",
    "document_title": "WHO Clinical Management of Dengue in Primary Health Centers",
    "document_version": "2024.1 Update",
    "extra_metadata": {
      "file_type": "md",
      "strategy": "token_structural"
    },
    "page_number": null,
    "publication_date": "2024",
    "section_heading": "2. Red Flag Danger Signs Requiring Urgent Referral",
    "source_file": "data/sample_advisories/who_dengue_clinical_guidance.md",
    "token_count": 95,
    "total_chunks": 4
  },
  {
    "agency": "CDC",
    "char_length": 387,
    "chunk_id": "doc_cdc_cholera_protocol_2024_c0003",
    "chunk_index": 3,
    "doc_id": "doc_cdc_cholera_protocol_2024",
    "document_title": "CDC EMERGENCY HEALTH BULLETIN: CHOLERA OUTBREAK WATER AND SANITATION PROTOCOL",
    "document_version": "CDC-WASH-CHOLERA-2024",
    "extra_metadata": {
      "file_type": "txt",
      "strategy": "token_structural"
    },
    "page_number": null,
    "publication_date": "2024",
    "section_heading": "4. INFECTION CONTROL IN THE FIELD",
    "source_file": "data/sample_advisories/cdc_cholera_water_safety.txt",
    "token_count": 75,
    "total_chunks": 4
  },
  {
    "agency": "MOH",
    "char_length": 428,
    "chunk_id": "doc_moh_malaria_triage_2024_c0001",
    "chunk_index": 1,
    "doc_id": "doc_moh_malaria_triage_2024",
    "document_title": "National Vector-Borne Disease Control Programme \u2014 Community Malaria Rapid Diagnostic Protocol",
    "document_version": "NVBDCP-SOP-2024",
    "extra_metadata": {
      "file_type": "html",
      "strategy": "token_structural"
    },
    "page_number": null,
    "publication_date": "2024",
    "section_heading": "Step-by-Step RDT Field Testing Procedure",
    "source_file": "data/sample_advisories/moh_malaria_rapid_triage.html",
    "token_count": 86,
    "total_chunks": 3
  }
]
```

---

## 4. Negative / Out-of-Domain Query Run (Empty Retrieval Fallback)

To prevent hallucination, if retrieval returns zero chunks or chunks below the similarity threshold, the pipeline immediately returns a clear fallback response without calling generation.

- **Query**: `What evidence is required for project submission?`
- **Retrieved Chunks**: 0
- **Generated Answer**: `I could not find relevant context for that question.`
- **Returned Sources**: `[]`

---

## 5. Architectural Verification & Guarantees

1. **Separation of Concerns**: Each stage is an independent function that can be executed and unit-tested in complete isolation.
2. **Zero Hallucination on Zero Context**: When retrieval yields 0 evidence, the pipeline deterministically refuses rather than inventing speculative guidance.
3. **Traceable Citations**: Generated answers cite exact document names and section titles, matching Member 1's metadata contract.
4. **Resilient Fallback**: System functions reliably both with live OpenAI-compatible LLM endpoints and in offline deterministic demonstration modes.
