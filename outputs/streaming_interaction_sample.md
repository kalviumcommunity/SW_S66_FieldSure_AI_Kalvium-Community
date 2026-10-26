# FieldSure AI — Streaming Responses & Citation Display (Concept 3.47)

> **API Contract & Execution Evidence**: Verified demonstration of progressive answer streaming over Server-Sent Events (SSE) and verifiable citation inspection.

---

## 1. Overview & Learning Objectives

- **Progressive Streaming**: Answers stream token-by-token to the client interface so users don't wait silently during generation.
- **Verifiable Citations**: Citations display visible markers (e.g. `[1]`), document titles, chunk IDs, and sections.
- **Inspectable Source Evidence**: Users can expand citations to inspect the exact retrieved source passage.
- **Resilient Interruption Handling**: If a stream is stopped or fails halfway, partial text and citations are preserved with a retry action.

---

## 2. Streaming Interaction Scenarios

### Scenario #1: Progressive Grounded Answer Streaming
- **Question**: `What evidence is required for project submission?`
- **Initial Event**: Citations delivered before token generation begins (`1 source(s)`).
- **Token Delivery**: `50 tokens` streamed progressively.
- **Final Accumulated Answer**:
```text
Based on verified protocol guidance :
Project submission requires evidence such as git commits, automated test execution reports, and source citation traceability.

Notice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).
```

#### Cited Source Evidence (Inspectable)
| Marker | Document | Chunk ID | Section | Source Passage Preview |
| :---: | :--- | :--- | :--- | :--- |
| **[1]** | `FieldSure AI Project Submission Standards` | `proj_sub_std_2026_c0001` | `Evidence Requirements` | *"Project submission requires evidence such as git commits, automated test executi..."* |


---

### Scenario #2: Safe Refusal Streaming (Out-of-Domain Query)
- **Question**: `What is the return policy for retail equipment?`
- **Guardrail Detection**: Low relevance context caught prior to generation.
- **Citations Event**: `0 citations` (no fabricated evidence).
- **Streamed Refusal**:
```text
I don't have enough reliable context to answer that.
```

---

### Scenario #3: Stream Interruption & Partial Output Preservation
- **Question**: `What evidence is required for project submission?`
- **Simulated Event**: Stream was aborted/interrupted at token `25` of `50`.
- **Preserved Partial Text**:
```text
Based on verified protocol guidance :
Project submission requires evidence such as git commits, automated test execution reports, and source citation traceability.

Notice: FieldSure AI ... [Incomplete - Stream Interrupted]
```
- **Preserved Citations**: `1 source citation(s)` kept intact.
- **User Interface State**: Re-enables input, displays retry button, avoids false success state.

---

## 3. Server-Sent Events (SSE) Wire Format

```http
POST /query/stream HTTP/1.1
Host: localhost:8001
Content-Type: application/json
Accept: text/event-stream

{"question": "What evidence is required for project submission?"}

HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"type": "citations", "sources": [{"id": "source-1", "label": "[1]", "document": "project_submission_policy.pdf", "chunk_id": "proj_sub_std_2026_c0001", "text": "..."}]}

data: {"type": "token", "text": "Project "}
data: {"type": "token", "text": "submission "}
data: {"type": "token", "text": "requires "}
data: {"type": "token", "text": "evidence..."}

data: {"type": "done"}
```

---

## 4. Verification Checklist

- [x] **Task 1 — Stream answers progressively**: `POST /query/stream` emits progressive token events.
- [x] **Task 2 — Display citations clearly**: Visible markers `[1]`, document names, chunk IDs, and sections.
- [x] **Task 3 — Let users view cited sources**: Users expand source details to inspect exact chunk text.
- [x] **Task 4 — Handle streaming errors**: Interruptions preserve partial answer and citations with retry action.
- [x] **Task 5 — Commit sample interaction**: Verified evidence committed in JSON and Markdown.
