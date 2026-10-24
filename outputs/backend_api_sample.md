# FieldSure AI — Backend API for RAG Service (Concept 3.44)

> **API Contract & Execution Evidence**: Verified demonstration of the FastAPI backend service exposing the FieldSure AI RAG pipeline for frontend and client consumption.

---

## 1. API Specification Summary

- **Endpoint**: `POST /query`
- **Request Content-Type**: `application/json`
- **Response Content-Type**: `application/json`
- **Health Check**: `GET /health`
- **Config Inspection**: `GET /config` (Masked credentials)

### Environment Configuration (Zero Hardcoding)
| Variable | Value / Source | Description |
|---|---|---|
| `OPENAI_API_KEY` | `[NOT SET]` | Authentication credential for LLM provider |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Target embedding model for vector search |
| `VECTOR_DB_URL` | `http://localhost:6333` | Remote or local vector database address |
| `COLLECTION_NAME` | `rag_chunks` | Vector collection storing ingested chunks |

---

## 2. Sample Request & Response (Grounded Answer)

### Request
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What evidence is required for project submission?"}'
```

### Response (HTTP 200 OK)
```json
{
  "answer": "Based on verified protocol guidance :\nProject submission requires evidence such as git commits, automated test execution reports, and source citation traceability.\n\nNotice: FieldSure AI is an informational protocol assistant, not a clinical diagnostic system. Always refer patients with severe or worsening symptoms to the nearest primary health center (PHC).",
  "sources": [
    {
      "source": "project_submission_policy.pdf",
      "chunk_id": "proj_sub_std_2026_c0001",
      "score": 0.75
    }
  ],
  "status": "answered",
  "metadata": {
    "top_score": 0.75,
    "chunks_used": 1,
    "is_refused": false,
    "total_sources": 1
  }
}
```

---

## 3. Sample Request & Response (Refusal Guardrail)

### Request
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy for products?"}'
```

### Response (HTTP 200 OK)
```json
{
  "answer": "I don't have enough reliable context to answer that.",
  "sources": [],
  "status": "refused_weak_context",
  "metadata": {
    "top_score": 0.3333,
    "chunks_used": 0,
    "is_refused": true,
    "total_sources": 0
  }
}
```

---

## 4. Input Validation & Error Handling

### Request (Invalid Question Length < 3)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "hi"}'
```

### Response (HTTP 422 Unprocessable Entity)
```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": [
        "body",
        "question"
      ],
      "msg": "String should have at least 3 characters",
      "input": "hi",
      "ctx": {
        "min_length": 3
      }
    }
  ]
}
```

---

## 5. Verification Checklist

- [x] **Task 1 — Query endpoint**: `POST /query` operational and serving requests.
- [x] **Task 2 — Structured JSON**: Reliable response schema with `answer`, `sources`, `status`, and `metadata`.
- [x] **Task 3 — Validation & Error Handling**: Pydantic validation handles length breaches (422), ValueErrors (400), and unhandled errors (500).
- [x] **Task 4 — Environment Configuration**: All settings loaded from environment with zero hardcoded secrets.
- [x] **Task 5 — Committed Sample Evidence**: Exported to `backend_api_sample.json` and `backend_api_sample.md`.
