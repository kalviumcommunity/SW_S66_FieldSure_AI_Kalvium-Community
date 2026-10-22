# Generating Embeddings via API — Sample Output

Run with `uv run python -m src.embed_corpus` after configuring `API_KEY` and
`EMBEDDING_MODEL` in `.env`. `API_BASE_URL` is optional; when omitted the
existing config uses the OpenAI-compatible default. The command reads the first
three chunks from `outputs/ingested_chunks.jsonl` and writes full vectors with
the original text and metadata to `outputs/embedded_chunks.jsonl`.

The run below used the prepared corpus but a **mocked API response**. The
8-dimensional values are illustrative test vectors, not embeddings returned by
`text-embedding-3-small`. A live model's vector dimension and values depend on
the configured API and model.

```text
Model: mocked-model
Chunks embedded: 3
Vector length: 8
Stored vectors: outputs\embedded_chunks.jsonl

Chunk: doc_who_dengue_guidance_2024_c0000 | Source: data/sample_advisories/who_dengue_clinical_guidance.md | Index: 0 | Section: Document Information
Text preview: "[Section: Document Information] # WHO Clinical Management of Dengue in Primary Health Centers ## Document Information - **Source**: World Health Organization / Regional Disease Surveillance - **Version**: 2024.1 Update - **Category**: Vector-Borne..."
Vector preview: [0.012300, -0.045600, 0.078900, 0.101100, -0.121300, 0.141500, -0.161700, 0.181900]

Chunk: doc_who_dengue_guidance_2024_c0001 | Source: data/sample_advisories/who_dengue_clinical_guidance.md | Index: 1 | Section: 1. Case Definition & Diagnostic Phases
Text preview: "[Section: 1. Case Definition & Diagnostic Phases] Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes. The clinical course consists of three distinct phases: 1. **Febrile Phase**: Abrupt onset of high fever (>38.5°C), severe..."
Vector preview: [0.021000, -0.043000, 0.065000, -0.087000, 0.109000, -0.121000, 0.143000, -0.165000]

Chunk: doc_who_dengue_guidance_2024_c0002 | Source: data/sample_advisories/who_dengue_clinical_guidance.md | Index: 2 | Section: 2. Red Flag Danger Signs Requiring Urgent Referral
Text preview: "[Section: 2. Red Flag Danger Signs Requiring Urgent Referral] Community health workers must evaluate patients daily during the critical phase for warning signs: - Persistent vomiting (> 3 episodes in 24 hours). - Severe abdominal pain or tenderness. - Fluid..."
Vector preview: [-0.031000, 0.052000, 0.073000, -0.094000, 0.115000, 0.136000, -0.157000, 0.178000]
```

Each JSONL record written by the command contains the complete `chunk_id`,
`text`, `metadata` object (including source file, chunk index, section, and page
when present), `embedding` array, and `vector_dimension`. Text and metadata are
only shortened in the console preview above; the saved record keeps the full
values.
