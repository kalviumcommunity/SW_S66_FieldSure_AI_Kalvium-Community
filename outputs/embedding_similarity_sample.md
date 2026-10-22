# Embedding Similarity & Distance Metrics — Sample Ranking

Run `uv run python -m src.rank_chunks` after generating
`outputs/embedded_chunks.jsonl` with `uv run python -m src.embed_corpus`. Set
`API_KEY` and `EMBEDDING_MODEL` in `.env`; `API_BASE_URL` is optional. Provide a
different query with `--query`.

Cosine similarity is used because it compares the direction of embedding
vectors, making it useful for semantic matching without letting vector
magnitude dominate. Higher scores indicate closer directional alignment.

The example below uses three actual sample corpus chunks and illustrative
mocked chunk and query vectors. It demonstrates ranking output only; scores are
not from a live embedding model.

```text
Metric: cosine similarity (compares vector direction independent of magnitude)
Model: mocked-model | Query: "How can field workers reduce dengue risk?"
Chunks ranked: 3

#1 | cosine=0.993884 | Chunk: doc_who_dengue_guidance_2024_c0001 | Source: data/sample_advisories/who_dengue_clinical_guidance.md | Index: 1 | Section: 1. Case Definition & Diagnostic Phases
Text preview: "[Section: 1. Case Definition & Diagnostic Phases] Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes. The clinical course consists of three distinct phases: 1. **Febrile Phase**: Abrupt onset of high fever (..."

#2 | cosine=0.968364 | Chunk: doc_moh_malaria_triage_2024_c0000 | Source: data/sample_advisories/moh_malaria_rapid_triage.html | Index: 0 | Section: Clinical Triage & Risk Stratification
Text preview: "[Section: Clinical Triage & Risk Stratification] National Vector-Borne Disease Control Programme Standard Operating Procedure: Community Malaria Rapid Diagnostic Testing (RDT) 1. Target Audience: Auxiliary Nurse Midwives (ANMs), Accredit..."

#3 | cosine=0.000000 | Chunk: doc_cdc_cholera_protocol_2024_c0001 | Source: data/sample_advisories/cdc_cholera_water_safety.txt | Index: 1 | Section: 2. WATER TREATMENT STANDARDS
Text preview: "[Section: 2. WATER TREATMENT STANDARDS] 2. WATER TREATMENT STANDARDS - All drinking water distributed in settlement camps must be disinfected with sodium dichloroisocyanurate (NaDCC) or calcium hypochlorite. - Target Free Residual Chlori..."

Most similar: doc_who_dengue_guidance_2024_c0001
Least similar: doc_cdc_cholera_protocol_2024_c0001
```

The dengue guidance is ranked first, the related vector-borne malaria triage
chunk second, and the unrelated cholera water-treatment chunk last.
