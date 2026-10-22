# Retrieval Evaluation & Recall Testing

## Result

The offline run evaluated 3 labeled queries at `k=3` using deterministic
illustrative vectors in a temporary ChromaDB collection.

| Measure | Result |
| --- | ---: |
| Queries | 3 |
| Macro precision@3 | 0.333 |
| Macro recall@3 | 1.000 |
| Micro precision@3 | 0.333 |
| Micro recall@3 | 1.000 |
| Hit@3 rate | 1.000 |
| Top-1 hit rate | 0.000 |
| Diagnosed top-1 misses | 3 |

Precision@k uses the requested `k` as its denominator. Each query returns one
labeled relevant chunk in the top three, so recall is 1.0 and precision is
1/3. For each query, the deliberately non-relevant fixture distractor scores
1.00 at rank 1; the labeled chunk scores approximately 0.95 at rank 2.

| Query | Labeled relevant chunk | Source | Relevant rank / score |
| --- | --- | --- | --- |
| Dengue warning signs | `doc_who_dengue_guidance_2024_c0002` | `data/sample_advisories/who_dengue_clinical_guidance.md` | 2 / 0.950 |
| Cholera water treatment | `doc_cdc_cholera_protocol_2024_c0001` | `data/sample_advisories/cdc_cholera_water_safety.txt` | 2 / 0.950 |
| Malaria rapid test | `doc_moh_malaria_triage_2024_c0001` | `data/sample_advisories/moh_malaria_rapid_triage.html` | 2 / 0.950 |

## Failure analysis and limitations

All three top-1 misses are intentional: fixture vectors place the first
non-relevant corpus chunk ahead of the labeled relevant chunk. This validates
that the evaluator reports a top-1 ranking failure even when recall@3 is
complete. It is a synthetic diagnostic case, not a measured failure of a live
embedding model or the production collection. The query-level JSON report
includes retrieved scores, source text, metadata, and likely-cause notes.

For real evaluation, ensure the collection contains the labeled chunks and
configure `EMBEDDING_MODEL` to match the model used to index them:

```bash
uv run python -m src.evaluate_retrieval
```

To reproduce the offline report without API credentials:

```bash
uv run python -m src.evaluate_retrieval --self-test
```
