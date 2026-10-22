# Retrieval Relevance Tuning

## Experiment setup

Three queries were evaluated against prepared chunks with expected relevant
**source documents**:

| Query | Expected source |
|---|---|
| What dengue guidance should field workers use? | `data/sample_advisories/who_dengue_clinical_guidance.md` |
| How should drinking water be treated during a cholera outbreak? | `data/sample_advisories/cdc_cholera_water_safety.txt` |
| What is the malaria rapid diagnostic testing procedure? | `data/sample_advisories/moh_malaria_rapid_triage.html` |

The repository does not contain live generated chunk embeddings or configured
API credentials. This reproducible ChromaDB experiment therefore uses
deterministic illustrative vectors: one deliberately high-scoring off-source
distractor per query and vectors for all chunks in the expected source. The
resulting scores validate the retrieval-setting comparison and metrics, not
real semantic quality.

## Compared settings

| Setting | k | Source metadata filter | Minimum cosine similarity |
|---|---:|---|---:|
| `baseline_vector_k2` | 2 | None | 0.00 |
| `source_filter_k5_threshold_0_5` | 5 | `source_file` equals expected source | 0.50 |

Metrics treat chunks from the expected source as relevant. Precision, recall,
and F1 are macro-averaged over the three queries.

## Results

| Setting | Mean precision@k | Mean recall@k | Mean F1@k | Top-1 hit rate | Hit@k rate |
|---|---:|---:|---:|---:|---:|
| `baseline_vector_k2` | 0.500 | 0.278 | 0.356 | 0.000 | 1.000 |
| `source_filter_k5_threshold_0_5` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

For each query, unfiltered vector search ranked an unrelated off-source
distractor first (similarity `1.000000`) and one expected-source chunk second
(similarity `0.980000`), yielding precision@2 `0.5` and top-1 hit `false`.
Filtering to the expected source and using `k=5` with threshold `0.5` returned
all expected-source chunks: dengue `4/4`, cholera `4/4`, malaria `3/3`. Every
returned chunk scored `0.980000`; per-query precision and recall were `1.0`.

## Selected configuration

Choose **`source_filter_k5_threshold_0_5`** for this experiment. It achieved
the best mean F1@k (`1.000`), precision, recall, and top-1 hit rate. The source
filter removed the deliberately plausible but wrong-source first result, `k=5`
covered all expected source chunks, and threshold `0.5` retained the relevant
fixture results. This is a candidate for evaluation on real embeddings and
annotated relevance judgments before production use.

The expected-source filter assumes the caller already knows which source
document should be in scope. It is appropriate for an explicitly scoped search,
not as an automatic filter when a query's source is unknown.

The machine-readable query-by-query report, including each retrieved chunk's
ID, score, source text, and metadata, is in
`outputs/retrieval_tuning_report.json`. Re-run with:

```bash
uv run python -m src.tune_retrieval
```
