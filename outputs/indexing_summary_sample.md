# Embedding Indexing Summary

## Sample run

This local ChromaDB run used all **11 prepared source chunks** from
`outputs/ingested_chunks.jsonl`. No model-generated embedding file or API
credentials were available in the workspace, so the 8-dimensional vectors
below were illustrative fixtures created only to exercise insertion,
count-validation, and readback. They are not semantic embeddings.

| Check | Result |
| --- | ---: |
| Source chunks | 11 |
| Embedded input records | 11 |
| Indexed ChromaDB records | 11 |
| Indexed count matches source | Yes |
| Vector dimension | 8 |
| Spot-checks passed | 2 / 2 |
| Failures | None |

## Spot-check readback

### First record

- **ID:** `doc_who_dengue_guidance_2024_c0000`
- **Source text:** `[Section: Document Information] # WHO Clinical Management of Dengue in Primary Health Centers ...`
- **Metadata:** source `data/sample_advisories/who_dengue_clinical_guidance.md`; section `Document Information`; chunk index `0`; agency `WHO`
- **Vector length:** 8
- **Vector preview:** `[0.0100, 0.0200, 0.0300, 0.0400, 0.0500, 0.0600, 0.0700, 0.0800]`
- **Integrity check:** passed; ID, text, metadata, and vector match the input record.

### Last record

- **ID:** `doc_moh_malaria_triage_2024_c0002`
- **Source text:** `[Section: Treatment & Urgent Referral Protocols] P. falciparum positive: Initiate Artemisinin-based Combination Therapy ...`
- **Metadata:** source `data/sample_advisories/moh_malaria_rapid_triage.html`; section `Treatment & Urgent Referral Protocols`; chunk index `2`; agency `MOH`
- **Vector length:** 8
- **Vector preview:** `[0.1100, 0.2200, 0.3300, 0.4400, 0.5500, 0.6600, 0.7700, 0.8800]`
- **Integrity check:** passed; ID, text, metadata, and vector match the input record.

## Failures

None. The sample index completed with 11 records in the collection, matching
the 11 source chunks.

For production indexing, generate real embeddings first and run
`uv run python -m src.index_embeddings`. The command validates the complete
embedding file against the source corpus and writes a detailed JSON report to
`outputs/indexing_summary.json`.
