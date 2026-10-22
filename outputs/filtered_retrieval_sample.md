# Metadata Filtering & Hybrid Search Sample

## Configuration

- **Query:** `dengue mosquito prevention`
- **Metadata filter:** `source_file = data/sample_advisories/who_dengue_clinical_guidance.md`
- **Embedding model label:** `illustrative-test-model`
- **Candidate limit:** 2
- **Hybrid weighting:** 70% normalized cosine similarity + 30% exact keyword coverage

The repository does not include `outputs/embedded_chunks.jsonl` or live API
credentials. This end-to-end ChromaDB example used the 11 prepared source
chunks with illustrative vectors and a mocked query embedding. The example
proves filter and reranking behavior; its scores are not real embedding
similarities or a claim about production retrieval quality.

## Unfiltered vector search — top 2

| Rank | Vector similarity | ID | Source | Chunk index | Section |
|---:|---:|---|---|---:|---|
| 1 | 1.000000 | `doc_moh_malaria_triage_2024_c0000` | `data/sample_advisories/moh_malaria_rapid_triage.html` | 0 | Clinical Triage & Risk Stratification |
| 2 | 1.000000 | `doc_moh_malaria_triage_2024_c0002` | `data/sample_advisories/moh_malaria_rapid_triage.html` | 2 | Treatment & Urgent Referral Protocols |

Retrieved text previews:

1. `[Section: Clinical Triage & Risk Stratification] National Vector-Borne Disease Control Programme Standard Operating Procedure: Community Malaria Rapid Diagnostic Testing...`
2. `[Section: Treatment & Urgent Referral Protocols] P. falciparum positive: Initiate Artemisinin-based Combination Therapy...`

For this dengue-specific case, neither unfiltered top result is from the dengue
source; precision@2 is **0/2 (0%)**.

## Filtered vector search — top 2

Filter applied by ChromaDB before vector ranking:
`{"source_file": "data/sample_advisories/who_dengue_clinical_guidance.md"}`

| Rank | Vector similarity | ID | Source | Chunk index | Section |
|---:|---:|---|---|---:|---|
| 1 | 0.953583 | `doc_who_dengue_guidance_2024_c0003` | `data/sample_advisories/who_dengue_clinical_guidance.md` | 3 | 3. Immediate Field Interventions |
| 2 | 0.913812 | `doc_who_dengue_guidance_2024_c0001` | `data/sample_advisories/who_dengue_clinical_guidance.md` | 1 | 1. Case Definition & Diagnostic Phases |

Retrieved text previews:

1. `[Section: 3. Immediate Field Interventions] Administer oral rehydration with ORS, coconut water, or fruit juices...`
2. `[Section: 1. Case Definition & Diagnostic Phases] Dengue is a severe arboviral infection transmitted by *Aedes aegypti* mosquitoes...`

Both results match the selected source; precision@2 is **2/2 (100%)**.

## Filtered hybrid results — top 2

Hybrid score: `0.7 * ((cosine_similarity + 1) / 2) + 0.3 * keyword_coverage`.
Keyword coverage is the fraction of distinct query terms found as exact words
in the chunk text or metadata.

| Rank | Hybrid score | Vector similarity | Keyword coverage | ID | Source / section |
|---:|---:|---:|---:|---|---|
| 1 | 0.783754 | 0.953583 | 0.333333 | `doc_who_dengue_guidance_2024_c0003` | `who_dengue_clinical_guidance.md` — 3. Immediate Field Interventions |
| 2 | 0.769834 | 0.913812 | 0.333333 | `doc_who_dengue_guidance_2024_c0001` | `who_dengue_clinical_guidance.md` — 1. Case Definition & Diagnostic Phases |

The hybrid output retains the dengue source constraint and reports both its
combined score components and the original chunk metadata. The meaningful
precision gain in this fixture comes from the metadata filter, which removes
the loosely related malaria records that occupied the unfiltered top two.
Hybrid results refer to the same chunks, text previews, and source metadata
shown in the filtered vector-search section above.
