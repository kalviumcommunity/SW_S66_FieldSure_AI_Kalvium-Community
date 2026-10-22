# Embedding Fundamentals Sample

Run the live demonstration with `uv run python -m src.embedding_fundamentals` after setting `API_KEY` and (optionally) `EMBEDDING_MODEL` in `.env`. It sends three sample texts to the configured OpenAI-compatible `/embeddings` endpoint, reports vector dimensions, previews vector values, and compares cosine similarity.

The captured values below use a mocked API response because no live API credentials were configured for this run. They demonstrate the output format and comparison logic; they are not embeddings produced by the named model.

```text
Model: illustrative-mock-model
Vector dimension: 4
All 3 vectors have the same length: yes

Text: "How can I prevent dengue fever?"
Vector (first 4 values): [0.800000, 0.200000, 0.100000, 0.000000]

Text: "What steps help avoid getting dengue?"
Vector (first 4 values): [0.750000, 0.250000, 0.120000, 0.050000]

Text: "How do I renew a library book?"
Vector (first 4 values): [0.050000, 0.100000, 0.950000, 0.200000]

Similar pair cosine similarity: 0.994717
Unrelated pair cosine similarity: 0.190944
```

Embedding vectors are numeric representations of meaning learned by an embedding model. They are not random IDs or keyword counts: related texts tend to map near one another in the model's vector space. Cosine similarity compares the angle between two vectors, with larger values indicating greater directional similarity.
