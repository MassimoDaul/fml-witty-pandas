# Papers Dataset

This dataset contains academic papers used for the research-paper recommendation component of the Student Research Helper.

Given a user query describing a research idea, we retrieve similar papers using vector embeddings and nearest-neighbor search.

---

## Data Source

We will use publicly available metadata from the **arXiv dataset**, which contains paper titles, abstracts, and subject categories.

Primary source:

- https://www.kaggle.com/datasets/Cornell-University/arxiv
- https://huggingface.co/datasets/arxiv_dataset

Relevant fields:

- `title`
- `abstract`
- `categories`

The text used for embeddings will be:

text = title & abstract

## Embedding Plan

Each paper will be converted into a vector representation.

Steps:

1. Clean text (remove newlines, normalize whitespace)
2. Combine title and abstract
3. Generate vector embeddings
4. Store embeddings for nearest-neighbor search

We will experiment with:

- TF-IDF vectors (baseline) (this is a classic linguistic method)
- Sentence embeddings (optional extension)

---

## Retrieval Method

Given a user query:

1. Convert the query into an embedding
2. Compute similarity between the query and all paper embeddings
3. Return the top-k most similar papers

Similarity will be computed using:

- cosine similarity
- Euclidean distance (comparison)

---

## Next Steps

- Download arXiv metadata
- Filter to relevant fields
- Generate embeddings
- Save processed dataset for retrieval

