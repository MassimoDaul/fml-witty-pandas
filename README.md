# fml-witty-pandas

# TO DO LIST

| Task | Person | Notes |
|-----|-----|-----|
| test search API for papers |  |  |
| papers embedding improvements |  | relevant search considerations beyond semantics: "impact"/citations, topics (e.g. currently searching for "attention is all you need" returns astrophysics papers) |
| add authors to semantic search |  | have to meaningfully embed authors in the same space as papers somehow (combine embeddings from papers?) |
| add events/conferences to semantic search |  |  |
| improve web app search UI |  | web app can be rebuilt with another framework (e.g. Vite+React, Next JS); consider extracting db schema out of papers/setup_db.sql into jsonschema file |
| design more + improve existing data visualizations |  |  |
| brainstorm one algorithm to implement by hand |  | e.g. search intent classification (paper? author? event?) |

# Student Research Helper

A tool that helps students explore academic research opportunities by recommending relevant resources based on a short description of a research idea.

Given a short text description (e.g. *"machine learning for endangered languages"*), the system retrieves related academic resources using similarity search over several curated datasets.

These resources include:

- Relevant research papers
- Conferences where the work might be published
- Scholarships or fellowships
- Research labs or groups

The goal is to make it easier for students to move from **an idea** to **concrete research directions and opportunities**.

---

## How it works

The system takes a text query and embeds it into a vector representation.  
It then performs nearest-neighbor search over multiple datasets of embedded resources.

Each dataset contains textual descriptions (titles, abstracts, topics).  
We compute similarity between the query embedding and dataset embeddings and return the closest matches.

Example workflow:

1. User enters research idea
2. System computes embedding
3. Nearest neighbors are retrieved from:
   - papers dataset
   - conference dataset
   - scholarship dataset
4. Results are returned ranked by similarity

---

## Algorithms Used

This project implements algorithms from the course including:

- **Vector embeddings of text**
- **Nearest Neighbor Search**
- **Distance metrics (cosine / Euclidean)**
- **Clustering or indexing methods for retrieval (optional extension)**

Nearest neighbor retrieval is implemented directly rather than relying on a library wrapper.

---

## Example Query

Input: "machine learning methods for predicting baby genetics"


Output:

- Relevant papers on machine vision and facial detection
- Conferences such as ....
- Scholarships and fellowships related to genetics and imaging

---

## Project Structure

TBD

---

## Contributors

- [Massimo Daul](https://github.com/MassimoDaul)
- [Benny Yuan](https://github.com/bennyyuan12)
- [Andrew Jiang](https://github.com/minghanminghan)
- [Audrey McKnight](https://github.com/audrey-mcknight)
- [Audrey McKnight](https://github.com/rushil-johal)

---

Thank you to arXiv for use of its open access interoperability.
