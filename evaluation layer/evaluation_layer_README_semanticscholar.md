# Evaluation Layer for Semantic Scholar API–Backed Research Paper Search

## 1. Purpose

This README defines the evaluation layer for a group project on research paper search using the Semantic Scholar Academic Graph API as the shared upstream data source.

The evaluation layer is designed to compare **full-pipeline search systems** whose final outputs are generated from the same Semantic Scholar search backend, with the final goal of selecting the most useful search pipeline for literature review use cases.

The main user profile is:

- beginner students trying to understand a field
- graduate students doing literature review

The evaluation focuses on the **top 10 returned papers**, because that is the part of the system users are most likely to inspect.

The benchmark is intended to answer questions like:

- Which pipeline gives cleaner top-10 results?
- Which pipeline ranks strong papers better?
- Which pipeline is better for broad exploratory search?
- Which pipeline handles explicit year filters correctly?
- Which pipeline is strongest for different query types?

This README covers:

- what the evaluated system boundary is
- how the benchmark is organized
- what teams must submit
- how the judging layer works
- which metrics are used
- how results are reported and interpreted

This README does **not** cover training methods, embedding internals, or UI design.

---

## 2. Evaluated System Boundary

For this project, the evaluated object is the **final search pipeline output**, not any single internal component.

A system is considered the **evaluated pipeline** if it includes everything from:

- the benchmark query input
- any explicit query parameters provided by the evaluator
- local query preprocessing or rewriting
- Semantic Scholar retrieval
- pagination or candidate gathering from the API if needed
- reranking
- post-processing
- final display ordering

up to:

- the final ranked top 10 paper list returned to the evaluator

In other words, the evaluation layer scores the **final ranked result list** that the user would see.

This means the evaluation layer is **black-box** with respect to the team pipelines. Different team members may use different internal methods, such as:

- dense embeddings
- lexical retrieval
- hybrid retrieval
- query expansion
- learned rerankers
- heuristic reranking
- post-retrieval filtering

The evaluation layer does not need access to those internals. It only needs the final ranked outputs on the benchmark queries.

---

## 3. Benchmark Design

### 3.1 Shared Semantic Scholar backend assumptions

All systems are expected to use the Semantic Scholar Academic Graph API as the shared search backend.

For version 1, the benchmark is designed around the **paper bulk search** endpoint:

- base URL: `https://api.semanticscholar.org/graph/v1`
- search endpoint: `/paper/search/bulk`
- authentication header: `x-api-key: <YOUR_KEY>`

Semantic Scholar documents the paper bulk search endpoint as the recommended keyword-search endpoint for most cases, with:

- required query parameter: `query`
- optional query parameters such as `token`, `fields`, `sort`, `publicationTypes`, `openAccessPdf`, `minCitationCount`, `publicationDateOrYear`, `year`, `venue`, and `fieldsOfStudy`
- response structure with top-level `total`, optional `token`, and `data`
- paper objects identified by `paperId`

The evaluator and the team pipelines should use **Semantic Scholar field names** directly wherever practical, especially:

- `paperId`
- `query`
- `year`
- `publicationDate`
- `fieldsOfStudy`
- `s2FieldsOfStudy`
- `authors`

The benchmark also assumes:

- all teams use the same benchmark query file
- all teams use the same required `fields` string unless the evaluator explicitly changes it
- all teams use the same Semantic Scholar API service, even if they apply different local reranking logic

### 3.2 Query set overview

The benchmark contains **100 public queries**.

The benchmark is:

- public to all participating team members
- balanced by design
- frozen before final comparison

Version 1 evaluates **single-turn search**, not multi-turn refinement.

### 3.3 Query types

Version 1 uses two main query types.

#### Broad exploration

These queries are intended to explore a field, topic, or area.

Examples:

- machine learning for healthcare
- graph learning in biology
- fairness in AI for education

A good result list for a broad query should:

- stay clearly on topic
- surface relevant papers
- help the user understand the range of the field
- include meaningfully different subtopics or directions within the broad area

A broad result list should **not** collapse into ten papers from the same narrow subtopic unless that subtopic is actually the whole query.

#### Specific technical

These queries are intended to retrieve papers aligned with a particular method, task, or framing.

Examples:

- contrastive learning for chest x ray report generation
- retinal image segmentation with transformer-based domain adaptation

A good result list for a specific query should:

- be tightly aligned with the exact topic
- contain little off-topic drift
- place the strongest papers early in the ranking

### 3.4 Year-filter scenarios

Year filtering is treated as a **scenario modifier**, not as a separate main query bucket.

The benchmark therefore spans four scenarios:

- broad, no year filter
- broad, year-filtered
- specific, no year filter
- specific, year-filtered

This design reflects the actual product more accurately than a separate "recent-work" bucket. If the benchmark query includes a `year` parameter, the evaluation should treat that parameter as a **constraint** on the search rather than a vague notion of recency.

### 3.5 Public benchmark policy

All 100 benchmark queries are public. There is no hidden final set in version 1.

This makes iteration simpler for a class project, but it also means that systems can overfit to the benchmark. To keep the comparison fair:

- the benchmark should be frozen before final comparison
- each team should submit one clean final run for scoring

---

## 4. Submission Contract

This section defines the exact handoff between the evaluation layer and the team pipelines.

The evaluation layer should be treated as a **benchmark harness**. Teammates will run their systems locally, call Semantic Scholar using the evaluator-provided benchmark inputs, and submit only the final ranked outputs. The evaluator then validates those outputs, hydrates paper metadata from Semantic Scholar by `paperId`, and sends structured judge packets to the LLM.

### 4.1 What the evaluator provides to the team

The evaluator provides a public benchmark query file, typically:

`queries/benchmark_queries.jsonl`

Each query row should contain only the fields the search system needs at inference time, and those fields should use Semantic Scholar parameter names whenever possible.

Required public query fields:

- `queryId` — unique evaluator identifier for the benchmark query
- `query` — the search string passed to Semantic Scholar bulk search
- `year` — either `null` or a Semantic Scholar-compatible year-range string
- `fields` — comma-separated Semantic Scholar fields that teams are expected to request

Optional public query fields, if the evaluator decides to use them later:

- `sort`
- `publicationTypes`
- `openAccessPdf`
- `minCitationCount`
- `publicationDateOrYear`
- `venue`
- `fieldsOfStudy`

Recommended minimum `fields` value for evaluation:

`paperId,title,abstract,authors,year,publicationDate,fieldsOfStudy,s2FieldsOfStudy,url`

Expected public queries format:

```json
{"queryId":"q_001","query":"machine learning for healthcare","year":null,"fields":"paperId,title,abstract,authors,year,publicationDate,fieldsOfStudy,s2FieldsOfStudy,url"}
{"queryId":"q_002","query":"clinical foundation models","year":"2023-","fields":"paperId,title,abstract,authors,year,publicationDate,fieldsOfStudy,s2FieldsOfStudy,url"}
```

Notes:

- Teammates are expected to run **all** queries in this file.
- Teammates should treat `year` as a hard search constraint when it is present.
- Query type labels such as `broad` or `specific` are evaluator metadata and are **not required** as model inputs.
- `token` is **not** part of the public benchmark input. It is returned by the API at runtime if a team chooses to paginate through more results.

### 4.2 What the teammates must submit

Each teammate submits **one JSONL file per system run**.

Recommended filename:

`submissions/<runId>.jsonl`

Each line in the submission file corresponds to **one query** and must contain:

- `runId` — unique identifier for the submitted run; should be the same on every line in the file
- `queryId` — must exactly match a query in `benchmark_queries.jsonl`
- `results` — ordered list of exactly 10 returned papers

Each item inside `results` must contain:

- `rank` — integer rank from 1 to 10
- `paperId` — Semantic Scholar `paperId` for the returned paper
- `score` — optional numeric score from the team system; used only for debugging and later analysis, not shown to the LLM judge

Required submission shape:

```json
{"runId":"team_a_model_1","queryId":"q_001","results":[{"rank":1,"paperId":"649def34f8be52c8b66281af98ae884c09aef38b","score":12.43},{"rank":2,"paperId":"001720a782840652b573bb4794774aee826510ca","score":11.92},{"rank":3,"paperId":"0019e876188f781fdca0c0ed3bca39d0c70c2ad2","score":11.10},{"rank":4,"paperId":"833ff07d2d1be9be7b12e88487d5631c141a2e95","score":10.88},{"rank":5,"paperId":"144b8d9c10ea111598aa239100cd6ed5c6137b1c","score":10.44},{"rank":6,"paperId":"02138d6d094d1e7511c157f0b1a3dd4e5b20ebee","score":10.11},{"rank":7,"paperId":"018f58247a20ec6b3256fd3119f57980a6f37748","score":9.87},{"rank":8,"paperId":"0045ad0c1e14a4d1f4b011c92eb36b8df63d65bc","score":9.43},{"rank":9,"paperId":"630642b7040a0c396967e4dab93cf73094fa4f8f","score":9.01},{"rank":10,"paperId":"9f0f8dd5a0c39bb7d9e6d751248957f7f1c2b8aa","score":8.72}]}
```

The example above is only meant to show field names and structure. Real submissions must not repeat the same `paperId` within one query result list.

### 4.3 What is expected from the teammate pipeline

For every query in `benchmark_queries.jsonl`, the team pipeline is expected to:

1. read `queryId`, `query`, `year`, and `fields`
2. call Semantic Scholar bulk search using the provided parameters
3. optionally continue pagination using the API response `token` if the team needs a larger candidate set
4. run the full local search pipeline
5. return exactly 10 ranked `paperId`s
6. respect the `year` filter if it is present

The evaluation layer expects the team system to output the **final user-facing ranking**, not intermediate retrieval candidates.

That means the submitted top 10 should reflect all of the team's own processing decisions, including any:

- query rewriting
- hybrid search logic
- reranking
- filtering
- deduplication
- result cleanup

### 4.4 What is *not* expected from the team

Teams do **not** need to submit:

- paper titles
- abstracts
- authors
- publication dates
- full raw API responses
- `token`
- `total`
- embedding vectors
- retrieved candidate pools larger than top 10
- internal prompts or reasoning traces
- screenshots or UI output

The evaluator will hydrate the submitted `paperId`s against Semantic Scholar paper details data before sending anything to the LLM judge.

### 4.5 Validity rules

A submission is valid only if:

- the file is valid JSONL
- every benchmark query appears exactly once in the run
- each query returns exactly 10 ranked results
- ranks are integers from 1 through 10 with no gaps or duplicates
- all returned `paperId` values are non-empty strings
- no `paperId` is repeated within the same top-10 list
- the `runId` is consistent across the file

For filtered queries, year-filter compliance is scored as part of evaluation rather than treated as a file-format validation rule. However, teams are still expected to honor the filter when producing results.

### 4.6 Why JSONL is the required format

JSONL is used because the unit of submission is **one query plus its ordered top-10 result list**.

This structure is naturally nested and maps cleanly to later LLM judge packets. CSV is acceptable for exported reports, but JSONL is the required format for run submissions.

---

## 5. Judging Layer

The evaluation layer separates **judging** from **metrics**.

Judging produces labels. Metrics summarize those labels.

This separation is important because the quality of the evaluation depends first on how well the judge interprets the query and the returned list.

### 5.1 Paper-level judging

Paper-level judging operates on a **query-paper pair**.

The main paper-level label is **relevance** on a graded 0–3 scale:

- **0 = irrelevant**
- **1 = loosely related**
- **2 = relevant**
- **3 = highly relevant / strong fit**

Interpretation:

- A score of 0 means the paper should not meaningfully count as a successful answer.
- A score of 1 means the paper is in the neighborhood of the topic but is not a strong answer.
- A score of 2 means the paper is clearly useful for the query.
- A score of 3 means the paper is among the strongest fits for the query.

For **specific technical queries**, the judge should emphasize exactness of fit and penalize broad or weakly related papers.

For **broad exploration queries**, the judge should still score papers based on topical relevance, but list-level breadth should be evaluated separately.

### 5.2 List-level judging for broad queries

Broad queries require more than paper-level relevance. A list can contain individually relevant papers and still be a weak exploratory result set if it covers only one narrow slice of the field.

For that reason, broad queries receive a list-level score called:

### Exploration Range Score

The Exploration Range Score measures how well the top-10 result list covers **meaningfully different subtopics or directions within the broad query**, while still remaining relevant to the query.

This score is **not** just about semantic distance between papers. The goal is not to reward arbitrary difference. The goal is to reward **useful breadth inside the relevant search space**.

Recommended raw scoring scale:

- **1 = very narrow**
- **2 = limited range**
- **3 = moderate range**
- **4 = good range**
- **5 = strong range across meaningful subtopics**

A low Exploration Range Score should be assigned when the list is heavily concentrated in one narrow subtopic, even if many of the papers are individually relevant.

A high Exploration Range Score should be assigned when the list gives a student multiple meaningful entry points into the field.

### 5.3 What the judge sees

The judge may see hydrated paper metadata derived from Semantic Scholar, including:

- query text
- year filter if present
- `paperId`
- `title`
- `abstract`
- author names
- `year`
- `publicationDate`
- `fieldsOfStudy`
- `s2FieldsOfStudy`
- `url`

For prompting convenience, the evaluator may flatten `authors` from a list of author objects into a simple list of names before sending the packet to the LLM.

### 5.4 What the judge must not see

To reduce bias, the judge must not see:

- team name
- algorithm name
- embedding type
- internal retrieval score
- reranker score
- any statement of which system produced the results

### 5.5 Semantic Scholar implementation note

Version 1 may use an LLM as the primary judge. The evaluator should fetch or cache paper details by `paperId` so all systems are judged on the same canonical metadata shape instead of relying on teammate-submitted paper text.

A secondary embedding- or clustering-based diversity proxy can be added later as a diagnostic tool. However, the main Exploration Range Score should remain query-conditioned and relevance-aware rather than relying only on pairwise paper distance.

---

## 6. Metrics

Version 1 uses a small set of metrics that align with the main product goals.

### 6.1 Metrics used for all queries

#### Precision@10

Precision@10 measures how clean the top 10 is.

To compute Precision@10:

- treat papers with relevance score **2 or 3** as relevant
- divide the number of relevant papers in the top 10 by 10

This metric directly captures one of the main failure modes: too many weak or irrelevant papers in the user-facing results.

#### nDCG@10

nDCG@10 measures ranking quality within the top 10.

It uses the graded relevance labels from the paper-level judge and rewards systems that place highly relevant papers earlier in the list.

nDCG@10 is important because two systems can have similar top-10 relevance counts while differing substantially in how well they order their strongest results.

### 6.2 Broad-query metric

#### Exploration Range Score

The Exploration Range Score applies only to **broad exploration queries**.

It is a list-level metric that measures how well the top 10 covers meaningful subtopics or directions within the broad area.

This score is meant to capture exploratory usefulness that Precision@10 and nDCG@10 do not fully capture by themselves.

If a broad-query summary score is needed, use:

#### Broad Exploration Summary Score

`BroadSummary = 0.7 * Precision@10 + 0.3 * ExplorationRangeNorm`

where `ExplorationRangeNorm` is the normalized Exploration Range Score on a 0–1 scale.

If the raw Exploration Range Score is on a 1–5 scale, normalize it as:

`ExplorationRangeNorm = (RangeRaw - 1) / 4`

Even when this summary score is computed, the underlying parts should still be reported separately.

### 6.3 Year-filter metric

#### FilterCompliance@10

Filtered queries are evaluated for whether the returned papers satisfy the explicit year constraint.

FilterCompliance@10 is defined as:

- number of returned papers in the top 10 that satisfy the benchmark `year` filter
- divided by 10

This metric is important because year filtering is an explicit user constraint, not just a soft preference.

### 6.4 Metrics not claimed in version 1

Version 1 does **not** claim:

- true corpus-wide recall
- full completeness over all relevant papers in Semantic Scholar
- citation-graph quality
- full-text retrieval quality
- online user satisfaction

Recall is intentionally not a headline metric in version 1 because the evaluator only judges surfaced outputs, not the full relevant set for each query.

---

## 7. Reporting

The reporting layer should support model comparison without forcing one single universal winner.

### 7.1 Per-query scoring

For each query and each run, the evaluation layer should compute the relevant metrics for that scenario.

At minimum:

- Precision@10
- nDCG@10

And when applicable:

- Exploration Range Score for broad queries
- FilterCompliance@10 for year-filtered queries

### 7.2 Per-bucket reporting

Results should be aggregated separately for:

- broad exploration
- specific technical

And may also be broken out by:

- filtered
- unfiltered

This allows the group to see whether one pipeline is better for exploratory use and another is better for precise technical retrieval.

### 7.3 Reporting rule for broad queries

For broad queries, do **not** rely only on a single composite number.

Report the key components separately:

- Precision@10
- nDCG@10
- Exploration Range Score
- FilterCompliance@10 if the query is year-filtered

If the Broad Exploration Summary Score is used, treat it as a convenience summary rather than the only decision signal.

### 7.4 Best-by-bucket interpretation

The evaluation layer is explicitly allowed to conclude that different pipelines are best for different query types or scenarios.

The project does **not** require one overall winner if the results show meaningful tradeoffs.

---

## 8. Evaluation Workflow

The intended workflow is:

1. Freeze the benchmark query file with Semantic Scholar-style parameter names such as `query`, `year`, and `fields`.
2. Distribute `benchmark_queries.jsonl` and the submission format to the team.
3. Each team runs its local full pipeline against Semantic Scholar bulk search and produces one JSONL run file containing final `paperId` rankings.
4. The evaluator validates each run file for JSONL format, query coverage, rank structure, and `paperId` validity.
5. The evaluator collects unique submitted `paperId` values across all runs.
6. The evaluator resolves missing paper metadata by calling Semantic Scholar paper details endpoints with the shared API key and caching the responses.
7. The LLM performs **paper-level judging** on unique `(queryId, paperId)` pairs and assigns 0–3 relevance labels.
8. For broad queries, the LLM performs **list-level judging** on the submitted top 10 and assigns an Exploration Range Score.
9. The evaluator computes Precision@10, nDCG@10, Exploration Range Score where applicable, and FilterCompliance@10 for year-filtered queries.
10. The reporting layer aggregates results by query type and scenario and produces comparison tables.
11. The group reviews the report and selects the best pipeline overall or best pipeline by bucket.

Briefly, the LLM is used as the **judge of labels**, not as the direct calculator of metrics. The LLM decides relevance and exploration range; the evaluation scripts then turn those labels into metric values.

---

## 9. Limitations

Version 1 has several important limitations.

### Public benchmark overfitting

Because all 100 queries are public, teams can tune directly against the evaluation set.

This is acceptable for a class project, but it means the benchmark is better understood as a shared comparison set than as a strict unseen test set.

### No true recall claim

The evaluator scores returned outputs, not the complete relevant set in Semantic Scholar. As a result, version 1 should not claim strong recall performance.

### LLM judge variance

If an LLM is used as the main judge, some labels may vary with prompt design or model behavior. Judge prompts should therefore be kept stable and versioned.

### API drift and missing fields

Because Semantic Scholar is a live API service, paper metadata can change over time, and some fields may be missing or null for some papers.

### Rate limits and hydration cost

Semantic Scholar recommends using an API key, limiting requested fields, and using bulk or batch endpoints when possible. Version 1 should therefore cache paper metadata aggressively to avoid unnecessary repeated calls.

### Single-turn only

Version 1 evaluates one-shot search, not follow-up query refinement or conversational retrieval behavior.

---

## 10. Appendix Pointers

The README should be supported by companion files rather than trying to contain every implementation detail itself.

Recommended companion files:

- `queries/benchmark_queries.jsonl` — public benchmark query file using Semantic Scholar-style query parameter names
- `schemas/submission_schema.json` — run submission format
- `judging/paper_judge_prompt.md` — paper-level judge instructions
- `judging/range_judge_prompt.md` — broad-query range judge instructions
- `metrics/metric_definitions.md` — exact formulas and edge-case handling
- `reports/` — generated evaluation outputs

These files should be versioned together with the README so the evaluation contract stays stable across model comparisons.
