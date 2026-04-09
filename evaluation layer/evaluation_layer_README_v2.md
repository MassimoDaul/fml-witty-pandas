# Evaluation Layer for Closed-Corpus Research Paper Search

## 1. Purpose

This README defines the evaluation layer for a group project on research paper search. The evaluation layer is designed to compare **full-pipeline search systems** for a **closed corpus** of papers, with the final goal of selecting the most useful search pipeline for literature review use cases.

This evaluation is built around the actual product setting rather than generic information retrieval benchmarks. The main user profile is:

- beginner students trying to understand a field
- graduate students doing literature review

The evaluation focuses on the **top 10 returned papers**, because that is the part of the system users are most likely to inspect.

The benchmark is intended to answer questions like:

- Which pipeline gives cleaner top-10 results?
- Which pipeline ranks strong papers better?
- Which pipeline is better for broad exploratory search?
- Which pipeline handles explicit time filters correctly?
- Which pipeline is strongest for different query types?

This README covers:

- what the evaluated system boundary is
- how the benchmark is organized
- what teams must submit
- how the judging layer works
- which metrics are used
- how results are reported and interpreted

This README does **not** cover training methods, retriever implementation details, or UI design.

---

## 2. Evaluated System Boundary

For this project, the evaluated object is the **final search pipeline output**, not any single internal component.

A system is considered the **evaluated pipeline** if it includes everything from:

- the input query text
- any explicit user-selected time filter
- query preprocessing or rewriting
- retrieval
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

### 3.1 Corpus assumptions

The benchmark assumes a **closed corpus**.

That means:

- all teams evaluate against the same frozen paper collection
- each paper has a canonical paper ID
- the evaluator uses a shared metadata table for judging and scoring
- the comparison target is "best system for this corpus," not "best system for all papers in the world"

At minimum, the evaluation metadata available for each paper should include:

- paper ID
- title
- abstract
- authors
- year or published date
- arXiv category or categories if available

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

### 3.4 Time-filter scenarios

Time filtering is treated as a **scenario modifier**, not as a separate main query bucket.

The benchmark therefore spans four scenarios:

- broad, no time filter
- broad, time-filtered
- specific, no time filter
- specific, time-filtered

This design reflects the actual product more accurately than a separate "recent-work" bucket. If the user explicitly selects a date range, the evaluation should treat that date range as a **constraint** on the search rather than a vague notion of recency.

### 3.5 Public benchmark policy

All 100 benchmark queries are public. There is no hidden final set in version 1.

This makes iteration simpler for a class project, but it also means that systems can overfit to the benchmark. To keep the comparison fair:

- the benchmark should be frozen before final comparison
- each team should submit one clean final run for scoring

---

## 4. Submission Contract

This section defines the exact handoff between the evaluation layer and the team pipelines.

The evaluation layer should be treated as a **benchmark harness**. Teammates will run their systems locally and submit only the final ranked outputs. The evaluator then validates those outputs, joins them with canonical metadata, and sends structured judge packets to the LLM.

### 4.1 What the evaluator provides to the team

The evaluator provides these files before scoring begins:

1. A list of 100 queries that measure different aspect of the machine learning models


The benchmark query file is the **input** every team pipeline is expected to read.

Each query object should include only the fields the search system needs at inference time:

- `query_id` — unique string identifier for the query
- `query_text` — the natural-language search query to run
- `time_filter` — either `null` or an object describing the allowed year range

Expected public queries format:

```json
{"query_id":"q_001","query_text":"machine learning for healthcare","time_filter":null}
{"query_id":"q_002","query_text":"clinical foundation models","time_filter":{"start_year":2023,"end_year":2026}}
```

Notes:

- Teammates are expected to run **all** queries in this file.
- Teammates should treat `time_filter` as a hard search constraint when it is present.
- Query type labels such as `broad` or `specific` are evaluator metadata and are **not required** as model inputs.

The paper metadata file is provided so teams can resolve the paper IDs used by the evaluator. At minimum it should contain:

- `paper_id`
- `title`
- `abstract`
- `authors`
- `year`
- `categories` if available

### 4.2 What the teammates must submit

Each teammates submits **one JSONL file per system run**.

Recommended filename:

`submissions/<run_id>.jsonl`

Each line in the submission file corresponds to **one query** and must contain:

- `run_id` — unique identifier for the submitted run; should be the same on every line in the file
- `query_id` — must exactly match a query in `benchmark_queries.jsonl`
- `results` — ordered list of exactly 10 returned papers

Each item inside `results` must contain:

- `rank` — integer rank from 1 to 10
- `paper_id` — canonical paper ID from the frozen corpus
- `score` — optional numeric score from the team system; used only for debugging and later analysis, not shown to the LLM judge

Required submission shape:

```json
{"run_id":"team_a_model_1","query_id":"q_001","results":[{"rank":1,"paper_id":"p_1023","score":12.43},{"rank":2,"paper_id":"p_0883","score":11.92},{"rank":3,"paper_id":"p_4451","score":11.10},{"rank":4,"paper_id":"p_3011","score":10.88},{"rank":5,"paper_id":"p_2190","score":10.44},{"rank":6,"paper_id":"p_1447","score":10.11},{"rank":7,"paper_id":"p_7810","score":9.87},{"rank":8,"paper_id":"p_6502","score":9.43},{"rank":9,"paper_id":"p_5122","score":9.01},{"rank":10,"paper_id":"p_9033","score":8.72}]}
{"run_id":"team_a_model_1","query_id":"q_002","results":[{"rank":1,"paper_id":"p_1111","score":14.12},{"rank":2,"paper_id":"p_2433","score":13.44},{"rank":3,"paper_id":"p_8888","score":12.77},{"rank":4,"paper_id":"p_4545","score":12.10},{"rank":5,"paper_id":"p_6767","score":11.89},{"rank":6,"paper_id":"p_7878","score":11.12},{"rank":7,"paper_id":"p_9898","score":10.67},{"rank":8,"paper_id":"p_2121","score":10.22},{"rank":9,"paper_id":"p_3434","score":9.98},{"rank":10,"paper_id":"p_5656","score":9.55}]}
```

### 4.3 What is expected from the teammate pipeline

For every query in `benchmark_queries.jsonl`, the team pipeline is expected to:

1. read `query_id`, `query_text`, and `time_filter`
2. run the full local search pipeline
3. return exactly 10 ranked papers from the frozen corpus
4. respect the time filter if it is present
5. output only canonical `paper_id` values from the shared corpus mapping

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
- years
- embedding vectors
- retrieved candidate pools larger than top 10
- internal prompts or reasoning traces
- screenshots or UI output

The evaluator will join the submitted `paper_id`s against the canonical metadata table before sending anything to the LLM judge.

### 4.5 Validity rules

A submission is valid only if:

- the file is valid JSONL
- every benchmark query appears exactly once in the run
- each query returns exactly 10 ranked results
- ranks are integers from 1 through 10 with no gaps or duplicates
- all returned paper IDs exist in the frozen corpus
- no paper ID is repeated within the same top-10 list
- the `run_id` is consistent across the file

For filtered queries, filter compliance is scored as part of evaluation rather than treated as a file-format validation rule. However, teams are still expected to honor the filter when producing results.

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

The judge may see:

- query text
- time-filter condition if present
- paper title
- abstract
- authors
- year
- arXiv categories if available

### 5.4 What the judge must not see

To reduce bias, the judge must not see:

- team name
- algorithm name
- embedding type
- internal retrieval score
- reranker score
- any statement of which system produced the results

### 5.5 Judge implementation note

Version 1 may use an LLM as the primary judge. If desired, a secondary embedding- or clustering-based diversity proxy can be added later as a diagnostic tool. However, the main Exploration Range Score should remain query-conditioned and relevance-aware rather than relying only on pairwise paper distance.

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

If a broad-query composite summary is needed, use:

#### Broad Exploration Summary Score

`BroadSummary = 0.7 * Precision@10 + 0.3 * ExplorationRangeNorm`

where `ExplorationRangeNorm` is the normalized Exploration Range Score on a 0–1 scale.

If the raw Exploration Range Score is on a 1–5 scale, normalize it as:

`ExplorationRangeNorm = (RangeRaw - 1) / 4`

Even when this summary score is computed, the underlying parts should still be reported separately.

### 6.3 Time-filter metric

#### Filter Compliance@10

Filtered queries are evaluated for whether the returned papers satisfy the explicit date constraint.

Filter Compliance@10 is defined as:

- number of returned papers in the top 10 that satisfy the filter
- divided by 10

This metric is important because time filtering is an explicit user constraint, not just a soft preference.

### 6.4 Metrics not claimed in version 1

Version 1 does **not** claim:

- true corpus-wide recall
- full completeness over all relevant papers in the corpus
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
- Filter Compliance@10 for filtered queries

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
- Filter Compliance@10 if the query is filtered

If the Broad Exploration Summary Score is used, treat it as a convenience summary rather than the only decision signal.

### 7.4 Best-by-bucket interpretation

The evaluation layer is explicitly allowed to conclude that different pipelines are best for different query types or scenarios.

The project does **not** require one overall winner if the results show meaningful tradeoffs.

---

## 8. Evaluation Workflow

The intended workflow is:

1. Freeze the corpus snapshot, canonical paper IDs, and benchmark query file.
2. Distribute `benchmark_queries.jsonl`, corpus metadata, and the submission format to the team.
3. Each team runs its local full pipeline on every benchmark query and produces one JSONL run file.
4. The evaluator validates each run file for JSONL format, query coverage, rank structure, and paper ID validity.
5. The evaluator joins each submitted `paper_id` with the canonical metadata needed for judging.
6. The LLM performs **paper-level judging** on unique `(query_id, paper_id)` pairs and assigns 0–3 relevance labels.
7. For broad queries, the LLM performs **list-level judging** on the submitted top 10 and assigns an Exploration Range Score.
8. The evaluator computes Precision@10, nDCG@10, Exploration Range Score where applicable, and Filter Compliance@10 for filtered queries.
9. The reporting layer aggregates results by query type and scenario and produces comparison tables.
10. The group reviews the report and selects the best pipeline overall or best pipeline by bucket.

Briefly, the LLM is used as the **judge of labels**, not as the direct calculator of metrics. The LLM decides relevance and exploration range; the evaluation scripts then turn those labels into metric values.

---

## 9. Limitations

Version 1 has several important limitations.

### Public benchmark overfitting

Because all 100 queries are public, teams can tune directly against the evaluation set.

This is acceptable for a class project, but it means the benchmark is better understood as a shared comparison set than as a strict unseen test set.

### No true recall claim

The evaluator scores returned outputs, not the complete relevant set in the corpus. As a result, version 1 should not claim strong recall performance.

### LLM judge variance

If an LLM is used as the main judge, some labels may vary with prompt design or model behavior. Judge prompts should therefore be kept stable and versioned.

### Range score subjectivity

The Exploration Range Score is more rubric-dependent than classical metrics such as Precision@10. This is acceptable because broad exploratory usefulness is one of the main product goals, but it should be acknowledged explicitly.

### Single-turn only

Version 1 evaluates one-shot search, not follow-up query refinement or conversational retrieval behavior.

---

## 10. Appendix Pointers

The README should be supported by companion files rather than trying to contain every implementation detail itself.

Recommended companion files:

- `queries/benchmark_queries.jsonl` — public benchmark query file
- `schemas/submission_schema.json` — run submission format
- `judging/paper_judge_prompt.md` — paper-level judge instructions
- `judging/range_judge_prompt.md` — broad-query range judge instructions
- `metrics/metric_definitions.md` — exact formulas and edge-case handling
- `reports/` — generated evaluation outputs

These files should be versioned together with the README so the evaluation contract stays stable across model comparisons.
