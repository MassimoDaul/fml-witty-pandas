# Evaluation Summary

## Runs Evaluated
- `andrew`
- `audrey`
- `massimo`
- `nomic`

## Overall Summary
| run_id | query_count | precision_at_10 | ndcg_at_10 | mean_relevance_at_10 | mean_relevance_norm_at_10 | high_relevance_rate_at_10 | weak_or_irrelevant_rate_at_10 | reciprocal_rank_at_10 | exploration_range_raw | exploration_range_norm | broad_summary_score | field_subfield_diversity_at_10 | venue_pairwise_match_rate_at_10 | pooled_relevant_recall_at_10 | pooled_coverage_at_10 | unique_paper_count_at_10 | unique_relevant_count_at_10 | overlap_jaccard_mean_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| andrew | 100 | 0.2630 | 0.7342 | 1.0540 | 0.3513 | 0.1510 | 0.7370 | 0.4564 | 3.0400 | 0.5100 | 0.4554 | 0.3883 | 0.0342 | 0.3944 | 0.3542 | 6.9900 | 1.2400 | 0.1076 |
| audrey | 100 | 0.2050 | 0.6235 | 0.8130 | 0.2710 | 0.0930 | 0.7950 | 0.3523 | 2.8000 | 0.4500 | 0.3926 | 0.3383 | 0.0207 | 0.2132 | 0.3542 | 8.5100 | 1.2100 | 0.0575 |
| massimo | 100 | 0.4060 | 0.8726 | 1.4990 | 0.4997 | 0.2410 | 0.5940 | 0.7485 | 3.5000 | 0.6250 | 0.6271 | 0.3227 | 0.0372 | 0.6528 | 0.3542 | 2.5800 | 0.7000 | 0.2508 |
| nomic | 100 | 0.4010 | 0.8801 | 1.4960 | 0.4987 | 0.2270 | 0.5990 | 0.7443 | 3.5200 | 0.6300 | 0.6216 | 0.3296 | 0.0303 | 0.6473 | 0.3542 | 2.2500 | 0.5600 | 0.2588 |

## Query Type Summary
| run_id | query_type | query_count | precision_at_10 | ndcg_at_10 | mean_relevance_at_10 | mean_relevance_norm_at_10 | high_relevance_rate_at_10 | weak_or_irrelevant_rate_at_10 | reciprocal_rank_at_10 | exploration_range_raw | exploration_range_norm | broad_summary_score | field_subfield_diversity_at_10 | venue_pairwise_match_rate_at_10 | pooled_relevant_recall_at_10 | pooled_coverage_at_10 | unique_paper_count_at_10 | unique_relevant_count_at_10 | overlap_jaccard_mean_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| andrew | broad | 50 | 0.4320 | 0.7819 | 1.4840 | 0.4947 | 0.2580 | 0.5680 | 0.6167 | 3.0400 | 0.5100 | 0.4554 | 0.3892 | 0.0295 | 0.3494 | 0.3599 | 6.8800 | 2.3000 | 0.1097 |
| andrew | specific | 50 | 0.0940 | 0.6865 | 0.6240 | 0.2080 | 0.0440 | 0.9060 | 0.2960 | N/A | N/A | N/A | 0.3874 | 0.0389 | 0.4468 | 0.3484 | 7.1000 | 0.1800 | 0.1054 |
| audrey | broad | 50 | 0.3680 | 0.7642 | 1.3000 | 0.4333 | 0.1680 | 0.6320 | 0.5734 | 2.8000 | 0.4500 | 0.3926 | 0.3208 | 0.0198 | 0.2724 | 0.3599 | 8.1800 | 2.3200 | 0.0695 |
| audrey | specific | 50 | 0.0420 | 0.4828 | 0.3260 | 0.1087 | 0.0180 | 0.9580 | 0.1312 | N/A | N/A | N/A | 0.3558 | 0.0217 | 0.1445 | 0.3484 | 8.8400 | 0.1000 | 0.0456 |
| massimo | broad | 50 | 0.6280 | 0.8886 | 1.9700 | 0.6567 | 0.3900 | 0.3720 | 0.9117 | 3.5000 | 0.6250 | 0.6271 | 0.3365 | 0.0326 | 0.5342 | 0.3599 | 2.6000 | 1.2600 | 0.2502 |
| massimo | specific | 50 | 0.1840 | 0.8567 | 1.0280 | 0.3427 | 0.0920 | 0.8160 | 0.5854 | N/A | N/A | N/A | 0.3089 | 0.0419 | 0.7907 | 0.3484 | 2.5600 | 0.1400 | 0.2515 |
| nomic | broad | 50 | 0.6180 | 0.8978 | 1.9300 | 0.6433 | 0.3640 | 0.3820 | 0.9133 | 3.5200 | 0.6300 | 0.6216 | 0.3491 | 0.0258 | 0.5232 | 0.3599 | 2.1000 | 0.9800 | 0.2625 |
| nomic | specific | 50 | 0.1840 | 0.8623 | 1.0620 | 0.3540 | 0.0900 | 0.8160 | 0.5753 | N/A | N/A | N/A | 0.3101 | 0.0349 | 0.7916 | 0.3484 | 2.4000 | 0.1400 | 0.2551 |

Best-by-bucket conclusions are allowed; different runs may be strongest for broad versus specific queries.

## Corpus Coverage
- Unique papers returned across all runs: 1826
- Corpus coverage rate: 0.0730

## Win Rate
| run_id | strict_wins | strict_win_rate | fractional_wins | fractional_win_rate |
| --- | --- | --- | --- | --- |
| andrew | 9 | 0.0900 | 9.0000 | 0.0900 |
| audrey | 1 | 0.0100 | 1.0000 | 0.0100 |
| massimo | 46 | 0.4600 | 46.0000 | 0.4600 |
| nomic | 44 | 0.4400 | 44.0000 | 0.4400 |
