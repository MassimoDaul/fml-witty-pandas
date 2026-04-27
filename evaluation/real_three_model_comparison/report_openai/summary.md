# Evaluation Summary

## Runs Evaluated
- `andrew`
- `audrey`
- `massimo`

## Overall Summary
| run_id | query_count | precision_at_10 | ndcg_at_10 | mean_relevance_at_10 | mean_relevance_norm_at_10 | high_relevance_rate_at_10 | weak_or_irrelevant_rate_at_10 | reciprocal_rank_at_10 | exploration_range_raw | exploration_range_norm | broad_summary_score | field_subfield_diversity_at_10 | venue_pairwise_match_rate_at_10 | pooled_relevant_recall_at_10 | pooled_coverage_at_10 | unique_paper_count_at_10 | unique_relevant_count_at_10 | overlap_jaccard_mean_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| andrew | 100 | 0.2630 | 0.7342 | 1.0540 | 0.3513 | 0.1510 | 0.7370 | 0.4564 | 3.0400 | 0.5100 | 0.4554 | 0.3883 | 0.0342 | 0.4128 | 0.3820 | 7.5700 | 1.4600 | 0.0836 |
| audrey | 100 | 0.2050 | 0.6235 | 0.8130 | 0.2710 | 0.0930 | 0.7950 | 0.3523 | 2.8000 | 0.4500 | 0.3926 | 0.3383 | 0.0207 | 0.2243 | 0.3820 | 8.7000 | 1.2700 | 0.0493 |
| massimo | 100 | 0.4060 | 0.8726 | 1.4990 | 0.4997 | 0.2410 | 0.5940 | 0.7485 | 3.5000 | 0.6250 | 0.6271 | 0.3227 | 0.0372 | 0.6956 | 0.3820 | 6.9700 | 2.4700 | 0.1027 |

## Query Type Summary
| run_id | query_type | query_count | precision_at_10 | ndcg_at_10 | mean_relevance_at_10 | mean_relevance_norm_at_10 | high_relevance_rate_at_10 | weak_or_irrelevant_rate_at_10 | reciprocal_rank_at_10 | exploration_range_raw | exploration_range_norm | broad_summary_score | field_subfield_diversity_at_10 | venue_pairwise_match_rate_at_10 | pooled_relevant_recall_at_10 | pooled_coverage_at_10 | unique_paper_count_at_10 | unique_relevant_count_at_10 | overlap_jaccard_mean_at_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| andrew | broad | 50 | 0.4320 | 0.7819 | 1.4840 | 0.4947 | 0.2580 | 0.5680 | 0.6167 | 3.0400 | 0.5100 | 0.4554 | 0.3892 | 0.0295 | 0.3741 | 0.3855 | 7.6000 | 2.6400 | 0.0837 |
| andrew | specific | 50 | 0.0940 | 0.6865 | 0.6240 | 0.2080 | 0.0440 | 0.9060 | 0.2960 | N/A | N/A | N/A | 0.3874 | 0.0389 | 0.4577 | 0.3785 | 7.5400 | 0.2800 | 0.0835 |
| audrey | broad | 50 | 0.3680 | 0.7642 | 1.3000 | 0.4333 | 0.1680 | 0.6320 | 0.5734 | 2.8000 | 0.4500 | 0.3926 | 0.3208 | 0.0198 | 0.2922 | 0.3855 | 8.4800 | 2.4400 | 0.0577 |
| audrey | specific | 50 | 0.0420 | 0.4828 | 0.3260 | 0.1087 | 0.0180 | 0.9580 | 0.1312 | N/A | N/A | N/A | 0.3558 | 0.0217 | 0.1452 | 0.3785 | 8.9200 | 0.1000 | 0.0410 |
| massimo | broad | 50 | 0.6280 | 0.8886 | 1.9700 | 0.6567 | 0.3900 | 0.3720 | 0.9117 | 3.5000 | 0.6250 | 0.6271 | 0.3365 | 0.0326 | 0.5843 | 0.3855 | 6.8400 | 3.9400 | 0.1090 |
| massimo | specific | 50 | 0.1840 | 0.8567 | 1.0280 | 0.3427 | 0.0920 | 0.8160 | 0.5854 | N/A | N/A | N/A | 0.3089 | 0.0419 | 0.8250 | 0.3785 | 7.1000 | 1.0000 | 0.0965 |

Best-by-bucket conclusions are allowed; different runs may be strongest for broad versus specific queries.

## Corpus Coverage
- Unique papers returned across all runs: 1666
- Corpus coverage rate: 0.0666

## Win Rate
| run_id | strict_wins | strict_win_rate | fractional_wins | fractional_win_rate |
| --- | --- | --- | --- | --- |
| andrew | 16 | 0.1600 | 16.0000 | 0.1600 |
| audrey | 3 | 0.0300 | 3.0000 | 0.0300 |
| massimo | 81 | 0.8100 | 81.0000 | 0.8100 |
