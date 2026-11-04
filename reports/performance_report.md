# AgroScope Performance Report

Built by `scripts/build_performance_report.py` from saved results. No number here is typed by hand.

## Summary

| Metric | Value | Source |
| :--- | :--- | :--- |
| Model: macro F1 on later dates | 0.9242 | `reports/evaluation_result.json` |
| Business: hectares flagged Priority 1 this cycle | 117,243.8 ha (97.2%) | `data/processed/cycle_statistics.json` |
| p95 latency, all requests | 150.5 ms | `reports/load_test_results.json` |
| Throughput | 170.79 requests per second | `reports/load_test_results.json` |
| Error rate | 0.00% | `reports/load_test_results.json` |
| Drift: share of inputs drifted on the published date | 100% | `data/processed/drift_summary.json` |

## Model performance

The promotion gate uses `temporal_macro_f1`, with a minimum of 0.85. Latest decision: promoted, lightgbm version 1. No champion exists yet.

| Model | Unseen cells | Later dates | Later dates without label inputs |
| :--- | :--- | :--- | :--- |
| random_forest | 0.9877 | 0.8845 | 0.774 |
| xgboost | 0.9967 | 0.915 | 0.7323 |
| lightgbm | 0.9966 | 0.9242 | 0.6707 |

**These scores don't show real-world accuracy.** Labels are calculated by a formula from three of the model's own inputs, so they show how well a model reproduces that formula. The last column removes those inputs, and is the more honest guide.

### As labels become available

The lightgbm model was trained only on 308,726 rows before 2024-02-01, then scored on each later date, as if each date's labels had just arrived.

| Date | Rows | Macro F1 | Accuracy | HIGH recall | Classes present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2024-02-03 | 21,448 | 0.9919 | 0.9935 | 0.9922 | 3 |
| 2024-02-08 | 20,868 | 0.9922 | 0.9958 | 0.9956 | 3 |
| 2024-02-13 | 21,251 | 0.9862 | 0.9936 | 0.9942 | 3 |
| 2024-02-18 | 21,103 | 0.9848 | 0.994 | 0.995 | 3 |
| 2024-02-23 | 21,028 | 0.8859 | 0.9781 | 0.9917 | 3 |
| 2024-02-28 | 18,870 | 0.9458 | 0.9897 | 0.9961 | 3 |
| 2024-03-04 | 20,796 | 0.8063 | 0.9795 | 0.9891 | 3 |
| 2024-03-09 | 20,304 | 0.8597 | 0.9643 | 0.9728 | 2 |
| 2024-03-14 | 20,047 | 0.8179 | 0.9553 | 0.9616 | 2 |
| 2024-03-19 | 19,574 | 0.7669 | 0.9606 | 0.9659 | 2 |
| 2024-03-24 | 19,300 | 0.7633 | 0.962 | 0.9661 | 2 |
| 2024-03-29 | 6,537 | 0.7849 | 0.9608 | 0.9713 | 2 |

Macro F1 covers only the classes present on that date.

### By canal reach

The data has no demographic variables, so results are compared along the canal, in three equal-width bands from the headworks in the west.

| Reach | Longitude | Rows | Macro F1 | Accuracy | HIGH recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| head | 74.054 to 74.374 | 51,173 | 0.9145 | 0.9864 | 0.99 |
| middle | 74.375 to 74.693 | 95,596 | 0.9239 | 0.9788 | 0.9827 |
| tail | 74.695 to 75.014 | 84,357 | 0.9269 | 0.9736 | 0.9765 |

The largest macro F1 gap between reaches is 0.0124.

## Business metric

**Hectares flagged for priority irrigation per cycle.** For the cycle of 2024-03-24:

| Priority | Hectares | Share |
| :--- | :--- | :--- |
| 1, urgent | 117,243.8 | 97.2% |
| 2, scheduled | 3,381.2 | 2.8% |
| 3, normal | 0.0 | 0.0% |

**This isn't validated against field outcomes.** No farmer or canal team has acted on these flags, so it counts what the system recommends, not water saved or crops protected. With most of the area in Priority 1, it can't yet help choose where water goes first. The total area also looks too large, because it exceeds the 66,250 ha culturable area in the study area file.

## Serving performance

10 simulated users for 60.1 seconds against `http://localhost:8000`, model `lightgbm-ac395666`, measured 2026-11-04T12:01:24.947417+00:00. Run on one local computer, with the client and API sharing its resources.

| Request | Requests | Errors | Throughput (per second) | p50 ms | p95 ms | p99 ms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| All | 10257 | 0 | 170.79 | 39.7 | 150.5 | 187.4 |
| statistics | 3129 | 0 | 52.1 | 26.2 | 59.4 | 80.9 |
| cell_details | 3033 | 0 | 50.5 | 27.0 | 59.6 | 79.8 |
| predict | 3023 | 0 | 50.34 | 118.1 | 176.7 | 211.8 |
| map_sample | 1072 | 0 | 17.85 | 47.1 | 82.8 | 106.5 |

`/predict` includes a SHAP explanation for each request. Its p95 was 176.7 ms.

## Drift

**Published date:** 100% of inputs drifted on 2024-03-24, against the period before 2024-02-01. The reference data covers one past season only, so drift here can reflect normal seasonal change as well as a real problem.

**Simulated drift:** to show the monitor responds to a known change, two samples of 20,000 rows from the training period were compared. This is simulated data, not a real result.

| Comparison | Inputs drifted |
| :--- | :--- |
| Unchanged control | 0% |
| Temperature +4.0 °C, rainfall +20.0 mm, NDWI -0.1 | 36% |

It flagged 9 of the 9 shifted inputs and 0 unshifted ones.

The overall alert, at 30% of inputs, fired on this change.

## Limits of these measurements

- Everything ran on one computer, with the load test sharing its resources with the API, so these aren't production figures.
- The data covers one season, from October 2023 to March 2024.
