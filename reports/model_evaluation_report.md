# AgroScope Phase 12 — Model Training & Evaluation Report

## 1. Validation Strategy
- **Method:** Spatial Block Holdout (75% train / 25% test across 5,607 unseen spatial grid cells)
- **Train Observations:** `354,801`
- **Holdout Test Observations:** `118,761`
- **Data Leakage Safeguard:** Spatial Group / Block partitioning guarantees unseen fields in test evaluation; ground truth deficit features excluded from input space.

---

## 2. Model Performance Benchmarks

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) | Low Recall | Mod Recall | High Recall |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Random Forest Baseline** | **99.86%** | 0.9984 | 0.9986 | 1.0 | 99.9% | 99.9% | 99.8% |
| **XGBoost Classifier (Main)** | **99.88%** | 0.9982 | 0.9988 | 1.0 | 99.9% | 99.8% | 99.9% |

---

## 3. Confusion Matrix (XGBoost Classifier)

| Actual \ Predicted | Predicted LOW | Predicted MODERATE | Predicted HIGH |
|:---|:---|:---|:---|
| **Actual LOW** | `10,098` | `7` | `0` |
| **Actual MODERATE** | `57` | `45,042` | `20` |
| **Actual HIGH** | `0` | `60` | `63,477` |

---

## 4. Top Predictive Features (XGBoost Gain)

| Rank | Feature | Relative Importance | Domain Meaning |
|:---|:---|:---|:---|
| 1 | `rainfall_8day` | `0.1750` | Weather / Evaporative Demand |
| 2 | `humidity_pct` | `0.1521` | Weather / Evaporative Demand |
| 3 | `ndvi_mean` | `0.1334` | Vegetation Vigor |
| 4 | `solar_radiation_mj` | `0.1228` | Weather / Evaporative Demand |
| 5 | `et0_8day_sum` | `0.0807` | Weather / Evaporative Demand |
| 6 | `rainfall_16day` | `0.0643` | Weather / Evaporative Demand |
| 7 | `ndwi_mean` | `0.0595` | Satellite Canopy Water |
| 8 | `rainfall_30day` | `0.0498` | Weather / Evaporative Demand |
| 9 | `temp_8day_max` | `0.0470` | Weather / Evaporative Demand |
| 10 | `temp_max_c` | `0.0273` | Weather / Evaporative Demand |

---

## 5. Model Artifacts Ready for API Deployment
- `xgboost_model.pkl` & `xgboost_model.json` — Optimized multi-class classifier ready for FastAPI containerization.
- `feature_metadata.json` — Schema and feature expectations for payload validation.