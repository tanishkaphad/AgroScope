# AgroScope Phase 12 — Model Training & Evaluation Report

## 1. Validation Strategy
- **Method:** Spatial Block Holdout (75% train / 25% test across 5,607 unseen spatial grid cells)
- **Train Observations:** `354,801`
- **Holdout Test Observations:** `118,761`
- **Known Limitation:** Labels are calculated from `ndwi_mean`, `rainfall_8day` and `et0_8day_sum`, which are also model inputs. These scores show how well the model reproduces that formula, not real-world accuracy.

---

## 2. Model Performance Benchmarks

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) | Low Recall | Mod Recall | High Recall |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Random Forest Baseline** | **99.87%** | 0.9984 | 0.9987 | 1.0 | 99.9% | 99.9% | 99.9% |
| **XGBoost Classifier (Main)** | **99.88%** | 0.9983 | 0.9988 | 1.0 | 99.9% | 99.8% | 99.9% |

---

## 3. Confusion Matrix (XGBoost Classifier)

| Actual \ Predicted | Predicted LOW | Predicted MODERATE | Predicted HIGH |
|:---|:---|:---|:---|
| **Actual LOW** | `10,097` | `8` | `0` |
| **Actual MODERATE** | `52` | `45,049` | `18` |
| **Actual HIGH** | `0` | `60` | `63,477` |

---

## 4. Top Predictive Features (XGBoost Gain)

| Rank | Feature | Relative Importance | Domain Meaning |
|:---|:---|:---|:---|
| 1 | `rainfall_8day` | `0.1775` | Weather / Evaporative Demand |
| 2 | `humidity_pct` | `0.1502` | Weather / Evaporative Demand |
| 3 | `ndvi_mean` | `0.1330` | Vegetation Vigor |
| 4 | `solar_radiation_mj` | `0.1173` | Weather / Evaporative Demand |
| 5 | `et0_8day_sum` | `0.0799` | Weather / Evaporative Demand |
| 6 | `rainfall_16day` | `0.0638` | Weather / Evaporative Demand |
| 7 | `ndwi_mean` | `0.0601` | Satellite Canopy Water |
| 8 | `temp_8day_max` | `0.0519` | Weather / Evaporative Demand |
| 9 | `rainfall_30day` | `0.0506` | Weather / Evaporative Demand |
| 10 | `temp_max_c` | `0.0283` | Weather / Evaporative Demand |

---

## 5. Model Artifacts Ready for API Deployment
- `xgboost_model.pkl` & `xgboost_model.json` — Optimized multi-class classifier ready for FastAPI containerization.
- `feature_metadata.json` — Schema and feature expectations for payload validation.