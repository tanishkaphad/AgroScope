# AgroScope Phase 12 — Model Training & Evaluation Report

## 1. Validation Strategy
- **Method:** Spatial Block Holdout (75% train / 25% test across 5,626 unseen spatial grid cells)
- **Train Observations:** `405,157`
- **Holdout Test Observations:** `134,695`
- **Known Limitation:** Labels are calculated from `ndwi_mean`, `rainfall_8day` and `et0_8day_sum`, which are also model inputs. These scores show how well the model reproduces that formula, not real-world accuracy.

---

## 2. Model Performance Benchmarks

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) | Low Recall | Mod Recall | High Recall |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Random Forest Baseline** | **98.82%** | 0.9877 | 0.9882 | 0.9998 | 99.0% | 99.2% | 98.5% |
| **XGBoost Classifier (Main)** | **99.70%** | 0.9967 | 0.997 | 1.0 | 99.8% | 99.6% | 99.7% |

---

## 3. Confusion Matrix (XGBoost Classifier)

| Actual \ Predicted | Predicted LOW | Predicted MODERATE | Predicted HIGH |
|:---|:---|:---|:---|
| **Actual LOW** | `24,714` | `48` | `0` |
| **Actual MODERATE** | `90` | `38,539` | `49` |
| **Actual HIGH** | `0` | `217` | `71,038` |

---

## 4. Top Predictive Features (XGBoost Gain)

| Rank | Feature | Relative Importance | Domain Meaning |
|:---|:---|:---|:---|
| 1 | `ndvi_mean` | `0.3052` | Vegetation Vigor |
| 2 | `ndwi_mean` | `0.1410` | Satellite Canopy Water |
| 3 | `solar_radiation_mj` | `0.1056` | Weather / Evaporative Demand |
| 4 | `rainfall_8day` | `0.0811` | Weather / Evaporative Demand |
| 5 | `et0_8day_sum` | `0.0693` | Weather / Evaporative Demand |
| 6 | `evi_mean` | `0.0559` | Vegetation Vigor |
| 7 | `rainfall_16day` | `0.0452` | Weather / Evaporative Demand |
| 8 | `temp_max_c` | `0.0365` | Weather / Evaporative Demand |
| 9 | `b12_mean` | `0.0277` | Weather / Evaporative Demand |
| 10 | `et0_16day_sum` | `0.0232` | Weather / Evaporative Demand |

---

## 5. Model Artifacts Ready for API Deployment
- `xgboost_model.pkl` & `xgboost_model.json` — Optimized multi-class classifier ready for FastAPI containerization.
- `feature_metadata.json` — Schema and feature expectations for payload validation.