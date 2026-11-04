# AgroScope Model Evaluation Report

## Validation
- **Unseen cells:** Spatial Block Holdout (75% train / 25% test across 5,626 unseen spatial grid cells)
- **Later dates:** Temporal Holdout Split (< 2024-02-01 vs >= 2024-02-01)
- **Unseen-cell rows:** `405,157` train, `134,695` test
- **Known limitation:** labels are calculated from `ndwi_mean`, `rainfall_8day` and `et0_8day_sum`, which are also model inputs. These scores show how well a model reproduces that formula, not real-world accuracy. The last score column removes those inputs.

## Model comparison (macro F1)

| Model | Role | Unseen cells | Later dates | Later dates without label inputs |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | baseline | 0.9877 | 0.8845 | 0.774 |
| XGBoost | candidate | 0.9967 | 0.915 | 0.7323 |
| LightGBM **(selected)** | candidate | 0.9966 | 0.9242 | 0.6707 |

The candidate is chosen on the later-dates score: LightGBM.

## Confusion matrix on unseen cells (LightGBM)

| Actual \ Predicted | LOW | MODERATE | HIGH |
| :--- | :--- | :--- | :--- |
| **LOW** | `24,703` | `59` | `0` |
| **MODERATE** | `98` | `38,543` | `37` |
| **HIGH** | `0` | `208` | `71,047` |

## Top features (LightGBM)

| Rank | Feature | Share of importance |
| :--- | :--- | :--- |
| 1 | `ndwi_mean` | `0.2839` |
| 2 | `et0_8day_sum` | `0.1843` |
| 3 | `rainfall_8day` | `0.1042` |
| 4 | `et0_16day_sum` | `0.0364` |
| 5 | `evi_mean` | `0.0355` |
| 6 | `b12_mean` | `0.0337` |
| 7 | `solar_radiation_mj` | `0.0319` |
| 8 | `b11_mean` | `0.0273` |
| 9 | `rainfall_16day` | `0.0236` |
| 10 | `humidity_pct` | `0.0227` |
