# AgroScope Responsible AI Statement

## Intended use

AgroScope gives a satellite and weather-based estimate of crop water stress across the Nira Left Bank Canal area, with irrigation advice in English, Marathi and Hindi. In its current form, it's a course project and a demonstration of machine learning operations, using one past season from 2023-24.

## Decision support only

**AgroScope must not be the only basis for a water decision.** A wrong recommendation can cost a crop or waste scarce water. Any decision should also use field checks, local knowledge and advice from agricultural experts.

The results aren't ready for real use, because:

- They're from a past season and aren't live.
- The labels come from a formula, and nothing has been checked against field measurements.
- No agronomist has reviewed the advice text or the stress thresholds.
- No farmer or canal team has used it, so its real effect is unknown.

The dashboard shows the data date, and labels the what-if simulator as simulation only.

## Not intended for

- Allocating, restricting or charging for water.
- Insurance, credit, compensation or any decision about a person or farm.
- Judging an individual field, since each 6.25 ha cell covers several farms.
- Any period, crop or area outside this data without new testing.

## Fairness

**The data has no demographic variables,** such as gender, caste, income or farm size, so fairness between groups of people can't be measured. Leaving them out doesn't make the system fair, because it can still treat different places or farmers unevenly.

Results are compared by area instead. This matters on a canal, where farmers at the tail often get less water than those at the head.

| Reach | Macro F1 on later dates | HIGH recall |
| :--- | :--- | :--- |
| Head | 0.9145 | 0.99 |
| Middle | 0.9239 | 0.9827 |
| Tail | 0.9269 | 0.9765 |

The largest macro F1 gap is 0.0124. **This shows the model agrees with its formula about equally along the canal, not that outcomes are fair.** The tail has slightly lower HIGH recall, so it's slightly more likely to have real stress missed, under the formula's own definition. Canal water supply isn't an input, so the model can't see whether water actually reaches an area.

Other known gaps:

- **Language:** advice is in three languages, but no native speaker has reviewed it for farmers. Some dashboard labels are still English only.
- **Access:** it's a web dashboard, which needs a device, a connection and reading ability.
- **Crops and soils:** everything is treated alike, which may suit some farmers less well.

## Transparency

- **Explanations:** `/predict` shows the top inputs behind each answer, using SHAP. Map results don't show them yet.
- **Limits:** the [model card](MODEL_CARD.md) and [data card](DATA_CARD.md) state the leakage finding and known problems.
- **No invented data:** when results are missing, the API returns an error and the dashboard shows it, rather than showing made-up values.
- **Release record:** MLflow records each model's scores and gate decision.

## Privacy

- **Data used:** satellite imagery at 10-20 m, summarised to 250 m cells, plus public weather data. There's no personal data.
- **Users:** the app has no accounts and stores no personal details. It records request counts and timings, not who made a request.
- **Secrets:** keys are kept in `.env`, outside git.
- **Future use:** a planned database table holds field feedback, but the app doesn't use it yet. Collecting farm locations or farmer feedback would need informed consent and its own privacy review first.

## Human oversight

A person starts the pipeline and retraining. The gate blocks a model below the minimum score or below the champion, and a person can review each decision in MLflow. Retraining through the API needs an admin token. Drift alerts inform a person and don't retrain anything automatically.

## Data credits

- **Satellite:** contains modified Copernicus Sentinel data 2023-2024, accessed through Microsoft Planetary Computer.
- **Weather:** [Open-Meteo](https://open-meteo.com/), under CC BY 4.0.
- **Maps:** © OpenStreetMap contributors, © CARTO, and EOxCloudless 2025 by EOX IT Services GmbH, containing modified Copernicus Sentinel data.

Map and weather services are used under free, non-commercial terms, so ads, fees or running this as a business would need different licences (D-22).
