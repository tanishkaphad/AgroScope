# AgroScope Viva Notes

Answers for each component, and the reasons behind each choice. Every number comes from a saved result as of 2026-11-04. Learn the reasoning, not the wording.

## Thirty-second pitch

AgroScope estimates crop water stress for 250 m cells across the Nira Left Bank Canal from Sentinel-2 and weather data, then gives irrigation advice in three languages. The MLOps work is the main point: DVC versions the data, Airflow runs the pipeline, MLflow tracks three models, a gate controls release, FastAPI serves results, and Prometheus, Grafana and Evidently monitor it. The scores have a known weakness, and the project measures and states it.

## The system in order

| Step | What happens | Tool |
| :--- | :--- | :--- |
| Ingestion | 27 Sentinel-2 dates and weather for 24 points | Planetary Computer, Open-Meteo |
| Raw validation | Checks columns, ranges, gaps and date coverage | Custom script |
| Preprocessing | 22,580 cells of 250 m, with cloud-masked band and index values | rasterio |
| Features and labels | Joins each cell to its nearest weather point and calculates stress | pandas |
| Feature validation | Checks the 566,319-row table before training | Custom script |
| Training | Three models, each with three tests, each its own run | MLflow |
| Evaluation | Records whether the best score passes 0.85 | Custom script |
| Registration | Applies the gate, and exports the champion | MLflow registry |
| Prediction | Maps and statistics for all 27 dates | LightGBM, SHAP |
| Drift | Published date against the training period | Evidently |

## Why each tool

| Choice | Why | Alternative rejected |
| :--- | :--- | :--- |
| DVC | Versions data with git commits, and skips unchanged stages | Git LFS: no pipeline, and 10 GB exceeds its free limits |
| Google Drive remote | Free, and enough space for 11 GB | S3: costs money, and the project stays on free tiers |
| Airflow as one container | Meets the requirement with the least setup | Separate scheduler and database containers: more memory on an 8 GB limit |
| Airflow calling DVC | One definition of each step, and caching stays | Copying the steps into Airflow: two versions to keep in sync |
| Separate Python environment in the Airflow image | Airflow pins packages that conflict with MLflow and Evidently | Sharing one environment |
| MLflow server | Local runs and Airflow share one store | A local file: Airflow couldn't reach it |
| Docker volume for MLflow | SQLite can have locking problems on folders shared from Windows | A Windows folder |
| FastAPI | Typed requests, automatic docs and fast enough | Flask: fewer built-in checks |
| Prometheus and Grafana | Standard and free, and the dashboard loads automatically | Hosted tools: paid tiers |
| Evidently | Ready-made per-input drift tests and a report | Writing the statistics by hand |

## Models and scores

Macro F1, because the classes are imbalanced and accuracy would flatter a model that favours HIGH.

| Model | Role | Unseen cells | Later dates | Later dates without label inputs |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest | Baseline | 0.9877 | 0.8845 | 0.774 |
| XGBoost | Candidate | 0.9967 | 0.915 | 0.7323 |
| LightGBM | Champion | 0.9966 | 0.9242 | 0.6707 |

- **Unseen cells:** 25% of cells held out across the same dates. This tests geography, not the future.
- **Later dates:** trained before 2024-02-01 and tested from that date on. This is closest to real use, so the gate uses it.
- **Without label inputs:** removes `ndwi_mean`, `rainfall_8day` and `et0_8day_sum`. This shows how much of the score came from relearning the formula.

## Hard questions

**Your F1 is 0.92. Is the model good?**
It reproduces its label formula well, and that isn't the same as being accurate. Labels are calculated from three of its inputs. Without them, the score is 0.6707, and nothing has been checked in the field.

**If the labels come from a formula, why use a model at all?**
For this version, the formula alone would give the same answer, so the model adds little. Its value here is that it lets the MLOps system work end to end. The next step is to predict the next clear satellite reading, which removes the circularity.

**Why LightGBM, if it's weakest without the label inputs?**
The rule set before training says to choose on the later-dates score, so I followed it and didn't change it after seeing results. It's a real weakness, because LightGBM relies most on the formula's inputs. Random Forest holds up best at 0.774.

**Why is the minimum 0.85?**
It's below all three current scores, so it catches a real fall without blocking normal retraining. There's no field data to set it from, so it's a judgement call.

**Scores drop late in the season. Does the gate catch that?**
No. Macro F1 falls from 0.9919 on 2024-02-03 to 0.7633 on the published date, while the gate sees the 0.9242 overall. A per-date minimum would catch it, and that's a known gap (B-40).

**Drift is 100%. Isn't that a failure?**
It's real, and mostly seasonal: March compared with October to January. With one season, seasonal change and a real problem look the same. More years of data would separate them. Raising the alert level would only hide it.

**How do you know the drift monitor works, then?**
I compared two random samples from the training period. Unchanged, 0% drifted. After shifting temperature, rainfall and NDWI, 36% drifted, and it flagged exactly the 9 shifted inputs and no others. The first version of the 50% alert missed this, so I lowered it to 30%.

**Is it fair?**
I can't measure fairness between people, because there are no demographic variables. I compared head, middle and tail reaches instead, and the macro F1 gap is 0.0124. That shows agreement with the formula is even, not that outcomes are fair. Canal supply isn't an input, so the model can't see whether water reaches the tail.

**Where did the canal boundary come from?**
I marked it by hand from the canal's known dimensions, so it's approximate. The file's "Validated Official Geometry" label overstates that, and I've logged it to fix. An official command-area map would be the next improvement. It may also explain why the reported 120,625 ha is more than the 66,250 ha culturable area.

**Why is 97.2% of the area HIGH?**
Late March has high demand and almost no rain, so the formula gives mostly HIGH. It means that date can't rank areas for water. Earlier dates vary far more: 2023-12-10 is 9.2% HIGH. The thresholds also need expert review.

**What's the business metric, and is it meaningful?**
Hectares flagged Priority 1 per cycle: 117,243.8 ha on the published date. It counts recommendations, not water saved or crops protected, because no one has acted on them yet.

**Do the past-date maps show the model works?**
No. The model trained on 75% of cells across those dates, so most of each map repeats its training data.

**Why isn't AWS built?**
I decided to put the time into the other required parts, and the project stays on free tiers. It's a known gap in the submission. The README maps each part to its AWS service: ECR and ECS for the API, S3 for DVC and MLflow files, and RDS for MLflow's database.

**What would you do with another month?**
Replace the labels with next-pass satellite change, so the score becomes meaningful. Add a per-date gate, field checks with farmers and an agronomist, and multiple seasons of data.

## Problems found and fixed

These make good answers to "what went wrong?"

| Problem | Effect | Fix |
| :--- | :--- | :--- |
| Sentinel-2's +1000 offset was never removed | Mean NDVI read 0.249 instead of 0.405 | Removed, based on each scene's processing version |
| One weather point for every cell | No weather variation across the canal | 24 points, and each cell uses its nearest |
| Wind was very likely in km/h, labelled m/s | Wrong input values | Requests now ask for m/s |
| Fake data shown when the API failed | Believable but false results | Removed, and errors now show |
| Training took 29 minutes in Docker, against 2.5 on Windows | A slow pipeline | Fixed at 8 threads, which took it to 3 min 23 sec. Using all 22 cores was the likely cause |
| `/predict` blocked other requests | Lower throughput | Runs in a worker thread |

## Numbers to remember

| Measure | Value |
| :--- | :--- |
| Cells and size | 22,580 cells, 250 m, 6.25 ha each |
| Dates | 27, from 2023-10-06 to 2024-03-29 |
| Training table | 566,319 rows, 25 inputs |
| Champion | LightGBM, later-dates macro F1 0.9242 |
| Gate | Minimum 0.85, and at least match the champion |
| p95 latency | 150.5 ms |
| Throughput | 170.79 requests per second |
| Error rate | 0% of 10,257 requests |
| Published-date drift | 100% |
| Simulated drift | 0% control, 36% shifted |
| Alert level | 30% of inputs |
| Tests | 76 |
| Pipeline stages | 10 |

## Things not to say

- **"99.7% accurate":** that score mostly reflects the formula.
- **"Validated boundary":** you marked it by hand.
- **"It's ready for farmers":** it has no field validation.
- **"It updates daily":** it uses one past season, started by hand.
- **"Past-date maps prove it works":** they include training data.
- **"The drift is low" or "I reduced drift":** the alert became more sensitive, and drift didn't change.
