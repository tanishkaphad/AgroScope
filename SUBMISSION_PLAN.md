# AgroScope Submission Plan

How we'll meet the MLOps course rubric in the 2 days left from 2026-11-04. The long-term product roadmap is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), and pauses until this is done.

**Status:** all sections done. Your remaining steps are a GitHub Actions screenshot, `dvc status -c`, the final commit, and practising the demo and viva.

**Rules:** Claude runs no git commands, and every service stays free.

**Decisions:** AgroScope is confirmed as the project, as you reported. AWS isn't being built for now, so state that clearly in the README and viva, because the rubric lists it as mandatory.

## Order of work

| # | Work | Rubric section | Size | When |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Docker Compose, MLflow server, Prometheus and Grafana | 4, 6 | Medium | Day 1 |
| 2 | One baseline, two candidates and a minimum-score gate | 2, 5 | Small | Day 1 |
| 3 | Data validation and Airflow DAG | 3 | Large | Day 1 |
| 4 | CI Docker build and prediction tests | 5 | Small | Day 2 |
| 5 | Model card, data card and responsible-AI statement | 7 | Small | Day 2 |
| 6 | Load test, simulated drift and performance report | 6, submission | Medium | Day 2 |
| 7 | Frontend improvements | Demo and usability | Medium | Day 2 |
| 8 | README, screenshots and final checks | Submission | Small | Day 2 |

---

## 1. Docker Compose and monitoring

**Goal:** one command starts the whole system, with live monitoring.

- [x] Add `docker-compose.yml` with the API, MLflow, Prometheus and Grafana.
- [x] Add a Prometheus metrics endpoint to the API. The existing `/metrics` stays as it is.
- [x] Record request count, error rate, inference latency, prediction distribution and drift share.
- [x] Add a ready-made Grafana dashboard that loads automatically.
- [x] Run MLflow as a server, so local runs and Airflow use the same store. Start the registry again on this server if moving old runs proves unreliable.

**Files:** `docker-compose.yml`, `monitoring/prometheus.yml`, `monitoring/grafana/`, `api/app/main.py`.

**Done when:** `docker compose up` starts all services, and test requests appear on the Grafana dashboard.

## 2. Models and promotion

**Goal:** one baseline and two candidates, compared fairly in MLflow.

- [x] Keep Random Forest as the baseline and XGBoost as candidate 1.
- [x] Add LightGBM as candidate 2, and confirm SHAP works with it.
- [x] Log every model as its own MLflow run, with parameters, metrics and files.
- [x] Choose the best model using the later-dates score.
- [x] Add a minimum score to the gate as well as the champion comparison.
- [x] Add version metadata to the registered model.

**Files:** `src/training/train.py`, `src/training/promote.py`, `configs/settings.yaml`.

**Done when:** MLflow shows all three models, and the gate rejects a model below the minimum in a test.

## 3. Data validation and Airflow

**Goal:** an Airflow DAG runs every required step and finishes successfully.

- [x] Add `src/validation/validate_data.py`, checking required columns, value ranges, missing values and date coverage. A failure stops the run.
- [x] Add it to the DVC pipeline.
- [x] Add a small evaluation step that reads the latest scores and records whether they pass.
- [x] Add Airflow to Docker Compose as a single container, the simplest free setup.
- [x] Build the DAG with these tasks: ingestion, validation, preprocessing, feature engineering, training, evaluation and model registration.
- [x] Have tasks call DVC stages, so unchanged steps are skipped and raw data isn't reprocessed needlessly.

**Files:** `airflow/dags/agroscope_pipeline.py`, `airflow/Dockerfile`, `src/validation/validate_data.py`, `dvc.yaml`.

**Main risk:** running a Linux container against Windows folders can be slow or cause file problems, and DVC or the geospatial packages may need extra setup. Check Docker Desktop's memory allowance before starting.

**Fallback:** if the full raw-data flow won't run reliably inside Airflow, its ingestion task will use the cloud and already-versioned data rather than the 10 GB of raw scenes. The task list stays the same, and the limit gets written down.

**Done when:** every DAG task is green in the Airflow UI.

**Result:** the full raw-data flow ran inside Airflow, so the fallback wasn't needed. A rerun with nothing changed took about 90 seconds, with all 10 stages skipped.

## 4. CI additions

- [x] Add a job that builds the Docker image in GitHub Actions, using small placeholder model and data files.
- [x] Add preprocessing and prediction unit tests that use a tiny test model, so they run in CI instead of skipping.
- [x] Run the model-quality check in CI against a sample result.

**Files:** `.github/workflows/ci.yml`, `tests/`.

**Done when:** all CI jobs are green on GitHub, including the image build.

## 5. Responsible AI documents

- [x] Write `docs/MODEL_CARD.md`: purpose, data, all three scores, the leakage finding and limits.
- [x] Write `docs/DATA_CARD.md`: sources, licences, dates, labels, known problems and storage.
- [x] Write `docs/RESPONSIBLE_AI.md`: intended use, decision-support-only use, privacy, ethical limits and data credits.
- [x] Add a fairness note: the data has no demographic variables, so results are compared by area instead.
- [x] Add a script that reports model scores for each area of the canal.
- [x] Confirm SHAP output works and include an example.

**Done when:** all three documents match the current results.

## 6. Performance report

**Goal:** real measurements for every number the rubric asks for.

- [x] Add `scripts/load_test.py` to measure p95 latency, throughput and error rate.
- [x] Add `scripts/simulate_drift.py`, shifting temperature, rainfall and NDWI, then show the results in Evidently and Grafana.
- [x] Measure model performance when labels become available, using later dates held back from training.
- [x] Define the business metric: hectares flagged for priority irrigation per cycle, with a clear note that it isn't validated against field outcomes.
- [x] Write `reports/performance_report.md` with the model metric, business metric, p95 latency, throughput, error rate and drift metric.

**Done when:** every number in the report comes from a saved run, with nothing estimated by hand.

## 7. Frontend improvements

**Goal:** a clear, polished dashboard that holds up in the five-minute demo. This improves what's already there, and doesn't add the paused farmer features.

- [x] Add your Carto key, or confirm the OpenStreetMap fallback looks good, and adjust map colours so results stay readable on the light base map.
- [x] Show the data date prominently, and make clear that it's a past date.
- [x] Show how many cells the map displays, such as 1,500 of 19,300.
- [x] Add clear loading states for the first page load and for each panel.
- [x] Tidy spacing and hierarchy in the KPI bar, sidebar and Model Status window.
- [x] Show drift as a short list of the most changed inputs, not all 25 names.
- [x] Add links from Model Status to MLflow, Grafana and the API docs.
- [x] Translate Model Status and other English-only labels into Marathi and Hindi.
- [x] Make the layout work at phone and tablet widths.
- [x] Improve keyboard use, contrast and labels.
- [x] Fix the `oxlint` warnings in the files changed.

**Files:** `frontend/src/App.jsx`, `frontend/src/components/`, `frontend/src/index.css`.

**Done when:** the dashboard is checked in the browser at desktop, tablet and phone widths, in all three languages, with the API on and off.

## 8. README and evidence

- [x] Rewrite `README.md` with an architecture diagram, one-command setup, stack and limitations.
- [x] Update `DEMO_GUIDE.md` into a five-minute demo with Airflow and Grafana.
- [x] Prepare viva notes covering each component, and why each choice was made.
- [x] Collect the screenshots listed below into `docs/screenshots/`.

- [x] State in the README and viva notes that the AWS extension isn't built, and what it would involve.

**You'll need to do these:** take the screenshots, or tell Claude if you'd like help with them, and run the final commits.

### Submission evidence

| Item | Evidence | Status |
| :--- | :--- | :--- |
| GitHub repository | Commit history | Final commit to run |
| README | Architecture and one-command setup | Done |
| DVC | `dvc.lock`, `docs/screenshots/dvc_dag.txt`, `dvc_status.txt` | Done, except `dvc status -c`, which needs your Drive sign-in |
| MLflow | `mlflow_runs.png`, `mlflow_registry.png` | Done |
| Airflow | `airflow/dags/agroscope_pipeline.py`, `airflow_dag_success.jpg` | Done |
| FastAPI | `swagger_api_docs.png` | Done |
| Docker | `Dockerfile`, `docker-compose.yml` | Done |
| GitHub Actions | Successful workflow screenshot | Passing, needs your screenshot |
| Monitoring | `grafana_monitoring.png`, `prometheus_targets.png`, `evidently_drift_report.png` | Done |
| Responsible AI | `docs/MODEL_CARD.md`, `DATA_CARD.md`, `RESPONSIBLE_AI.md` | Done |
| Performance | `reports/performance_report.md` | Done |
| Demo | `DEMO_GUIDE.md` | Written, needs your practice |
| Viva | `docs/VIVA_NOTES.md` | Written, needs your practice |
| AWS | Not built for now, stated in the README | Deferred |

## What gets cut if time runs short

Cut from the top of this list first:

1. Scores by area, keeping the written fairness note.
2. Lower-priority frontend items: translations, phone layout and lint cleanup.
3. Airflow's full raw-data flow, using the fallback above.

**Don't cut:** Airflow itself, Prometheus and Grafana, Docker Compose, the performance report, or the model and data cards, because the rubric asks for them directly. Finish these before polishing the frontend.

## Not part of this plan

These stay paused until after submission: AWS, crop type classification, Sentinel-1 radar, farmer field advice, Supabase, scheduled live data and public deployment. They're real limits, so the README and viva notes will state them openly.

## Commit points

Claude will give you the commands at each of these:

1. After Compose and monitoring.
2. After models and Airflow.
3. After CI and documents.
4. After frontend improvements.
5. Final submission.
