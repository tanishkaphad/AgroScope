# AgroScope Implementation Plan

**Goal:** turn AgroScope into a tool people use daily to answer one question: *which areas need water next, and how much?* Every forecast gets checked against what the satellite later observes.

**Status:** Phase 1 done. Phase 0 needs only the DVC remote. Phase 2 is next. Last updated 2026-11-03.

**Git rule:** Claude never runs git operations. The project owner runs them all.

## Assumptions

Change these first if they're wrong, because they affect the later phases.

| Decision | Assumption | Affects |
| :--- | :--- | :--- |
| First user | Canal section officers, water user associations and extension staff | Phases 3 and 6 |
| Geography | Nira Left Bank Canal only, until it's useful there | All phases |
| Grid | Keep the 250 m grid and report by section or village | Phases 2, 3 and 6 |
| Infrastructure | Scheduled jobs and Postgres, with no Celery or Kubernetes | Phases 2 and 5 |

## Starting point

Checked in the repo on 2026-11-03:

- **Data:** 27 satellite dates from 2023-10-06 to 2024-03-29, across 22,580 cells. Weather ends on 2024-05-31.
- **Weather:** the rows sampled from `daily_weather.csv` all use one point (18.1132, 74.4997).
- **Pipeline:** no stage in `dvc.yaml` produces `data/features/features.csv`.
- **MLOps tools:** `src/`, `api/` and `tests/` never import Evidently or MLflow, and `src/monitoring/` has only an `__init__.py`.
- **API access:** no authentication, and CORS allows every origin.

Reported in `PROJECT_ROAST.md`, to confirm when working on each item:

- Hardcoded responses from `/monitoring/drift`, `/retrain` and `/metrics`.
- Mock data fallbacks in `frontend/src/services/api.js`.
- Invented weather, crop and soil values in `api/app/routes/inference.py`.
- Labels calculated from three of the model's own input features.

## Phase overview

| Phase | Outcome | Depends on | Effort |
| :--- | :--- | :--- | :--- |
| 0. Stabilize the repo | Code is committed and rebuilds cleanly | None | Small |
| 1. Make it honest | The app never shows invented data | 0 | Small |
| 2. Make it live | Results refresh daily without manual work | 1 | Large |
| 3. Trustworthy advice | Daily water-need estimates for each cell | 2 | Medium |
| 4. A real model job | A model tested against future observations | 2, 3 | Large |
| 5. Real MLOps loop | Models are monitored, retrained and promoted on evidence | 2, 4 | Medium |
| 6. Daily habit | Users get a morning summary and can give feedback | 3 | Medium |

Phases 4 and 6 can run in parallel once Phase 3 is done. Phase 6 needs Phase 5 only for its model accuracy display.

**User research runs alongside all of this.** Speak to 3–5 target users before Phase 3 is finished, and ask how water decisions are made today.

---

## Phase 0: Stabilize the repo

**Goal:** a clean base that anyone can run and rebuild.

- [x] Commit the untracked work: `api/`, `frontend/`, `tests/`, `supabase/`, `dvc.yaml`, `Dockerfile`, `railway.toml` and CI.
- [x] Pin dependency versions, add the missing `psutil`, and move test and DVC tools to `requirements-dev.txt`.
- [x] Add `.env.example` files, load `.env` in the API and sync script, and add a README.
- [x] Fix `dvc.yaml`: the weather flag, missing outputs and missing dependencies, and add the `generate_labels` stage.
- [x] Remove the hardcoded score check from `test_metrics_endpoint`.
- [x] Skip tests that need a model or frontend build when those files are missing.
- [x] Initialize DVC and track `data/raw/satellite`.
- [x] Run `dvc repro` from raw data to model.
- [ ] Set up a DVC remote, then make CI run `dvc pull` so the skipped tests run there.

**Done when:** a fresh clone can get the data and run `dvc repro` from raw data to model, and CI passes without relying on files only you have.

## Phase 1: Make it honest

**Goal:** everything on screen is real or clearly marked as unavailable.

- [x] Remove the mock fallbacks from `frontend/src/services/api.js`.
- [x] Add error, empty and loading states to the frontend components.
- [x] Show the data date and model version wherever results appear.
- [x] Make `api/app/routes/inference.py` return an error or "no data" instead of invented weather, crop and soil values.
- [x] Make `/monitoring/drift` and `/retrain` return `501 Not Implemented` until Phase 5.
- [x] Read `/metrics` model scores from `models/feature_metadata.json` or the evaluation output, not from constants.
- [x] Remove the 99.86% claim, Evidently claims and "any canal in India" claim from the docs and slides, or mark them as planned. Done with a status note at the top of each.
- [x] Add tests that shut off the data source and confirm the API returns an error, not values.

**Done when:** with the backend off, the frontend shows a clear error and no numbers.

## Phase 2: Make it live

**Goal:** the system updates itself every day.

### Configuration
- [ ] Move URLs, thresholds, feature lists, crop tables and dates into `configs/settings.yaml`.

### Weather
- [ ] Replace the single weather point with a coarse grid across the canal area, interpolated to cells.
- [ ] Add 7-day forecasts to `src/ingestion/fetch_weather.py`.
- [ ] Check the weather provider's terms and rate limits for this use.

### Satellite
- [ ] Make `src/ingestion/fetch_satellite.py` incremental, so it only handles scenes it hasn't seen.
- [ ] Read only the needed area from cloud-hosted scenes, and keep derived features rather than raw downloads.
- [ ] Record each scene's cloud cover and whether it was used.

### Storage and scheduling
- [ ] Apply and extend `supabase/migrations/001_initial_schema.sql` with tables for runs, daily features, predictions and model versions.
- [ ] Remove the Pune and crop defaults from the schema.
- [ ] Change `api/app/services/geo_service.py` to read the latest results from Postgres, not local GeoJSON or a fixed date list.
- [ ] Fix the `Dockerfile`, which copies `models/` and `data/processed/` even though neither is in a fresh clone.
- [ ] Add a scheduled daily job, starting with GitHub Actions if the runtime fits.
- [ ] Add input checks that stop publishing on missing, stale or out-of-range data.
- [ ] Alert on failed runs, and show the last successful update in the UI.

**Done when:** the app has run for 14 days in a row without manual steps, and shows today's weather-based results with the latest clear satellite date.

## Phase 3: Trustworthy advice

**Goal:** replace the circular stress label with a transparent, physically based estimate. This also gives Phase 4 its baseline.

- [ ] Create `src/waterbalance/` with a daily FAO-56 soil water balance for each cell.
- [ ] Add soil water capacity from a soil dataset such as SoilGrids.
- [ ] Add crop type and sowing dates. Start with user or section-level input and a season calendar, not a crop classifier.
- [ ] Use the existing crop coefficient table in `src/target/generate_labels.py`, moved to config, by growth stage.
- [ ] Use NDVI to adjust for how much crop cover each cell actually has.
- [ ] Add canal rotation dates and section boundaries if users can provide them.
- [ ] Change the output to "irrigate within N days, about X mm", with a confidence level.
- [ ] Rewrite `src/advisory/advisory_engine.py` to use this output, and move Marathi and Hindi text to locale files.
- [ ] Add section and village rankings to the API.
- [ ] Review the results with an agronomist or extension officer.

**Done when:** target users review a week of rankings and say they broadly match what they see in the field. Record what they disagree with.

## Phase 4: A real model job

**Goal:** a model that beats simple baselines on data from the future.

- [ ] Backfill several years of Sentinel-2 features, including at least two monsoons.
- [ ] Add Sentinel-1 radar features to help cover cloudy periods.
- [ ] Define the target as the change in NDWI and NDVI at the next clear satellite pass.
- [ ] Use only inputs known at forecast time. Train on archived forecasts where possible, not on the weather that later happened.
- [ ] Remove the old label and leaking features from `src/training/train.py`.
- [ ] Train on earlier seasons and test on later ones, and hold out whole areas to avoid nearby cells leaking information.
- [ ] Compare with two baselines: "nothing changes" and the Phase 3 water balance.
- [ ] Report error by season, crop stage and section, and include prediction intervals.
- [ ] Rewrite `reports/model_evaluation_report.md` with these results.

**Known limit:** NDWI and NDVI show canopy condition, not soil moisture directly, and they often lag behind water stress. Check against field reports or soil probes when you can.

**Decision rule:** ship the model only if it clearly beats both baselines on later seasons. If it doesn't, keep the water balance and try again with more data.

**Done when:** the evaluation report shows a win over both baselines on a later season, or records why the model wasn't shipped.

## Phase 5: Real MLOps loop

**Goal:** model changes are driven by measured performance.

- [ ] Log every training run to MLflow: data version, code version, parameters and metrics.
- [ ] Register models with candidate, production and archived stages.
- [ ] Save every forecast with its model version and forecast date.
- [ ] Score earlier forecasts automatically when a new clear scene arrives.
- [ ] Add `src/monitoring/` with real Evidently drift reports on live and training data, allowing for normal seasonal change.
- [ ] Make `/monitoring/drift` return the latest stored report.
- [ ] Make `/retrain` start a real job, protected by login, and run retraining seasonally or when scores drop.
- [ ] Promote a new model only if it beats the current one on later data by an agreed margin.
- [ ] Add one-step rollback to the previous model.
- [ ] Add a model health page showing forecast accuracy over time.

**Done when:** you've run one full cycle on real data: drift or a score drop is detected, a model is retrained, it's promoted or rejected, and rollback has been tested.

## Phase 6: Daily habit

**Goal:** useful information reaches people without them having to open a dashboard.

- [ ] Add login with roles for operator, section staff and admin, and restrict CORS to your frontends.
- [ ] Let users choose their sections, language and alert preferences.
- [ ] Send a morning summary, starting with one channel. WhatsApp is the likely fit, but check its costs and approval steps first.
- [ ] Add a mobile-first page with a ranked list, and keep the map as a secondary view.
- [ ] Add "Did you irrigate?" and "Was this right?" feedback, stored with the forecast it refers to.
- [ ] Feed that feedback into Phase 5 evaluation as field evidence.
- [ ] Add basic usage analytics, and get consent for any personal data you store.

**Done when:** at least 3 real users have used it weekly for 4 weeks and given feedback, and at least one has made a water decision using it.

---

## Later

Only start these once the Nira canal version is in regular use:

- **Other canals:** automatic UTM zones, cell names from the input boundary, and no fixed map centre or Veer Dam marker.
- **Advice for individual farms:** 10–20 m pixels and farm boundaries.
- **Crop classifier:** replaces manual crop input once there are enough confirmed crop labels.

## Ongoing measures

| Measure | What it shows |
| :--- | :--- |
| Weekly active users, and how many act on the advice | Whether people find it useful |
| Forecast error compared with the baselines | Whether the model adds value |
| Days with on-time results | Whether people can rely on it |
| Share of forecasts later scored | Whether the evaluation loop is working |
| Feedback marked "right" or "wrong" | Whether the advice matches the field |

## First tasks

1. Set up the DVC remote and make CI run `dvc pull`.
2. Replace the watermarked base map (B-26).
3. Book the first conversations with target users.
4. Start Phase 2 with `configs/settings.yaml`.
