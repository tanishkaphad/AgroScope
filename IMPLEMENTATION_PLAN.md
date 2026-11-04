# AgroScope Implementation Plan

**Goal:** turn AgroScope into a tool people use daily to answer one question: *should I water my farm, when, and how much?* Every forecast gets checked against what the satellite later observes.

**Status:** Submission priority, with 2 days left from 2026-11-04. Phase 1 done, and the core of Phase 5 built ahead of order. Last updated 2026-11-04.

**Until submission:** finish tests, Docker and the demo. Phases 2, 3, 4 and 6 resume afterwards.

**Git rule:** Claude never runs git operations. The project owner runs them all.

**Cost rule:** every external service stays on a free tier. Check current limits before adopting one.

## Key decisions

| Decision | Choice | Status | Affects |
| :--- | :--- | :--- | :--- |
| First user | Farmers who control their own water, such as well, borewell and drip users. Canal farmers and canal teams come later. | Confirmed | Phases 3 and 6 |
| Cost | Free tiers only, and the app stays non-commercial for now | Confirmed | All phases |
| Data storage | Google Drive as the DVC remote | Confirmed | Phase 0 |
| Geography | Nira Left Bank Canal only, until it's useful there | Assumed | All phases |
| Farm detail | Farmers mark their own field, and it's read at 10–20 m. The 250 m grid stays for area overviews. | Assumed | Phases 2, 3 and 6 |
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
| 3. Advice for one farm | A farmer marks a field and gets daily water advice for it | 2 | Large |
| 4. A real model job | A model tested against future observations | 2, 3 | Large |
| 5. Real MLOps loop | Models are monitored, retrained and promoted on evidence | 2, 4 | Medium |
| 6. Daily habit | Farmers get advice on their phones in their language and give feedback | 3 | Large |

Phases 4 and 6 can run in parallel once Phase 3 is done. Phase 6 needs Phase 5 only for its model accuracy display.

**Farmer conversations run alongside all of this.** Speak to 5–10 farmers before Phase 3 is finished, starting with those who control their own water. Ask how they decide to water today, what phone they use, and what would make them trust the advice. For expert review, approach a Krishi Vigyan Kendra or an agricultural university.

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
- [x] Add the Google Drive remote and run `dvc push`.
- [ ] Add a service account and the `GDRIVE_CREDENTIALS_DATA` secret, so CI's new `dvc pull` step runs the skipped tests.

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
- [x] Move URLs, thresholds, feature lists, crop tables and dates into `configs/settings.yaml`. Advisory rules stay in code until Phase 3 rewrites them.

### Weather
- [x] Replace the single weather point with a 24-point grid across the canal area, with each cell using its nearest point.
- [x] Add 7-day forecasts to `src/ingestion/fetch_weather.py`.
- [x] Check the weather provider's terms and rate limits for this use: under 10,000 calls a day, non-commercial, with credit shown.
- [x] Run `dvc repro` with the new weather grid and the reflectance fix.

### Satellite
- [x] Make `src/ingestion/fetch_satellite.py` incremental, so it only handles scenes it hasn't seen.
- [x] Calculate features directly from cloud-hosted scenes and keep only the results, about 0.3–0.8 MB per scene instead of about 200 MB. Tested on an older and the newest scene.
- [ ] Process the new scenes, extend the weather dates, and feed the results into the pipeline. Check whether a scene now takes less than the 8–13 minutes measured during `dvc push`.
- [x] Record each scene's cloud cover and download status in the manifest.

### Storage and scheduling
- [ ] Check the study area boundary, because the published vegetated area is larger than the official command area (B-35).
- [ ] Apply and extend `supabase/migrations/001_initial_schema.sql` with tables for runs, daily features, predictions and model versions.
- [ ] Remove the Pune and crop defaults from the schema.
- [ ] Change `api/app/services/geo_service.py` to read the latest results from Postgres, not local GeoJSON or a fixed date list.
- [ ] Fix the `Dockerfile`, which copies `models/` and `data/processed/` even though neither is in a fresh clone.
- [ ] Check free-tier limits for API hosting, Postgres and scheduled jobs, and replace any that aren't free in practice.
- [ ] Add a scheduled daily job, starting with GitHub Actions if the runtime fits.
- [ ] Add input checks that stop publishing on missing, stale or out-of-range data.
- [ ] Alert on failed runs, and show the last successful update in the UI.

**Done when:** the app has run for 14 days in a row without manual steps, and shows today's weather-based results with the latest clear satellite date.

## Phase 3: Advice for one farm

**Goal:** a farmer marks a field and gets advice they can act on. This replaces the circular stress label with a transparent estimate, and gives Phase 4 its baseline.

### The farm
- [ ] Let a farmer mark a field by tapping the map or using phone location, then adjust its boundary.
- [ ] Ask for crop, sowing date and water source: canal, well, borewell, drip or rain only.
- [ ] Read that field's Sentinel-2 pixels only when needed, at 10–20 m, instead of processing the whole canal in advance.
- [ ] Say so when a field is too small or too cloudy to read reliably.

### The advice
- [ ] Create `src/waterbalance/` with a daily FAO-56 soil water balance for each field.
- [ ] Add soil water capacity from a free soil dataset such as SoilGrids.
- [ ] Use crop coefficients by growth stage, moved to config from `src/target/generate_labels.py`.
- [ ] Use NDVI to adjust for how much crop cover the field actually has.
- [ ] Use the 7-day forecast, so the app can say "rain is likely, wait".
- [ ] Change the output to "water within N days, about X mm", in units farmers use, with a confidence level.
- [ ] Fit advice to the water source. Canal users need advice that fits their next water turn, so add rotation dates where available.
- [ ] Rewrite `src/advisory/advisory_engine.py` to use this output, and move Marathi and Hindi text to locale files.

### Safety and trust
- [ ] Show the reason for each piece of advice in plain words, and when the data was last updated.
- [ ] Review the method and wording with an agronomist or agricultural extension officer before farmers rely on it.
- [ ] Say clearly that it's guidance, and check against what farmers see in their fields.

**Done when:** 5 or more farmers have followed advice for their own fields for two weeks, and an agronomist has reviewed it. Record every case where they disagreed with it.

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

- [x] Log every training run to MLflow: data version, parameters and metrics. Code version isn't recorded yet.
- [x] Register every candidate in the MLflow registry, with a `champion` alias for the live model.
- [ ] Save every forecast with its model version and forecast date.
- [ ] Score earlier forecasts automatically when a new clear scene arrives.
- [x] Add `src/monitoring/` with real Evidently drift reports on the published date and training data. It can't yet allow for normal seasonal change, because there's only one season of data.
- [x] Make `/monitoring/drift` return the latest stored report.
- [x] Make `/retrain` start a real job, protected by an admin token.
- [ ] Run retraining seasonally or when scores drop.
- [x] Promote a new model only if it matches or beats the current one on later data.
- [ ] Add one-step rollback to the previous model.
- [x] Add a model status window with scores, the promotion decision and drift results.
- [ ] Show forecast accuracy over time.

**Done when:** you've run one full cycle on real data: drift or a score drop is detected, a model is retrained, it's promoted or rejected, and rollback has been tested.

## Phase 6: Daily habit

**Goal:** farmers get what they need on their phones without having to open a dashboard.

- [ ] Build a mobile-first installable web app that shows "my farm today" first, with the map as a secondary view.
- [ ] Make it work on low-end phones and weak networks, and show the last saved advice when offline.
- [ ] Use Marathi by default in this area, with simple words, icons and read-aloud where possible.
- [ ] Choose a low-cost sign-in. SMS codes usually cost money, so check free options first.
- [ ] Send daily or as-needed alerts through a free channel. Check web push and Telegram first; WhatsApp and SMS may not stay free at scale.
- [ ] Restrict CORS to your frontends, and let farmers view and delete their farm data.
- [ ] Add "Did you water?" and "Was this right?" feedback, stored with the forecast it refers to.
- [ ] Feed that feedback into Phase 5 evaluation as field evidence.
- [ ] Add basic usage analytics, and get consent for any personal data you store.

**Done when:** at least 10 farmers have used it weekly for 4 weeks and given feedback, and most say it helped a real water decision.

---

## Later

Only start these once the Nira canal version is in regular use:

- **Other canals:** automatic UTM zones, cell names from the input boundary, and no fixed map centre or Veer Dam marker.
- **Canal teams and other users:** views built from the same data, such as area demand and rotation planning.
- **Crop classifier:** replaces manual crop input once there are enough confirmed crop labels.

## Ongoing measures

| Measure | What it shows |
| :--- | :--- |
| Weekly active farmers, and how many act on the advice | Whether farmers find it useful |
| Forecast error compared with the baselines | Whether the model adds value |
| Days with on-time results | Whether people can rely on it |
| Share of forecasts later scored | Whether the evaluation loop is working |
| Feedback marked "right" or "wrong" | Whether the advice matches the field |

## First tasks

1. Add the Google Drive remote and run `dvc push`.
2. Add a free Carto key.
3. Arrange the first conversations with farmers.
4. Start Phase 2 with `configs/settings.yaml`.
