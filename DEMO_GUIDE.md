# AgroScope Demo Guide

How to run and present AgroScope's MLOps work. Run everything from the project root, with the virtual environment active.

## One-line summary

AgroScope is a satellite and weather pipeline for crop water stress on the Nira Left Bank Canal. It's versioned with DVC, tracked in MLflow, released through a promotion gate, and monitored for data drift with Evidently.

## Before the demo

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Add a local retraining token to `.env`, choosing your own value:

```text
RETRAIN_TOKEN=choose-a-private-value
```

Start each of these in its own terminal, with the environment active:

```powershell
uvicorn api.app.main:app
```

```powershell
npm run dev --prefix frontend
```

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

| Page | Address |
| :--- | :--- |
| Dashboard | http://localhost:5173 |
| API docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Drift report | http://localhost:8000/api/v1/monitoring/drift/report |

## Demo order

### 1. The pipeline

```powershell
dvc dag
```

```powershell
dvc status
```

**Point out:** every stage from weather to drift is defined in `dvc.yaml`, settings are in `configs/settings.yaml`, and data is stored on Google Drive.

### 2. Experiment tracking

Open MLflow, then the `agroscope-water-stress` experiment.

**Point out:** each run records its settings, scores, data version hash and model files.

### 3. Model registry and promotion

In MLflow, open **Models**, then `agroscope-stress-classifier`, and show the `champion` alias. Then show `reports/promotion_decision.json`.

**Point out:** training only creates a candidate. It replaces the champion only if its later-dates score is at least as good. Otherwise, the champion stays live.

### 4. Drift monitoring

Open the dashboard, click **Model Status**, then **Open full report**.

**Point out:** this is a real Evidently comparison between the training period and the published date. All 25 inputs show drift, because March is much hotter and drier than October to January. With only one season of data, this can't separate normal seasonal change from a real problem.

### 5. Retraining

In PowerShell, use your own token:

```powershell
curl.exe -X POST -H "X-Admin-Token: your-token" http://localhost:8000/api/v1/retrain
```

```powershell
curl.exe http://localhost:8000/api/v1/retrain/status
```

**Point out:** this takes about 2 minutes. It retrains, applies the gate, rebuilds the maps and drift report, then reloads the API. A missing or wrong token gets a 403.

### 6. Testing

```powershell
pytest tests -v
```

**Point out:** the tests cover the API's error handling, weather and satellite ingestion, the promotion gate rejecting a worse model, and drift parsing. GitHub Actions runs them too.

### 7. Packaged API

```powershell
docker build -t agroscope-api:local .
```

```powershell
docker run --rm -p 8000:8000 agroscope-api:local
```

Stop the normal API first, because both use port 8000. The dashboard is then at http://localhost:8000/dashboard/.

**Point out:** the image carries the champion model, results and API, but not the training tools, so `/retrain` is disabled there. The image is about 2.7 GB. Building it needs `dvc pull` and `npm run build --prefix frontend` first on a fresh clone.

## Honest results

These are measured against labels calculated from a formula, not field observations.

| XGBoost test | Macro F1 | What it means |
| :--- | :--- | :--- |
| Unseen cells, same dates | 0.9967 | Looks excellent, but the labels use three of the inputs |
| Later dates | 0.9176 | A fairer test of future dates |
| Later dates, without those three inputs | 0.7439 | Shows how much of the score came from reproducing the formula |

**Key message:** the gate uses the later-dates score, not the most flattering one.

## Problems found and fixed

- **Reflectance bug:** Sentinel-2's +1000 offset was never removed. Mean NDVI across all dates went from 0.249 to 0.405 after the fix.
- **Wind units:** requests didn't ask for m/s, so values were very likely km/h.
- **Single weather point:** every cell shared one point. There are now 24.
- **Fake outputs:** invented API and dashboard data is gone. Missing data now shows a clear error.

## Likely questions

**Is this accurate enough for farmers to use?**
No. There's no field validation yet, and an agronomist should review the thresholds first.

**Why not report 99.7%?**
Because that mostly reflects the model reproducing its own label formula.

**Does the app update daily?**
Not yet. Weather forecasts, new-scene search and cloud feature extraction work by hand, but no scheduled job connects them to the app yet.

**How big is the storage saving from cloud features?**
On one tested scene, 827 KB instead of 198 MB.

## What isn't built

Say this before you're asked:

- **From the full problem statement:** crop type classification, growth-stage stress and Sentinel-1 radar data.
- **Live operation:** scheduled runs and a public deployment. The API runs in Docker locally.
- **Data limits:** one season of model data, and no field validation.
- **Reported area:** the 120,625 ha vegetated area is larger than the 84,500 ha gross command area in the study area file, so the boundary and NDVI threshold need checking.
- **MLflow history:** runs are stored locally, so a fresh clone starts with no champion.
