# AgroScope

Satellite and weather-based crop water stress estimates and irrigation advice for the Nira Left Bank Canal in Maharashtra, India.

**Status:** early development. The data is from one past season (Oct 2023 to Mar 2024), the stress labels are calculated from a formula rather than field measurements, and drift monitoring and retraining aren't built yet. See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the roadmap.

## Requirements

- Python 3.10
- Node.js 20

## Setup

Run these from the project root. The commands below are for Windows PowerShell.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

`.env` is only needed for Supabase. The API runs from local files without it.

## Run the API

The API needs model and map files in `models/` and `data/processed/`. Get them with `dvc pull` if a remote is set up, or build them with `dvc repro`. Without them, data endpoints return 503.

```powershell
uvicorn api.app.main:app --reload
```

API docs are at http://localhost:8000/docs.

## Run the frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

The dashboard opens at http://localhost:5173.

## Run the pipeline

Activate the virtual environment first, because the stages run with `python`.

```powershell
dvc repro
```

| Stage | Output |
| :--- | :--- |
| `acquire_weather` | `data/raw/weather/daily_weather.csv` |
| `preprocess_satellite` | 250 m grid and satellite features in `data/processed/` |
| `generate_labels` | `data/processed/target_labels.csv` and `data/features/features.csv` |
| `train_evaluate` | Models and metadata in `models/`, plus `reports/model_evaluation_report.md` |
| `predict_cycle` | Stress map, irrigation map and cycle statistics in `data/processed/` |

Satellite downloads aren't a pipeline stage. To fetch scenes into `data/raw/satellite/`, run:

```powershell
python src/ingestion/fetch_satellite.py --start-date 2023-10-01 --end-date 2024-03-31
```

## Run the tests

```powershell
pytest tests/test_api.py -v
```

Tests that need a trained model, prediction results or a frontend build are skipped when those files are missing.
