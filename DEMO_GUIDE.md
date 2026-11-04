# AgroScope Demo Guide

A five-minute demo of AgroScope's MLOps work. Practise it once with a timer.

## One-line summary

AgroScope turns satellite and weather data into crop water stress estimates for the Nira Left Bank Canal, with a versioned pipeline, tracked experiments, a release gate, orchestration and live monitoring.

## Before the demo

**Do this at least 10 minutes before.**

1. Close the old Windows copies of the API and MLflow. They answer on `127.0.0.1`, and could show old data:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' -and $_.CommandLine -match 'AgroScope|api.app.main' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

2. Start the stack, and wait until the API shows as healthy:

```bash
docker compose up -d
```
```bash
docker compose ps
```

3. Send some traffic so Grafana has something to show. Do this about 3 minutes before you start, because its charts cover recent activity:

```bash
.venv/Scripts/python scripts/load_test.py --users 5 --duration 30
```

The load test overwrites `reports/load_test_results.json`. To keep the submitted numbers, add `--output reports/demo_load_test.json` and don't commit that file.

4. Open these tabs in order, using `localhost`, not `127.0.0.1`:

| Tab | Address |
| :--- | :--- |
| Dashboard | http://localhost:8000/dashboard/ |
| Airflow, signed in as `admin` / `admin` | http://localhost:8080 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| API docs | http://localhost:8000/docs |

5. Open a terminal in the project root, and have `dvc dag` ready to run. Activate the environment first with `.venv\Scripts\Activate.ps1`.

## The five minutes

| Time | Show | Say |
| :--- | :--- | :--- |
| 0:00 | Dashboard | "This estimates crop water stress for the canal from satellite and weather data." |
| 0:30 | Airflow, then `dvc dag` | "One DAG runs every step, and each task is a DVC stage." |
| 1:30 | MLflow | "Three models, compared fairly, with a gate before release." |
| 2:30 | Grafana | "Live request, latency, prediction and drift metrics." |
| 3:15 | Dashboard, Model Status | "Here's what the system tells us about itself." |
| 4:00 | The honest results | "These scores have a catch, and here's what it is." |
| 4:30 | What isn't built | "This is what's missing, including AWS." |

## 0:00 Dashboard (30 seconds)

1. Point out the date bar: past data, with 1,500 of 19,300 cells shown.
2. Change the date to **10 December 2023**, then to **24 March 2024**. HIGH stress goes from 9.2% to 97.2% across the season.
3. Zoom into the map and click a cell, because cells are too small to click at full view. Switch to **मराठी** to show its advice in Marathi.

## 0:30 Airflow and DVC (1 minute)

1. Open `agroscope_pipeline` and show the successful run in the grid view.
2. Name the tasks: ingestion, validation, preprocessing, features, validation, training, evaluation, registration, prediction and drift.
3. Run `dvc dag` in the terminal.

**Point out:** each task calls a DVC stage, so unchanged steps are skipped. A rerun with nothing changed took about 90 seconds. Validation failures stop the run before training.

## 1:30 MLflow (1 minute)

1. Open the `agroscope-water-stress` experiment, and compare the `random_forest`, `xgboost` and `lightgbm` runs.
2. Open **Models**, then `agroscope-stress-classifier`, and show the `champion` alias and the version's tags.

**Point out:** the best later-dates score becomes the candidate. It's released only if it reaches 0.85 and at least matches the champion. A worse model is rejected and the champion stays, and tests prove it.

## 2:30 Grafana (45 seconds)

Show requests per second, error rate, inference latency, and the drift panel with `published` and `simulated`.

**Point out:** the load test measured 170.79 requests a second with a p95 of 150.5 ms and no errors, and those numbers are in the performance report.

## 3:15 Model Status (45 seconds)

Click **Model Status** on the dashboard. Show the three scores, the promotion decision and the top drifted inputs. Open the full Evidently report if time allows.

**Point out:** drift is 100% on the published date, mostly normal seasonal change, because the reference is one season. A simulated test shows the monitor works: 0% on unchanged data, and 36% with exactly the 9 shifted inputs flagged, which fires the 30% alert.

## 4:00 The honest results (30 seconds)

| Test | LightGBM macro F1 |
| :--- | :--- |
| Unseen cells | 0.9966 |
| Later dates, used by the gate | 0.9242 |
| Later dates, without the label inputs | 0.6707 |

**Say:** "The labels come from a formula that uses three of the model's inputs, so high scores mostly mean it has relearned the formula. The last row is the fairer guide, and nothing has been checked in the field."

## 4:30 What isn't built (30 seconds)

**Say this before you're asked:**

- **AWS:** not built, by decision. The README shows how each part would move.
- **Field validation:** none yet.
- **Live data:** one past season, and a person starts each run.
- **From the wider plan:** crop type classification, Sentinel-1 radar and advice for individual fields.

## If something goes wrong

| Problem | Fix |
| :--- | :--- |
| API not healthy | Wait a minute, then run `docker compose restart api` |
| Old data or an old look | Check the address says `localhost`, and close the old Windows copies |
| Dashboard shows old results after a DAG run | `docker compose restart api` |
| Empty Grafana charts | Set the time range to the last 30 minutes, and send traffic with the load test |
| MLflow is empty | Run `agroscope_pipeline`. Don't use `docker compose down -v`, which deletes this data |
| A tool won't open | Show the screenshot from `docs/screenshots/` |

The full viva preparation is in [docs/VIVA_NOTES.md](docs/VIVA_NOTES.md).
