"""AgroScope end-to-end pipeline.

Each task runs one DVC stage, so DVC skips any step whose inputs haven't changed and raw data
isn't reprocessed needlessly. `dvc.yaml` remains the single definition of what each step does.
A failed validation stops every task after it.

To show a real run when nothing has changed, trigger with `force_rerun` on. That reruns every
stage except the weather download, which needs `refetch_weather` as well, because new weather
values would change every result.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/agroscope"
PIPELINE_BIN = "/opt/pipeline-venv/bin"
WEATHER_STAGE = "acquire_weather"

# (task id, DVC stage, what the task does), in the order they run
STEPS = [
    ("ingestion", "acquire_weather", "Fetch weather data. Satellite scenes are versioned separately with DVC."),
    ("raw_data_validation", "validate_raw", "Check raw weather and the scene list, and stop the run on failure."),
    ("preprocessing", "preprocess_satellite", "Build the 250 m grid and per-cell satellite values."),
    ("feature_engineering", "generate_labels", "Join satellite and weather data, and calculate stress labels."),
    ("feature_validation", "validate_features", "Check the training table, and stop the run on failure."),
    ("training", "train_evaluate", "Train the baseline and two candidates, each as its own MLflow run."),
    ("evaluation", "evaluate_model", "Record whether the best model's scores pass the minimum."),
    ("model_registration", "promote_model", "Register the candidate, apply the gate and export the champion."),
    ("batch_prediction", "predict_cycle", "Publish the stress and irrigation maps from the champion."),
    ("drift_monitoring", "monitor_drift", "Compare the published date's inputs with the training period."),
]

with DAG(
    dag_id="agroscope_pipeline",
    description="Ingestion to model registration, using DVC stages",
    schedule=None,  # Started by hand: the data is one past season, so a timer would find nothing new.
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 0, "execution_timeout": timedelta(hours=2)},
    tags=["agroscope", "mlops"],
    params={
        "force_rerun": Param(False, type="boolean", title="Force rerun", description="Rerun every stage, even if nothing changed. Takes about 9 minutes."),
        "refetch_weather": Param(False, type="boolean", title="Refetch weather", description="Also download weather again. Needs internet, and new values would change every result."),
    },
    doc_md=__doc__,
) as dag:
    tasks = [
        BashOperator(
            task_id=task_id,
            bash_command=(
                "dvc repro {{ '--force' if params." + ("refetch_weather" if stage == WEATHER_STAGE else "force_rerun") + " else '' }} "
                f"--single-item {stage}"
            ),
            cwd=PROJECT_DIR,
            # DVC runs each stage's `python` from this PATH.
            env={"PATH": f"{PIPELINE_BIN}:/usr/local/bin:/usr/bin:/bin"},
            append_env=True,
            doc=description,
        )
        for task_id, stage, description in STEPS
    ]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
