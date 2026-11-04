"""AgroScope Model Training and Comparison.

Trains one baseline and two candidates on the same data, and scores each three ways:
on unseen cells, on later dates, and on later dates without the inputs the labels come from.
Each model becomes its own MLflow run, and the best later-dates score becomes the candidate
in models/candidate/ for the promotion gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle
import shutil
import tempfile

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.training.tracking import attach_run_artifact, file_md5, log_training_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.training")

TRAINING = settings["training"]

# Canonical feature list for training (excludes target and direct leakage columns)
FEATURE_COLUMNS = TRAINING["features"]

TARGET_COL = TRAINING["target_column"]
SPLIT_DATE_THRESHOLD = TRAINING["temporal_split_date"]
RANDOM_STATE = TRAINING["random_state"]
N_JOBS = TRAINING["n_jobs"]

MODEL_FILE = "model.pkl"
METADATA_FILE = "feature_metadata.json"
MODEL_SPECS = {
    "random_forest": {"title": "Random Forest", "role": "baseline"},
    "xgboost": {"title": "XGBoost", "role": "candidate"},
    "lightgbm": {"title": "LightGBM", "role": "candidate"},
}


def prepare_datasets(
    features_csv: Path,
    strategy: str = "spatial",
    temporal_split_date: str = SPLIT_DATE_THRESHOLD,
    sample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], str]:
    """Load, filter, and perform spatial block or temporal holdout split."""
    logger.info("Loading dataset from: %s", features_csv)
    df = pd.read_csv(features_csv)

    # Filter to active agricultural canopy (NDVI >= 0.15)
    if "is_vegetated" in df.columns:
        initial_len = len(df)
        df = df[df["is_vegetated"] == True].copy()
        logger.info("Filtered non-vegetated/fallow cells: %d -> %d rows", initial_len, len(df))

    # Verify all feature columns exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns in dataset: {missing_cols}")

    # Optional downsampling for rapid local experimentation if requested
    if sample_size and len(df) > sample_size:
        logger.info("Sampling %d rows for rapid training...", sample_size)
        df = df.sample(n=sample_size, random_state=RANDOM_STATE).copy()

    if strategy == "spatial" and "cell_id" in df.columns:
        # Spatial Block Holdout (Phase 11): 25% of agricultural fields held out as unseen blocks
        gss = GroupShuffleSplit(n_splits=1, test_size=TRAINING["spatial_test_size"], random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(df, groups=df["cell_id"]))
        X_train = df.iloc[train_idx][FEATURE_COLUMNS]
        y_train = df.iloc[train_idx][TARGET_COL].astype(int)
        X_test = df.iloc[test_idx][FEATURE_COLUMNS]
        y_test = df.iloc[test_idx][TARGET_COL].astype(int)
        split_desc = f"Spatial Block Holdout (75% train / 25% test across {df.iloc[test_idx]['cell_id'].nunique():,} unseen spatial grid cells)"
        logger.info(
            "Spatial Block Split: Train = %d rows (%.1f%%) | Test Holdout = %d rows (%.1f%%)",
            len(X_train),
            (len(X_train) / len(df)) * 100,
            len(X_test),
            (len(X_test) / len(df)) * 100,
        )
    else:
        # Temporal Holdout Split
        train_mask = df["date"] < temporal_split_date
        test_mask = df["date"] >= temporal_split_date
        X_train = df.loc[train_mask, FEATURE_COLUMNS]
        y_train = df.loc[train_mask, TARGET_COL].astype(int)
        X_test = df.loc[test_mask, FEATURE_COLUMNS]
        y_test = df.loc[test_mask, TARGET_COL].astype(int)
        split_desc = f"Temporal Holdout Split (< {temporal_split_date} vs >= {temporal_split_date})"
        logger.info(
            "Temporal Split (< %s): Train = %d rows (%.1f%%) | Test Holdout = %d rows (%.1f%%)",
            temporal_split_date,
            len(X_train),
            (len(X_train) / len(df)) * 100,
            len(X_test),
            (len(X_test) / len(df)) * 100,
        )

    return X_train, X_test, y_train, y_test, FEATURE_COLUMNS, split_desc


def build_model(name: str) -> Any:
    """Create an untrained model from its settings. All three correct for class imbalance."""
    params = TRAINING[name]
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", n_jobs=N_JOBS, random_state=RANDOM_STATE, **params)
    if name == "xgboost":
        return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", n_jobs=N_JOBS, random_state=RANDOM_STATE, **params)
    if name == "lightgbm":
        return LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced", n_jobs=N_JOBS, random_state=RANDOM_STATE, verbose=-1, **params)
    raise ValueError(f"Unknown model: {name}")


def train_model(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train one model. XGBoost has no class_weight option, so it takes sample weights."""
    model = build_model(name)
    if name == "xgboost":
        model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(X_train, y_train)
    return model


def select_best_model(scores: Dict[str, float]) -> str:
    """Pick the highest score. On a tie, the earliest in MODEL_SPECS wins, so the baseline is preferred."""
    return max(scores, key=lambda name: (scores[name], -list(MODEL_SPECS).index(name)))


def ranked_importances(model: Any, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Rank features by share of importance. LightGBM reports split counts, so all are normalised."""
    importances = np.asarray(model.feature_importances_, dtype=float)
    shares = importances / (importances.sum() or 1.0)
    return sorted(zip(feature_names, [float(x) for x in shares]), key=lambda x: x[1], reverse=True)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """Evaluate model with full metrics suite: Accuracy, Macro F1, Weighted F1, Multi-class ROC-AUC, and Confusion Matrix."""
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    labels = [0, 1, 2]
    target_names = ["0: LOW", "1: MODERATE", "2: HIGH"]

    accuracy = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, preds, labels=labels, average="weighted", zero_division=0))

    # Multi-class ROC-AUC (One-vs-Rest)
    roc_auc_ovr = 0.0
    per_class_auc: Dict[str, float] = {}
    if probs is not None:
        try:
            roc_auc_ovr = float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro", labels=labels))
            for i, name in enumerate(target_names):
                per_class_auc[name] = float(roc_auc_score((y_test == i).astype(int), probs[:, i]))
        except Exception as e:
            logger.warning("Could not calculate ROC-AUC: %s", e)

    report_dict = classification_report(
        y_test,
        preds,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=labels).tolist()

    logger.info("[%s Evaluation on Unseen Spatial Holdout]", model_name)
    logger.info("  Overall Accuracy:                %.2f%%", accuracy * 100)
    logger.info("  Macro F1 Score:                  %.4f", macro_f1)
    logger.info("  Weighted F1 Score:               %.4f", weighted_f1)
    logger.info("  Multi-class ROC-AUC (OvR Macro): %.4f", roc_auc_ovr)
    logger.info(
        "  Class 0 (LOW)      Recall: %5.2f%% | Precision: %5.2f%% | ROC-AUC: %.4f",
        report_dict["0: LOW"]["recall"] * 100,
        report_dict["0: LOW"]["precision"] * 100,
        per_class_auc.get("0: LOW", 0.0),
    )
    logger.info(
        "  Class 1 (MODERATE) Recall: %5.2f%% | Precision: %5.2f%% | ROC-AUC: %.4f",
        report_dict["1: MODERATE"]["recall"] * 100,
        report_dict["1: MODERATE"]["precision"] * 100,
        per_class_auc.get("1: MODERATE", 0.0),
    )
    logger.info(
        "  Class 2 (HIGH)     Recall: %5.2f%% | Precision: %5.2f%% | ROC-AUC: %.4f",
        report_dict["2: HIGH"]["recall"] * 100,
        report_dict["2: HIGH"]["precision"] * 100,
        per_class_auc.get("2: HIGH", 0.0),
    )

    print(f"\n  Confusion Matrix ({model_name}):")
    print("                    Pred LOW    Pred MOD    Pred HIGH")
    print(f"    Actual LOW:     {cm[0][0]:<11} {cm[0][1]:<11} {cm[0][2]:<11}")
    print(f"    Actual MOD:     {cm[1][0]:<11} {cm[1][1]:<11} {cm[1][2]:<11}")
    print(f"    Actual HIGH:    {cm[2][0]:<11} {cm[2][1]:<11} {cm[2][2]:<11}\n")

    return {
        "model_name": model_name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "roc_auc_ovr": round(roc_auc_ovr, 4),
        "per_class_auc": per_class_auc,
        "classification_report": report_dict,
        "confusion_matrix": cm,
    }


def run_training_pipeline(
    features_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    strategy: str = "spatial",
    sample_size: Optional[int] = None,
) -> Path:
    """Train every model, log each as its own run, and save the best one as the candidate."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if features_csv is None:
        features_csv = base_dir / "data" / "features" / "features.csv"
    if output_dir is None:
        output_dir = base_dir / "models" / "candidate"

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names, split_desc = prepare_datasets(
        features_csv=features_csv,
        strategy=strategy,
        sample_size=sample_size,
    )
    X_past, X_future, y_past, y_future, _, temporal_desc = prepare_datasets(
        features_csv=features_csv, strategy="temporal", sample_size=sample_size
    )
    label_inputs = TRAINING["label_input_features"]
    data_md5 = file_md5(features_csv)

    results: Dict[str, Dict[str, Any]] = {}
    for name, spec in MODEL_SPECS.items():
        title = spec["title"]
        logger.info("Training %s (%s)...", title, spec["role"])
        model = train_model(name, X_train, y_train)
        results[name] = {
            "model": model,
            "spatial": evaluate_model(model, X_test, y_test, model_name=title),
            # Date-based check: train on earlier dates and test on later ones
            "temporal": evaluate_model(train_model(name, X_past, y_past), X_future, y_future, model_name=f"{title} Date-Based Check"),
            # Leakage check: the same test without the inputs the labels are calculated from
            "temporal_no_label_inputs": evaluate_model(
                train_model(name, X_past.drop(columns=label_inputs), y_past),
                X_future.drop(columns=label_inputs),
                y_future,
                model_name=f"{title} Date-Based Check Without Label Inputs",
            ),
        }

    best_name = select_best_model({name: r["temporal"]["macro_f1"] for name, r in results.items()})
    logger.info("Best model on later dates: %s (macro F1 %.4f).", best_name, results[best_name]["temporal"]["macro_f1"])

    shared_metadata = {
        "features": feature_names,
        "classes": [0, 1, 2],
        "class_labels": ["LOW", "MODERATE", "HIGH"],
        "validation_strategy": split_desc,
        "temporal_validation": temporal_desc,
        "selected_model": best_name,
        "metrics": {name: r["spatial"] for name, r in results.items()},
        "temporal_metrics": {name: r["temporal"] for name, r in results.items()},
        "temporal_metrics_without_label_inputs": {name: r["temporal_no_label_inputs"] for name, r in results.items()},
    }

    # Each model becomes its own MLflow run. Only the best one is kept on disk as the candidate.
    with tempfile.TemporaryDirectory() as tmp:
        best_dir = None
        for name, result in results.items():
            run_dir = Path(tmp) / name
            run_dir.mkdir()
            model_path = run_dir / MODEL_FILE
            meta_path = run_dir / METADATA_FILE
            with open(model_path, "wb") as f:
                pickle.dump(result["model"], f)

            metadata = {**shared_metadata, "model_type": name, "top_features": ranked_importances(result["model"], feature_names)[:10]}
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            run_id = log_training_run(
                params={**TRAINING[name], "model_type": name, "feature_count": len(feature_names), "train_rows": len(X_train), "temporal_split_date": SPLIT_DATE_THRESHOLD},
                metrics={
                    "spatial_accuracy": result["spatial"]["accuracy"],
                    "spatial_macro_f1": result["spatial"]["macro_f1"],
                    "temporal_accuracy": result["temporal"]["accuracy"],
                    "temporal_macro_f1": result["temporal"]["macro_f1"],
                    "temporal_macro_f1_without_label_inputs": result["temporal_no_label_inputs"]["macro_f1"],
                },
                tags={"model_type": name, "role": MODEL_SPECS[name]["role"], "selected": str(name == best_name).lower(), "data_md5": data_md5, "validation": split_desc},
                artifact_paths=[model_path],
                run_name=name,
            )
            # The metadata names its own run, so it can only be attached once the run exists.
            metadata["mlflow_run_id"] = run_id
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            attach_run_artifact(run_id, meta_path)

            if name == best_name:
                shutil.copy2(model_path, output_dir / MODEL_FILE)
                shutil.copy2(meta_path, output_dir / METADATA_FILE)
                best_dir = output_dir

    write_evaluation_report(reports_dir / "model_evaluation_report.md", results, best_name, feature_names, split_desc, temporal_desc, len(X_train), len(X_test))
    return best_dir / MODEL_FILE


def write_evaluation_report(
    report_file: Path,
    results: Dict[str, Dict[str, Any]],
    best_name: str,
    feature_names: List[str],
    split_desc: str,
    temporal_desc: str,
    train_rows: int,
    test_rows: int,
) -> None:
    """Write the comparison of all models, with details for the selected one."""
    best = results[best_name]
    matrix = best["spatial"]["confusion_matrix"]
    lines = [
        "# AgroScope Model Evaluation Report",
        "",
        "## Validation",
        f"- **Unseen cells:** {split_desc}",
        f"- **Later dates:** {temporal_desc}",
        f"- **Unseen-cell rows:** `{train_rows:,}` train, `{test_rows:,}` test",
        "- **Known limitation:** labels are calculated from `ndwi_mean`, `rainfall_8day` and `et0_8day_sum`, which are also model inputs. These scores show how well a model reproduces that formula, not real-world accuracy. The last score column removes those inputs.",
        "",
        "## Model comparison (macro F1)",
        "",
        "| Model | Role | Unseen cells | Later dates | Later dates without label inputs |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for name, result in results.items():
        marker = " **(selected)**" if name == best_name else ""
        lines.append(
            f"| {MODEL_SPECS[name]['title']}{marker} | {MODEL_SPECS[name]['role']} | {result['spatial']['macro_f1']} "
            f"| {result['temporal']['macro_f1']} | {result['temporal_no_label_inputs']['macro_f1']} |"
        )
    lines += [
        "",
        f"The candidate is chosen on the later-dates score: {MODEL_SPECS[best_name]['title']}.",
        "",
        f"## Confusion matrix on unseen cells ({MODEL_SPECS[best_name]['title']})",
        "",
        "| Actual \\ Predicted | LOW | MODERATE | HIGH |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for label, row in zip(("LOW", "MODERATE", "HIGH"), matrix):
        lines.append(f"| **{label}** | `{row[0]:,}` | `{row[1]:,}` | `{row[2]:,}` |")
    lines += [
        "",
        f"## Top features ({MODEL_SPECS[best_name]['title']})",
        "",
        "| Rank | Feature | Share of importance |",
        "| :--- | :--- | :--- |",
    ]
    for rank, (feature, importance) in enumerate(ranked_importances(best["model"], feature_names)[:10], start=1):
        lines.append(f"| {rank} | `{feature}` | `{importance:.4f}` |")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Evaluation report saved to: %s", report_file)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AgroScope Crop Water Stress ML Models.")
    parser.add_argument("--strategy", type=str, default="spatial", choices=["spatial", "temporal"], help="Validation strategy: 'spatial' block (default) or 'temporal'")
    parser.add_argument("--sample-size", type=int, default=None, help="Downsample rows for rapid testing (e.g., 50000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(strategy=args.strategy, sample_size=args.sample_size)
