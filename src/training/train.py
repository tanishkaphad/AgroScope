"""AgroScope Phase 10 & 11 — Machine Learning Training & Validation Pipeline.

Implements the complete ML training workflow:
1. Loads unified features dataset (data/features/features.csv).
2. Filters for active agricultural vegetation (is_vegetated == True).
3. Strictly isolates input features (satellite indices + bands + weather) from target labels
   (excludes water_deficit_8day_mm to prevent data leakage).
4. Implements Temporal Holdout Validation:
   - Train Set: Oct 2023 – Jan 2024 (Early & mid Rabi season)
   - Test Set: Feb 2024 – Mar 2024 (Late Rabi maturity season)
   - Prevents spatial-temporal autocorrelation leakage across 8-day cycles.
5. Balances class weights to address 88% / 8% / 4% class distribution.
6. Trains:
   - Baseline: Balanced Random Forest Classifier
   - Main Model: Tuned XGBoost Classifier
7. Computes evaluation benchmarks:
   - Macro-averaged F1 Score
   - Per-class Recall (especially High & Moderate Stress Recall)
   - Confusion Matrix
   - Feature Importances
8. Persists serialized models and metadata to models/ for backend API inference.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pickle
import numpy as np
import pandas as pd
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


def train_baseline_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = TRAINING["random_forest"]["n_estimators"],
    max_depth: int = TRAINING["random_forest"]["max_depth"],
) -> RandomForestClassifier:
    """Train class-balanced Random Forest baseline."""
    logger.info("Training Baseline Random Forest (n_estimators=%d, max_depth=%d)...", n_estimators, max_depth)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest baseline training completed.")
    return rf


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = TRAINING["xgboost"]["n_estimators"],
    max_depth: int = TRAINING["xgboost"]["max_depth"],
    learning_rate: float = TRAINING["xgboost"]["learning_rate"],
) -> XGBClassifier:
    """Train weighted XGBoost classifier."""
    logger.info("Training Main Model: XGBoost Classifier (n_estimators=%d, max_depth=%d, lr=%.2f)...", n_estimators, max_depth, learning_rate)

    # Compute sample weights to balance class frequencies
    sample_weights = compute_sample_weight("balanced", y_train)

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=TRAINING["xgboost"]["subsample"],
        colsample_bytree=TRAINING["xgboost"]["colsample_bytree"],
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    logger.info("XGBoost training completed.")
    return xgb


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
    """Execute complete Phase 10 & 11 training and evaluation pipeline."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    if features_csv is None:
        features_csv = base_dir / "data" / "features" / "features.csv"
    if output_dir is None:
        output_dir = base_dir / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, feature_names, split_desc = prepare_datasets(
        features_csv=features_csv,
        strategy=strategy,
        sample_size=sample_size,
    )

    # 1. Train Baseline Random Forest
    rf_model = train_baseline_random_forest(X_train, y_train)
    rf_eval = evaluate_model(rf_model, X_test, y_test, model_name="Random Forest Baseline")

    # 2. Train Main XGBoost Classifier
    xgb_model = train_xgboost_classifier(X_train, y_train)
    xgb_eval = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost Classifier")

    # 3. Extract Feature Importances from XGBoost
    importances = xgb_model.feature_importances_
    feat_imp = sorted(zip(feature_names, [float(x) for x in importances]), key=lambda x: x[1], reverse=True)

    # 4. Save Serialized Models & Inference Metadata
    rf_path = output_dir / "baseline_random_forest.pkl"
    xgb_json_path = output_dir / "xgboost_model.json"
    xgb_pkl_path = output_dir / "xgboost_model.pkl"
    meta_path = output_dir / "feature_metadata.json"

    with open(rf_path, "wb") as f:
        pickle.dump(rf_model, f)

    with open(xgb_pkl_path, "wb") as f:
        pickle.dump(xgb_model, f)
    xgb_model.save_model(xgb_json_path)

    logger.info("Saved models to:\n  %s\n  %s\n  %s", rf_path, xgb_pkl_path, xgb_json_path)

    metadata = {
        "features": feature_names,
        "classes": [0, 1, 2],
        "class_labels": ["LOW", "MODERATE", "HIGH"],
        "validation_strategy": split_desc,
        "metrics": {
            "random_forest": rf_eval,
            "xgboost": xgb_eval,
        },
        "top_features": feat_imp[:10],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 5. Generate Model Evaluation Report
    report_file = reports_dir / "model_evaluation_report.md"
    lines = [
        "# AgroScope Phase 12 — Model Training & Evaluation Report",
        "",
        "## 1. Validation Strategy",
        f"- **Method:** {split_desc}",
        f"- **Train Observations:** `{len(X_train):,}`",
        f"- **Holdout Test Observations:** `{len(X_test):,}`",
        "- **Known Limitation:** Labels are calculated from `ndwi_mean`, `rainfall_8day` and `et0_8day_sum`, which are also model inputs. These scores show how well the model reproduces that formula, not real-world accuracy.",
        "",
        "---",
        "",
        "## 2. Model Performance Benchmarks",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC (OvR) | Low Recall | Mod Recall | High Recall |",
        "|:---|:---|:---|:---|:---|:---|:---|:---|",
        f"| **Random Forest Baseline** | **{rf_eval['accuracy']*100:.2f}%** | {rf_eval['macro_f1']} | {rf_eval['weighted_f1']} | {rf_eval['roc_auc_ovr']} | {rf_eval['classification_report']['0: LOW']['recall']*100:.1f}% | {rf_eval['classification_report']['1: MODERATE']['recall']*100:.1f}% | {rf_eval['classification_report']['2: HIGH']['recall']*100:.1f}% |",
        f"| **XGBoost Classifier (Main)** | **{xgb_eval['accuracy']*100:.2f}%** | {xgb_eval['macro_f1']} | {xgb_eval['weighted_f1']} | {xgb_eval['roc_auc_ovr']} | {xgb_eval['classification_report']['0: LOW']['recall']*100:.1f}% | {xgb_eval['classification_report']['1: MODERATE']['recall']*100:.1f}% | {xgb_eval['classification_report']['2: HIGH']['recall']*100:.1f}% |",
        "",
        "---",
        "",
        "## 3. Confusion Matrix (XGBoost Classifier)",
        "",
        "| Actual \\ Predicted | Predicted LOW | Predicted MODERATE | Predicted HIGH |",
        "|:---|:---|:---|:---|",
        f"| **Actual LOW** | `{xgb_eval['confusion_matrix'][0][0]:,}` | `{xgb_eval['confusion_matrix'][0][1]:,}` | `{xgb_eval['confusion_matrix'][0][2]:,}` |",
        f"| **Actual MODERATE** | `{xgb_eval['confusion_matrix'][1][0]:,}` | `{xgb_eval['confusion_matrix'][1][1]:,}` | `{xgb_eval['confusion_matrix'][1][2]:,}` |",
        f"| **Actual HIGH** | `{xgb_eval['confusion_matrix'][2][0]:,}` | `{xgb_eval['confusion_matrix'][2][1]:,}` | `{xgb_eval['confusion_matrix'][2][2]:,}` |",
        "",
        "---",
        "",
        "## 4. Top Predictive Features (XGBoost Gain)",
        "",
        "| Rank | Feature | Relative Importance | Domain Meaning |",
        "|:---|:---|:---|:---|",
    ]

    for rank, (feat, imp) in enumerate(feat_imp[:10], start=1):
        meaning = "Satellite Canopy Water" if "ndwi" in feat else ("Vegetation Vigor" if "ndvi" in feat or "evi" in feat else "Weather / Evaporative Demand")
        lines.append(f"| {rank} | `{feat}` | `{imp:.4f}` | {meaning} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Model Artifacts Ready for API Deployment",
        f"- `{xgb_pkl_path.name}` & `{xgb_json_path.name}` — Optimized multi-class classifier ready for FastAPI containerization.",
        f"- `{meta_path.name}` — Schema and feature expectations for payload validation.",
    ])

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Evaluation report saved to: %s", report_file)

    # Print summary highlights
    print("\n" + "=" * 75)
    print("AGROSCOPE PHASE 10 & 11 — MODEL TRAINING SUMMARY")
    print("=" * 75)
    print(f"Validation Strategy:     {split_desc}")
    print("-" * 75)
    print("Random Forest Baseline:")
    print(f"  Accuracy:              {rf_eval['accuracy']*100:.2f}%")
    print(f"  Macro F1 Score:        {rf_eval['macro_f1']:.4f}")
    print(f"  Multi-class ROC-AUC:   {rf_eval['roc_auc_ovr']:.4f}")
    print("XGBoost Classifier (Main):")
    print(f"  Accuracy:              {xgb_eval['accuracy']*100:.2f}%")
    print(f"  Macro F1 Score:        {xgb_eval['macro_f1']:.4f}")
    print(f"  Multi-class ROC-AUC:   {xgb_eval['roc_auc_ovr']:.4f}")
    print("-" * 75)
    print("Top 5 Predictive Features (Satellite & Climate Interaction):")
    for f_name, f_val in feat_imp[:5]:
        print(f"  {f_name:<22}: {f_val:.4f}")
    print("=" * 75 + "\n")

    return xgb_pkl_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train AgroScope Crop Water Stress ML Models.")
    parser.add_argument("--strategy", type=str, default="spatial", choices=["spatial", "temporal"], help="Validation strategy: 'spatial' block (default) or 'temporal'")
    parser.add_argument("--sample-size", type=int, default=None, help="Downsample rows for rapid testing (e.g., 50000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(strategy=args.strategy, sample_size=args.sample_size)
