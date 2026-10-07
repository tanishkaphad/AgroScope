"""AgroScope Evaluation Visualization & Figure Generator.

Generates presentation-ready, high-resolution figures from models/feature_metadata.json:
1. Confusion Matrix Heatmaps (XGBoost & Random Forest)
2. Feature Importance Horizontal Bar Chart
3. Model Benchmark Comparison Bar Chart
All figures are saved to reports/figures/ for documentation and presentation.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.visualization")


def generate_evaluation_figures(
    metadata_json: Path,
    output_dir: Path,
) -> List[Path]:
    """Generate and save publication-quality evaluation figures."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set publication aesthetic style
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 16,
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: List[Path] = []

    with open(metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    metrics = meta.get("metrics", {})
    class_labels = meta.get("class_labels", ["LOW", "MODERATE", "HIGH"])

    # -------------------------------------------------------------
    # 1. Confusion Matrix Heatmaps (XGBoost & Random Forest)
    # -------------------------------------------------------------
    for model_key, title_name in [("xgboost", "XGBoost Classifier"), ("random_forest", "Random Forest Baseline")]:
        model_data = metrics.get(model_key, {})
        cm = np.array(model_data.get("confusion_matrix", []))
        if cm.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        
        # Calculate percentage per actual class (row normalization)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        # Annotations with both count and percentage
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"

        palette = "Blues" if model_key == "xgboost" else "Greens"
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap=palette,
            cbar=True,
            xticklabels=[f"Pred {c}" for c in class_labels],
            yticklabels=[f"Actual {c}" for c in class_labels],
            ax=ax,
            linewidths=1,
            linecolor="white",
            annot_kws={"size": 11, "weight": "bold"},
        )

        acc = model_data.get("accuracy", 0.0) * 100
        macro_f1 = model_data.get("macro_f1", 0.0)
        ax.set_title(f"{title_name} — Confusion Matrix\n(Accuracy: {acc:.2f}% | Macro F1: {macro_f1:.4f})", pad=15, weight="bold")
        ax.set_ylabel("Ground Truth Class", labelpad=10, weight="semibold")
        ax.set_xlabel("Predicted Crop Water Stress Class", labelpad=10, weight="semibold")
        plt.tight_layout()

        out_file = output_dir / f"confusion_matrix_{model_key}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", out_file)
        generated_paths.append(out_file)

    # -------------------------------------------------------------
    # 2. Top Feature Importances (XGBoost)
    # -------------------------------------------------------------
    top_features = meta.get("top_features", [])
    if top_features:
        feat_names = [x[0] for x in top_features][::-1]
        feat_scores = [x[1] for x in top_features][::-1]

        # Categorize by feature domain
        colors = []
        for name in feat_names:
            if "ndvi" in name or "ndwi" in name or "evi" in name or name.startswith("b"):
                colors.append("#2ca02c")  # Green for satellite features
            else:
                colors.append("#1f77b4")  # Blue for weather / climate drivers

        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
        bars = ax.barh(feat_names, feat_scores, color=colors, edgecolor="black", linewidth=0.5, height=0.65)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.003, bar.get_y() + bar.get_height() / 2, f"{width:.4f}",
                    va="center", ha="left", fontsize=10, weight="semibold")

        ax.set_title("XGBoost Crop Water Stress — Top Predictive Features", pad=15, weight="bold")
        ax.set_xlabel("Relative Feature Importance (Gain / Weight)", labelpad=10, weight="semibold")
        ax.set_xlim(0, max(feat_scores) * 1.18)

        # Custom legend for domain categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ca02c", edgecolor="black", label="Satellite Biophysical / Spectral"),
            Patch(facecolor="#1f77b4", edgecolor="black", label="Agro-Meteorological / Climate"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)
        plt.tight_layout()

        feat_file = output_dir / "feature_importance_xgboost.png"
        fig.savefig(feat_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", feat_file)
        generated_paths.append(feat_file)

    # -------------------------------------------------------------
    # 3. Model Performance Comparison Chart
    # -------------------------------------------------------------
    rf = metrics.get("random_forest", {})
    xgb = metrics.get("xgboost", {})
    if rf and xgb:
        metric_names = ["Overall Accuracy", "Macro F1 Score", "Weighted F1", "Low Recall", "Mod Recall", "High Recall"]
        
        rf_vals = [
            rf.get("accuracy", 0.0) * 100,
            rf.get("macro_f1", 0.0) * 100,
            rf.get("weighted_f1", 0.0) * 100,
            rf.get("classification_report", {}).get("0: LOW", {}).get("recall", 0.0) * 100,
            rf.get("classification_report", {}).get("1: MODERATE", {}).get("recall", 0.0) * 100,
            rf.get("classification_report", {}).get("2: HIGH", {}).get("recall", 0.0) * 100,
        ]

        xgb_vals = [
            xgb.get("accuracy", 0.0) * 100,
            xgb.get("macro_f1", 0.0) * 100,
            xgb.get("weighted_f1", 0.0) * 100,
            xgb.get("classification_report", {}).get("0: LOW", {}).get("recall", 0.0) * 100,
            xgb.get("classification_report", {}).get("1: MODERATE", {}).get("recall", 0.0) * 100,
            xgb.get("classification_report", {}).get("2: HIGH", {}).get("recall", 0.0) * 100,
        ]

        x = np.arange(len(metric_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        rects1 = ax.bar(x - width/2, rf_vals, width, label="Random Forest Baseline", color="#55a868", edgecolor="black", linewidth=0.5)
        rects2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost Classifier (Main)", color="#4c72b0", edgecolor="black", linewidth=0.5)

        ax.set_ylabel("Score (%)", labelpad=10, weight="semibold")
        ax.set_title("Model Benchmark Comparison across Unseen Spatial Holdout Cells", pad=15, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, weight="semibold")
        ax.set_ylim(95, 101)  # Zoom in on high-performance range
        ax.legend(loc="lower left", frameon=True)

        for rects in [rects1, rects2]:
            for bar in rects:
                height = bar.get_height()
                ax.annotate(f"{height:.2f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, weight="bold")

        plt.tight_layout()
        comp_file = output_dir / "model_benchmark_comparison.png"
        fig.savefig(comp_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", comp_file)
        generated_paths.append(comp_file)

    print("\n" + "=" * 70)
    print("AGROSCOPE EVALUATION FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    for p in generated_paths:
        print(f"  • {p.relative_to(output_dir.parent.parent)}")
    print("=" * 70 + "\n")

    return generated_paths


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent.parent
    metadata_json = base_dir / "models" / "feature_metadata.json"
    output_dir = base_dir / "reports" / "figures"

    if not metadata_json.exists():
        logger.error("Metadata JSON not found at: %s. Please train models first.", metadata_json)
        sys.exit(1)

    generate_evaluation_figures(metadata_json=metadata_json, output_dir=output_dir)


if __name__ == "__main__":
    main()
