"""AgroScope Evaluation Visualization & Figure Generator.

Generates presentation-ready, high-resolution figures from models/feature_metadata.json:
1. Confusion matrix heatmaps for every model
2. Feature importance for the selected model
3. Comparison of every model on all three tests
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

MODEL_TITLES = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
# Fixed so a model keeps its colour between charts. Checked for colour-blind separation.
MODEL_COLORS = {"random_forest": "#2a78d6", "xgboost": "#eb6834", "lightgbm": "#1baf7a"}


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
    # 1. Confusion Matrix Heatmaps
    # -------------------------------------------------------------
    for model_key, model_data in metrics.items():
        title_name = MODEL_TITLES.get(model_key, model_key)
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

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
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
    # 2. Top Feature Importances, for the selected model
    # -------------------------------------------------------------
    top_features = meta.get("top_features", [])
    selected = meta.get("model_type", "model")
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

        ax.set_title(f"{MODEL_TITLES.get(selected, selected)}: Top Predictive Features", pad=15, weight="bold")
        ax.set_xlabel("Share of feature importance", labelpad=10, weight="semibold")
        ax.set_xlim(0, max(feat_scores) * 1.18)

        # Custom legend for domain categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ca02c", edgecolor="black", label="Satellite Biophysical / Spectral"),
            Patch(facecolor="#1f77b4", edgecolor="black", label="Agro-Meteorological / Climate"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True)
        plt.tight_layout()

        feat_file = output_dir / f"feature_importance_{selected}.png"
        fig.savefig(feat_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved: %s", feat_file)
        generated_paths.append(feat_file)

    # -------------------------------------------------------------
    # 3. Every model on all three tests. The axis starts at zero, so the fall between tests is visible.
    # -------------------------------------------------------------
    temporal = meta.get("temporal_metrics", {})
    no_label_inputs = meta.get("temporal_metrics_without_label_inputs", {})
    if temporal and no_label_inputs:
        tests = ["Unseen cells", "Later dates", "Later dates,\nwithout label inputs"]
        models = list(metrics)
        x = np.arange(len(tests))
        width = 0.26

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
        for index, model in enumerate(models):
            scores = [metrics[model]["macro_f1"], temporal[model]["macro_f1"], no_label_inputs[model]["macro_f1"]]
            selected_note = " (selected)" if model == selected else ""
            bars = ax.bar(
                x + (index - (len(models) - 1) / 2) * width,
                scores,
                width * 0.92,
                label=f"{MODEL_TITLES.get(model, model)}{selected_note}",
                color=MODEL_COLORS.get(model),
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color="#52514e")

        ax.set_ylabel("Macro F1", labelpad=10)
        ax.set_title("Scores fall once the test gets harder", pad=15, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(tests)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(models), frameon=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        fig.text(0.01, -0.03, "Measured against formula-based labels, not field observations.", fontsize=9, color="#52514e")
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
