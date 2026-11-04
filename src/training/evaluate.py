"""AgroScope Model Evaluation Step.

Reads the candidate's scores and records whether they pass the minimum, for the pipeline
and for CI. The promotion gate makes the release decision; this records the quality check.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
logger = logging.getLogger("agroscope.evaluation")

PROMOTION = settings["promotion"]


def evaluate_candidate(metadata: Dict[str, Any], min_score: float) -> Dict[str, Any]:
    """Compare the selected model's later-dates score with the minimum."""
    selected = metadata["selected_model"]
    score = metadata["temporal_metrics"][selected]["macro_f1"]
    return {
        "selected_model": selected,
        "metric": PROMOTION["metric"],
        "score": score,
        "min_score": min_score,
        "passed": score >= min_score,
        "scores_by_model": {
            name: {
                "unseen_cells_macro_f1": metadata["metrics"][name]["macro_f1"],
                "later_dates_macro_f1": metadata["temporal_metrics"][name]["macro_f1"],
                "later_dates_without_label_inputs_macro_f1": metadata["temporal_metrics_without_label_inputs"][name]["macro_f1"],
            }
            for name in metadata["metrics"]
        },
    }


def run_evaluation(metadata_path: Optional[Path] = None, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the candidate's metadata and save the result."""
    metadata_path = metadata_path or (PROJECT_ROOT / "models" / "candidate" / "feature_metadata.json")
    output_path = output_path or (PROJECT_ROOT / "reports" / "evaluation_result.json")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    result = evaluate_candidate(metadata, PROMOTION["min_score"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "%s: %s scored %.4f against a minimum of %.4f.",
        "PASSED" if result["passed"] else "FAILED",
        result["selected_model"], result["score"], result["min_score"],
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record whether the candidate's scores pass the minimum.")
    parser.add_argument("--metadata", type=Path, default=None, help="Metadata file to read, instead of the candidate's")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the result")
    parser.add_argument("--strict", action="store_true", help="Exit with an error when the check fails, as CI needs")
    args = parser.parse_args()

    outcome = run_evaluation(args.metadata, args.output)
    if args.strict and not outcome["passed"]:
        sys.exit(1)
