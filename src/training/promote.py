"""AgroScope Champion-Challenger Promotion Gate.

Compares the newly trained candidate with the current champion on the date-based
holdout score, then exports whichever model wins to models/ for prediction.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow
from mlflow.exceptions import MlflowException

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.training.tracking import TRACKING, get_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("agroscope.promotion")

PROMOTION = settings["promotion"]
GATE_METRIC = PROMOTION["metric"]
CHAMPION_ALIAS = "champion"
EXPORTED_FILES = ("model.pkl", "feature_metadata.json")


def decide_promotion(candidate_score: float, champion_score: Optional[float], min_improvement: float, min_score: float) -> Tuple[bool, str]:
    """Promote a candidate only if it reaches the minimum, and either there's no champion or it scores at least as well."""
    if candidate_score < min_score:
        return False, f"Candidate {candidate_score:.4f} is below the minimum score of {min_score:.4f}."
    if champion_score is None:
        return True, "No champion exists yet."
    required = champion_score + min_improvement
    if candidate_score >= required:
        return True, f"Candidate {candidate_score:.4f} meets the required {required:.4f}."
    return False, f"Candidate {candidate_score:.4f} is below the required {required:.4f}."


def run_promotion_gate(
    candidate_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    decision_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Register the candidate, apply the gate and export the winning model."""
    candidate_dir = candidate_dir or (PROJECT_ROOT / "models" / "candidate")
    output_dir = output_dir or (PROJECT_ROOT / "models")
    decision_path = decision_path or (PROJECT_ROOT / "reports" / "promotion_decision.json")

    with open(candidate_dir / "feature_metadata.json", "r", encoding="utf-8") as f:
        candidate_metadata = json.load(f)
    candidate_run_id = candidate_metadata["mlflow_run_id"]

    client = get_client()
    registered_name = TRACKING["registered_model"]
    try:
        client.create_registered_model(registered_name)
    except MlflowException:
        logger.info("Registered model %s already exists.", registered_name)

    candidate_run = client.get_run(candidate_run_id)
    candidate_score = candidate_run.data.metrics[GATE_METRIC]

    champion = None
    champion_score = None
    try:
        champion = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
        champion_score = client.get_run(champion.run_id).data.metrics[GATE_METRIC]
    except MlflowException:
        logger.info("No champion is registered yet.")

    existing = client.search_model_versions(f"name='{registered_name}' and run_id='{candidate_run_id}'")
    version = existing[0] if existing else client.create_model_version(
        registered_name,
        source=f"{candidate_run.info.artifact_uri}/model_files",
        run_id=candidate_run_id,
    )

    is_current_champion = champion is not None and champion.run_id == candidate_run_id
    promoted, reason = decide_promotion(candidate_score, None if is_current_champion else champion_score, PROMOTION["min_improvement"], PROMOTION["min_score"])
    if promoted:
        client.set_registered_model_alias(registered_name, CHAMPION_ALIAS, version.version)

    version_tags = {
        "gate_decision": "promoted" if promoted else "rejected",
        "gate_reason": reason,
        "model_type": candidate_metadata.get("model_type", "unknown"),
        GATE_METRIC: f"{candidate_score:.4f}",
        "min_score": str(PROMOTION["min_score"]),
        "data_md5": candidate_run.data.tags.get("data_md5", "unknown"),
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(registered_name, version.version, key, value)
    client.update_model_version(registered_name, version.version, description=f"{version_tags['model_type']}: {reason}")
    logger.info("%s %s", "PROMOTED:" if promoted else "REJECTED:", reason)

    decision = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metric": GATE_METRIC,
        "min_score": PROMOTION["min_score"],
        "candidate_model": version_tags["model_type"],
        "candidate_version": int(version.version),
        "candidate_score": round(candidate_score, 4),
        "previous_champion_version": int(champion.version) if champion else None,
        "previous_champion_score": round(champion_score, 4) if champion_score is not None else None,
        "promoted": promoted,
        "reason": reason,
        "active_version": None,
    }
    decision_path.parent.mkdir(parents=True, exist_ok=True)

    # With no champion to fall back on, a rejected first model leaves nothing safe to publish.
    if not promoted and champion is None:
        with open(decision_path, "w", encoding="utf-8") as f:
            json.dump(decision, f, indent=2)
        raise RuntimeError(f"No model can be published: {reason}")

    winner = client.get_model_version_by_alias(registered_name, CHAMPION_ALIAS)
    downloaded = Path(mlflow.artifacts.download_artifacts(run_id=winner.run_id, artifact_path="model_files"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPORTED_FILES:
        shutil.copy2(downloaded / name, output_dir / name)
    logger.info("Exported champion version %s to %s.", winner.version, output_dir)

    decision["active_version"] = int(winner.version)
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    return decision


if __name__ == "__main__":
    run_promotion_gate()
