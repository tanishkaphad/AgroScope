"""AgroScope pipeline settings, loaded from configs/settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


def load_settings(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read pipeline settings from a YAML file."""
    with open(path or SETTINGS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


settings = load_settings()
