"""Load the YAML config and resolve project paths."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    paths["data_dir"] = REPO_ROOT / paths["data_dir"]
    paths["results_dir"] = REPO_ROOT / paths["results_dir"]
    paths["raw_dir"] = paths["data_dir"] / "raw"
    paths["processed_dir"] = paths["data_dir"] / "processed"
    return cfg
