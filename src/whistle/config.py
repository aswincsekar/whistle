"""Typed config loader. Reads configs/default.yaml into a SimpleNamespace tree."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _to_ns(raw)
    cfg._raw = raw  # type: ignore[attr-defined]
    cfg._path = str(path)  # type: ignore[attr-defined]
    return cfg


def repo_root() -> Path:
    """Repo root = parent of src/."""
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()
