"""Trình tải cấu hình dựa trên YAML có hỗ trợ gộp (merge)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Gộp đệ quy *override* vào *base* (in-place) và trả về *base*."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    default_path: str = "config/default.yaml",
    paths_path: str = "config/paths.yaml",
    override_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Tải và gộp các file cấu hình YAML.

    Thứ tự ưu tiên (cao nhất thắng): override > paths > default.
    """
    root = Path(__file__).resolve().parents[2]  # project root

    def _load(p: str) -> Dict[str, Any]:
        full = Path(p) if Path(p).is_absolute() else root / p
        if not full.exists():
            return {}
        with open(full) as f:
            return yaml.safe_load(f) or {}

    cfg = _load(default_path)
    paths_cfg = _load(paths_path)
    _deep_merge(cfg, paths_cfg)

    if override_path:
        _deep_merge(cfg, _load(override_path))

    return cfg
