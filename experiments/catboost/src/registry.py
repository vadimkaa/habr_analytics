"""Реестр экспериментов: один запуск - одна строка + JSON.

Колонки реестра соответствуют протоколу; статус различает exploratory /
validation / audit_exposed. audit_exposed обозначает тестовую выборку, на которой уже изучались
результаты и поэтому отсутствует независимое подтверждение.
Повторный запуск с теми же входами даёт идемпотентные артефакты: новый
run_id не стирает прежние результаты.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np

from . import config

REGISTRY_COLUMNS = [
    "run_id",
    "protocol_version",
    "code_hash",
    "input_sha256",
    "created_at",
    "corpus",
    "split_id",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
    "test_start",
    "test_end",
    "n_train",
    "n_val",
    "n_test",
    "loss_function",
    "target_transform",
    "model_family",
    "cat_mode",
    "feature_set",
    "density_mode",
    "seed",
    "params_json",
    "fit_time_sec",
    "best_iteration",
    "rho_all",
    "rho_within_month",
    "mae_log",
    "bias_log",
    "r2_log",
    "delta_rho",
    "delta_rho_ci95",
    "top_q_ratio",
    "coverage",
    "ci_method",
    "status",
    "stage",
]


def code_sources() -> list[Path]:
    """Files that define an experiment (including the stage entry points)."""
    root = config.REPO
    return [
        *sorted((config.CATBOOST_DIR / "src").glob("*.py")),
        *sorted((config.CATBOOST_DIR / "stages").glob("*.py")),
        *sorted((config.CATBOOST_DIR / "configs").glob("*.yaml")),
        config.CATBOOST_DIR / "run_experiments.py",
        root / "habrpub" / "legacy_model.py",
        root / "habrpub" / "analysis.py",
        root / "habrpub" / "calendar_analysis.py",
    ]


def code_hash() -> str:
    """Short SHA-256 digest of the computation code and configurations.

    Include relative paths as well as file bytes to detect both moves and edits.
    """
    h = hashlib.sha256()
    root = config.REPO
    for path in code_sources():
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _finite(v):
    if v is None:
        return ""
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else ""
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def append_registry_row(row: dict) -> str:
    """Добавить строку реестра (файл дописывается, история не стирается)."""
    run_id = row.get("run_id") or _make_run_id(row)
    row = dict(row)
    row.setdefault("protocol_version", config.protocol()["protocol"])
    row.setdefault("code_hash", code_hash())
    row.setdefault("input_sha256", json.dumps(config.input_hashes()))
    row.setdefault("created_at", dt.datetime.now().isoformat(timespec="seconds"))
    row.setdefault("status", config.STATUS_EXPLORATORY)
    row["run_id"] = run_id
    path = config.REGISTRY_CSV
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: _finite(row.get(k)) for k in REGISTRY_COLUMNS})
    return run_id


def _make_run_id(row: dict) -> str:
    seed_src = "|".join(
        str(row.get(k, ""))
        for k in (
            "stage",
            "corpus",
            "split_id",
            "loss_function",
            "model_family",
            "feature_set",
            "cat_mode",
            "density_mode",
            "seed",
        )
    )
    return hashlib.sha256(seed_src.encode()).hexdigest()[:10]


def save_metrics(name: str, payload: dict) -> None:
    """JSON со всеми метриками прогона (allow_nan=False: NaN -> None)."""

    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (float, np.floating)):
            return float(v) if np.isfinite(v) else None
        return v

    path = config.METRICS_DIR / f"{name}.json"
    path.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_metrics(name: str) -> dict:
    """Прочитать сохранённые метрики (для слияния волн экспериментов)."""
    path = config.METRICS_DIR / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
