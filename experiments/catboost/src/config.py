"""Конфигурация протокола cb-research-v1 и общие константы нового ML-контура.

Модуль читает YAML-конфиги из configs/ и даёт пути к результатам.
Результаты эксперимента сохраняются в experiments/catboost/results/;
подготовленные входные данные основного проекта не изменяются.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# .../experiments/catboost/src/config.py -> .../experiments/catboost
CATBOOST_DIR = Path(__file__).resolve().parent.parent
# Корень репозитория habr_analytics
REPO = CATBOOST_DIR.parent.parent
RESULTS = CATBOOST_DIR / "results"
METRICS_DIR = RESULTS / "metrics"
MODELS_DIR = RESULTS / "models"
PREDICTIONS_DIR = RESULTS / "predictions"
FIGURES_DIR = RESULTS / "figures"
REGISTRY_CSV = RESULTS / "registry.csv"
POLICY_JSON = RESULTS / "policy.json"
SELECTED_POLICIES_JSON = RESULTS / "selected_policies.json"

for _d in (RESULTS, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 20261111
N_BOOT = 1000
THREAD_COUNT = 2
EARLY_STOPPING_ROUNDS = 100
ITERATIONS_CAP = 1000

STATUS_EXPLORATORY = "exploratory"
STATUS_VALIDATION = "validation"
STATUS_AUDIT_EXPOSED = "audit_exposed"


@dataclass(frozen=True)
class FoldSpec:
    name: str
    start: str | None
    end: str | None
    role: str


@lru_cache(maxsize=1)
def protocol() -> dict:
    return yaml.safe_load((CATBOOST_DIR / "configs" / "protocol_v1.yaml").read_text())


@lru_cache(maxsize=1)
def losses_cfg() -> dict:
    return yaml.safe_load((CATBOOST_DIR / "configs" / "losses.yaml").read_text())


@lru_cache(maxsize=1)
def splits_cfg() -> dict:
    return yaml.safe_load((CATBOOST_DIR / "configs" / "splits.yaml").read_text())


def folds(corpus: str) -> dict[str, FoldSpec]:
    out = {}
    for name, spec in splits_cfg()["corpora"][corpus]["folds"].items():
        out[name] = FoldSpec(name, spec.get("start"), spec.get("end"), spec["role"])
    return out


def rolling_folds(corpus: str) -> list[dict]:
    return splits_cfg()["corpora"][corpus].get("rolling", [])


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes() -> dict[str, str]:
    """SHA-256 входных выгрузок: реестр привязан к конкретному срезу данных."""
    manifest_path = REPO / "data" / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    declared_map = manifest.get("sha256", {})
    out = {}
    for corpus, rel in protocol()["inputs"]["corpora"].items():
        p = REPO / rel
        if not p.exists():
            # Фоллбэк релиза: данные в data/raw/
            alt = REPO / "data" / "raw" / Path(rel).name
            if alt.exists():
                p = alt
        declared = declared_map.get(p.name)
        actual = file_sha256(p)
        out[corpus] = actual
        out[corpus + "_declared_matches"] = (
            None if declared is None else bool(declared == actual)
        )
    return out
