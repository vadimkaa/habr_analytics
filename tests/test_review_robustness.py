"""Контроль отчёта об устойчивости без изменения базового протокола."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from habrpub.analysis import load_corpus
from habrpub import legacy_model

ROOT = Path(__file__).resolve().parents[1]


def _review_module():
    spec = importlib.util.spec_from_file_location(
        "review_robustness", ROOT / "16_review_robustness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_and_catboost_shared_features_match_on_real_subset():
    from experiments.catboost.src import features

    corpus = load_corpus("ai").iloc[:180].copy()
    base = legacy_model.prepare(corpus)
    cb = features.prepare(corpus)
    shared = [
        "prior_count", "has_prior", "hour_sin", "hour_cos", "is_holiday",
        "t_len", "t_words", "t_digit", "t_qmark", "t_colon", "is_weekend", "is_corp",
    ]
    for col in shared:
        assert np.allclose(base[col], cb[col], equal_nan=True), col


def test_review_report_has_calendar_and_episode_checks():
    result = json.loads((ROOT / "data/results/16_review_robustness.json").read_text())
    for corpus in ("ai", "wide"):
        item = result["corpora"][corpus]
        reg = item["regression"]
        assert 1 < reg["age_bands"]["ratio_views_plus1"] < 1.5
        assert 1 < reg["quarter_calendar"]["ratio_views_plus1"] < 1.5
        assert item["holidays"]["loo_min_pp"] > 0
        assert item["holidays"]["loo_max_pp"] >= item["holidays"]["loo_min_pp"]
        assert len(item["holidays"]["leave_one_episode_out"]) == item["holidays"]["n_episodes"]
    assert result["corpora"]["wide"]["holidays"]["n_episodes"] == 7


def test_review_baseline_matches_published_regression():
    # Небольшой тест точного контракта отчёта со штатной регрессией 09.
    prior = json.loads((ROOT / "data/results/09_robustness_results.json").read_text())
    result = json.loads((ROOT / "data/results/16_review_robustness.json").read_text())
    for corpus in ("ai", "wide"):
        current = result["corpora"][corpus]["regression"]["baseline"]
        expected = prior["corpora"][corpus]["adjusted_weekend"]
        assert abs(current["ratio_views_plus1"] - expected["ratio_target_plus1"]) < 1e-6
        assert np.allclose(current["ci95_cluster_author"], expected["ratio_ci95"], atol=1e-6)
