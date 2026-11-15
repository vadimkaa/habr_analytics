"""Финалисты и победитель только по validation, без аудита.

Проверяется: финалисты выводятся из validation-метрик по правилу протокола;
L05 участвует на общих основаниях; аудит не влияет на выбор; контроль LightGBM
в отборе не участвует; победитель в policy.json берётся из заморозки, а не по
максимуму delta_rho на аудите.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src import selection as SEL


def _val(rho, mae=1.0):
    return {
        "validation": {
            "rho_within_month_mean": rho,
            "rho_all": rho * 0.8,
            "mae_log": mae,
            "baseline_mae_log": 1.05,
        },
        "final_audit": {"rho_within_month_mean": None, "delta_rho": None},
    }


def _summary():
    return {
        "ai:L01_rmse_log1p": _val(0.087),
        "ai:L02_quantile_views_1000": _val(0.089),
        "ai:L03_poisson_views_1000": _val(0.079),
        "ai:L05_quantile_log1p": _val(0.107),
        "wide:L01_rmse_log1p": _val(0.326),
        "wide:L02_quantile_views_1000": _val(0.299),
        "wide:L05_quantile_log1p": _val(0.311),
        "ai:lgb_poisson_v24": _val(0.031),
        "wide:lgb_rmse_log1p": _val(0.274),
    }


def test_finalists_from_validation_only():
    sel = SEL.derive_selection(_summary())
    # ai: L05 лучший (0.107); L02 в пределах tie_gap 0.02 (0.089); L01 ровно 0.02 - не в gap
    assert sel["finalists_by_corpus"]["ai"] == [
        "L05_quantile_log1p",
        "L02_quantile_views_1000",
    ]
    # wide: L01 лучший; L05 в пределах 0.015; L02 - вне
    assert sel["finalists_by_corpus"]["wide"] == [
        "L01_rmse_log1p",
        "L05_quantile_log1p",
    ]
    assert sel["finalists"] == [
        "L05_quantile_log1p",
        "L02_quantile_views_1000",
        "L01_rmse_log1p",
    ]
    assert sel["winner_by_corpus"] == {
        "ai": "L05_quantile_log1p",
        "wide": "L01_rmse_log1p",
    }


def test_audit_metrics_do_not_affect_selection():
    """Изменение результатов аудита не может повлиять на выбор."""
    summary = _summary()
    sel1 = SEL.derive_selection(summary)
    # Аудит «шумит» в пользу L02 на обоих корпусах
    for key in ("ai:L02_quantile_views_1000", "wide:L02_quantile_views_1000"):
        summary[key]["final_audit"] = {
            "rho_within_month_mean": 0.999,
            "delta_rho": 0.9,
            "mae_log": 0.1,
        }
    summary["ai:L05_quantile_log1p"]["final_audit"] = {
        "rho_within_month_mean": 0.0,
        "delta_rho": -0.5,
    }
    sel2 = SEL.derive_selection(summary)
    assert sel1["finalists"] == sel2["finalists"]
    assert sel1["winner_by_corpus"] == sel2["winner_by_corpus"]


def test_l05_same_rules_as_others():
    """L05_quantile_log1p не исключён и стал победителем ai как лучший по validation."""
    sel = SEL.derive_selection(_summary())
    assert "L05_quantile_log1p" in sel["finalists"]
    assert sel["winner_by_corpus"]["ai"] == "L05_quantile_log1p"


def test_library_control_never_selected():
    """Даже лучший LightGBM-контроль не становится финалистом или победителем."""
    summary = _summary()
    summary["ai:lgb_poisson_v24"] = _val(0.99)  # формально лучший на validation
    sel = SEL.derive_selection(summary)
    assert "lgb_poisson_v24" not in sel["finalists"]
    assert sel["winner_by_corpus"]["ai"] != "lgb_poisson_v24"


def test_tie_break_prefers_simpler_wave():
    """Ничья внутри tie_gap: волна 1 (проще) впереди волны 2 при равной метрике."""
    summary = _summary()
    summary["ai:L05_quantile_log1p"] = _val(0.107)
    summary["ai:L01_rmse_log1p"] = _val(0.107)  # равные на validation
    sel = SEL.derive_selection(summary)
    assert sel["winner_by_corpus"]["ai"] == "L01_rmse_log1p"


def test_selected_policies_roundtrip(tmp_path, monkeypatch):
    """Заморозка пишется в файл и читается; победитель зафиксирован до аудита."""
    monkeypatch.setattr(
        config, "SELECTED_POLICIES_JSON", tmp_path / "selected_policies.json"
    )
    monkeypatch.setattr(
        SEL,
        "build_selection_doc",
        lambda corpora=None: {
            "protocol": "cb-research-v1",
            "frozen_at": "2026-11-13 00:00:00",
            "finalists": ["L05_quantile_log1p", "L01_rmse_log1p"],
            "finalists_by_corpus": {
                "ai": ["L05_quantile_log1p"],
                "wide": ["L01_rmse_log1p"],
            },
            "winner_by_corpus": {"ai": "L05_quantile_log1p", "wide": "L01_rmse_log1p"},
            "policy_freeze": {"q_fixed": 0.2},
            "winner_locked_before_audit": True,
        },
    )
    doc = SEL.write_selected_policies(["ai", "wide"])
    assert doc["winner_locked_before_audit"] is True
    loaded = SEL.load_selection()
    assert loaded["winner_by_corpus"]["ai"] == "L05_quantile_log1p"


def test_load_selection_requires_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SELECTED_POLICIES_JSON", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        SEL.load_selection()


def test_policy_json_winner_locked_not_max_audit(tmp_path, monkeypatch):
    """policy.json берёт победителя из заморозки: большой delta_rho другой
    модели на аудите не меняет победителя."""
    from src import config as _config  # noqa: F401
    from stages import policy

    monkeypatch.setattr(config, "POLICY_JSON", tmp_path / "policy.json")
    sel = {
        "winner_by_corpus": {"ai": "L05_quantile_log1p"},
        "finalists_by_corpus": {"ai": ["L05_quantile_log1p", "L01_rmse_log1p"]},
    }
    report = {
        "ai": {
            "L05_quantile_log1p|density=none": {
                "thresholds_train": {"q_fixed": 0.2, "terciles": [1.0, 2.0]},
                "delta_rho": -0.05,
                "coverage": 0.86,
            },
            "L01_rmse_log1p|density=none": {
                "thresholds_train": {"q_fixed": 0.2, "terciles": [1.0, 2.0]},
                "delta_rho": 0.95,
                "coverage": 0.86,  # «победитель» по аудиту
            },
        }
    }
    policy.write_policy_json(report, sel)
    out = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    assert out["corpora"]["ai"]["model"] == "L05_quantile_log1p|density=none"
    assert out["winner_locked_before_audit"] is True
    assert out["corpora"]["ai"]["audit_delta_rho_informational"] == -0.05
