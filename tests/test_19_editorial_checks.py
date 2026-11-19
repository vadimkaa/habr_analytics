"""Validate the rounded published diagnostics against the frozen data."""
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_editorial_checks():
    mod = importlib.import_module("19_editorial_checks")
    expected = json.loads((ROOT / "data/results/19_editorial_checks.json").read_text(encoding="utf-8"))
    assert mod.compute() == expected
    short = (ROOT / "short_article/article.md").read_text(encoding="utf-8")
    for key in ("ed_ai_8_vpd", "ed_ai_8_n", "ed_ai_union_rho", "ed_wide_union_rho",
                "ed_ai_wd_day", "ed_ai_wd_eve", "ed_wide_wd_day", "ed_wide_wd_eve",
                "ed_ai_we_day", "ed_ai_we_eve", "ed_wide_we_day", "ed_wide_we_eve"):
        assert expected[key] in short, key

def test_short_article_key_values():
    # Additional independently generated values, rounded to the precision of the short text.
    values = json.loads((ROOT / "data/results/article_values.json").read_text(encoding="utf-8"))
    short = (ROOT / "short_article/article.md").read_text(encoding="utf-8")
    for key in ("unique", "read_n", "read_n_ai", "read_n_wide", "ai_recent_n",
                "wide_recent_n", "hour_top_n", "leak_orig", "leak_count", "leak_none",
                "ml_overpred_ai", "ml_overpred_wide", "s_fe_ai", "s_fe_wide",
                "s_age_final_train_ai", "s_age_final_train_wide",
                "s_age_audit_exposed_ai", "s_age_audit_exposed_wide"):
        assert str(values[key]) in short, key
    assert f"{float(values['auc75']):.2f}" in short
