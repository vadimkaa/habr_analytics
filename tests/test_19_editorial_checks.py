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
                "ed_ai_wd_day", "ed_ai_wd_eve", "ed_wide_wd_day", "ed_wide_wd_eve"):
        assert expected[key] in short, key
