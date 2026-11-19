"""Frozen exploratory check and its reported uncertainty."""
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_morning_check():
    mod = importlib.import_module("20_morning_check")
    data = mod.frozen()
    mod.validate(mod.compute(), data)
    assert round(100 * data["weekend_not_above_weekday_median"]["ai"], 1) == 39.1
    assert round(100 * data["weekend_not_above_weekday_median"]["wide"], 1) == 38.5
    v = mod.publication_values()
    short = (ROOT / "short_article/article.md").read_text(encoding="utf-8")
    assert v["am_ai_ratio_ci"] in short
