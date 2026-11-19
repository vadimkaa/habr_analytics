"""Numerical and contextual checks for the publication diagnostics."""
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_editorial_checks():
    mod = importlib.import_module("19_editorial_checks")
    mod.validate(mod.compute(), mod.frozen())
    values = mod.publication_values()
    short = (ROOT / "short_article/article.md").read_text(encoding="utf-8")
    ai = f"{values['ed_ai_wd_eve']} против {values['ed_ai_wd_day']} в ИИ/МЛ"
    wide = f"{values['ed_wide_wd_eve']} против {values['ed_wide_wd_day']} в пяти хабах"
    weekend = (f"({values['ed_ai_we_eve']} против {values['ed_ai_we_day']} и "
               f"{values['ed_wide_we_eve']} против {values['ed_wide_we_day']})")
    assert ai in short and wide in short and weekend in short
    assert f"VPD {values['ed_ai_8_vpd']} на {values['ed_ai_8_n']} статьях ИИ/МЛ" in short
    assert f"({values['ed_ai_union_rho']} и {values['ed_wide_union_rho']})" in short


def test_short_article_key_values():
    values = json.loads((ROOT / "data/results/article_values.json").read_text(encoding="utf-8"))
    short = (ROOT / "short_article/article.md").read_text(encoding="utf-8")
    expected = (
        f"уникальных — {values['unique']}.",
        f"{values['ai_recent_n']} публикаций ИИ/МЛ и {values['wide_recent_n']} из пяти хабов",
        f"осталось {values['read_n']} публикаций ({values['read_n_ai']} из ИИ/МЛ и {values['read_n_wide']} из пяти хабов)",
        f"там всего {values['hour_top_n']} статей",
        f"равнялась {values['leak_orig']}.",
        f"{values['leak_count']}, если",
        f"и {values['leak_none']}, если",
        f"в медиане в {values['ml_overpred_ai']} и {values['ml_overpred_wide']} раза",
        f"составили {values['s_fe_ai']} и {values['s_fe_wide']} (ДИ",
        f"{values['s_age_final_train_ai']} и {values['s_age_audit_exposed_ai']} дней в ИИ/МЛ, "
        f"{values['s_age_final_train_wide']} и {values['s_age_audit_exposed_wide']} в пяти хабах",
        f"AUC = {float(values['auc75']):.2f};",
    )
    for fragment in expected:
        assert fragment in short, fragment
