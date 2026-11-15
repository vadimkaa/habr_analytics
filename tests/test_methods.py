"""Регрессионные проверки методических ошибок, найденных при аудите."""

import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from habrpub.analysis import (
    prior_density,
    previous_counts,
    bh,
    finite_json,
    paired_authors,
    loo_median,
)

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_density_excludes_simultaneous_and_includes_left_boundary():
    times = pd.Series(
        pd.to_datetime(
            [
                "2026-01-01T00:00Z",
                "2026-01-02T00:00Z",
                "2026-01-02T00:00Z",
                "2026-01-02T00:01Z",
            ]
        )
    )
    assert prior_density(times).tolist() == [0, 1, 1, 2]


def test_unknown_authors_and_simultaneous_history():
    d = pd.DataFrame(
        {
            "author_key": ["a", "a", "a", "__unknown_1", "__unknown_2"],
            "dt_utc": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
                utc=True,
            ),
        }
    )
    assert previous_counts(d).tolist() == [0, 0, 2, 0, 0]


def test_history_features_do_not_depend_on_outcome():
    pytest.importorskip("lightgbm")
    m = module("08_model.py")
    d = pd.DataFrame(
        {
            "id": ["1", "2"],
            "dt_utc": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "author_key": ["a", "a"],
            "hour": [1, 2],
            "year": [2026, 2026],
            "date_msk": [
                pd.Timestamp("2026-01-01").date(),
                pd.Timestamp("2026-01-02").date(),
            ],
            "title": ["A", "B"],
            "is_weekend": [False, False],
            "is_corp": [False, False],
            "vpd": [1, 2],
        }
    )
    before = m.prepare(d)
    d["vpd"] = [100000, 0]
    after = m.prepare(d)
    pd.testing.assert_frame_equal(before[m.AUTHOR], after[m.AUTHOR])
    assert "prior_med_vpd" not in m.FULL


def test_html_nested_formulas_and_single_letter_words():
    pytest.importorskip("razdel")
    pytest.importorskip("bs4")
    m = module("06_text_sample.py")
    prose, c = m.prepare_prose(
        '<p>Я и он.</p><span class="katex"><span>REMOVE</span>REMOVE2</span><pre>CODE</pre><p>Да!</p>'
    )
    assert "REMOVE" not in prose and "CODE" not in prose
    assert c["formula_n"] == 1
    assert m.compute_metrics(prose)["words"] == 4
    assert m.compute_metrics(prose)["sentences"] == 2


def test_sentence_abbreviations_do_not_split_every_dot():
    pytest.importorskip("razdel")
    pytest.importorskip("bs4")
    m = module("06_text_sample.py")
    text = "Он видел версии 3.14 и 3.15. Это работает, т. е. результат верен."
    assert m.compute_metrics(text)["sentences"] == 2


def test_bh_known_example_and_missing():
    np.testing.assert_allclose(bh([0.01, 0.04, 0.03, np.nan])[:3], [0.03, 0.04, 0.04])
    assert np.isnan(bh([0.01, np.nan])[1])


def test_strict_json_removes_nan():
    assert finite_json({"a": np.nan, "b": np.inf, "c": np.int64(2)}) == {
        "a": None,
        "b": None,
        "c": 2,
    }


def test_paired_authors_require_both_conditions():
    d = pd.DataFrame(
        {
            "author_key": ["a"] * 4 + ["b"] * 4 + ["only_weekday"] * 4,
            "is_weekend": [False, False, True, True] * 2 + [False] * 4,
            "vpd": [10, 10, 20, 20, 10, 10, 30, 30, 100, 100, 100, 100],
        }
    )
    r = paired_authors(d)
    assert r["n_authors"] == 2 and r["median_ratio"] == 2.5


def test_loo_median_keeps_row_order():
    """Leave-one-out медиана должна удалять значение текущей строки, а не
    i-е по величине. Сортировка перед delete подменяла удалённую статью:
    медиана «других статей автора» включала исход текущей статьи, разрыв
    терцилей раздувался и доля силы автора выходила 29–33% вместо 18–26%."""
    out = loo_median([5.0, 1.0, 9.0])
    # строка 0 — медиана [1, 9] = 5; строка 1 — медиана [5, 9] = 7;
    # строка 2 — медиана [5, 1] = 3 (с сортировкой вышло бы [7, 3, 5])
    assert np.allclose(out, [5.0, 7.0, 3.0])
    df = pd.DataFrame({"g": ["a"] * 4, "x": [10.0, 2.0, 7.0, 4.0]})
    res = df.groupby("g").x.transform(loo_median)
    assert np.allclose(
        res,
        [
            np.median([2, 7, 4]),
            np.median([10, 7, 4]),
            np.median([10, 2, 4]),
            np.median([10, 2, 7]),
        ],
    )


def test_adjusted_weekend_log10_targets_recover_known_ratio():
    """Все цели условной регрессии — в одной шкале log10(1+x), и при
    известном эффекте 1,5 коэффициент переводится в отношение 1,5.
    Для комментариев и закладок применяется та же логарифмическая шкала,
    а переводились через 10**b — отношение завышалось в exp(ln 10 * b) раза."""
    pytest.importorskip("statsmodels")
    m = module("09_robustness.py")
    rng = np.random.default_rng(11)
    n = 400
    d = pd.DataFrame(
        {
            "author_key": [f"a{i % 40}" for i in range(n)],
            "month_id": [f"2026-{i % 2 + 1:02d}" for i in range(n)],
            "is_weekend": rng.integers(0, 2, n),
            "is_corp": 0,
            "age_days": rng.uniform(30, 3000, n),
            "hubs": [[] for _ in range(n)],
        }
    )
    d["log_comments"] = 1 + np.log10(1.5) * d.is_weekend
    r = m.adjusted_weekend(d, use_density=False, target="log_comments")
    assert abs(r["ratio_target_plus1"] - 1.5) < 1e-6
    # та же шкала для просмотров: единство преобразования по всем исходам
    d["log_views"] = np.log10(51) + np.log10(1.5) * d.is_weekend
    r2 = m.adjusted_weekend(d, use_density=False, target="log_views")
    assert abs(r2["ratio_target_plus1"] - 1.5) < 1e-6


def test_holiday_calendar_does_not_count_dates_outside_window():
    from habrpub.calendar_analysis import holiday_analysis

    dates = pd.date_range("2026-04-01", "2026-05-20", tz="Europe/Moscow")
    d = pd.DataFrame(
        {
            "dt_msk": dates,
            "date_msk": dates.date,
            "views": 100.0,
            "month_id": dates.strftime("%Y-%m"),
            "vpd": np.arange(len(dates)) + 1,
            "id": range(len(dates)),
            "is_corp": [False] * len(dates),
        }
    )
    r = holiday_analysis(d)
    # 2026: нерабочие будни в окне - 1 мая (пятница) и 11 мая (перенос с 9 мая, субботы)
    assert r["holiday_days"] == 2
    assert str(pd.Timestamp("2026-05-11").date()) in r["holiday_dates"]
    assert r["volume_holiday"] == 1 and r["volume_control"] == 1
    # перцентиль внутри месяца - основная метрика; на константных просмотрах
    # он определяется числом статей месяца и обязан быть конечным
    assert np.isfinite(r["pct_holiday"]) and 0 < r["pct_holiday"] <= 1
    assert (
        r["by_kind_titled"]["n_articles"] + r["by_kind_transferred"]["n_articles"] == 2
    )


def test_holiday_blocks_have_own_nearby_controls():
    """Контроли каждого блока лежат в пределах 21 дня от его дат и не являются
    нерабочими днями (каждый блок имеет собственные контрольные даты
    всех праздников, и «8 марта» доставался декабрь с маем)."""
    from habrpub.analysis import offday_sets
    from habrpub.calendar_analysis import holiday_blocks

    dates = pd.date_range("2026-01-01", "2026-12-31", tz="Europe/Moscow")
    frame = pd.DataFrame(
        {
            "dt_msk": dates,
            "date_msk": dates.date,
            "views": 100.0,
            "month_id": dates.strftime("%Y-%m"),
            "vpd": 1.0,
            "id": range(len(dates)),
            "is_corp": False,
            "author_key": "a",
        }
    )
    off, _ = offday_sets()
    rows = holiday_blocks(frame)
    by_name = {r["block"]: r for r in rows}
    assert "1 мая" in by_name and "23 февраля" in by_name and "Переносы" in by_name
    for r in rows:
        block_dates = [pd.Timestamp(d).date() for d in r["dates"]]
        ctrl_dates = [pd.Timestamp(d).date() for d in r["control_dates"]]
        assert r["dates"] and ctrl_dates, r["block"]
        for cd in ctrl_dates:
            assert cd not in off, (r["block"], cd)
            assert min(abs((cd - d).days) for d in block_dates) <= 21, (
                r["block"],
                cd,
            )
    # контроли блока локальны, а не общий пул
    # только календарные даты соответствующего праздничного блока
    months = {pd.Timestamp(d).month for d in by_name["1 мая"]["control_dates"]}
    assert months <= {4, 5}, months
    months_ny = {pd.Timestamp(d).month for d in by_name["Новый год"]["control_dates"]}
    assert months_ny <= {12, 1, 2}, months_ny


def test_working_saturday_is_not_weekend():
    """Рабочая суббота 01.11.2025 не входит в группу выходных,
    обычная суббота остаётся выходной."""
    from habrpub.analysis import load_corpus

    w = load_corpus("wide")
    work_sat = w[w.date_msk.map(str) == "2025-11-01"]
    assert len(work_sat) > 0 and not work_sat.is_weekend.any()
    usual_sat = w[w.date_msk.map(str) == "2025-11-08"]
    assert len(usual_sat) > 0 and usual_sat.is_weekend.all()


def test_formula_counter_sees_data_tex_images():
    pytest.importorskip("razdel")
    pytest.importorskip("bs4")
    m = module("06_text_sample.py")
    prose, c = m.prepare_prose(
        '<p>Формула:</p><img data-tex="x^2" src="f.svg" alt=""><p>Далее.</p>'
    )
    assert c["formula_n"] == 1
    assert "x^2" not in prose  # alt без класса не попадает в прозу


def test_mattr_resists_length_growth():
    pytest.importorskip("razdel")
    pytest.importorskip("bs4")
    m = module("06_text_sample.py")
    vocab = [f"слово{i}" for i in range(300)]
    assert m.mattr([]) is None
    # короткий текст (<= окна): MATTR совпадает с обычным TTR
    assert abs(m.mattr(vocab) - 1.0) < 1e-9
    # длинный циклический текст: TTR падает до 300/4000, MATTR держится уровня окна
    long_text = [vocab[i % 300] for i in range(4000)]
    long_ttr = len(set(long_text)) / len(long_text)
    assert long_ttr < 0.1
    assert m.mattr(long_text) > 0.5


def test_production_calendar_transfers_and_working_saturday():
    from datetime import date
    from habrpub.analysis import offday_sets, weekday_holidays

    off, workweekend = offday_sets()
    hol = weekday_holidays()
    # переносы 2025 года по постановлению N 1335
    assert date(2025, 5, 2) in hol and date(2025, 12, 31) in hol
    # переносы 2026: 9 марта и 11 мая
    assert date(2026, 3, 9) in hol and date(2026, 5, 11) in hol
    # рабочие субботы не считаются выходными
    assert date(2025, 11, 1) in workweekend and date(2024, 11, 2) in workweekend
    assert date(2025, 11, 1) not in off
    # 13 июня 2024 - рабочий день; 9 мая 2026 (суббота) - не "праздник на буднях"
    assert date(2024, 6, 13) not in off
    assert date(2026, 5, 9) not in hol
    # карантинные и голосные дни указов 2020 года - не праздники, но и не рабочие
    assert date(2020, 4, 15) not in hol and date(2020, 7, 1) not in hol
    assert date(2020, 4, 15) in off


def test_week_cluster_bootstrap_runs_and_covers_point():
    from habrpub.analysis import compare_weekend

    n = 60
    d = pd.DataFrame(
        {
            "author_key": [f"a{i % 6}" for i in range(n)],
            "week_start": pd.to_datetime(
                [f"2026-0{(i // 15) % 3 + 1}-01" for i in range(n)]
            ),
            "is_weekend": [i % 2 == 0 for i in range(n)],
            "vpd": [20.0 if i % 2 == 0 else 10.0 for i in range(n)],
        }
    )
    r = compare_weekend(d)
    assert r["ratio"] == 2
    lo, hi = r["ratio_ci_week"]
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= 2 <= hi


def test_poisson_fit_beats_baseline_on_synthetic():
    pytest.importorskip("lightgbm")
    m = module("08_model.py")
    rng = np.random.default_rng(7)
    n = 300
    tr = pd.DataFrame(
        {
            "is_weekend": rng.integers(0, 2, n),
            "dow": rng.integers(0, 7, n),
            "hour": rng.integers(0, 24, n),
        }
    )
    tr["views"] = (
        np.where(tr.is_weekend == 1, 200, 50) * rng.lognormal(0, 0.3, n)
    ).astype(int)
    te = tr.copy()
    te["views"] = (
        np.where(te.is_weekend == 1, 200, 50) * rng.lognormal(0, 0.3, n)
    ).astype(int)
    _, met = m.fit(tr, te, ["is_weekend", "dow", "hour"])
    assert np.isfinite(met["mae_log"]) and met["mae_log"] < met["baseline_mae_log"]
    assert met["rho"] > 0.5


def test_slot_group_thresholds_do_not_use_test_outcomes(monkeypatch):
    pytest.importorskip("lightgbm")
    m = module("08_model.py")
    # Статистика теперь находится в именованном модуле, а не в CLI-скрипте.
    import habrpub.legacy_model as legacy

    monkeypatch.setattr(
        legacy,
        "bootstrap_cluster",
        lambda df, statistic, cluster="author_key": [statistic(df), statistic(df)],
    )
    grid = pd.DataFrame(
        {
            "dow": [0, 1, 2],
            "hour": [12] * 3,
            "relative_prediction": [0.5, 1, 2],
            "supported": [True] * 3,
        }
    )
    train = pd.DataFrame(
        {
            "id": list("abc") * 10,
            "dow": [0, 1, 2] * 10,
            "hour": [12] * 30,
            "is_corp": [0] * 30,
        }
    )
    test = train.copy()
    test["views"] = [10, 20, 30] * 10
    test["vpd"] = [1, 2, 3] * 10
    test["age_days"] = [60.0] * 30
    test["author_key"] = np.arange(30).astype(str)
    a, _ = m.policy_groups(train, test, grid)
    test["views"] = [100, 20, 1] * 10
    b, _ = m.policy_groups(train, test, grid)
    assert a["thresholds_train"] == b["thresholds_train"]
    assert a["top_bottom_ratio"] == 3 and b["top_bottom_ratio"] == 0.01


def test_count_images_skips_formula_images():
    """Шаг 14 считает иллюстрации, но не формулы, вставленные картинками."""
    m = module("14_extra_checks.py")
    html = (
        '<p>текст</p><img src="a.png"><figure><img src="b.png"></figure>'
        '<img class="formula" src="f.svg"><img data-tex="x^2" src="g.svg">'
    )
    assert m.count_images(html) == 2
    assert m.count_images("") == 0
    assert m.count_images(None) == 0
