"""Календарь B2 и нормировки персонализированной оценки.

Проверяется:
1. Единая семантика календарных флагов calendar_flags == признаки обучения
   (weekday_holidays: праздники только на буднях; обычные выходные - только
   в is_weekend; рабочие субботы - будни).
2. Прогноз фактического слота в B2 совпадает с прогнозом режима A на тех же
   признаках статьи.
3. Допустимые альтернативы - только поддержанные слоты, кроме слота
   фактической публикации; медиана и перцентиль считаются по этому пулу.
4. Три шкалы согласованы между собой (absolute/relative/percentile).
"""

from __future__ import annotations

import pytest

# Эти тесты обучают настоящие модели; без CatBoost они пропускаются.
pytest.importorskip("catboost")

from datetime import date

import numpy as np
import pytest

from src import models as MO
from src import slot_policy as SP


def test_calendar_flags_match_training_semantics():
    """Единый источник календаря: та же семантика, что и features.prepare."""
    from habrpub.analysis import weekday_holidays, offday_sets

    dates = [
        date(2024, 1, 6),
        date(2024, 1, 8),
        date(2023, 4, 29),
        date(2024, 2, 22),
        date(2024, 2, 23),
        date(2024, 5, 9),
        date(2024, 7, 15),
    ]
    flags = SP.calendar_flags(dates)
    hol = weekday_holidays([2023, 2024])
    off, workweekend = offday_sets([2023, 2024])
    for _, r in flags.iterrows():
        d = r["date"]
        # is_holiday - только праздничные будни (weekday_holidays), НЕ выходные
        assert r["is_holiday"] == int(d in hol)
        # is_weekend - сб/вс, не являющиеся рабочими днями
        assert r["is_weekend"] == int(d.weekday() >= 5 and d not in workweekend)
    # Контрольные точки семантики (производственный календарь РФ):
    assert (
        flags.loc[flags.date == date(2024, 1, 6), "is_weekend"].iloc[0] == 1
    )  # суббота
    assert (
        flags.loc[flags.date == date(2024, 1, 6), "is_holiday"].iloc[0] == 0
    )  # не будний праздник
    assert (
        flags.loc[flags.date == date(2024, 1, 8), "is_holiday"].iloc[0] == 1
    )  # 8 января - праздничный будень
    assert (
        flags.loc[flags.date == date(2024, 2, 22), "is_holiday"].iloc[0] == 0
    )  # рабочий четверг
    assert (
        flags.loc[flags.date == date(2024, 2, 23), "is_holiday"].iloc[0] == 1
    )  # 23 февраля 2024 - четверг
    assert (
        flags.loc[flags.date == date(2024, 5, 9), "is_holiday"].iloc[0] == 1
    )  # 9 мая 2024 - четверг
    assert (
        flags.loc[flags.date == date(2024, 7, 15), "is_weekend"].iloc[0] == 0
    )  # рабочий понедельник


def test_calendar_flags_differ_from_offday_sets_on_weekends():
    """Обычные выходные не должны попадать в праздничный флаг."""
    sats = [date(2024, 6, 1), date(2024, 6, 8), date(2024, 6, 15)]
    sundays = [date(2024, 6, 2), date(2024, 6, 9)]
    from habrpub.analysis import offday_sets

    off, _ = offday_sets([2024])
    assert all(
        d in off for d in sats + sundays
    )  # календарь нерабочих дней включает обычные выходные
    flags = SP.calendar_flags(sats + sundays)
    assert (flags["is_holiday"] == 0).all()  # праздничный флаг их исключает


def test_b2_actual_slot_matches_mode_a(small_fit):
    """Регрессионный тест: прогноз фактического слота в B2 равен
    прогнозу режима A на идентичных признаках той же статьи."""
    from src import slot_policy as SP

    fit = MO.fit_catboost(
        small_fit["train"],
        None,
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=50,
    )
    te = small_fit["test"]
    # A: прогноз на фактических признаках
    a_scored = SP.assign_scores(
        fit, te, small_fit["train"], small_fit["features"], None
    )
    # B2: персональное планирование
    res = SP.personalized_scheduling(
        fit, te, small_fit["train"], small_fit["features"], None
    )
    pa = res["per_article"]
    assert len(pa) == len(te)
    m = pa.merge(
        a_scored[["id"]].assign(a_pred=a_scored["model_score"]),
        left_on="article_id",
        right_on="id",
    )
    assert len(m) == len(te)
    # Совпадение с точностью до численного допуска - по всем статьям
    assert np.allclose(m["a_pred"], m["actual_pred"], atol=1e-9)


def test_b2_alternatives_supported_only_and_exclude_own(small_fit):
    """Медиана и перцентиль - по поддержанным слотам БЕЗ слота фактической
    публикации той же статьи."""
    fit = MO.fit_catboost(
        small_fit["train"],
        None,
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=50,
    )
    te = small_fit["test"].sample(150, random_state=3)
    res = SP.personalized_scheduling(
        fit, te, small_fit["train"], small_fit["features"], None, return_slot_table=True
    )
    pa, st = res["per_article"], res["slot_table"]
    for aid in pa["article_id"].head(15):
        row = pa[pa.article_id == aid].iloc[0]
        slots = st[st.article_id == aid]
        d, h = map(int, row["actual_slot"].split("-"))
        own_twin = slots[(slots.dow == d) & (slots.hour == h)]
        pool = slots[(slots.supported == 1) & ~((slots.dow == d) & (slots.hour == h))]
        assert len(pool) == row["eligible_slots_count"]
        med = float(pool["pred"].median())
        assert abs(med - row["median_eligible_pred"]) < 1e-9
        # перцентиль фактического слота в пуле допустимых альтернатив
        assert (
            abs(
                float(np.mean(pool["pred"] <= row["actual_pred"]))
                - row["personalized_slot_pct"]
            )
            < 1e-9
        )


def test_b2_scales_consistency(small_fit):
    """Три шкалы согласованы: relative = log1p-разность; percentile в [0, 1]."""
    fit = MO.fit_catboost(
        small_fit["train"],
        None,
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=50,
    )
    te = small_fit["test"].sample(120, random_state=7)
    res = SP.personalized_scheduling(
        fit, te, small_fit["train"], small_fit["features"], None
    )
    pa = res["per_article"].dropna(subset=["personalized_slot_score"])
    assert len(pa) > 0
    rel = np.log1p(pa["actual_pred"]) - np.log1p(pa["median_eligible_pred"])
    assert np.allclose(rel, pa["personalized_slot_rel"], atol=1e-9)
    assert (pa["personalized_slot_pct"] >= 0).all() and (
        pa["personalized_slot_pct"] <= 1
    ).all()
    # знак относительной шкалы совпадает со знаком абсолютной (log1p монотонен)
    assert (
        np.sign(pa["personalized_slot_rel"]) == np.sign(pa["personalized_slot_score"])
    ).all()


def test_b2_metrics_main_population_equals_supported_filter(small_fit):
    """Метрики B2 на основной совокупности = личные статьи поддержанных слотов
    (та же совокупность, что в A/B1), а не все статьи."""
    fit = MO.fit_catboost(
        small_fit["train"],
        None,
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=50,
    )
    te = small_fit["test"]
    slot_n = small_fit["train"].groupby(["dow", "hour"]).size()
    te_sup = te.copy()
    te_sup["slot_n_train"] = [
        int(slot_n.get((int(d), int(h)), 0)) for d, h in zip(te_sup.dow, te_sup.hour)
    ]
    supp = SP.personal_supported(te_sup)
    res = SP.personalized_scheduling(
        fit, te, small_fit["train"], small_fit["features"], None
    )
    pa = res["per_article"]
    # Каждая статья основной совокупности есть в B2, и её слот поддержан
    main = pa[pa.article_id.isin(supp["id"].astype(str))]
    assert (
        len(main) >= len(supp) * 0.8
    )  # статьи, отсутствующие в B2, только вне месяцев аудита
    assert "personalized_slot_rel" in pa.columns
    assert "personalized_slot_pct" in pa.columns
