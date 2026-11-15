"""Разделение задач A / B1 / B2 в slot_policy.

Проверяется: режим A - прогноз на фактических признаках (контрольная точка);
B1 - у фиксированного профиля score зависит ТОЛЬКО от календаря строки;
B2 - альтернативные слоты календарно согласованы (месяц, dow, праздничный
статус соответствуют реальной дате), рекомендации и метрики диагностические.
"""

from __future__ import annotations

import pytest

# Эти тесты обучают настоящие модели; без CatBoost они пропускаются.
pytest.importorskip("catboost")

import numpy as np
import pandas as pd
import pytest

from src import models as MO
from src import slot_policy as SP


@pytest.fixture()
def fitted(small_fit):
    """Быстрый CatBoost (60 деревьев) на синтетике с сильным календарным сигналом."""
    fit = MO.fit_catboost(
        small_fit["train"],
        None,
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=60,
    )
    return fit, small_fit


def test_mode_a_full_prediction(fitted):
    """Режим A: score = декодированный прогноз на фактических признаках статьи."""
    fit, env = fitted
    te = env["test"].head(300)
    scored = SP.assign_scores(fit, te, env["train"], env["features"], None)
    direct = fit.target.decode(fit.model.predict(te[env["features"]]))
    assert np.allclose(scored["model_score"].to_numpy(), direct)


def test_assign_scores_rejects_non_a_modes(fitted):
    """assign_scores - только режим A; B1/B2 живут в отдельных функциях."""
    fit, env = fitted
    with pytest.raises(ValueError):
        SP.assign_scores(
            fit,
            env["test"],
            env["train"],
            env["features"],
            None,
            mode="fixed_profile_time_score",
        )


def test_b1_score_depends_only_on_calendar(fitted):
    """B1: одинаковый календарь -> одинаковый score независимо от статьи.

    У профиля некалендарные-признаки зафиксированы, поэтому внутри группы по
    ВСЕМ календарным признакам (dow, hour, is_weekend, is_holiday) дисперсия
    score обязана быть нулевой; is_holiday - тоже календарь (реальная дата),
    поэтому не группировать по нему нельзя. Между группами сигнал есть
    (синтетика: выходные +800, вечер +300)."""
    fit, env = fitted
    te = env["test"].sample(400, random_state=0)
    profile = SP.fixed_profiles(env["train"])["novice"]
    out = SP.b1_fixed_profile_scores(fit, te, env["features"], profile)
    g = out.groupby(["dow", "hour", "is_weekend", "is_holiday"])["model_score"]
    assert float(g.std().fillna(0).max()) < 1e-9, (
        "B1 score зависит от некалендарные-признаков статьи"
    )
    assert float(g.mean().max() - g.mean().min()) > 0, (
        "календарный сигнал между слотами отсутствует (нарушение синтетики)"
    )


def test_b1_weekend_slot_scores_higher(fitted):
    """B1 на синтетике: выходной вечерний слот дороже будничного утреннего."""
    fit, env = fitted
    te = env["test"].sample(400, random_state=0)
    profile = SP.fixed_profiles(env["train"])["novice"]
    out = SP.b1_fixed_profile_scores(fit, te, env["features"], profile)
    means = out.groupby(["dow", "hour"])["model_score"].mean()
    slots = set(means.index)
    we = [means[k] for k in slots if k[0] in (5, 6) and k[1] in (20, 21, 22)]
    wd = [means[k] for k in slots if k[0] in (1, 2, 3) and k[1] in (3, 4, 5)]
    assert we and wd, "в подвыборке нет слотов нужных типов"
    assert float(np.mean(we)) > float(np.mean(wd))


def test_b2_month_frame_calendar_consistency(fitted):
    """B2: слоты месяца опираются на реальные даты - dow и праздничный статус
    канонической даты соответствуют колонкам слота."""
    fit, env = fitted
    te = env["test"]
    dates_by = SP._canonical_month_dates(te.dt_msk)
    (year, month) = sorted(dates_by)[0]
    sf = SP.b2_month_slot_frame(
        env["train"], year, month, None, dates_by[(year, month)]
    )
    assert len(sf) == 168
    assert (sf["month"] == month).all()
    dd = pd.to_datetime(sf["date"])
    assert (dd.dt.dayofweek.to_numpy() == sf["dow"].to_numpy()).all()
    from habrpub.analysis import offday_sets, weekday_holidays

    _, workweekend = offday_sets([year])
    hol = weekday_holidays([year])
    exp_weekend = ((sf["dow"] >= 5) & (~dd.dt.date.isin(workweekend))).astype(int)
    assert (sf["is_weekend"].to_numpy() == exp_weekend.to_numpy()).all()
    assert (
        sf["is_holiday"].to_numpy() == dd.dt.date.isin(hol).astype(int).to_numpy()
    ).all()


def test_personalized_scheduling_metrics(fitted):
    """B2: рекомендации по всем статьям, правило выходных детерминировано,
    доля отличий и месячная согласованность в допустимых диапазонах."""
    fit, env = fitted
    te = env["test"].sample(250, random_state=1)
    res = SP.personalized_scheduling(fit, te, env["train"], env["features"], None)
    assert res["n_articles"] == len(te)
    assert 0.0 <= res["share_reco_differs_weekend_rule"] <= 1.0
    n_months = 0
    for key, vec in res["month_rank_vectors"].items():
        assert len(vec) == 168
        n_months += 1
    assert n_months >= 1
    # правило выходных детерминировано внутри месяца: один rule_slot на месяц
    per_month = {}
    for aid, (reco, rule, mkey) in res["reco_per_article"].items():
        per_month.setdefault(mkey, set()).add(rule)
        assert 0 <= reco < 168
    for mkey, rules in per_month.items():
        assert len(rules) == 1, f"правило выходных нестабильно в {mkey}"


def test_personalized_scheduling_profile_override(fitted):
    """B2 с другим профилем (устойчивость к профилю) выполняется и даёт структуру."""
    fit, env = fitted
    te = env["test"].sample(150, random_state=2)
    res = SP.personalized_scheduling(
        fit,
        te,
        env["train"],
        env["features"],
        None,
        profile_override={"prior_count": 10.0, "has_prior": 1},
    )
    assert res["n_articles"] == len(te)
    assert res["profile_override"]["prior_count"] == "10.0"


def test_b2_never_uses_actual_density(fitted):
    """B2 при density-free наборе запрещает dens24 в признаках (как режим A)."""
    fit, env = fitted
    te = env["test"].sample(120, random_state=3)
    feats_bad = env["features"] + ["dens24"]
    with pytest.raises(Exception):
        SP.personalized_scheduling(fit, te, env["train"], feats_bad, None)
