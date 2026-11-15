"""Инвариантность невременных признаков в B2.

При генерации альтернативных слотов у статьи должны меняться ТОЛЬКО календарные
признаки; заголовок, история автора, тип блога и прочее остаются её собственными.
Плюс: personalized_slot_score = прогноз фактического слота минус медиана
прогнозов допустимых слотов того же месяца.
"""

from __future__ import annotations

import pytest

# Эти тесты обучают настоящие модели; без CatBoost они пропускаются.
pytest.importorskip("catboost")

import numpy as np
import pandas as pd
import pytest

from src import slot_policy as SP


def test_b2_alternatives_keep_article_features(synthetic):
    """Невременные признаки статьи одинаковы во всех её альтернативных слотах."""
    from src import features as feats

    df = feats.prepare(synthetic)
    train = df[df.dt_msk < "2024-11-01 00:00:00+03:00"]
    te = df[df.dt_msk >= "2024-11-01 00:00:00+03:00"].sample(60, random_state=0)
    dates_by = SP._canonical_month_dates(te.dt_msk)
    (year, month) = sorted(dates_by)[0]
    sf = SP.b2_month_slot_frame(train, year, month, None, dates_by[(year, month)])
    cal_cols = ["dow", "hour", "is_weekend", "is_holiday", "hour_sin", "hour_cos"]
    art_cols = [
        c for c in SP.fixed_profiles(train)["novice"] if c not in SP.CALENDAR_FEATURES
    ]
    arts = te[
        [(y, m) == (year, month) for y, m in zip(te.dt_msk.dt.year, te.dt_msk.dt.month)]
    ]
    arts = arts.head(10)
    big = SP._alternatives_frame(arts, sf, cal_cols, art_cols)
    n_slots = len(sf)
    for aid, grp in big.groupby("_art"):
        assert len(grp) == n_slots
        for c in art_cols:  # невременные признаки НЕ меняются между слотами
            assert grp[c].nunique() == 1, f"признак {c} изменился между слотами"
        for c in cal_cols:  # календарные признаки совпадают с сеткой месяца
            assert np.allclose(
                np.sort(grp[c].to_numpy()),
                np.sort(sf[c].to_numpy())
                if c in sf.columns
                else np.sort(grp[c].to_numpy()),
            ), f"календарный признак {c} не совпадает с сеткой"


def test_b2_alternatives_calendar_matches_month(synthetic):
    """Месяц, dow и праздничный статус слота соответствуют реальной дате."""
    train = synthetic[synthetic.dt_msk < "2024-11-01 00:00:00+03:00"]
    te = synthetic[synthetic.dt_msk >= "2024-11-01 00:00:00+03:00"].head(50)
    dates_by = SP._canonical_month_dates(te.dt_msk)
    for (year, month), dby in sorted(dates_by.items()):
        sf = SP.b2_month_slot_frame(train, year, month, None, dby)
        assert len(sf) == 168
        dd = pd.to_datetime(sf["date"])
        assert (dd.dt.dayofweek.to_numpy() == sf["dow"].to_numpy()).all()
        assert (sf["month"] == month).all()
        from habrpub.analysis import weekday_holidays, offday_sets

        _, workweekend = offday_sets([year])
        hol = weekday_holidays([year])
        exp_we = ((sf["dow"] >= 5) & (~dd.dt.date.isin(workweekend))).astype(int)
        assert (sf["is_weekend"].to_numpy() == exp_we.to_numpy()).all()
        # праздничный флаг = праздники на БУДНЯХ (семантика обучения)
        assert (
            sf["is_holiday"].to_numpy() == dd.dt.date.isin(hol).astype(int).to_numpy()
        ).all()


def test_personalized_slot_score_definition(fitted):
    """personalized_slot_score = прогноз фактического слота минус медиана
    прогнозов допустимых (поддержанных) слотов месяца статьи."""
    fit, env = fitted
    te = env["test"].sample(120, random_state=5)
    res = SP.personalized_scheduling(
        fit, te, env["train"], env["features"], None, return_slot_table=True
    )
    pa = res["per_article"]
    assert len(pa) == len(te)
    recomputed = pa["actual_pred"] - pa["median_eligible_pred"]
    assert np.allclose(recomputed, pa["personalized_slot_score"], atol=1e-9)
    # Медиана допустимых альтернатив воспроизводится из таблицы слотов
    st = res["slot_table"]
    for aid in pa["article_id"].head(20):
        slots = st[st.article_id == aid]
        d, h = map(int, pa.loc[pa.article_id == aid, "actual_slot"].iloc[0].split("-"))
        # пул = поддержанные слоты БЕЗ слота фактической публикации
        pool = slots[(slots.supported == 1) & ~((slots.dow == d) & (slots.hour == h))]
        med = float(pool["pred"].median())
        row = pa[pa.article_id == aid].iloc[0]
        assert abs(med - row["median_eligible_pred"]) < 1e-9
        assert (
            abs(
                float(np.mean(pool["pred"] <= row["actual_pred"]))
                - row["personalized_slot_pct"]
            )
            < 1e-9
        )


@pytest.fixture()
def fitted(small_fit):
    from src import models as MO

    fit = MO.fit_catboost(
        small_fit["train"],
        None,
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=50,
    )
    return fit, small_fit
