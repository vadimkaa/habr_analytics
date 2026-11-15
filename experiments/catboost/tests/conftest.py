"""Общие фикстуры: маленькие синтетические данные с теми же полями, что у корпуса.

Реальные выгрузки в тесты не загружаются (быстро и без зависимостей от среза);
интеграционные стадии запускаются run_experiments.py отдельно.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CATBOOST_DIR = Path(__file__).resolve().parent.parent
REPO = CATBOOST_DIR.parent.parent
for p in (str(REPO), str(CATBOOST_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture()
def synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 12000
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="Europe/Moscow")
    dow = ts.dayofweek.to_numpy()
    hour = ts.hour.to_numpy()
    authors = rng.choice(["a", "b", "c", "d", "e"], size=n)
    views = (
        1000
        + np.where(dow >= 5, 800, 0)
        + np.where(hour >= 19, 300, 0)
        + rng.integers(0, 2000, size=n)
    ).astype(float)
    df = pd.DataFrame(
        dict(
            id=[str(i) for i in range(n)],
            dt_utc=ts.tz_convert("UTC"),
            dt_msk=ts,
            date_msk=ts.date,
            year=ts.year,
            month=ts.month,
            month_id=ts.strftime("%Y-%m"),
            dow=dow,
            hour=hour,
            is_weekend=(dow >= 5).astype(int),
            is_corp=rng.integers(0, 2, size=n),
            author_key=authors,
            views=views,
            title=[
                f"Заголовок статьи {i} " + ("?" if i % 7 == 0 else "часть: обзор")
                for i in range(n)
            ],
            dens24=rng.integers(0, 10, size=n).astype(float),
            t_len=rng.integers(10, 90, size=n).astype(float),
            t_words=rng.integers(1, 10, size=n).astype(float),
            t_digit=rng.integers(0, 2, size=n),
            t_qmark=rng.integers(0, 2, size=n),
            t_colon=rng.integers(0, 2, size=n),
            age_days=rng.uniform(35, 300, size=n),
        )
    )
    df["week_start"] = df.dt_msk.dt.normalize() - pd.to_timedelta(df.dow, unit="D")
    df["is_translation"] = False
    df["hubs"] = "['test_hub']"
    df["month_pct"] = 0.5
    return df.sort_values(["dt_utc", "id"]).reset_index(drop=True)


@pytest.fixture()
def small_fit(synthetic):
    """Быстрое обучение CatBoost C1 на синтетике (50 деревьев)."""
    from src import features as feats

    df = feats.prepare(synthetic)
    train = df[df.dt_msk < "2024-11-01 00:00:00+03:00"]
    test = df[df.dt_msk >= "2024-11-01 00:00:00+03:00"]
    return dict(
        df=df,
        train=train,
        test=test,
        features=feats.FEATURE_SETS["F1_meta"]["features"],
    )
