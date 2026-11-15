"""Базовые ориентиры B0-B3: CatBoost обязан обойти простые правила.

Все базлайны строятся без тестовых исходов: медианы и таблицы - только train.
Контрольные LightGBM-модели обучаются через models.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

from .targets import TargetSpec


def b0_constant(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict:
    """Константа: медиана log1p(views) train (вариант среднего - для RMSE)."""
    med_log1p = float(np.median(np.log1p(train["views"].to_numpy())))
    mean_log1p = float(np.mean(np.log1p(train["views"].to_numpy())))
    target = TargetSpec("log1p")
    pred = {
        name: np.full(len(f), target.decode(np.array([med_log1p]))[0])
        for name, f in frames.items()
    }
    return dict(
        name="B0_constant",
        pred_views=pred,
        metadata=dict(constant_log1p_median=med_log1p, constant_log1p_mean=mean_log1p),
        score={k: v.copy() for k, v in pred.items()},
    )


def b1_calendar_median(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict:
    """Календарная медиана: месяц-в-году из train, fallback - общая медиана.

    Месяцы теста, которых нет в train (широкий корпус), получают fallback."""
    ly = np.log1p(train["views"].to_numpy())
    global_med = float(np.median(ly))
    by_month = pd.Series(ly).groupby(train["month"].to_numpy()).median().to_dict()
    target = TargetSpec("log1p")
    pred = {}
    for name, f in frames.items():
        vals = [float(by_month.get(int(m), global_med)) for m in f["month"]]
        pred[name] = target.decode(np.asarray(vals))
    return dict(
        name="B1_calendar_median",
        pred_views=pred,
        metadata=dict(by_month=by_month, global_fallback=global_med),
        score={k: v.copy() for k, v in pred.items()},
    )


def _small_lgbm():
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=30,
        random_state=config.SEED,
        n_jobs=2,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def b2_author_prior(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict:
    """Только авторские известные: prior_count, has_prior, тип блога."""
    feats = ["prior_count", "has_prior", "is_corp"]
    target = TargetSpec("log1p")
    m = _small_lgbm()
    m.fit(train[feats], target.encode(train["views"].to_numpy()))
    pred = {n: target.decode(m.predict(f[feats])) for n, f in frames.items()}
    return dict(
        name="B2_author_prior",
        pred_views=pred,
        metadata=dict(features=feats),
        score={k: v.copy() for k, v in pred.items()},
    )


def b3_title_only(train: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict:
    """Только заголовок + тип блога: простейшая регуляризованная модель."""
    feats = ["t_len", "t_words", "t_digit", "t_qmark", "t_colon", "is_corp"]
    target = TargetSpec("log1p")
    m = _small_lgbm()
    m.fit(train[feats], target.encode(train["views"].to_numpy()))
    pred = {n: target.decode(m.predict(f[feats])) for n, f in frames.items()}
    return dict(
        name="B3_title_only",
        pred_views=pred,
        metadata=dict(features=feats),
        score={k: v.copy() for k, v in pred.items()},
    )


def run_baselines(
    train: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> dict[str, dict]:
    out = {}
    for fn in (b0_constant, b1_calendar_median, b2_author_prior, b3_title_only):
        try:
            out[fn.__name__[:2]] = fn(train, frames)
        except Exception as exc:  # недостаточно данных для базлайна - честная запись
            out[fn.__name__[:2]] = dict(name=fn.__name__, error=str(exc))
    return out
