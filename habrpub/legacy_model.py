"""Контрольная модель LightGBM и общие функции ретроспективной проверки.

Этот модуль доступен для основного анализа и диагностик без динамической
загрузки скриптов по имени файла. Все входные признаки и параметры обучения
совпадают с исторически зафиксированным экспериментом.
"""

import html
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from .analysis import SEED, bootstrap_cluster, corr, previous_counts, weekday_holidays

TIME = [
    "dow",
    "hour",
    "hour_sin",
    "hour_cos",
    "month",
    "is_weekend",
    "is_holiday",
    "dens24",
]
# Абляция «без плотности и месяца»: dens24 и month работают
# прокси эпохи в растущем корпусе; без них сетка слотов чище от эффекта даты.
TIME_NO_EPOCH = ["dow", "hour", "hour_sin", "hour_cos", "is_weekend", "is_holiday"]
TITLE = ["t_len", "t_words", "t_digit", "t_qmark", "t_colon"]
AUTHOR = ["prior_count", "has_prior"]
FULL = TIME + TITLE + AUTHOR + ["is_corp"]
PARAMS = dict(
    objective="poisson",
    n_estimators=300,
    learning_rate=0.04,
    num_leaves=15,
    min_child_samples=60,
    colsample_bytree=1.0,
    subsample=1.0,
    random_state=SEED,
    n_jobs=2,
    verbosity=-1,
    deterministic=True,
    force_col_wise=True,
    importance_type="gain",
)


def prepare(df):
    df = df.sort_values(["dt_utc", "id"]).reset_index(drop=True).copy()
    prior = previous_counts(df)
    df["prior_count"] = np.log1p(prior)
    df["has_prior"] = (prior > 0).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    # Производственный календарь: нерабочие будни (праздники и переносы).
    hol = weekday_holidays(df.year.unique())
    df["is_holiday"] = df.date_msk.isin(hol).astype(int)
    title = df.title.fillna("").map(lambda t: html.unescape(re.sub("<[^>]+>", " ", t)))
    df["t_len"] = title.str.len()
    df["t_words"] = title.str.split().str.len()
    df["t_digit"] = title.str.contains(r"\d").astype(int)
    df["t_qmark"] = title.str.contains(r"\?").astype(int)
    df["t_colon"] = title.str.contains(r"[:—-]").astype(int)
    for c in ["is_weekend", "is_corp"]:
        df[c] = df[c].astype(int)
    return df


def fit(train, test, features, target="views"):
    """Пуассоновская регрессия с лог-связью на накопленных просмотрах.

    Метрики на тесте: Спирмен по просмотрам; MAE и R² считаются на log1p,
    потому что на сырых просмотрах их доминируют единичные хиты."""
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(
        train[features],
        train[target],
        categorical_feature=[c for c in ["dow", "hour", "month"] if c in features],
    )
    pred = m.predict(test[features])
    y = test[target].to_numpy()
    ly = np.log1p(y)
    lp = np.log1p(np.maximum(pred, 0))
    base = np.median(np.log1p(train[target]))
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "rho": corr(pred, y)["rho"],
        "r2_log": 1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2),
        "mae_log": np.mean(abs(lp - ly)),
        "baseline_mae_log": np.mean(abs(base - ly)),
        "importance_gain": dict(zip(features, m.feature_importances_)),
    }
    # Честная к возрасту метрика: Спирмен внутри календарных
    # месяцев теста, усреднённый по месяцам с >= 30 статьями. Возраст внутри
    # месяца почти постоянен, поэтому общий Спирмен на накопленных просмотрах
    # частично измеряет возраст, а не качество слота.
    if "month_id" in test.columns:
        groups = test.reset_index(drop=True).groupby("month_id").groups
        rhos = [
            corr(pred[np.asarray(v)], y[np.asarray(v)])["rho"]
            for v in groups.values()
            if len(v) >= 30
        ]
        rhos = [r for r in rhos if r is not None]
        if rhos:
            metrics["rho_within_month_mean"] = float(np.mean(rhos))
            metrics["rho_within_month_median"] = float(np.median(rhos))
            metrics["rho_within_month_n"] = len(rhos)
    return m, metrics


def make_grid(model, train):
    profile = {c: float(train[c].median()) for c in FULL}
    profile.update(
        is_corp=0,
        is_holiday=0,
        has_prior=0,
        prior_count=0,
        t_digit=0,
        t_qmark=0,
        t_colon=0,
    )
    density = train.groupby(["dow", "hour"]).dens24.median()
    count = train.groupby(["dow", "hour"]).size()
    rows = []
    for d in range(7):
        for h in range(24):
            r = dict(
                profile,
                dow=d,
                hour=h,
                is_weekend=int(d >= 5),
                hour_sin=np.sin(2 * np.pi * h / 24),
                hour_cos=np.cos(2 * np.pi * h / 24),
                dens24=float(density.get((d, h), np.nan)),
            )
            r["n_train"] = int(count.get((d, h), 0))
            rows.append(r)
    grid = pd.DataFrame(rows)
    grid["pred_views"] = model.predict(grid[FULL])
    supported = (grid.n_train >= 20) & grid.dens24.notna()
    grid["supported"] = supported
    grid["relative_prediction"] = (
        grid.pred_views / grid.loc[supported, "pred_views"].median()
    )
    return grid


def policy_groups(train, test, grid):
    """Границы по train; группы по прогнозу; test outcomes не выбирают победителей."""
    columns = ["dow", "hour", "relative_prediction", "supported"]
    tr = train.merge(grid[columns], on=["dow", "hour"])
    tr = tr[(tr.is_corp == 0) & tr.supported]
    thresholds = np.quantile(tr.relative_prediction, [1 / 3, 2 / 3])
    te = test.merge(grid[columns], on=["dow", "hour"])
    te = te[(te.is_corp == 0) & te.supported].copy()
    te["group"] = np.searchsorted(thresholds, te.relative_prediction, side="right")
    groups = (
        te.groupby("group")
        .agg(
            size=("id", "size"),
            views_median=("views", "median"),
            vpd_median=("vpd", "median"),
            age_median=("age_days", "median"),
            # Близкие медианы возраста не гарантируют одинаковых распределений
            # Рядом идут квартили.
            age_q25=("age_days", lambda s: float(s.quantile(0.25))),
            age_q75=("age_days", lambda s: float(s.quantile(0.75))),
        )
        .reindex([0, 1, 2])
    )

    def ratio(z):
        return (
            z.loc[z.group == 2, "views"].median()
            / z.loc[z.group == 0, "views"].median()
        )

    return {
        "thresholds_train": thresholds.tolist(),
        "groups": groups.reset_index().to_dict("records"),
        "top_bottom_ratio": ratio(te),
        "ci95_author": bootstrap_cluster(te, ratio),
        "n_test_personal_supported": len(te),
        "n_test_personal_total": int((test.is_corp == 0).sum()),
    }, te
