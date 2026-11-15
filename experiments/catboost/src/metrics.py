"""Метрики задач A и B. Полный набор зафиксирован до
обучения; Спирмен общий - справочная метрика, зависящая от возраста, основной
критерий A - средний Спирмен внутри календарных месяцев (n>=30, равные веса).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def _clean(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def rho_all(x, y) -> float | None:
    """Спирмен по всему тесту; зависит от возраста - только справочная."""
    x, y = _clean(x, y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(stats.spearmanr(x, y).statistic)


def rho_within_month(
    df: pd.DataFrame,
    score_col: str = "score",
    views_col: str = "views",
    min_n: int = 30,
) -> dict:
    """Средний Спирмен внутри месяцев теста, равные веса месяцев (как в исходной модели).

    Внутри месяца возраст почти постоянен, поэтому эта метрика измеряет
    ранжирование слотов, а не знание даты публикации."""
    rhos = []
    n_months = 0
    for _, sub in df.groupby("month_id"):
        if len(sub) < min_n:
            continue
        r = rho_all(sub[score_col], sub[views_col])
        if r is not None:
            rhos.append(r)
    n_months = len(rhos)
    return {
        "rho_within_month_mean": float(np.mean(rhos)) if rhos else None,
        "rho_within_month_median": float(np.median(rhos)) if rhos else None,
        "rho_within_month_n": n_months,
    }


def level_metrics(
    pred_views: np.ndarray, y_views: np.ndarray, train_log1p_median: float
) -> dict:
    """Абсолютный уровень: MAE/RMSE/bias/R2 на log1p-шкале (хиты доминируют
    квадратичные ошибки на сырых просмотрах). MAE базовой линии - медиана
    log1p train."""
    p, y = _clean(pred_views, y_views)
    lp = np.log1p(np.maximum(p, 0))
    ly = np.log1p(np.maximum(y, 0))
    base = float(train_log1p_median)
    return {
        "mae_log": float(np.mean(np.abs(lp - ly))),
        "rmse_log": float(np.sqrt(np.mean((lp - ly) ** 2))),
        "bias_log": float(np.mean(lp - ly)),
        "r2_log": float(1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2)),
        "baseline_mae_log": float(np.mean(np.abs(base - ly))),
    }


def top_quartile_precision_recall(
    df: pd.DataFrame,
    score_col: str = "score",
    views_col: str = "views",
    min_n: int = 30,
) -> dict:
    """Качество верхнего хвоста: precision/recall верхнего квартиля месяца."""
    precs, recs = [], []
    for _, sub in df.groupby("month_id"):
        if len(sub) < min_n:
            continue
        k = max(1, len(sub) // 4)
        true_top = set(sub.nlargest(k, views_col).index)
        pred_top = set(sub.nlargest(k, score_col).index)
        inter = len(true_top & pred_top)
        precs.append(inter / k)
        recs.append(inter / k)
    return {
        "top25_precision_mean": float(np.mean(precs)) if precs else None,
        "top25_recall_mean": float(np.mean(recs)) if recs else None,
        "top25_months": len(precs),
    }


def pinball_and_coverage(
    pred: np.ndarray, y: np.ndarray, alpha: float, scale: float
) -> dict:
    """Pinball loss и покрытие квантиля на шкале вьюх.

    pred/y передаются в единицах цели (тысячи/десятки тысяч); scale - множитель
    для пересчёта в просмотры."""
    p, t = _clean(pred, y)
    d = t - p
    loss = np.mean(np.maximum(alpha * d, (alpha - 1) * d)) * scale
    return {
        "pinball_views": float(loss),
        "coverage_quantile": float(np.mean(t <= p)),
        "n": len(p),
    }


def prediction_health(pred: np.ndarray) -> dict:
    """Численная диагностика: inf/NaN, дисперсия, диапазон."""
    p = np.asarray(pred, float)
    return {
        "n_finite": int(np.isfinite(p).sum()),
        "n_total": int(len(p)),
        "frac_finite": float(np.isfinite(p).mean()) if len(p) else None,
        "pred_var": float(np.var(p[np.isfinite(p)])) if np.isfinite(p).any() else None,
        "pred_min": float(np.min(p[np.isfinite(p)])) if np.isfinite(p).any() else None,
        "pred_max": float(np.max(p[np.isfinite(p)])) if np.isfinite(p).any() else None,
    }


# ---------------------------------------------------------------------------
# Задача B: метрики policy (на личных статьях поддержанных слотов)
# ---------------------------------------------------------------------------


def rule_metrics(te: pd.DataFrame, score_col: str, n_boot: int | None = None) -> dict:
    """rho_slot, rho_weekend, отношения верх/низ для заданного score.

    te должен содержать колонки: views, is_weekend, author_key, score, slot_ok.
    Разбиения не подбираются по test: границы подаются снаружи (train-only)."""
    from .statistics import paired_delta_rho_ci, ratio_ci_author

    out = {}
    # Число наблюдений и уникальных авторов для каждой таблицы.
    out["n"] = int(len(te))
    out["n_authors"] = (
        int(te["author_key"].nunique()) if "author_key" in te.columns else None
    )
    out["rho_slot"] = rho_all(te[score_col], te["views"])
    out["rho_weekend"] = rho_all(te["is_weekend"].astype(float), te["views"])
    out["delta_rho"] = (
        None
        if out["rho_slot"] is None or out["rho_weekend"] is None
        else out["rho_slot"] - out["rho_weekend"]
    )
    if n_boot:
        out["delta_rho_ci_author"] = paired_delta_rho_ci(te, score_col, n_boot=n_boot)
    # Отношение медиан верх/низ по терцилям score (границы заданы заранее).
    if "tercile" in te.columns:
        hi = te.loc[te.tercile == 2, "views"]
        lo = te.loc[te.tercile == 0, "views"]
        if len(hi) and len(lo):
            out["hi_lo_tercile"] = float(hi.median() / lo.median())
            if n_boot:
                out["hi_lo_tercile_ci_author"] = ratio_ci_author(
                    te, te.tercile == 2, te.tercile == 0, n_boot=n_boot
                )
    return out


def equal_share_ratio(
    te: pd.DataFrame, score_col: str, q: float, n_boot: int | None = None
) -> dict:
    """Верхняя доля q по score против остальных: медианы просмотров.

    q задаётся заранее (из train), например как доля выходных."""
    from .statistics import ratio_ci_author

    k = max(1, int(np.ceil(q * len(te))))
    order = te.sort_values(score_col, ascending=False).index[:k]
    hi_mask = te.index.isin(order)
    hi = te.loc[hi_mask, "views"]
    lo = te.loc[~hi_mask, "views"]
    if len(hi) == 0 or len(lo) == 0:
        return {"top_q_ratio": None, "q": q, "n_hi": len(hi)}
    out = {
        "top_q_ratio": float(hi.median() / lo.median()),
        "q": q,
        "n_hi": int(len(hi)),
    }
    if n_boot:
        out["top_q_ratio_ci_author"] = ratio_ci_author(
            te, hi_mask, ~hi_mask, n_boot=n_boot
        )
    return out


def weekend_rule_ratio(te: pd.DataFrame, q: float, n_boot: int | None = None) -> dict:
    """Правило выходных при том же покрытии q: weekend-first порядок,
    ничьи - фиксированный детерминированный порядок (hour, id)."""
    from .statistics import ratio_ci_author

    rank = te.assign(_r=te.is_weekend.astype(int) * 1_000_000 - te.hour).sort_values(
        "_r", ascending=False
    )
    k = max(1, int(np.ceil(q * len(te))))
    hi_mask = te.index.isin(rank.index[:k])
    hi = te.loc[hi_mask, "views"]
    lo = te.loc[~hi_mask, "views"]
    naive = (
        te.loc[te.is_weekend.astype(bool), "views"].median()
        / te.loc[~te.is_weekend.astype(bool), "views"].median()
    )
    out = {
        "weekend_q_ratio": float(hi.median() / lo.median()),
        "weekend_naive_ratio": float(naive),
        "q": q,
        "n_hi": int(len(hi)),
    }
    if n_boot:
        out["weekend_q_ratio_ci_author"] = ratio_ci_author(
            te, hi_mask, ~hi_mask, n_boot=n_boot
        )
    return out


def weekday_only_metrics(
    te: pd.DataFrame, score_col: str, train_terciles: np.ndarray
) -> dict:
    """Только будничные публикации: различает ли модель время внутри будней."""
    wd = te[~te.is_weekend.astype(bool)]
    if len(wd) < 100:
        return {"weekday_only_hi_lo": None, "n_weekday": int(len(wd))}
    g = np.searchsorted(train_terciles, wd[score_col].to_numpy(), side="right")
    wd = wd.assign(tercile=g)
    hi = wd.loc[wd.tercile == 2, "views"]
    lo = wd.loc[wd.tercile == 0, "views"]
    return {
        "weekday_only_hi_lo": float(hi.median() / lo.median())
        if len(hi) and len(lo)
        else None,
        "weekday_only_rho": rho_all(wd[score_col], wd["views"]),
        "n_weekday": int(len(wd)),
    }


def summarize_A(
    te: pd.DataFrame,
    pred_views: np.ndarray,
    train_log1p_median: float,
    score_col: str = "score",
) -> dict:
    """Полный набор метрик задачи A для одного тестового фолда."""
    te = te.assign(score=np.asarray(pred_views, float))
    out = {"rho_all": rho_all(te[score_col], te["views"])}
    out.update(rho_within_month(te, score_col))
    out.update(level_metrics(pred_views, te["views"].to_numpy(), train_log1p_median))
    out.update(top_quartile_precision_recall(te, score_col))
    out.update(prediction_health(pred_views))
    return out


# ---------------------------------------------------------------------------
# Контроль возраста и операционный порог
# ---------------------------------------------------------------------------


def month_pct_views(df: pd.DataFrame) -> pd.Series:
    """Перцентиль просмотров ВНУТРИ календарного месяца публикации.

    Контроль возраста: внутри месяца возраст почти постоянен, поэтому
    ранжирование по перцентилю месяца не зависит от даты снятия счётчиков."""
    return df.groupby("month_id")["views"].rank(pct=True)


def b_metrics_age(
    te: pd.DataFrame,
    score_col: str,
    n_boot: int | None = None,
    min_n: int = 30,
    seed: int = config.SEED,
    cluster: str = "author_key",
) -> dict:
    """Метрики задачи B для одного score с контролем возраста.

    Возвращает корреляции score и weekend-признака:
     - с накопленными просмотрами;
     - с перцентилем просмотров внутри месяца (контроль возраста);
     - средний внутри-месячный Спирмен и парная разность внутри месяцев;
     - всё то же только на буднях.
    Одинаковые наблюдения для score и weekend-правила гарантированы: обе
    корреляции считаются на одних строках, ДИ - парный бутстреп."""
    from .statistics import paired_delta_rho_ci

    te = te.copy()
    te["views_pct_month"] = month_pct_views(te)
    wk_col = "is_weekend"
    wk = te[wk_col].astype(float)
    out = {}
    # Число наблюдений и уникальных авторов для каждой таблицы.
    out["n"] = int(len(te))
    out["n_authors"] = int(te[cluster].nunique()) if cluster in te.columns else None
    out["rho_slot_views"] = rho_all(te[score_col], te["views"])
    out["rho_weekend_views"] = rho_all(wk, te["views"])
    out["delta_rho_views"] = (
        None
        if out["rho_slot_views"] is None or out["rho_weekend_views"] is None
        else out["rho_slot_views"] - out["rho_weekend_views"]
    )
    out["rho_slot_pct"] = rho_all(te[score_col], te["views_pct_month"])
    out["rho_weekend_pct"] = rho_all(wk, te["views_pct_month"])
    out["delta_rho_pct"] = (
        None
        if out["rho_slot_pct"] is None or out["rho_weekend_pct"] is None
        else out["rho_slot_pct"] - out["rho_weekend_pct"]
    )

    # Внутри календарных месяцев: парная разность в каждом месяце, среднее.
    def _within_month_delta(frame: pd.DataFrame, target: str) -> float | None:
        ds = []
        for _, s in frame.groupby("month_id"):
            if len(s) < min_n:
                continue
            a = rho_all(s[score_col], s[target])
            b = rho_all(s[wk_col].astype(float), s[target])
            if a is not None and b is not None:
                ds.append(a - b)
        return float(np.mean(ds)) if ds else None

    ns = []
    for _, sub in te.groupby("month_id"):
        if len(sub) < min_n:
            continue
        ns.append(len(sub))
    out["delta_rho_within_month_views"] = _within_month_delta(te, "views")
    out["delta_rho_within_month_pct"] = _within_month_delta(te, "views_pct_month")
    out["within_month_n"] = len(ns)
    # Только будни (временная оценка, не полный прогноз).
    wd = te[~te[wk_col].astype(bool)]
    out["n_weekday"] = int(len(wd))
    out["rho_slot_views_weekday"] = rho_all(wd[score_col], wd["views"])
    out["rho_slot_pct_weekday"] = rho_all(wd[score_col], wd["views_pct_month"])
    out["delta_rho_within_month_views_weekday"] = (
        _within_month_delta(wd, "views") if len(wd) >= min_n else None
    )
    out["delta_rho_within_month_pct_weekday"] = (
        _within_month_delta(wd, "views_pct_month") if len(wd) >= min_n else None
    )
    if n_boot:
        out["delta_rho_ci_author_views"] = paired_delta_rho_ci(
            te,
            score_col,
            n_boot=n_boot,
            seed=seed,
            cluster=cluster,
        )
        out["delta_rho_ci_author_pct"] = paired_delta_rho_ci(
            te,
            score_col,
            n_boot=n_boot,
            seed=seed,
            cluster=cluster,
            target_col="views_pct_month",
        )
        out["delta_rho_ci_week_views"] = paired_delta_rho_ci(
            te,
            score_col,
            cluster="week_start",
            n_boot=n_boot,
            seed=seed,
        )
    return out


def fixed_threshold_ratio(
    te: pd.DataFrame,
    score_col: str,
    threshold: float,
    n_boot: int | None = None,
) -> dict:
    """ОПЕРАЦИОННАЯ метрика: порог, ЗАФИКСИРОВАННЫЙ на train, применяется к
    тесту без пересчёта.

    Отличать от equal_share_ratio (диагностика: верхняя доля переопределяется
    на тесте). Порог - квантиль train score, в тесте не подбирается."""
    from .statistics import ratio_ci_author

    s = te[score_col].to_numpy()
    hi_mask = s >= threshold
    lo_mask = ~hi_mask
    n_hi, n_lo = int(hi_mask.sum()), int(lo_mask.sum())
    if n_hi == 0 or n_lo == 0:
        return {
            "threshold": float(threshold),
            "n_hi": n_hi,
            "n_lo": n_lo,
            "fixed_threshold_ratio": None,
            "selected_share": None,
        }
    hi_v, lo_v = te.loc[hi_mask, "views"], te.loc[lo_mask, "views"]
    out = {
        "threshold": float(threshold),
        "n_hi": n_hi,
        "n_lo": n_lo,
        "median_hi_views": float(hi_v.median()),
        "median_lo_views": float(lo_v.median()),
        "fixed_threshold_ratio": float(hi_v.median() / lo_v.median()),
        "selected_share": float(n_hi / len(te)),
        "rho_score_views_selected": rho_all(te.loc[hi_mask, score_col], hi_v),
        "rho_score_views_all": rho_all(te[score_col], te["views"]),
        "n": int(len(te)),
    }
    if n_boot:
        out["fixed_threshold_ratio_ci_author"] = ratio_ci_author(
            te,
            pd.Series(hi_mask, index=te.index),
            pd.Series(lo_mask, index=te.index),
            n_boot=n_boot,
        )
    return out
