"""Статистическая неопределённость.

Парный бутстреп по авторам: в каждом повторе одновременно пересчитываются
обе величины (две корреляции, две медианы), разность берётся на одной и той же
перевыборке. Дополнительная кластеризация по неделям - чувствительность к
общему информационному фону; простой авторский бутстреп не учитывает все
зависимости, и это оговаривается в результатах.

Во всех кластерных процедурах перевыбираются целые кластеры. Для
отношения медиан автор может присутствовать только в одной из сравниваемых
групп; такой автор тоже входит в пул перевыборки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def _author_groups(df: pd.DataFrame, cluster: str = "author_key"):
    return [x.index.to_numpy() for _, x in df.groupby(cluster, sort=True)]


def _resample_idx(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in pick])


def paired_delta_rho_ci(
    te: pd.DataFrame,
    score_col: str,
    cluster: str = "author_key",
    n_boot: int = 1000,
    seed: int = config.SEED,
    target_col: str = "views",
) -> list[float]:
    """95% ДИ разности rho(score) - rho(weekend) на общих бутстреп-выборках.

    Положительная разность - score ранжирует лучше простого правила выходных.
    target_col позволяет контролировать возраст: "views" (накопленные просмотры)
    или "views_pct_month" (перцентиль просмотров внутри месяца публикации).
    Обе корреляции каждого повтора считаются на одних и тех же строках."""
    groups = _author_groups(te, cluster)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = _resample_idx(groups, rng)
        part = te.loc[idx]
        a = stats.spearmanr(part[score_col], part[target_col]).statistic
        b = stats.spearmanr(
            part["is_weekend"].astype(float), part[target_col]
        ).statistic
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(float(a - b))
    if not diffs:
        return [None, None]
    return [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]


def ratio_ci_author(
    te: pd.DataFrame,
    hi_mask: pd.Series,
    lo_mask: pd.Series,
    cluster: str = "author_key",
    n_boot: int = 1000,
    seed: int = config.SEED,
) -> list[float]:
    """95% ДИ отношения медиан верх/низ при перевыборке целых авторов.

    В пул входят все авторы, представленные в любой из двух групп.
    Отдельные авторы могут не иметь наблюдений в одной группе. На каждом
    повторе авторы выбираются с возвращением, с сохранением всех их статей.
    Повторы с пустой группой или некорректным знаменателем исключаются."""
    hi_idx = set(te.index[np.asarray(hi_mask, bool)])
    lo_idx = set(te.index[np.asarray(lo_mask, bool)])
    pool: dict = {}
    for a, sub in te.groupby(cluster, sort=True):
        h = sub.loc[sub.index.isin(hi_idx), "views"].to_numpy(float)
        low_values = sub.loc[sub.index.isin(lo_idx), "views"].to_numpy(float)
        if len(h) or len(low_values):
            pool[a] = (h, low_values)
    if not pool:
        return [None, None]
    keys = list(pool)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        h = np.concatenate([pool[keys[i]][0] for i in pick])
        low_values = np.concatenate([pool[keys[i]][1] for i in pick])
        if len(h) == 0 or len(low_values) == 0:
            continue
        mh, ml = float(np.median(h)), float(np.median(low_values))
        if not np.isfinite(mh) or not np.isfinite(ml) or ml == 0:
            continue
        v = mh / ml
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [None, None]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def rho_ci_author(
    te: pd.DataFrame,
    col: str,
    cluster: str = "author_key",
    n_boot: int = 1000,
    seed: int = config.SEED,
) -> tuple[float | None, list[float]]:
    """Спирмен колонки с просмотрами и 95% ДИ по авторам."""
    rho = (
        float(stats.spearmanr(te[col], te["views"]).statistic) if len(te) > 2 else None
    )
    groups = _author_groups(te, cluster)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        part = te.loc[_resample_idx(groups, rng)]
        r = stats.spearmanr(part[col], part["views"]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    ci = (
        [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
        if vals
        else [None, None]
    )
    return rho, ci


def delta_rho_two_scores(
    te: pd.DataFrame,
    col_a: str,
    col_b: str,
    cluster: str = "author_key",
    n_boot: int = 1000,
    seed: int = config.SEED,
) -> dict:
    """Разность корреляций двух score-колонок на общих выборках (A-B)."""
    groups = _author_groups(te, cluster)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        part = te.loc[_resample_idx(groups, rng)]
        a = stats.spearmanr(part[col_a], part["views"]).statistic
        b = stats.spearmanr(part[col_b], part["views"]).statistic
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(float(a - b))
    ci = (
        [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
        if diffs
        else [None, None]
    )
    return {"delta_rho": float(np.mean(diffs)) if diffs else None, "ci95": ci}


def shuffle_labels_within_month(te: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Негативный контроль: случайно перемешать целевые значения ВНУТРИ периода.

    Сохраняет распределение (дополнительный сигнал модели должен исчезнуть в
    среднем); применяется к копии, исходная рамка не меняется."""
    rng = np.random.default_rng(seed)
    out = te.copy()
    shuffled = np.empty(len(out))
    for _, idx in out.groupby("month_id").groups.items():
        vals = out.loc[idx, "views"].to_numpy()
        shuffled[out.index.get_indexer(idx)] = rng.permutation(vals)
    out["views"] = shuffled
    return out


def bootstrap_median_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = config.SEED,
    cluster: np.ndarray | None = None,
) -> list[float]:
    """ДИ медианы (кластерный, если переданы кластеры).

    Без cluster перевыбираются наблюдения; с cluster — целые группы."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if cluster is None:
        samples = [v[rng.integers(len(v), size=len(v))] for _ in range(n_boot)]
    else:
        groups = [x for _, x in pd.DataFrame({"v": v, "c": cluster}).groupby("c")["v"]]
        samples = []
        for _ in range(n_boot):
            pick = rng.integers(len(groups), size=len(groups))
            samples.append(np.concatenate([groups[i].to_numpy() for i in pick]))
    meds = [np.median(sample) for sample in samples if len(sample)]
    meds = [m for m in meds if np.isfinite(m)]
    return (
        [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]
        if meds
        else [None, None]
    )
