"""Общие определения выборок и статистик. Наблюдательные сравнения, не эффекты вмешательства."""

import json
import numpy as np
import pandas as pd
from scipy import stats
from . import DATA, RESULTS, FIGS, SNAP_AI, SNAP_WIDE

# Общий сид всех случайных процедур и число бутстреп-повторений.
SEED = 20261111
N_BOOT = 1000
# Общее покрытие всех пяти хабов: первый полный день после начала infosecurity.
WIDE_COMMON_START = pd.Timestamp("2025-06-24", tz="Europe/Moscow")
RECENT_START = pd.Timestamp("2025-11-09", tz="Europe/Moscow")


def load_corpus(corpus):
    name = f"habr_{'ai_ml' if corpus == 'ai' else 'wide'}_raw.jsonl.gz"
    path = DATA / name
    if not path.exists():
        # Фоллбэк для структуры релиза research/data/raw/:
        # данные лежат в data/raw, код ищет их без ручных правок путей.
        path = DATA / "raw" / name
    df = pd.read_json(path, lines=True, compression="gzip")
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates("id").copy()
    df["dt_utc"] = pd.to_datetime(df["timePublished"], utc=True, errors="coerce")
    snap = SNAP_AI if corpus == "ai" else SNAP_WIDE
    df["views"] = pd.to_numeric(df["readingCount"], errors="coerce")
    df = df[
        (df.lang == "ru")
        & (df.postType == "article")
        & df.dt_utc.notna()
        & df.views.notna()
        & (df.views >= 0)
        & (df.dt_utc <= snap)
    ].copy()
    if corpus == "wide":
        df = df[df.dt_utc >= pd.Timestamp("2024-11-09", tz="UTC")].copy()
    df["dt_msk"] = df.dt_utc.dt.tz_convert("Europe/Moscow")
    df["date_msk"] = df.dt_msk.dt.date
    df["year"] = df.dt_msk.dt.year
    df["month"] = df.dt_msk.dt.month
    df["month_id"] = df.dt_msk.dt.strftime("%Y-%m")
    df["dow"] = df.dt_msk.dt.dayofweek
    df["hour"] = df.dt_msk.dt.hour
    # Выходной = нерабочий день недели по производственному календарю: рабочие
    # субботы/воскресья (is_workweekend) в группу выходных не попадают. Для дат
    # вне покрытия календаря (до 2016 года) флаг совпадает с днём недели.
    _, workweekend = offday_sets()
    df["is_weekend"] = (df.dow >= 5) & (~df.date_msk.isin(workweekend))
    # Перевод чужой статьи: стиль и успех частично за счёт оригинала. В замороженной
    # выгрузке признак есть только у ИИ/МЛ; в широком корпусе поле отсутствует.
    if "is_translation" in df.columns:
        df["is_translation"] = df["is_translation"].fillna(False).astype(bool)
    else:
        df["is_translation"] = False
    # Начало календарной недели (понедельник) - кластер для бутстрапа по неделям:
    # авторский бутстрап не учитывает общий новостной фон конкретной недели.
    df["week_start"] = df.dt_msk.dt.normalize() - pd.to_timedelta(df.dow, unit="D")
    df["is_corp"] = df.isCorporative.fillna(False).astype(bool)
    df["age_days"] = (snap - df.dt_utc).dt.total_seconds() / 86400
    df["mature"] = df.age_days >= 30
    df["vpd"] = df.views / df.age_days.clip(lower=1)
    df["log_vpd"] = np.log10(1 + df.vpd)
    df["log_views"] = np.log10(1 + df.views)
    for source, target in [
        ("commentsCount", "comments"),
        ("favoritesCount", "favorites"),
        ("score", "score"),
    ]:
        df[target] = pd.to_numeric(df[source], errors="coerce")
    # Неизвестный автор не становится общим псевдоавтором всех анонимных статей.
    df["author_key"] = df.author.fillna("").astype(str)
    missing = df.author_key.eq("")
    df.loc[missing, "author_key"] = "__unknown_" + df.loc[missing, "id"]
    df["corpus"] = corpus
    df = df.sort_values(["dt_utc", "id"]).reset_index(drop=True)
    df["dens24"] = prior_density(df.dt_utc)
    df["dens24_fwd"] = forward_density(df.dt_utc)
    df["density_complete"] = df.dt_utc >= df.dt_utc.min() + pd.Timedelta(days=1)
    # Прямое окно [t, t+24 ч) полно только там, где после статьи есть ещё сутки данных.
    df["density_complete_fwd"] = df.dt_utc <= df.dt_utc.max() - pd.Timedelta(days=1)
    if corpus == "wide":
        df["density_complete"] &= df.dt_msk >= WIDE_COMMON_START + pd.Timedelta(days=1)
    return df


def prior_density(times):
    """Число публикаций в [t−24 ч, t), без публикаций с тем же timestamp.

    Приведение к наносекундам обязательно: при входном dtype datetime64[us]
    константа суток в наносекундах без приведения расширяет
    окно в 1000 раз — вместо суток получается интервал в 1000 дней."""
    ts = _ns_timestamps(times)
    return np.searchsorted(ts, ts, side="left") - np.searchsorted(
        ts, ts - 86400 * 10**9, side="left"
    )


def forward_density(times):
    """Число публикаций в (t, t+24 ч): статьи, вышедшие сразу после и способные
    вытеснить эту из хронологической ленты. Одновременные публикации, как и в
    обратном окне, конкурентами не считаются. Зеркальна prior_density; для
    последних суток корпуса окно неполно — отсекается density_complete_fwd."""
    ts = _ns_timestamps(times)
    return np.searchsorted(ts, ts + 86400 * 10**9, side="left") - np.searchsorted(
        ts, ts, side="right"
    )


def _ns_timestamps(times):
    """Общий вход prior_density/forward_density: отсортированные метки в нс."""
    ts = pd.to_datetime(times, utc=True)
    ts = (
        ts.dt.as_unit("ns")
        if isinstance(ts, pd.Series)
        else pd.Series(ts).dt.as_unit("ns")
    )
    ts = ts.astype("int64").to_numpy()
    if np.any(np.diff(ts) < 0):
        raise ValueError("times must be sorted")
    return ts


def previous_counts(df):
    """Только число ранее опубликованных статей; счётчики успеха не читаются."""
    out = pd.Series(0, index=df.index, dtype=int)
    for _, sub in df.groupby("author_key", sort=False):
        ts = sub.dt_utc.astype("int64").to_numpy()
        out.loc[sub.index] = np.searchsorted(ts, ts, side="left")
    return out


def recent(df):
    return df[df.mature & (df.dt_msk >= RECENT_START)].copy()


_PROD_CAL = None


def production_calendar():
    """Производственный календарь РФ 2016-2026 (xmlcalendar.ru, по постановлениям
    Правительства РФ о переносе выходных). Файл зафиксирован в data/; колонки
    is_off (нерабочий день) и is_workweekend (рабочая суббота/воскресенье)."""
    global _PROD_CAL
    if _PROD_CAL is None:
        path = DATA / "production_calendar_2016_2026.csv"
        cal = pd.read_csv(path)
        cal["date"] = pd.to_datetime(cal["date"]).dt.date
        _PROD_CAL = cal
    return _PROD_CAL


def offday_sets(years=None):
    """Нерабочие дни по производственному календарю и рабочие выходные.

    Возвращается пара множеств: праздники и переносы на любой день недели
    плюс обычные суббота/воскресенье; отдельно - рабочие выходные, которые
    перенесённым праздником делать нерабочими нельзя."""
    cal = production_calendar()
    if years is not None:
        cal = cal[cal.date.map(lambda d: d.year in set(years))]
    off = set(cal.loc[cal.is_off == 1, "date"])
    workweekend = set(cal.loc[cal.is_workweekend == 1, "date"])
    return off - workweekend, workweekend


def weekday_holidays(years=None):
    """Нерабочие праздничные и перенесённые дни, выпавшие на будни (пн-пт).

    Суббота и воскресенье в этот набор не входят: они описываются признаком
    выходных отдельно, иначе праздники поглотили бы эффект выходных.
    Одноразовые нерабочие дни указов Президента 2020–2021 годов (карантин COVID-19,
    дни голосования) не включаются: это не ежегодные праздники и не переносы,
    а отдельный период с собственным контекстом поведения; для контрольных
    дат они по-прежнему нерабочие."""
    off, _ = offday_sets(years)
    cal = production_calendar()
    titles = dict(zip(cal.date, cal.holiday_title.fillna("")))
    one_off = {d for d, t in titles.items() if ("Нерабоч" in t) or ("голосован" in t)}
    return {d for d in off if d.weekday() < 5 and d not in one_off}


def fixed_holidays(years):
    """Только фиксированные календарные даты, без переносов. Нужна шагу 11:
    он воспроизводит исходную модель, где праздники задавались так, чтобы
    сравнение авторских признаков не смешивалось со сменой календаря.
    В остальном анализе используется производственный календарь."""
    from datetime import date

    return {
        date(int(y), m, d)
        for y in years
        for m, d in (
            [(1, d) for d in range(1, 9)]
            + [(2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)]
        )
    }


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(
        json.dumps(finite_json(obj), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "rho": None, "p": None}
    r, p = stats.spearmanr(x, y)
    return {"n": len(x), "rho": float(r), "p": float(p)}


def bh(pvalues):
    """Benjamini–Hochberg в пределах явно заданного семейства проверок."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    n = len(order)
    if n:
        out[order] = np.minimum(
            1, np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
        )
    return out


def bootstrap_cluster(df, statistic, cluster="author_key", n_boot=N_BOOT):
    """Перевыборка целых авторских кластеров. Не учитывает зависимость между авторами."""
    groups = [x.index.to_numpy() for _, x in df.groupby(cluster, sort=True)]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        )
        val = statistic(df.loc[idx])
        if np.isfinite(val):
            vals.append(val)
    return np.quantile(vals, [0.025, 0.975]).tolist() if vals else [None, None]


def compare_weekend(df, with_ci=True):
    w = df.loc[df.is_weekend, "vpd"]
    d = df.loc[~df.is_weekend, "vpd"]
    u, p = stats.mannwhitneyu(w, d, alternative="two-sided")
    res = {
        "n_weekend": len(w),
        "n_weekday": len(d),
        "median_weekend": w.median(),
        "median_weekday": d.median(),
        "ratio": w.median() / d.median(),
        "p_iid": p,
        "rank_biserial": 2 * u / (len(w) * len(d)) - 1,
    }
    if with_ci:
        res["ratio_ci_author"] = bootstrap_cluster(
            df,
            lambda z: (
                z.loc[z.is_weekend, "vpd"].median()
                / z.loc[~z.is_weekend, "vpd"].median()
            ),
        )
        # Дополнительная проверка: зависимость по неделям (~104 выходных дней в году
        # собраны в десятки кластеров, поэтому авторской зависимости мало).
        res["ratio_ci_week"] = bootstrap_cluster(
            df,
            lambda z: (
                z.loc[z.is_weekend, "vpd"].median()
                / z.loc[~z.is_weekend, "vpd"].median()
            ),
            cluster="week_start",
        )
    return res


def paired_authors(df, min_each=2):
    """Один голос на автора; только авторы с обеими группами и >=min_each в каждой."""
    a = df.groupby(["author_key", "is_weekend"]).vpd.agg(["median", "size"]).unstack()
    a = a.dropna()
    a = a[(a["size"][False] >= min_each) & (a["size"][True] >= min_each)]
    ratios = a["median"][True] / a["median"][False]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    rng = np.random.default_rng(SEED)
    # Отдельный авторский bootstrap: 2000 повторов, как в опубликованном срезе.
    ci = np.quantile(
        [np.median(rng.choice(ratios, len(ratios), replace=True)) for _ in range(2000)],
        [0.025, 0.975],
    )
    lr = np.log(ratios)
    return {
        "n_authors": len(ratios),
        "median_ratio": ratios.median(),
        "ci95": ci.tolist(),
        "wilcoxon_p": float(stats.wilcoxon(lr).pvalue) if np.any(lr != 0) else 1.0,
        "min_each": min_each,
    }


def loo_median(vals):
    """Leave-one-out медиана: для каждой строки группы — медиана остальных.

    Порядок строк сохраняется: delete убирает i-е значение в том же
    порядке, в котором значения пришли из датафрейма. Сортировка здесь
    ломала соответствие: удалялась не статья текущей строки, а статья с
    i-м по величине значением, из-за чего медиана «других статей автора»
    механически включала исход текущей статьи и раздувала разрыв терцилей.
    """
    v = np.asarray(vals, dtype=float)
    return np.array([np.median(np.delete(v, i)) for i in range(len(v))])


def author_slot_descriptive(df, min_articles=3):
    """Описательное сравнение силы автора и силы слота на одном срезе.

    Сила автора — медиана log VPD его ДРУГИХ статей в том же окне
    (leave-one-out: включение текущей статьи механически связывает группу
    с исходом и раздувает разрыв терцилей; минимум min_articles статей
    означает хотя бы две других). Группы — по этой «чистой» силе; в силу
    автора входит всё постоянное у него: темы, хабы, блог, аудитория,
    продвижение. Группы составлены из СТАТЕЙ, а не из авторов: плодовитый
    автор попадает в группу каждым своим разделом по отдельности, и один
    автор может стоять в разных группах. Слот — выходные/будни; в ячейках
    медианы накопленных просмотров. Это описание среза, не разложение
    вкладов и не прогноз."""
    d = df.copy()
    # Для оценки силы автора логарифм со своим нижним порогом, не log10(1+VPD).
    d["strength_log"] = np.log(d.vpd.clip(lower=0.01))
    a = d.groupby("author_key").agg(n=("id", "size"))
    a = a[a.n >= min_articles]
    if len(a) < 30:
        return {"status": "too_few_authors", "n_authors": int(len(a))}
    d = d[d.author_key.isin(a.index)].copy()

    d["author_strength"] = d.groupby("author_key").strength_log.transform(loo_median)
    q1, q2 = d.author_strength.quantile([1 / 3, 2 / 3])

    def tercile(m):
        return 0 if m <= q1 else (1 if m <= q2 else 2)

    d["author_tercile"] = d.author_strength.map(tercile)
    labels = ["нижняя", "средняя", "верхняя"]
    rows = []
    for t in (0, 1, 2):
        for wk in (False, True):
            s = d[(d.author_tercile == t) & (d.is_weekend == wk)]
            cell = {
                "tercile": t,
                "tercile_label": labels[t],
                "weekend": bool(wk),
                "n_articles": len(s),
                "n_authors": int(s.author_key.nunique()),
                "views_median": float(s.views.median()) if len(s) else None,
            }
            if len(s):
                cell["views_ci95_author"] = bootstrap_cluster(
                    s, lambda z: float(z.views.median())
                )
            rows.append(cell)
    wk = {
        (r["tercile"], r["weekend"]): r["views_median"]
        for r in rows
        if r["views_median"]
    }
    ratios = {
        "weekend_vs_weekday": {
            labels[t]: wk[(t, True)] / wk[(t, False)]
            for t in range(3)
            if wk.get((t, False))
        },
        "strong_vs_weak_weekday": (
            wk[(2, False)] / wk[(0, False)] if wk.get((0, False)) else None
        ),
    }
    # Доли дисперсии log10(1+views): прирост R² сверх log возраста для флага
    # выходных и для силы автора (без текущей статьи).
    y = np.log10(1 + d.views).to_numpy()

    def r2(X):
        return 1 - np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2) / np.sum(
            (y - y.mean()) ** 2
        )

    age = np.column_stack([np.log(d.age_days), np.ones(len(d))])
    with_wk = np.column_stack(
        [np.log(d.age_days), d.is_weekend.astype(float), np.ones(len(d))]
    )
    with_au = np.column_stack([np.log(d.age_days), d.author_strength, np.ones(len(d))])
    shares = {
        "weekend_over_age": float(r2(with_wk) - r2(age)),
        "author_loo_over_age": float(r2(with_au) - r2(age)),
    }
    # Сила автора зависит от того, какая статья исключена
    # (leave-one-out), поэтому один автор может попасть в разные группы.
    # Считаем, у скольких авторов LOO-медианы пересекают границу группы.
    spanning = int((d.groupby("author_key").author_tercile.nunique() > 1).sum())
    return {
        "definition": (
            "article groups by leave-one-out median log VPD of the author's "
            "other articles in the window (>= "
            f"{min_articles} articles); groups are made of articles, so one "
            "author can appear in several groups; slot = weekend/weekday; "
            "medians of accumulated views; descriptive slice, no contribution decomposition"
        ),
        "min_articles": min_articles,
        "n_authors": int(len(a)),
        "n_authors_spanning_groups": spanning,
        "cells": rows,
        "ratios": ratios,
        "variance_shares": shares,
    }


def setup_plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    FIGS.mkdir(exist_ok=True)
    return plt
