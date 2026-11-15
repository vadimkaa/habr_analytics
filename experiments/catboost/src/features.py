"""Признаки: набор F0-F5, допубликационная доступность и dens24_expected.

Признаки сопоставимы с контрольной моделью LightGBM (08_model.prepare).
Каждый признак помечен как доступный до публикации (prepublication) либо
диагностический (diagnostic):
 - dens24_actual - ТОЛЬКО диагностика уже вышедших статей;
 - dens24_expected - train-only сглаженная плотность слота, применимая к будущим
   слотам без подглядывания; для обучающих строк считается leave-one-out,
   чтобы собственная фактическая плотность не просачивалась в признак.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

from . import config

# Признак -> доступность. dens24_actual и month/year известны для календарного
# анализа, но в deployable-наборы (F3) не входят.
AVAILABILITY = {
    "dow": "prepublication",
    "hour": "prepublication",
    "hour_sin": "prepublication",
    "hour_cos": "prepublication",
    "is_weekend": "prepublication",
    "is_holiday": "prepublication",
    "month": "prepublication",  # месяц-в-году целым числом 1-12 (10.2)
    "year_epoch": "diagnostic",  # прокси эпохи/возраста; только абляция H5
    "dens24": "diagnostic",  # фактическая плотность предыдущих 24 ч
    "dens24_expected": "prepublication",
    "dens24_next": "diagnostic",  # интенсивность следующих 24 ч (сенситивность)
    "t_len": "prepublication",
    "t_words": "prepublication",
    "t_digit": "prepublication",
    "t_qmark": "prepublication",
    "t_colon": "prepublication",
    "is_corp": "prepublication",
    "prior_count": "prepublication",  # только число ранее вышедших статей автора
    "has_prior": "prepublication",
    "hub_top": "diagnostic",  # принадлежность фиксированному списку хабов
}

# Именованные наборы. Цель набора - в комментарии.
FEATURE_SETS: dict[str, dict] = {
    "F0_time": dict(
        features=["dow", "hour", "hour_sin", "hour_cos", "is_weekend", "is_holiday"],
        purpose="базовый сигнал времени",
    ),
    "F1_meta": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="честный pre-publication baseline",
    ),
    "F2_dens_actual": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "dens24",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="диагностика: фактическая плотность ранее вышедших статей",
    ),
    "F3_dens_expected": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "dens24_expected",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="имитация выбора будущего времени: deployable-набор без подглядывания",
    ),
    "F4_season": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "month",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="разделить сезонность и исторический сдвиг (месяц-в-году)",
    ),
    "F4b_epoch": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "dens24",
            "month",
            "year_epoch",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="диагностика: явный прокси эпохи; общий Спирмен растёт, внутри месяцев нет",
    ),
    "F5_hubs": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="тема/состав: набор расширяется мультихабовыми флагами в prepare",
    ),
    # Набор признаков контрольной модели LightGBM (08_model.FULL).
    "V24_full": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "month",
            "is_weekend",
            "is_holiday",
            "dens24",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "prior_count",
            "has_prior",
            "is_corp",
        ],
        purpose="контрольный набор LightGBM с месяцем и плотностью",
    ),
    # Абляции (A3-A6) на базе F1/F2:
    "ABL_no_time": dict(
        features=[
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="B6-аналог: заголовок + авторские + тип блога; календаря нет",
    ),
    "ABL_no_author": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "dens24",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
        ],
        purpose="A5: не держится ли качество на суррогате регулярности автора",
    ),
    "ABL_no_title": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "dens24",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="A6: доступность двойных метаданных заголовка",
    ),
    "ABL_no_dens_month": dict(
        features=[
            "dow",
            "hour",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_holiday",
            "t_len",
            "t_words",
            "t_digit",
            "t_qmark",
            "t_colon",
            "is_corp",
            "prior_count",
            "has_prior",
        ],
        purpose="диагностика роли плотности и календарного месяца",
    ),
}

HUB_FLAG_LIMIT = 12
HUB_FLAG_PREFIX = "hub_top_"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Признаки статьи с теми же определениями, что в 08_model.prepare."""
    df = df.sort_values(["dt_utc", "id"]).reset_index(drop=True).copy()
    from habrpub.analysis import previous_counts, weekday_holidays

    prior = previous_counts(df)
    df["prior_count"] = np.log1p(prior)
    df["has_prior"] = (prior > 0).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)
    hol = weekday_holidays(df.year.unique())
    df["is_holiday"] = df.date_msk.isin(hol).astype(int)
    title = df.title.fillna("").map(lambda t: html.unescape(re.sub("<[^>]+>", " ", t)))
    df["t_len"] = title.str.len()
    df["t_words"] = title.str.split().str.len()
    df["t_digit"] = title.str.contains(r"\d").astype(int)
    df["t_qmark"] = title.str.contains(r"\?").astype(int)
    df["t_colon"] = title.str.contains(r"[:—-]").astype(int)
    df["month"] = df.dt_msk.dt.month.astype(int)
    df["year_epoch"] = (df.year - 2016).astype(int)
    for c in ["is_weekend", "is_corp"]:
        df[c] = df[c].astype(int)
    return df


def add_expected_density(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    loo: bool = False,
    k: int | None = None,
) -> pd.DataFrame:
    """dens24_expected: сглаженная слотовая медиана плотности из train.

    loo=True - строки df сами входят в train_df: признак каждой строки
    считается по медиане ДРУГИХ строк того же слота, чтобы собственная
    фактическая плотность не просачивалась в обучающий признак.
    loo=False - тестовые/сеточные строки: медиана слота по train целиком.
    Слоты без истории получают общую медиану train.
    """
    k = config.protocol()["policy"]["expected_density_shrinkage_k"] if k is None else k
    train = train_df
    global_med = float(train["dens24"].median())
    slot_stats = train.groupby(["dow", "hour"])["dens24"].agg(["size", "median"])

    def expected_from(df_part: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=df_part.index)
        # Группировка по исходным меткам индекса: .groups даёт именно метки,
        # а не позиции (reset_index() здесь ломал соответствие).
        for (d, h), idx in df_part.groupby(["dow", "hour"]).groups.items():
            n, med = (
                slot_stats.loc[(d, h)] if (d, h) in slot_stats.index else (0, np.nan)
            )
            vals = train.loc[(train.dow == d) & (train.hour == h), "dens24"].to_numpy()
            if loo and len(idx):
                # Медиана остальных: для каждой строки исключаем её значение.
                own = df_part.loc[idx, "dens24"].to_numpy()
                sorted_vals = np.sort(vals)
                exp = np.empty(len(idx))
                for j, x in enumerate(own):
                    pos = np.searchsorted(sorted_vals, x)
                    rest = np.delete(sorted_vals, pos)
                    exp[j] = np.median(rest) if len(rest) else global_med
                med_j = exp
            else:
                med_j = np.full(len(idx), med if np.isfinite(med) else global_med)
            w = n / (n + k)
            out.loc[idx] = w * np.asarray(med_j, float) + (1 - w) * global_med
        # Слоты, которых нет в train: чистая глобальная медиана.
        miss = out.index[out.isna()]
        out.loc[miss] = global_med
        return out

    res = df.copy()
    res["dens24_expected"] = expected_from(res).astype(float)
    # Интенсивность следующих 24 ч - только сенситивность, не признак политики.
    res["dens24_next"] = res.get("dens24_fwd", pd.Series(np.nan, index=res.index))
    return res


def add_hub_flags(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """F5: мультихабовые флаги за зафиксированным списком хабов.

    Список хабов фиксируется по train (частота), значение - бинарный флаг.
    Это проверка тем/состава, а не семантики.
    """
    import ast

    def hub_list(s):
        try:
            v = ast.literal_eval(s) if isinstance(s, str) else s
            return [str(x) for x in v] if v is not None else []
        except Exception:
            return []

    tl = train_df["hubs"].map(hub_list)
    from collections import Counter

    cnt = Counter(x for row in tl for x in row)
    top = [h for h, _ in cnt.most_common(HUB_FLAG_LIMIT)]
    res = df.copy()
    hs = res["hubs"].map(hub_list)
    for h in top:
        col = HUB_FLAG_PREFIX + h
        res[col] = hs.map(lambda row: int(h in row)).astype(int)
        AVAILABILITY[col] = "prepublication"
    res.attrs["hub_columns"] = [HUB_FLAG_PREFIX + h for h in top]
    return res


def materialize(corpus: str, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """Полный корпус с признаками именованного набора (без разбиения).

    Возвращает (df со всеми колонками-признаками, список признаков).
    Плотность expected считается по всему зрелому корпусу как train-источнику:
    при фактическом обучении передаётся train_df, поэтому здесь используется
    весь корпус ТОЛЬКО как заготовка колонок; численно это та же формула,
    но обучению запрещено видеть dens24_expected тестовых строк (splits.py).
    """
    from .data_contract import load_corpus_data

    df = prepare(load_corpus_data(corpus).df)
    if feature_set == "F5_hubs":
        df = add_hub_flags(df, df)
        fs = list(FEATURE_SETS["F5_hubs"]["features"]) + df.attrs["hub_columns"]
    else:
        fs = list(FEATURE_SETS[feature_set]["features"])
    if "dens24_expected" in fs or "dens24_next" in fs:
        df = add_expected_density(df, df)
    return df, fs


def check_availability(feature_set: str, features: list[str]) -> None:
    """Защитный фильтр: deployable-наборы не содержат post-publication полей.

    Негативный тест 16.1: dens24_actual в обучении с dens24_expected запрещён.
    """
    bad = [
        f
        for f in features
        if AVAILABILITY.get(f) == "diagnostic"
        and f not in ("dens24", "month", "year_epoch", "dens24_next", "hub_top")
    ]
    if bad:
        raise ValueError(f"diagnostic features in set {feature_set}: {bad}")
    if feature_set in ("F3_dens_expected",) and "dens24" in features:
        raise ValueError("F3 deployable не может включать dens24_actual")


def cat_features_for(features: list[str], cat_mode: str) -> list[str]:
    """Какие колонки подаются как категориальные при данном режиме кодирования."""
    if cat_mode == "categorical":
        return [c for c in ("dow", "hour", "month") if c in features]
    if cat_mode == "one_hot":
        return [c for c in ("dow", "hour") if c in features]
    return []


def cyclic_only_rename(features: list[str]) -> list[str]:
    """Режим cyclic: убрать сырые dow/hour, оставив sin/cos (абляция A7)."""
    return [f for f in features if f not in ("dow", "hour", "month")]
