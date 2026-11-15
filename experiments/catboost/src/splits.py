"""Временные фолды: rolling-origin без утечек между наборами.

Протокол 6.1-6.4: только хронологическое разбиение; ни один ID не попадает в
разные фолды; в deployable-наборе нет будущих входов. Функции здесь - единая
точка резки: обучение, валидация (выбор loss/гиперпараметров), доп. проверка
2025 (ИИ/МЛ) и зафиксированный аудит 2026 (audit_exposed).
"""

from __future__ import annotations

import pandas as pd

from . import config, features as feats
from .data_contract import load_corpus_data


def materialize_corpus(corpus: str) -> pd.DataFrame:
    """Корпус с базовыми признаками (без expected-плотности: она train-only)."""
    df = feats.prepare(load_corpus_data(corpus).df)
    return df


def split_frame(df: pd.DataFrame, corpus: str) -> dict[str, pd.DataFrame]:
    """Временные фолды для корпуса; границы - Europe/Moscow."""
    out = {}
    for name in config.folds(corpus):
        out[name] = _fold_slice(df, corpus, name)
    _assert_no_overlap(out)
    return out


def _fold_slice(df: pd.DataFrame, corpus: str, name: str) -> pd.DataFrame:
    spec = config.folds(corpus)[name]
    d = df
    if spec.start is not None:
        d = d[d.dt_msk >= pd.Timestamp(spec.start, tz="Europe/Moscow")]
    if spec.end is not None:
        d = d[d.dt_msk < pd.Timestamp(spec.end, tz="Europe/Moscow")]
    return d.copy()


def _assert_no_overlap(folds: dict[str, pd.DataFrame]) -> None:
    seen: set[str] = set()
    for name, f in folds.items():
        ids = set(f["id"])
        inter = seen & ids
        if inter:
            raise AssertionError(
                f"ID дублируется между фолдами: {name}: {sorted(inter)[:5]}"
            )
        seen |= ids


def check_chronology(train: pd.DataFrame, *later: pd.DataFrame) -> None:
    """train заканчивается раньше начала каждого следующего набора."""
    t_max = pd.Timestamp(train.dt_msk.max())
    for part in later:
        if len(part) == 0:
            continue
        p_min = pd.Timestamp(part.dt_msk.min())
        if p_min < t_max:
            raise AssertionError(
                f"фолды пересекаются по времени: train max {t_max} > next min {p_min}"
            )


def rolling_splits(df: pd.DataFrame, corpus: str) -> list[dict]:
    """Rolling-origin фолды для финалистов (гиперпараметры НЕ подбираются)."""
    out = []
    for spec in config.rolling_folds(corpus):
        cut = pd.Timestamp(spec["cut"], tz="Europe/Moscow")
        end = cut + pd.DateOffset(months=spec["test_months"])
        train = df[df.dt_msk < cut].copy()
        test = df[(df.dt_msk >= cut) & (df.dt_msk < end)].copy()
        if len(train) < 500 or len(test) < 100:
            continue
        out.append({"cut": spec["cut"], "train": train, "test": test})
    return out


def fit_expected_density_column(
    train: pd.DataFrame, frames: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """dens24_expected: формула из train, применяется к train (LOO) и к test.

    Возвращает те же фолды с колонкой dens24_expected. Тестовые строки не
    участвуют в оценке плотности ни в каком виде (негативный тест
    test_policy_train_only).
    """
    train = train.copy()
    train = feats.add_expected_density(train, train, loo=True)
    out = {"train": train}
    for name, f in frames.items():
        if name == "train":
            continue
        f2 = f.copy()
        f2 = feats.add_expected_density(f2, train, loo=False)
        out[name] = f2
    return out
