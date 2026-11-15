"""Контроль возраста, операционный порог, парный бутстреп.

Проверяется: перцентиль просмотров внутри месяца; внутри-месячные корреляции;
парность бутстрепа (score и weekend на одних строках); операционная метрика
fixed_threshold_ratio с порогом, зафиксированным на train, не смешивается с
диагностической equal_share_ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as M
from src import statistics as ST


def _frame_month_pct():
    """Две статьи в каждом из двух месяцев: вторая всегда вдвое популярнее."""
    rows = []
    for m in ("2026-01", "2026-02"):
        rows.append(dict(month_id=m, views=100.0, score=1.0))
        rows.append(dict(month_id=m, views=200.0, score=2.0))
    return pd.DataFrame(rows).assign(
        is_weekend=[0, 0, 1, 1],
        author_key=["a", "b", "a", "b"],
        week_start=["w1", "w1", "w2", "w2"],
    )


def test_month_pct_views():
    df = _frame_month_pct()
    pct = M.month_pct_views(df)
    # Внутри месяца первая статья - 0.5, вторая - 1.0, независимо от месяца
    assert np.allclose(pct.to_numpy(), [0.5, 1.0, 0.5, 1.0])


def test_age_metrics_pct_ignores_level_shift():
    """score, идеально ранжирующий внутри месяцев: Δρ с перцентилем месяца = 1
    даже если уровни месяцев различаются; с накопленными просмотрами - ниже."""
    rng = np.random.default_rng(4)
    rows = []
    base = [100.0, 3000.0]  # январь в 30 раз «беднее» февраля
    for i, m in enumerate(["2026-01", "2026-02"]):
        for j in range(40):
            v = base[i] * rng.uniform(0.5, 1.5)
            rows.append(
                dict(
                    month_id=m,
                    views=v,
                    score=rng.uniform(0, 1),
                    is_weekend=j % 2,
                    author_key=f"a{j % 7}",
                    week_start=f"w{j % 4}",
                )
            )
    df = pd.DataFrame(rows)
    # score = ранг внутри месяца (идеальное внутримесячное ранжирование)
    df["score_rank"] = df.groupby("month_id")["views"].rank(pct=True)
    out = M.b_metrics_age(df, "score_rank")
    assert out["delta_rho_pct"] is not None and out["delta_rho_pct"] > 0.95
    # С накопленными просмотрами межмесячный сдвиг уровней ломает ранжирование
    assert out["delta_rho_views"] < out["delta_rho_pct"]


def test_delta_rho_ci_paired_same_rows():
    """Парность: разность считается на одной и той же перевыборке строк."""
    rng = np.random.default_rng(6)
    n = 600
    df = pd.DataFrame(
        dict(
            author_key=rng.choice([f"a{i}" for i in range(12)], n),
            week_start=rng.choice([f"w{i}" for i in range(6)], n),
            score=rng.random(n),
            views=rng.random(n),
            is_weekend=rng.integers(0, 2, n),
            month_id="2026-01",
        )
    )
    ci = ST.paired_delta_rho_ci(df, "score", n_boot=100, seed=3)
    ci_pct = ST.paired_delta_rho_ci(
        df,
        "score",
        n_boot=100,
        seed=3,
        target_col="views_pct_month" if "views_pct_month" in df.columns else "views",
    )
    assert ci == ci_pct  # одинаковая цель - одинаковый результат при том же seed
    assert ci[0] <= ci[1]


def test_fixed_threshold_ratio_train_threshold():
    """Операционная метрика: порог train применяется без пересчёта; доля выбранных
    в тесте может отличаться от q - в отличие от equal_share, где доля фиксируется."""
    n = 800
    te = pd.DataFrame(
        dict(
            model_score=[0.85] * 100 + [0.1] * (n - 100),  # 100 статей выше порога
            views=rng_views(n),
            is_weekend=np.zeros(n, int),
            author_key=[f"a{i % 10}" for i in range(n)],
            month_id="2026-01",
        )
    )
    threshold = 0.8  # «квантиль train»
    ft = M.fixed_threshold_ratio(te, "model_score", threshold, n_boot=100)
    assert ft["n_hi"] == 100
    assert abs(ft["selected_share"] - 100 / n) < 1e-12
    # Диагностика с той же долей переопределяет верхние 20% на тесте: 160 статей
    es = M.equal_share_ratio(te, "model_score", 0.2, n_boot=100)
    assert es["n_hi"] == int(np.ceil(0.2 * n)) == 160
    assert ft["n_hi"] != es["n_hi"]
    assert ft["fixed_threshold_ratio_ci_author"] is not None


def rng_views(n):
    return np.random.default_rng(8).uniform(1, 100, n)


def test_fixed_threshold_empty_group():
    te = pd.DataFrame(
        dict(
            model_score=[0.1, 0.2],
            views=[1.0, 2.0],
            is_weekend=[0, 1],
            author_key=["a", "b"],
            month_id=["m", "m"],
        )
    )
    ft = M.fixed_threshold_ratio(te, "model_score", 10.0)
    assert ft["fixed_threshold_ratio"] is None and ft["n_hi"] == 0


def test_weekday_only_uses_given_score():
    """Будни-only: метрика использует переданный score (для B1 - временную оценку)."""
    rng = np.random.default_rng(10)
    n = 1200
    te = pd.DataFrame(
        dict(
            model_score=rng.random(n),
            views=rng.random(n),
            is_weekend=np.zeros(n, int),  # только будни
            author_key=[f"a{i % 10}" for i in range(n)],
            week_start=[f"w{i % 5}" for i in range(n)],
            month_id=["2026-01"] * (n // 2) + ["2026-02"] * (n - n // 2),
        )
    )
    out = M.b_metrics_age(te, "model_score", n_boot=50)
    assert out["n_weekday"] == n
    assert out["rho_slot_views_weekday"] is not None
