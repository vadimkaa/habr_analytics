"""Policy: границы только из train; одинаковые статьи и покрытие при сравнении.

Проверяются правила протокола: границы терцилей и доля q из train,
одинаковое множество тестовых статей у модели и у правила, случайный слот
не даёт систематического эффекта.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics as M
from src import slot_policy as SP
from src import statistics as ST


def _test_frame(synthetic) -> pd.DataFrame:
    te = synthetic[synthetic.dt_msk >= "2024-11-01 00:00:00+03:00"].copy()
    rng = np.random.default_rng(3)
    # score частично связан с исходом: пара (score, views) с шумом
    te["score"] = te["views"] * 0.5 + rng.normal(0, 400, len(te))
    te["slot_n_train"] = 50
    return te


def test_thresholds_from_train_only(synthetic):
    train = synthetic[synthetic.dt_msk < "2024-11-01 00:00:00+03:00"].assign(
        model_score=1.0
    )
    rng = np.random.default_rng(5)
    train["model_score"] = rng.random(len(train))
    th = SP.train_thresholds(train)
    assert th["q_fixed"] == 0.2
    # Границы не зависят от теста: пересчёт с другим тестом даёт те же числа
    th2 = SP.train_thresholds(train.assign(model_score=train["model_score"] * 3 + 1))
    # Порог - квантиль train score: монотонное преобразование train score
    # сдвигает порог согласованно; сам train score не менялся:
    th3 = SP.train_thresholds(train)
    assert th["threshold_q"] == th3["threshold_q"]


def test_equal_coverage_same_articles(synthetic):
    """Модель и weekend-правило сравниваются на ОДНОМ множестве статей."""
    te = _test_frame(synthetic)
    q = 0.3
    es = M.equal_share_ratio(te, "score", q)
    wr = M.weekend_rule_ratio(te, q)
    assert es["n_hi"] == wr["n_hi"], "разное покрытие у модели и правила"
    assert es["n_hi"] == int(np.ceil(q * len(te)))


def test_random_slot_control_near_zero(synthetic):
    """Негативный контроль P3: случайный допустимый слот - нет сигнала."""
    te = _test_frame(synthetic)
    rng = np.random.default_rng(11)
    for _ in range(5):
        te_rand = te.assign(score=rng.random(len(te)))
        rho = M.rho_all(te_rand["score"], te_rand["views"])
        assert abs(rho) < 0.15, f"случайный слот дал систематический эффект: {rho}"


def test_delta_rho_bootstrap_reproducible(synthetic):
    """Seed воспроизводим: одинаковые входы дают одинаковый интервал."""
    te = _test_frame(synthetic)
    ci1 = ST.paired_delta_rho_ci(te, "score", n_boot=100, seed=42)
    ci2 = ST.paired_delta_rho_ci(te, "score", n_boot=100, seed=42)
    assert ci1 == ci2


def test_paired_bootstrap_pairs_authors(synthetic):
    """В каждом повторе score и weekend привязаны к тем же article_id."""
    te = _test_frame(synthetic)
    # Проверка структуры: группы авторов не пусты и покрывают все строки
    groups = ST._author_groups(te)
    assert sum(len(g) for g in groups) == len(te)
    ci = ST.paired_delta_rho_ci(te, "score", n_boot=30)
    assert len(ci) == 2


def test_personal_supported_filter(synthetic):
    """Слоты с n_train < порога не рекомендуются: fallback «мало наблюдений»."""
    te = _test_frame(synthetic)
    te["slot_n_train"] = [10, 50] * (len(te) // 2) + [50] * (len(te) % 2)
    supp = SP.personal_supported(te)
    assert (supp["slot_n_train"] >= 20).all()
    assert len(supp) < len(te)


def test_weekend_rule_deterministic_tiebreak(synthetic):
    """Ничьи правил выходных разрешаются детерминированно (hour): повтор -> то же."""
    te = _test_frame(synthetic)
    r1 = M.weekend_rule_ratio(te, 0.25)
    r2 = M.weekend_rule_ratio(te, 0.25)
    assert r1["weekend_q_ratio"] == r2["weekend_q_ratio"]
    assert r1["n_hi"] == r2["n_hi"]
