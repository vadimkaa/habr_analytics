"""ratio_ci_author использует ПОЛНУЮ совокупность авторов.

В пул попадают авторы, представленные хотя бы в одной группе.
Проверяемые случаи: авторы только в одной группе; часть в обеих;
сильно различающееся число статей; пустая группа после перевыборки;
воспроизводимость при фиксированном seed. Плюс проверка, что paired-бутстреп
delta_rho перевыбирает исходные авторские кластеры целиком.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import statistics as ST


def _frame(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """rows: (author, views, group)."""
    return pd.DataFrame(
        {
            "author_key": [a for a, _, _ in rows],
            "views": [v for _, v, _ in rows],
            "grp": [g for _, _, g in rows],
        }
    )


def test_authors_only_in_one_group():
    """Все авторы представлены только в одной из групп - пул не пуст."""
    te = _frame(
        [
            ("a1", 2000.0, "hi"),
            ("a1", 2200.0, "hi"),
            ("a2", 1800.0, "hi"),
            ("a3", 1000.0, "lo"),
            ("a3", 900.0, "lo"),
            ("a4", 1100.0, "lo"),
        ]
    )
    hi = te.grp == "hi"
    lo = te.grp == "lo"
    ci = ST.ratio_ci_author(te, hi, lo, n_boot=300, seed=42)
    assert ci != [None, None], "дизъюнктные группы авторов исключили весь пул"
    lo_v, hi_v = ci
    assert hi_v > lo_v > 1.0


def test_some_authors_in_both_groups():
    """Часть авторов в обеих группах - интервал считается и не вырожден."""
    rng = np.random.default_rng(5)
    rows = []
    for a in ("a1", "a2"):  # в обеих группах
        for _ in range(20):
            rows.append((a, rng.uniform(1500, 2500), "hi"))
            rows.append((a, rng.uniform(700, 1200), "lo"))
    for a in ("a3", "a4", "a5"):  # только в одной
        for _ in range(10):
            rows.append((a, rng.uniform(1500, 2500), "hi"))
    for a in ("b1", "b2"):
        for _ in range(10):
            rows.append((a, rng.uniform(700, 1200), "lo"))
    te = _frame(rows)
    ci = ST.ratio_ci_author(te, te.grp == "hi", te.grp == "lo", n_boot=300, seed=7)
    assert ci != [None, None]
    assert ci[1] >= ci[0] > 1.0


def test_highly_unbalanced_author_sizes():
    """Один автор с сотнями статей и авторы с единичными - без вырождений."""
    rng = np.random.default_rng(11)
    rows = [("big", rng.uniform(900, 1100), "lo") for _ in range(500)]
    rows += [("big", rng.uniform(1800, 2200), "hi") for _ in range(500)]
    for i in range(30):  # мелкие авторы: только одна статья
        rows.append((f"m{i}", rng.uniform(1500, 2600), "hi" if i % 2 else "lo"))
    te = _frame(rows)
    ci = ST.ratio_ci_author(te, te.grp == "hi", te.grp == "lo", n_boot=300, seed=3)
    assert ci != [None, None]
    lo_v, hi_v = ci
    # В перевыборках без «big» мелкие авторы дают ratio около 1; с «big» - около 2:
    # интервал конечен, положителен и отражает сигнал крупного кластера сверху.
    assert 0 < lo_v <= hi_v
    assert hi_v > 1.5


def test_empty_group_after_resampling():
    """Перевыборка может оставить одну из групп пустой - повтор пропускается,
    функция возвращает конечный интервал по валидным повторам."""
    te = _frame(
        [
            ("a", 2000.0, "hi"),  # автор только в верхней
            ("b", 1000.0, "lo"),  # автор только в нижней
        ]
    )
    ci = ST.ratio_ci_author(te, te.grp == "hi", te.grp == "lo", n_boot=200, seed=42)
    assert ci != [None, None]
    lo_v, hi_v = ci
    # Единственный валидный вариант перевыборки: {a, b} -> отношение медиан 2
    assert abs(lo_v - 2.0) < 1e-9 and abs(hi_v - 2.0) < 1e-9


def test_all_replicates_invalid_returns_none():
    """Если группа пуста в КАЖДОМ повторе - честный [None, None] без исключения."""
    te = _frame([("a", 2000.0, "hi"), ("b", 1000.0, "hi")])
    ci = ST.ratio_ci_author(te, te.grp == "hi", te.grp == "lo", n_boot=100, seed=1)
    assert ci == [None, None]
    # Нулевой знаменатель: медиана нижней группы 0 -> повтор пропускается
    te0 = _frame([("a", 2000.0, "hi"), ("b", 0.0, "lo"), ("b2", 0.0, "lo")])
    ci0 = ST.ratio_ci_author(te0, te0.grp == "hi", te0.grp == "lo", n_boot=100, seed=1)
    assert ci0 == [None, None]


def test_seed_reproducibility():
    """Фиксированный seed - идентичный интервал (требование 6)."""
    rng = np.random.default_rng(21)
    rows = []
    for i in range(40):
        rows.append((f"a{i % 8}", rng.uniform(800, 2500), "hi" if i % 3 else "lo"))
    te = _frame(rows)
    hi, lo = te.grp == "hi", te.grp == "lo"
    assert ST.ratio_ci_author(te, hi, lo, n_boot=200, seed=123) == ST.ratio_ci_author(
        te, hi, lo, n_boot=200, seed=123
    )


def test_paired_delta_resamples_full_clusters():
    """paired_delta_rho_ci / _resample_idx перевыбирают исходные кластеры
    ЦЕЛИКОМ: каждый выбранный автор вносит все свои наблюдения."""
    rng = np.random.default_rng(9)
    n = 400
    te = pd.DataFrame(
        {
            "author_key": rng.choice([f"a{i}" for i in range(10)], size=n),
            "views": rng.uniform(0, 1000, n),
            "score": rng.uniform(0, 1000, n),
            "is_weekend": rng.integers(0, 2, n),
        }
    )
    sizes = te.groupby("author_key").size().to_dict()
    groups = ST._author_groups(te)
    rng2 = np.random.default_rng(0)
    idx = ST._resample_idx(groups, rng2)
    got = pd.Series(idx).value_counts()  # позиции строк
    # Собираем количество строк по авторам в перевыборке
    resampled_authors = te.iloc[idx].groupby("author_key").size().to_dict()
    # Каждому автору достаётся k*size_k строк его собственного кластера
    author_sizes = {a: s for a, s in sizes.items()}
    for a, cnt in resampled_authors.items():
        assert cnt % author_sizes[a] == 0, f"кластер {a} разорван: {cnt}"
    ci = ST.paired_delta_rho_ci(te, "score", n_boot=100, seed=5)
    assert len(ci) == 2
