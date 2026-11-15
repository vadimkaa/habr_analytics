"""Пин определения корпусов, из которого считаются 46 845 уникальных ID.

Фильтры load_corpus включают отсечение по моменту снятия счётчиков (снапшоту)
и нижнюю границу широкого корпуса 2024-11-09; без них наивный пересчёт даёт
29 451 запись широкого корпуса и 46 893 уникальных. Тест фиксирует и определение,
и совпадение с зафиксированным результатом 09_robustness."""

import json
from pathlib import Path

from habrpub.analysis import load_corpus


def test_identity_matches_article():
    a = set(load_corpus("ai").id)
    b = set(load_corpus("wide").id)
    res = json.loads(Path("data/results/09_robustness_results.json").read_text())[
        "identity"
    ]
    assert res["ai"] == len(a) == 21389
    assert res["wide"] == len(b) == 29399
    assert res["overlap"] == len(a & b) == 3943
    assert res["unique"] == len(a | b) == 46845
