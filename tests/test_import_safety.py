"""Проверки того, что пакеты исследования не затеняют стандартную библиотеку."""

import statistics


def test_standard_statistics_module_available():
    assert statistics.median([1, 3, 5]) == 3
    assert statistics.__file__.endswith("statistics.py")
    assert "experiments/catboost" not in statistics.__file__
