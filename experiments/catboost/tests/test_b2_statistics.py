"""Проверка персональных оценок на небольшой матрице альтернатив."""

import numpy as np

from src.slot_policy import _eligible_slot_statistics


def test_actual_slot_is_excluded_from_its_own_alternatives():
    predictions = np.array([[10.0, 20.0, 30.0], [3.0, 6.0, 9.0]])
    actual = np.array([21.0, 4.0])
    supported = np.array([0, 1, 2])
    actual_positions = np.array([1, -1])

    medians, relative, pct, count = _eligible_slot_statistics(
        predictions, actual, supported, actual_positions
    )
    assert count.tolist() == [2, 3]
    assert medians.tolist() == [20.0, 6.0]
    assert np.isclose(pct[0], 0.5)
    assert np.isclose(pct[1], 1.0 / 3.0)
    assert np.all(np.isfinite(relative))
