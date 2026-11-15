"""Целевые переменные: encode/decode согласованы, масштаб Poisson конечен."""

from __future__ import annotations

import numpy as np
import pytest

from src.targets import TargetSpec, VIEWS_PER_UNIT, VIEWS_PER_UNIT_ALT, check_roundtrip


def test_roundtrip_all_kinds():
    views = np.array([0.0, 1.0, 999.0, 1000.0, 166_000.0, 1_670_000.0])
    for kind in ("views_raw", "log1p", "divide_1000", "divide_10000"):
        spec = TargetSpec(kind)
        check_roundtrip(views, spec)
        # decode(encode(y)) == y для всех перечисленных значений
        assert np.allclose(spec.decode(spec.encode(views)), views)


def test_decode_clips_negatives():
    spec = TargetSpec("log1p")
    assert spec.decode(np.array([-5.0]))[0] == 0.0
    spec2 = TargetSpec("divide_1000")
    assert spec2.decode(np.array([-2.0]))[0] == 0.0


def test_poisson_scale_is_scaled():
    """Протокол 8.3: Poisson учится на views/1000; /10000 - проверка масштаба.

    На максимальных вьюхах корпуса (~1.67 млн) оба масштаба дают конечные
    значения; сырые просмотры численно неустойчивы (пробный запуск) и запрещены."""
    views = np.array([1_670_000.0])
    assert TargetSpec("divide_1000").encode(views)[0] == pytest.approx(1670.0)
    assert TargetSpec("divide_10000").encode(views)[0] == pytest.approx(167.0)
    assert VIEWS_PER_UNIT == 1000.0 and VIEWS_PER_UNIT_ALT == 10000.0


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        TargetSpec("month_pct")
