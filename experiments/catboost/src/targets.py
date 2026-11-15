"""Целевые переменные: encode/decode и правила обратного преобразования.

Протокол 5.2: каждую цель возвращает в вьюхах ровно один раз; Poisson -
обязательно на масштабированном счётчике (пробный запуск на сырых просмотрах давал
бесконечный loss и константную модель); Quantile на views/1000 возвращает
тысячи - умножить обратно; RMSE/Huber на log1p дают лог-число, views = expm1.
month_pct-ранги не используются как абсолютный прогноз.
"""

from __future__ import annotations

import numpy as np

VIEWS_PER_UNIT = 1000.0  # масштаб счётчика для Poisson/Quantile
VIEWS_PER_UNIT_ALT = 10000.0  # проверка масштаба на ограниченном наборе


class TargetSpec:
    """Спецификация цели: encode для обучения, decode для вьюх."""

    def __init__(self, kind: str):
        if kind not in ("views_raw", "log1p", "divide_1000", "divide_10000"):
            raise ValueError(f"unknown target kind: {kind}")
        self.kind = kind

    def encode(self, views: np.ndarray) -> np.ndarray:
        y = np.asarray(views, dtype=float)
        if self.kind == "views_raw":
            return y
        if self.kind == "log1p":
            return np.log1p(y)
        if self.kind == "divide_1000":
            return y / VIEWS_PER_UNIT
        if self.kind == "divide_10000":
            return y / VIEWS_PER_UNIT_ALT
        raise ValueError(self.kind)

    def decode(self, pred: np.ndarray) -> np.ndarray:
        """Единственное обратное преобразование; отрицательные отсекаются."""
        p = np.asarray(pred, dtype=float)
        if self.kind == "views_raw":
            return np.maximum(0.0, p)
        if self.kind == "log1p":
            return np.maximum(0.0, np.expm1(p))
        if self.kind == "divide_1000":
            return np.maximum(0.0, p) * VIEWS_PER_UNIT
        if self.kind == "divide_10000":
            return np.maximum(0.0, p) * VIEWS_PER_UNIT_ALT
        raise ValueError(self.kind)

    def __repr__(self) -> str:  # pragma: no cover
        return f"TargetSpec({self.kind})"


def check_roundtrip(views: np.ndarray, spec: TargetSpec) -> None:
    """encode/decode согласованы для 0, 1, 1000, 166 (тест 16.1)."""
    for v in (0.0, 1.0, 1000.0, 166.0):
        v_arr = np.array([v])
        decoded = spec.decode(spec.encode(v_arr))
        if not np.isclose(decoded[0], v, rtol=1e-9, atol=1e-9):
            raise AssertionError(
                f"{spec.kind}: encode/decode mismatch for {v}: got {decoded[0]}"
            )
