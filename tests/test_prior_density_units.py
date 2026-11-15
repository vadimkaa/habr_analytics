"""Единицы времени в prior_density/forward_density: вход ns и us обязан давать
одинаковый результат.

На pandas 3 метки приходят как datetime64[us]; без приведения к наносекундам
константа суток в нс расширяет окно в 1000 раз — вместо суток считались бы
1000 дней."""

import numpy as np
import pandas as pd

from habrpub.analysis import forward_density, prior_density


def _series(unit):
    stamps = pd.to_datetime(
        [
            "2026-01-01 00:00",
            "2026-01-01 01:00",
            "2026-01-01 02:00",
            "2026-01-02 01:00",
        ],
        utc=True,
    ).as_unit(unit)
    return pd.Series(stamps)


def test_ns_and_us_agree():
    assert np.array_equal(prior_density(_series("ns")), prior_density(_series("us")))


def test_window_is_24h():
    out = prior_density(_series("us"))
    # для последней метки окно [t−24 ч, t) содержит публикации 01:00 и 02:00
    assert out[0] == 0 and out[-1] == 2


def test_forward_density_window():
    assert np.array_equal(
        forward_density(_series("ns")), forward_density(_series("us"))
    )
    out = forward_density(_series("us"))
    # для первой метки окно [t, t+24 ч) содержит 01:00 и 02:00; для последней пусто
    assert out[0] == 2 and out[-1] == 0
