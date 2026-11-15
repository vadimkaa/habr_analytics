"""Календарь слотов: дата-кандидат даёт правильные dow, сезон, праздники, зону."""

from __future__ import annotations

import pandas as pd

from src.slot_policy import slot_calendar_rows


def test_working_saturday_is_weekday():
    """01.11.2025 - рабочая суббота производственного календаря: это будни."""
    rows = slot_calendar_rows(pd.Series([pd.Timestamp("2025-11-01 11:00:00+03:00")]))
    assert rows.iloc[0]["dow"] == 5
    assert rows.iloc[0]["is_weekend"] == 0


def test_transferred_day_off_is_holiday():
    """Нерабочий будень (например, 31.12.2024 - перенесённый выходной):
    праздничный флаг равен 1, день недели - будний."""
    rows = slot_calendar_rows(pd.Series([pd.Timestamp("2024-12-31 13:00:00+03:00")]))
    r = rows.iloc[0]
    assert r["dow"] < 5
    assert r["is_holiday"] == 1


def test_fixed_holiday_flag():
    rows = slot_calendar_rows(
        pd.Series(
            [
                pd.Timestamp("2026-01-01 10:00:00+03:00"),
                pd.Timestamp("2026-03-09 10:00:00+03:00"),
            ]
        )
    )
    assert rows.iloc[0]["is_holiday"] == 1  # Новый год
    assert rows.iloc[1]["is_holiday"] == 1  # перенос 8 марта
    assert rows.iloc[0]["month"] == 1 and rows.iloc[1]["month"] == 3


def test_month_is_integer_1_12():
    """Месяц - целое 1-12, а не 6.5 и не YYYY-MM."""
    ts = pd.Series(
        pd.date_range("2025-01-01", periods=12, freq="MS", tz="Europe/Moscow")
    )
    rows = slot_calendar_rows(ts)
    assert sorted(rows["month"].tolist()) == list(range(1, 13))
    assert rows["month"].dtype.kind in "iu"


def test_hour_and_timezone():
    """Час слота берётся по Москве; UTC-час до 21:00 соответствует следующему дню."""
    rows = slot_calendar_rows(pd.Series([pd.Timestamp("2025-06-10 21:30:00+00:00")]))
    r = rows.iloc[0]
    assert r["hour"] == 0  # 00:30 мск следующего дня
    assert r["dow"] == 2  # среда (11.06.2025)
    assert rows.index[0] == 0


def test_impossible_combinations_absent():
    """Все 168 слотов достижимы в году: нет невозможных комбинаций dow×hour."""
    ts = pd.Series(
        pd.date_range("2025-06-01", periods=366 * 24, freq="h", tz="Europe/Moscow")
    )
    rows = slot_calendar_rows(ts)
    assert len(rows.groupby(["dow", "hour"])) == 168
