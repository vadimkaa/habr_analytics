"""Фиксируем смысл календарных исключений, не зависящий от формата входных меток."""

import pandas as pd
from habrpub.analysis import weekday_holidays, offday_sets


def test_one_off_nonworking_days_not_treated_as_regular_holidays():
    holidays = weekday_holidays([2020, 2021])
    offdays, _ = offday_sets()
    assert all(day in offdays for day in holidays)
    assert pd.Timestamp("2020-05-06").date() not in holidays
    assert pd.Timestamp("2021-05-05").date() not in holidays


def test_calendar_captures_documented_one_off_days():
    from habrpub.analysis import production_calendar
    cal = production_calendar()
    flagged = {
        row.date for row in cal.itertuples()
        if "Нерабоч" in str(row.holiday_title) or "голосован" in str(row.holiday_title)
    }
    assert len(flagged) == 46
    assert sum(day.weekday() < 5 for day in flagged) == 36
