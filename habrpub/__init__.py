"""Пути и фиксированные опорные даты исходного исследования.

SNAP_* восстановлены из исходных age_days. В сырье нет timestamp каждого
запроса: это опорные даты расчёта возраста, а не доказанно точное время счётчика.
Новый сбор следует анализировать отдельно с observed_at каждой записи.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
FIGS = BASE / "figs"
RESULTS = DATA / "results"
SNAP_AI = pd.Timestamp("2026-11-10 16:12:14.312197+00:00")
SNAP_WIDE = pd.Timestamp("2026-11-10 17:29:18.334424+00:00")
