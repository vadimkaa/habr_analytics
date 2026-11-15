"""Data contract: загрузка корпусов, аудит и применения временных фолдов.

Загрузка данных идёт ТОЛЬКО через исходный загрузчик habrpub
(habrpub.analysis.load_corpus): сбор данных не переписывается.
Модуль добавляет аудит (4.1), выборку зрелых статей и резку по фолдам из
configs/splits.yaml. Никакие поля, известные только после публикации, не
попадают в допубликационные признаки: у каждого признака есть availability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# habrpub лежит в корне репозитория; обеспечим импортируемость из любого cwd.
_REPO = str(config.REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from habrpub.analysis import (  # noqa: E402  (исходный загрузчик habrpub)
    WIDE_COMMON_START,
    load_corpus,
)

AUDIT_MIN_AGE_DAYS = 30.0
WIDE_START = WIDE_COMMON_START + pd.Timedelta(days=1)  # окно плотности на день позже


class CorpusData:
    """Зрелый корпус с признаками и фолдами. Один ID - одна строка."""

    def __init__(self, corpus: str, df: pd.DataFrame):
        self.corpus = corpus
        self.df = df

    @property
    def ids(self) -> set[str]:
        return set(self.df["id"])

    def fold(self, name: str) -> pd.DataFrame:
        spec = config.folds(self.corpus)[name]
        d = self.df
        if spec.start is not None:
            d = d[d.dt_msk >= pd.Timestamp(spec.start, tz="Europe/Moscow")]
        if spec.end is not None:
            d = d[d.dt_msk < pd.Timestamp(spec.end, tz="Europe/Moscow")]
        return d.copy()

    def rolling(self, cut: str, test_months: int):
        """Rolling-origin: train < cut, тест - следующие test_months месяцев."""
        c = pd.Timestamp(cut, tz="Europe/Moscow")
        end = c + pd.DateOffset(months=test_months)
        train = self.df[self.df.dt_msk < c].copy()
        test = self.df[(self.df.dt_msk >= c) & (self.df.dt_msk < end)].copy()
        return train, test


def load_corpus_data(corpus: str) -> CorpusData:
    """Зрелые статьи корпуса по правилам сплита (mature + полная плотность)."""
    df = load_corpus(corpus)
    if corpus == "ai":
        eligible = df.mature & df.density_complete & (df.year >= 2016)
    else:
        eligible = df.mature & df.density_complete & (df.dt_msk >= WIDE_START)
    data = CorpusData(corpus, df[eligible].copy().reset_index(drop=True))
    return data


def corpus_overlap() -> dict:
    """Пересечение корпусов по ID: общий ID должен попадать
    только в один фонд объединённой выборки; здесь корпуса обучаются отдельно."""
    ai = set(load_corpus("ai")["id"])
    wide = set(load_corpus("wide")["id"])
    inter = ai & wide
    return {
        "n_ai": len(ai),
        "n_wide": len(wide),
        "n_union": len(ai | wide),
        "n_intersect": len(inter),
        "intersect_ids_sample": sorted(inter)[:20],
    }


def audit_frame(df: pd.DataFrame, name: str) -> dict:
    """Инвентаризация перед обучением: размеры, пропуски, распределения."""
    views = df["views"].to_numpy(float)
    def quant(v):
        return {
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
        "p99": float(np.quantile(v, 0.99)),
        "p99_9": float(np.quantile(v, 0.999)),
            "max": float(v.max()),
        }

    out = {
        "frame": name,
        "n_rows": int(len(df)),
        "n_unique_ids": int(df["id"].nunique()),
        "n_authors": int(df["author_key"].nunique()),
        "missing": {
            c: int(df[c].isna().sum())
            for c in ("views", "title", "author_key", "dens24", "is_corp", "t_len")
            if c in df.columns
        },
        "views_quantiles": quant(views),
        "n_zero_views": int((views == 0).sum()),
        "age_days": {
            "min": float(df.age_days.min()),
            "median": float(df.age_days.median()),
            "max": float(df.age_days.max()),
        },
        "share_corp": float(df.is_corp.mean()),
        "share_weekend": float(df.is_weekend.mean()),
        "share_translation": float(df.is_translation.mean()),
        "date_min": str(df.dt_msk.min()),
        "date_max": str(df.dt_msk.max()),
        "rows_per_year": {
            str(y): int(n) for y, n in df.year.value_counts().sort_index().items()
        },
        "dow_counts": {
            int(k): int(v) for k, v in df.dow.value_counts().sort_index().items()
        },
        "hour_p10_p50_p90": [float(np.quantile(df.hour, q)) for q in (0.1, 0.5, 0.9)],
    }
    return out


def write_quality_report(parts: list[str], path: Path | None = None) -> Path:
    path = path or (config.RESULTS / "data_quality_report.md")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def build_quality_report() -> Path:
    """Полный аудит корпуса -> results/data_quality_report.md (этап 1, P0)."""
    hashes = config.input_hashes()
    parts: list[str] = [
        "# Аудит корпуса",
        "",
        f"Версия протокола: `{config.protocol()['protocol']}`.",
        "",
        "## Контроль входов",
        "",
        f"- SHA-256 ИИ/МЛ: `{hashes.get('ai')}`",
        f"- SHA-256 wide: `{hashes.get('wide')}`",
        f"- Совпадение с input_manifest.json: ai={hashes.get('ai_declared_matches')}, wide={hashes.get('wide_declared_matches')}",
        "",
        "Правило: реестр экспериментов привязан к этим хешам; на изменённой",
        "выгрузке пересчёт старых чисел не гарантируется.",
        "",
    ]
    ov = corpus_overlap()
    parts += [
        "## Пересечение корпусов",
        "",
        f"- ИИ/МЛ: {ov['n_ai']} статей; широкий: {ov['n_wide']}; объединение: {ov['n_union']};",
        f"  пересечение: {ov['n_intersect']} ID. Корпуса обучаются отдельно,",
        "  поэтому один ID не может попасть в train и test одновременно;",
        "  проверка непересекающихся частей - этап 6 (robustness, A13).",
        "",
    ]
    for corpus in ("ai", "wide"):
        cd = load_corpus_data(corpus)
        parts.append(f"## Корпус {corpus}: зрелые статьи после фильтров")
        parts.append("")
        aud = audit_frame(cd.df, corpus)
        for k, v in aud.items():
            parts.append(f"- {k}: {v}")
        parts.append("")
        for fold_name, spec in config.folds(corpus).items():
            f = cd.fold(fold_name)
            parts.append(
                f"  - фонд `{fold_name}` ({spec.role}): {len(f)} строк, "
                f"{str(f.dt_msk.min())[:10]}..{str(f.dt_msk.max())[:10]}"
            )
        parts.append("")
    return write_quality_report(parts)
