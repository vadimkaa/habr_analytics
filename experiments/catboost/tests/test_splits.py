"""Фонды: только хронология, ни один ID не попадает в разные наборы."""

from __future__ import annotations

import pandas as pd
import pytest

from src import splits


def _ids(df):
    return set(df["id"])


def test_no_overlap_and_chronology(synthetic):
    df = synthetic.copy()
    frames = {
        "train": df[df.dt_msk < "2024-11-01 00:00:00+03:00"],
        "valid": df[df.dt_msk >= "2024-11-01 00:00:00+03:00"],
    }
    ids_tr, ids_va = _ids(frames["train"]), _ids(frames["valid"])
    assert not (ids_tr & ids_va), "один ID в train и в валидации"
    splits.check_chronology(frames["train"], frames["valid"])
    assert frames["train"].dt_msk.max() < frames["valid"].dt_msk.min()


def test_check_chronology_raises_on_overlap(synthetic):
    df = synthetic
    with pytest.raises(AssertionError):
        splits.check_chronology(df, df)


def test_assert_no_overlap_raises(synthetic):
    from src.splits import _assert_no_overlap

    with pytest.raises(AssertionError):
        _assert_no_overlap({"a": synthetic, "b": synthetic})


def test_protocol_split_bounds_are_chronological():
    """Границы из configs/splits.yaml монотонны и роли различаются."""
    from src import config

    for corpus in ("ai", "wide"):
        folds = list(config.folds(corpus).values())
        for a, b in zip(folds, folds[1:]):
            assert a.end == b.start or a.end is None or b.start is None
        roles = {f.role for f in folds}
        assert "train" in roles and "validation" in roles
        assert "audit_exposed" in roles


def test_wide_start_matches_protocol():
    """Широкий корпус: плотности нужен ещё день истории после 24.06.2025."""
    from src.data_contract import WIDE_START

    assert WIDE_START == pd.Timestamp("2025-06-25", tz="Europe/Moscow")
