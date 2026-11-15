"""Воспроизводимость: одинаковые входы -> идентичные артефакты.

Идемпотентность: повторный запуск с теми же входами выдаёт те же прогнозы;
изменение одного параметра даёт другой run_id, но не стирает прежние строки.
"""

from __future__ import annotations

import pytest

# Эти тесты обучают настоящие модели; без CatBoost они пропускаются.
pytest.importorskip("catboost")

import numpy as np

from src import features as feats
from src.models import fit_catboost


def test_same_seed_same_predictions(small_fit):
    train = small_fit["train"]
    feats_used = small_fit["features"]
    r1 = fit_catboost(
        train, None, feats_used, [], "L01_rmse_log1p", iterations_fixed=40
    )
    r2 = fit_catboost(
        train, None, feats_used, [], "L01_rmse_log1p", iterations_fixed=40
    )
    p1 = r1.target.decode(r1.model.predict(train[feats_used]))
    p2 = r2.target.decode(r2.model.predict(train[feats_used]))
    np.testing.assert_array_equal(p1, p2, err_msg="тот же seed дал другой прогноз")


def test_different_seed_changes_model(small_fit):
    train = small_fit["train"]
    feats_used = small_fit["features"]
    r1 = fit_catboost(
        train, None, feats_used, [], "L01_rmse_log1p", iterations_fixed=40, seed=1
    )
    r2 = fit_catboost(
        train, None, feats_used, [], "L01_rmse_log1p", iterations_fixed=40, seed=2
    )
    p1 = r1.model.predict(train[feats_used])
    p2 = r2.model.predict(train[feats_used])
    assert not np.allclose(p1, p2), (
        "разные seeds дают идентичные модели - подозрительно"
    )


def test_finite_predictions_enforced(small_fit):
    """Численное здоровье: прогноз конечен и невырожден (assert_finite)."""
    from src.models import assert_finite

    train = small_fit["train"]
    feats_used = small_fit["features"]
    fit = fit_catboost(
        train, None, feats_used, [], "L02_quantile_views_1000", iterations_fixed=40
    )
    pred = fit.model.predict(train[feats_used])
    assert_finite(pred, "smoke")
    assert np.isfinite(fit.target.decode(pred)).all()


def test_registry_row_appended_and_persisted(synthetic, tmp_path, monkeypatch):
    """Реестр: дописывание, детерминированный run_id, история не стирается."""
    from src import registry

    # Тестовый реестр в отдельном каталоге, чтобы не засорять результаты прогона
    monkeypatch.setattr(registry.config, "REGISTRY_CSV", tmp_path / "registry.csv")
    row = dict(
        stage="test",
        corpus="synthetic",
        split_id="synthetic__test",
        loss_function="L01_rmse_log1p",
        model_family="catboost:C1_symmetric",
        n_train=10,
        n_test=5,
        status="exploratory",
    )
    id1 = registry.append_registry_row(row)
    id2 = registry.append_registry_row(row)
    assert id1 == id2, "run_id должен быть детерминированным по содержимому"
    import pandas as pd

    reg = pd.read_csv(tmp_path / "registry.csv")
    assert len(reg) == 2
    assert (reg["run_id"] == id1).sum() == 2


def test_features_availability_guards():
    """Deployable-наборы состоят только из допубликационных признаков."""
    for fset in ("F1_meta", "F3_dens_expected"):
        for f in feats.FEATURE_SETS[fset]["features"]:
            assert feats.AVAILABILITY.get(f) == "prepublication", (
                f"{fset} содержит недоступный до публикации признак: {f}"
            )
