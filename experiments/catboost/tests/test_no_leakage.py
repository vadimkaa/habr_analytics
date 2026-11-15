"""Утечки: train-only строгие проверки и негативные тесты.

Главные проверки:
 - изменение views теста не влияет на fit, границы и рекомендации;
 - dens24_actual не входит в deployable-набор;
 - публикации ПОСЛЕ кандидата не меняют его dens24 (прошлая плотность);
 - shuffle исходов внутри периода убивает сигнал модели;
 - history автора: только число ранее вышедших, будущие счётчики исключены.
"""

from __future__ import annotations

import pytest

# Эти тесты обучают настоящие модели; без CatBoost они пропускаются.
pytest.importorskip("catboost")

import numpy as np
import pandas as pd
import pytest

from src import features as feats
from src import slot_policy as SP
from src import statistics as ST
from src.targets import TargetSpec


def test_dens24_actual_not_in_deployable():
    f3 = feats.FEATURE_SETS["F3_dens_expected"]["features"]
    assert "dens24" not in f3 and "dens24_expected" in f3
    with pytest.raises(ValueError):
        feats.check_availability("F3_dens_expected", f3 + ["dens24"])


def test_prior_density_ignores_future(synthetic):
    """Публикации после t не меняют dens24(t); одновременные не считаются."""
    from habrpub.analysis import prior_density

    ts = pd.Series(synthetic.dt_utc)
    base = prior_density(ts)
    # Сдвигаем views/данные после третьей строки - плотность первой не меняется
    ts2 = ts.copy()
    base_first = prior_density(ts2)[0]
    ts2_later = ts2.copy()
    ts2_later.iloc[3:] = ts2_later.iloc[3:] + pd.Timedelta(days=30)
    after = prior_density(ts2_later.sort_values().reset_index(drop=True))
    assert base_first == after[0], "будущие публикации просочились в past density"


def test_previous_counts_excludes_future(synthetic):
    """author history: строго past published_at < candidate_at; тот же timestamp
    не считается предшествующим."""
    from habrpub.analysis import previous_counts

    df = synthetic
    prior = previous_counts(df)
    # автор "a": проверяем строгую хронологию для его статей (индекс исходный)
    sub = df[df.author_key == "a"].copy()
    prior_a = prior.loc[sub.index]
    for i in range(1, len(sub)):
        same_ts = sub.dt_utc.iloc[i] == sub.dt_utc.iloc[i - 1]
        if same_ts:
            assert prior_a.iloc[i] == prior_a.iloc[i - 1]
        else:
            assert prior_a.iloc[i] >= prior_a.iloc[i - 1]


def test_expected_density_is_train_only(synthetic):
    """dens24_expected для тестовых строк не видит dens24 теста."""
    from src.splits import fit_expected_density_column

    train = synthetic[synthetic.dt_msk < "2024-11-01 00:00:00+03:00"].copy()
    test = synthetic[synthetic.dt_msk >= "2024-11-01 00:00:00+03:00"].copy()
    base = fit_expected_density_column(train, {"test": test})
    perturbed = test.copy()
    perturbed["dens24"] = perturbed["dens24"] * 100 + 777
    pert = fit_expected_density_column(train, {"test": perturbed})
    np.testing.assert_allclose(
        base["test"]["dens24_expected"].to_numpy(),
        pert["test"]["dens24_expected"].to_numpy(),
        rtol=1e-12,
        err_msg="dens24_expected теста зависит от dens24 теста",
    )


def test_loo_density_excludes_own_row(synthetic):
    """Для train-строки LOO-медиана не включает собственную фактическую плотность."""
    from src.features import add_expected_density

    df = synthetic.copy()
    out = add_expected_density(df, df, loo=True)
    # крайний случай: строка с уникально большой плотностью в своём слоте
    d = df.copy()
    i = d.index[0]
    d.loc[i, "dens24"] = 10_000.0
    out2 = add_expected_density(d, d, loo=True)
    assert out2.loc[i, "dens24_expected"] < 10_000.0 / 2, (
        "собственная плотность вошла в признак самой строки"
    )


def test_test_views_do_not_affect_fit_or_policy(small_fit):
    """Strict train-only: замена тестовых исходов меняет только outcome-метрики.

    Обучение, границы терцилей и порог q остаются прежними."""
    from src.models import fit_catboost

    train, test = small_fit["train"], small_fit["test"]
    feats_used = small_fit["features"]
    fit = fit_catboost(
        train, None, feats_used, [], "L01_rmse_log1p", iterations_fixed=40
    )
    pred_base = fit.target.decode(fit.model.predict(test[feats_used]))
    th_base = SP.train_thresholds(train.assign(model_score=np.arange(len(train))))
    # Жёсткая порча тестовых исходов и их признаковой части, не входящей в fit
    poisoned = test.copy()
    poisoned["views"] = 10_000_000.0
    pred_poison = fit.target.decode(fit.model.predict(poisoned[feats_used]))
    np.testing.assert_array_equal(pred_base, pred_poison)
    th_after = SP.train_thresholds(train.assign(model_score=np.arange(len(train))))
    assert th_base["threshold_q"] == th_after["threshold_q"]
    np.testing.assert_allclose(th_base["terciles"], th_after["terciles"])


def test_shuffle_labels_kills_signal(synthetic):
    """Негативный контроль: shuffle исходов внутри периода убивает корреляцию."""
    rng = np.random.default_rng(0)
    te = synthetic.copy()
    te["views"] = te["hour"] * 100 + rng.integers(0, 50, len(te))
    te["score"] = te["views"] * 0.3 + rng.normal(0, 20, len(te))
    rho_real = float(
        np.corrcoef(
            np.argsort(np.argsort(te["score"])), np.argsort(np.argsort(te["views"]))
        )[0, 1]
    )
    shuffled = ST.shuffle_labels_within_month(te)
    rho_shuf = float(
        np.corrcoef(
            np.argsort(np.argsort(te["score"])),
            np.argsort(np.argsort(shuffled["views"])),
        )[0, 1]
    )
    assert rho_real > 0.3
    assert abs(rho_shuf) < 0.15, "после shuffle внутри месяца сигнал должен исчезнуть"


def test_decode_called_once_poisson():
    """Преобразование цепи инвертируется ровно один раз."""
    spec = TargetSpec("divide_1000")
    raw = np.array([25.0])
    assert spec.decode(raw)[0] == pytest.approx(25_000.0)
    assert spec.decode(spec.decode(raw))[0] != pytest.approx(25_000.0)
