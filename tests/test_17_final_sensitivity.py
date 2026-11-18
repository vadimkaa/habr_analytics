"""Checks for the post-review diagnostic script (no model refit)."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

s = importlib.import_module('17_final_sensitivity')
ROOT = Path(__file__).resolve().parents[1]


def test_local_rank_ties_and_edges():
    days = np.repeat(np.arange(10), 3)
    views = np.tile([10, 10, 20], 10)
    rank, n = s.local_percentiles(days, views, 1, min_n=3)
    for i, day in enumerate(days):
        if day < 2 or day > 7:
            assert np.isnan(rank[i])
            continue
        mask = np.abs(days - day) <= 1
        expect = rankdata(views[mask])[np.flatnonzero(np.flatnonzero(mask) == i)[0]] / mask.sum()
        assert n[i] == mask.sum()
        assert abs(rank[i] - expect) < 1e-12
    too_few, _ = s.local_percentiles(days, views, 1, min_n=10)
    assert np.isnan(too_few).all()


def test_fixed_effects_matches_dummy_model():
    rng = np.random.default_rng(18)
    g = np.repeat(np.arange(40), 6)
    w = np.tile([0., 0., 0., 1., 1., 1.], 40)
    x = pd.DataFrame({'weekend': w, 'other': rng.normal(size=len(g))})
    d = pd.DataFrame({'author_key': g})
    y = pd.Series(0.13*w + x.other*.22 + rng.normal(size=40)[g] + rng.normal(size=len(g))*.1)
    fit = s.fe(d, x, y)
    design = pd.concat([x, pd.get_dummies(d.author_key, dtype=float)], axis=1)
    expected = sm.OLS(y, design).fit(cov_type='cluster', cov_kwds={'groups': g}, use_t=True)
    assert np.isclose(fit['ratio'], 10**expected.params['weekend'])
    assert np.allclose(fit['ci95'], 10**expected.conf_int().loc['weekend'])


def test_result_contract():
    result = json.loads((ROOT/'data/results/17_final_sensitivity.json').read_text())
    prior = json.loads((ROOT/'data/results/09_robustness_results.json').read_text())
    for c in ('ai', 'wide'):
        a = result['corpora'][c]
        assert set(a['holiday']) == {'0','3','7','14'}
        assert np.isclose(a['calendar']['ols']['ratio'], prior['corpora'][c]['adjusted_weekend']['ratio_target_plus1'])
        assert len(a['calendar']['age_windows']) == 22
        assert np.isfinite(a['calendar']['quantile_views']['ratio'])
        assert a['calendar']['quantile_views']['ci95'] is None
        assert a['calendar']['random_intercept']['converged']
    assert result['readability']['ai']['oborneva']['within_author']['3']['adjusted'] is None


def test_quantile_equivariance_to_included_age():
    rng = np.random.default_rng(45)
    age = rng.uniform(1.5, 3.5, 300)
    x = pd.DataFrame({"weekend": rng.integers(0, 2, len(age)).astype(float), "log_age": age})
    y = pd.Series(2 + x.weekend * .13 + age * .4 + rng.normal(0, .15, len(age)))
    a = s.quantile(x, y)
    b = s.quantile(x, y - age)
    assert np.isclose(a["ratio"], b["ratio"], atol=5e-4, rtol=1e-5)


def test_comparison_allows_numerical_drift_not_changed_results():
    assert not s.compare_results(1.295, 1.2951, "/calendar/quantile_views/ratio")
    assert s.compare_results(1.295, 1.31, "/calendar/quantile_views/ratio")
    assert not s.compare_results(4.5, 4.57, "/holiday/7/ci95_episode_week/0")
    assert s.compare_results(4.5, 5., "/holiday/7/ci95_episode_week/0")
    assert s.compare_results(2586, 2587, "/calendar/fe/n")
    assert not s.compare_results(650, 651, "/power/raw/per_arm/1.2")
    assert s.compare_results(650, 655, "/power/raw/per_arm/1.2")
    assert not s.compare_results(42, 987, "/calendar/quantile_views/iterations")


def test_published_sensitivity_numbers_and_no_working_notes():
    results = json.loads((ROOT / "data/results/17_final_sensitivity.json").read_text())
    short = (ROOT / "short_article/article.md").read_text(encoding="utf-8")
    full = (ROOT / "article.md").read_text(encoding="utf-8")
    for c in ("ai", "wide"):
        r = results["corpora"][c]
        fe = r["calendar"]["fe"]
        for n in (str(fe["n"]), str(fe["n_authors"]), f"{fe['ratio']:.2f}",
                  f"[{fe['ci95'][0]:.2f}; {fe['ci95'][1]:.2f}]"):
            assert n in short, n
        for w in ("3", "7", "14"):
            value = f"{r['holiday'][w]['diff_pp']:.1f}"
            assert value in short and value in full, value
    for value in ("после согласования редакции", "Для публикации (не забыть):", "/mnt/data/"):
        assert value not in short and value not in full
