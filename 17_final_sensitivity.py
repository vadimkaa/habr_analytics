"""Проверки чувствительности на фиксированном срезе, без переобучения CatBoost.
Команда --check сравнивает повторный расчёт с опубликованным JSON, не меняя его.
"""
import argparse
import math
from pathlib import Path
import hashlib
import importlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats, optimize
from patsy import dmatrix
from statsmodels.stats.power import TTestIndPower
from habrpub import DATA, RESULTS
from habrpub.analysis import load_corpus, recent, WIDE_COMMON_START, corr, finite_json
from habrpub.calendar_analysis import _holiday_days, _with_month_pct, holiday_episodes
from experiments.catboost.src.config import folds
SEED = 20261118
N_BOOT = 1000

def boot(df, col, stat):
    groups = list(df.groupby(col, sort=True).indices.values())
    rng = np.random.default_rng(SEED)
    a = [stat(np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])) for _ in range(N_BOOT)]
    return np.nanquantile(a, [0.025, 0.975]).tolist()

def rho(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x = x - x.mean()
    y = y - y.mean()
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den else np.nan

def ratio(y, w):
    return np.median(y[w]) / np.median(y[~w]) if w.any() and (~w).any() and np.median(y[~w]) else np.nan

def local_percentiles(day, views, k, min_n=20):
    """Midrank/n among mature same-corpus articles within ±k calendar days.

    Includes self and ties. Excludes incomplete edge windows and n < min_n.
    `day` consists of sorted integer Moscow calendar days.
    """
    day, views = (np.asarray(day), np.asarray(views))
    if len(day) != len(views) or np.any(np.diff(day) < 0):
        raise ValueError('misaligned or unsorted dates')
    pct = np.full(len(day), np.nan)
    ns = np.zeros(len(day), int)
    for u in np.unique(day):
        lo, hi = np.searchsorted(day, [u - k, u + k + 1])
        n = hi - lo
        ix = np.flatnonzero(day == u)
        ns[ix] = n
        if u - k < day.min() + 1 or u + k > day.max() - 1 or n < min_n:
            continue
        ref = np.sort(views[lo:hi])
        v = views[ix]
        pct[ix] = (np.searchsorted(ref, v, 'left') + np.searchsorted(ref, v, 'right') + 1) / (2 * n)
    return (pct, ns)

def holiday(d, c):
    start = pd.Timestamp('2016-01-01', tz='Europe/Moscow') if c == 'ai' else WIDE_COMMON_START
    d = _with_month_pct(d[d.mature & (d.dt_msk >= start)].copy())
    cal = _holiday_days(d)
    ep = {day: i for i, e in enumerate(holiday_episodes(cal['hol'])) for day in e}
    day = d.dt_msk.dt.tz_localize(None).dt.normalize().to_numpy(dtype='datetime64[D]').astype('int64')
    views = d.views.to_numpy()
    out = {}
    for k in (0, 7, 3, 14):
        pct = d.month_pct.to_numpy().copy()
        ns = np.zeros(len(d), int)
        if k:
            pct, ns = local_percentiles(day, views, k)
        s = d.assign(pct=pct, ns=ns).dropna(subset=['pct'])
        h = s[s.date_msk.isin(cal['hol'])].copy()
        z = s[s.date_msk.isin(cal['ctrl'])].copy()
        h['ep'] = h.date_msk.map(ep)
        hg = [g.pct.to_numpy() for _, g in h.groupby('ep')]
        cg = [g.pct.to_numpy() for _, g in z.groupby('week_start')]
        rng = np.random.default_rng(SEED)
        draws = [100 * (np.median(np.concatenate([hg[i] for i in rng.integers(len(hg), size=len(hg))])) - np.median(np.concatenate([cg[i] for i in rng.integers(len(cg), size=len(cg))]))) for _ in range(N_BOOT)]
        out[str(k)] = {'n_holiday': len(h), 'n_control': len(z), 'n_episodes': len(hg), 'n_control_weeks': len(cg), 'diff_pp': 100 * (h.pct.median() - z.pct.median()), 'ci95_episode_week': np.quantile(draws, [0.025, 0.975]), 'matched_month_diff_pp': 100 * (h.month_pct.median() - z.month_pct.median()), 'median_reference_n': s.ns.median() if k else None}
    return out

def design(d, hubs=15, calendar='month', age='log', density=True, weekend=True):
    x = pd.DataFrame(index=d.index)
    if weekend:
        x['weekend'] = d.is_weekend.astype(float)
    x['corp'] = d.is_corp.astype(float)
    if density:
        x['density'] = d.dens24.astype(float) / 10
    if age == 'spline':
        b = np.asarray(dmatrix('bs(z, df=5, degree=3, include_intercept=False)-1', {'z': np.log(d.age_days)}))
        for i in range(b.shape[1]):
            x[f'age_spline_{i}'] = b[:, i]
    else:
        x['log_age'] = np.log(d.age_days)
    if calendar == 'month':
        x = x.join(pd.get_dummies(d.month_id, prefix='month', drop_first=True, dtype=float))
    else:
        x = x.join(pd.get_dummies(d.month_id.str[:4], prefix='year', drop_first=True, dtype=float))
        x = x.join(pd.get_dummies(d.month_id.str[-2:], prefix='season', drop_first=True, dtype=float))
    for h in pd.Series([h for hs in d.hubs for h in hs]).value_counts().head(hubs).index:
        x['hub_' + str(h)] = d.hubs.map(lambda hs: float(h in hs))
    return x

def independent(x):
    """Remove constant/linearly dependent columns after within-group centering."""
    keep = []
    basis = []
    for name in x:
        v = x[name].to_numpy().copy()
        n = np.linalg.norm(v)
        if n < 1e-10:
            continue
        for _ in range(2):
            for q in basis:
                v -= q * (q @ v)
        nn = np.linalg.norm(v)
        if nn > n * 1e-09:
            keep.append(name)
            basis.append(v / nn)
    return x[keep]

def ols(d, x, y):
    return sm.OLS(y, independent(sm.add_constant(x, has_constant='add'))).fit(cov_type='cluster', cov_kwds={'groups': d.author_key})

def eff(m, k='weekend'):
    return {'ratio': 10 ** m.params[k], 'ci95': 10 ** m.conf_int().loc[k].to_numpy(), 'p': m.pvalues[k], 'n': int(m.nobs)}

def fe(d, x, y):
    x = independent(x - x.groupby(d.author_key).transform('mean'))
    y = y - y.groupby(d.author_key).transform('mean')
    m = sm.OLS(y, x).fit(cov_type='cluster', cov_kwds={'groups': d.author_key, 'use_correction': False})
    g = d.author_key.nunique()
    n = len(d)
    k = g + x.shape[1]
    b = m.params.weekend
    se = m.bse.weekend * np.sqrt(g / (g - 1) * (n - 1) / (n - k))
    cv = stats.t.ppf(0.975, g - 1)
    return {'ratio': 10 ** b, 'ci95': 10 ** np.array([b - cv * se, b + cv * se]), 'n': n, 'n_authors': g, 'sigma': np.sqrt(np.sum(m.resid ** 2) / (n - k))}

def random_intercept(d, x, y):
    """Gaussian author random intercept, ML profile over variance ratio."""
    a = independent(sm.add_constant(x, has_constant='add')).to_numpy()
    y = np.asarray(y)
    codes = pd.factorize(d.author_key)[0]
    counts = np.bincount(codes)
    n = len(y)
    ma = np.vstack([np.bincount(codes, weights=v) / counts for v in a.T]).T[codes]
    my = (np.bincount(codes, weights=y) / counts)[codes]

    def p(lam, details=False):
        s = (1 - 1 / np.sqrt(1 + lam * counts))[codes]
        aw = a - ma * s[:, None]
        yw = y - my * s
        b = np.linalg.lstsq(aw, yw, rcond=None)[0]
        e = yw - aw @ b
        s2 = e @ e / n
        loss = n * np.log(s2) + np.log1p(lam * counts).sum()
        return (loss, s2) if details else loss
    fit = optimize.minimize_scalar(lambda z: p(np.exp(z)), bounds=(-16, 12), method='bounded')
    lam = np.exp(fit.x)
    loss, s2 = p(lam, True)
    if p(0) < loss:
        lam = 0
        loss, s2 = p(0, True)
    return {'icc': lam / (1 + lam), 'between_variance': lam * s2, 'within_variance': s2, 'converged': fit.success}

def quantile(x, y):
    x = independent(sm.add_constant(x, has_constant='add'))
    import warnings
    from scipy import sparse
    from statsmodels.tools.sm_exceptions import IterationLimitWarning, ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', IterationLimitWarning)
        warnings.simplefilter('ignore', ConvergenceWarning)
        m = sm.QuantReg(y, x).fit(q=0.5, max_iter=3000, p_tol=1e-07)
    beta = m.params.weekend
    if m.iterations >= 3000 or not np.isfinite(beta):
        n, p = x.shape
        eye = sparse.eye(n, format='csc')
        fit = optimize.linprog(np.r_[np.zeros(p), np.ones(2 * n)], A_eq=sparse.hstack([sparse.csc_matrix(x), eye, -eye], format='csc'), b_eq=np.asarray(y), bounds=[(None, None)] * p + [(0, None)] * (2 * n), method='highs')
        if not fit.success:
            raise RuntimeError(f'Median regression failed: {fit.message}')
        beta = fit.x[x.columns.get_loc('weekend')]
    return {'ratio': 10 ** beta, 'iterations': m.iterations, 'n': len(y), 'ci95': None, 'note': 'Median-regression point estimate; no clustered CI'}

def calendar(d):
    x = design(d)
    y = d.log_views
    m = ols(d, x, y)
    groups = d.groupby('author_key').is_weekend.nunique()
    a = d[d.author_key.isin(groups[groups == 2].index)].copy()
    f = fe(a, x.loc[a.index], y.loc[a.index])
    med = d.groupby('author_key').vpd.agg(['size', 'median'])
    out = {'ols': eff(m), 'fe': f, 'same_sample_ols': eff(ols(a, x.loc[a.index], y.loc[a.index])), 'author_iqr': {str(k): med[med['size'] >= k]['median'].quantile([0.25, 0.5, 0.75]).tolist() for k in (3, 5)}, 'random_intercept': random_intercept(d, x, y), 'random_intercept_without_weekend': random_intercept(d, x.drop(columns='weekend'), y), 'spline': eff(ols(d, design(d, age='spline'), y))}
    print('FE/random/spline complete', flush=True)
    out['quantile_views'] = quantile(x, y)
    print('quantile views done', flush=True)
    positive = d[d.vpd > 0]
    out['quantile_vpd'] = quantile(x.loc[positive.index], np.log10(positive.vpd))
    print('quantile VPD done', flush=True)
    out['age_windows'] = {}
    out['age_windows_definition'] = '22 complete 15-day age bands [30,360); no partial final band; at least 10 observations per group'
    for lo in range(30, 360, 15):
        s = d[(d.age_days >= lo) & (d.age_days < lo + 15)]
        yy = s.vpd.to_numpy()
        w = s.is_weekend.to_numpy()
        if min(w.sum(), (~w).sum()) < 10:
            continue
        out['age_windows'][str(lo)] = {'n_weekend': int(w.sum()), 'n_weekday': int((~w).sum()), 'ratio': ratio(yy, w), 'ci95': boot(s, 'author_key', lambda i: ratio(yy[i], w[i]))}
    out['density'] = {}
    for k, s in d.groupby('is_weekend'):
        out['density']['weekend' if k else 'weekday'] = {'raw': corr(s.dens24, s.vpd), 'density_iqr': s.dens24.quantile([0.25, 0.5, 0.75]).tolist(), 'adjusted_per10': eff(ols(s, design(s, weekend=False), s.log_views), 'density')}
    xx = x.copy()
    xx['interaction'] = xx.weekend * xx.corp
    mi = ols(d, xx, y)
    names = ['weekend', 'interaction']
    b = mi.params[names].sum()
    se = np.sqrt(mi.cov_params().loc[names, names].to_numpy().sum())
    out['blog'] = {'personal': eff(mi), 'ratio_of_ratios': eff(mi, 'interaction'), 'corporate': {'ratio': 10 ** b, 'ci95': 10 ** np.array([b - 1.96 * se, b + 1.96 * se])}}
    out['engagement'] = {}
    no_density = xx.drop(columns='density')
    ni = ols(d, no_density, y)
    b = ni.params[names].sum()
    se = np.sqrt(ni.cov_params().loc[names, names].to_numpy().sum())
    out['blog_without_density'] = {'personal': eff(ni), 'ratio_of_ratios': eff(ni, 'interaction'), 'corporate': {'ratio': 10 ** b, 'ci95': 10 ** np.array([b - 1.96 * se, b + 1.96 * se])}}
    for k in ('comments', 'favorites'):
        s = d[(d.views > 0) & d[k].notna() & (d[k] >= 0)].copy()
        v = s.views.to_numpy()
        e = s[k].to_numpy()
        w = s.is_weekend.to_numpy()
        r = 1000 * e / v

        def diff(i):
            return np.median(r[i][w[i]]) - np.median(r[i][~w[i]]) if w[i].any() and (~w[i]).any() else np.nan
        out['engagement'][k] = {'n': len(s), 'weekday': np.median(r[~w]), 'weekend': np.median(r[w]), 'diff_ci95': boot(s, 'author_key', diff), 'aggregate_weekday': 1000 * e[~w].sum() / v[~w].sum(), 'aggregate_weekend': 1000 * e[w].sum() / v[w].sum()}
    out['power'] = {}
    for name, sig in [('raw', d.log_views.std(ddof=1)), ('adjusted', np.sqrt(np.mean(m.resid ** 2))), ('within_author', f['sigma'])]:
        out['power'][name] = {'sigma': sig, 'per_arm': {str(k): int(np.ceil(TTestIndPower().solve_power(effect_size=np.log10(k) / sig, alpha=0.05, power=0.8, ratio=1))) for k in (1.1, 1.2, 1.3, 1.4)}}
    return out

def age_folds(d, c):
    s = d[d.mature & d.density_complete & (d.year >= 2016 if c == 'ai' else d.dt_msk >= WIDE_COMMON_START + pd.Timedelta(days=1))]
    f = {}
    for name, sp in folds(c).items():
        a = s
        if sp.start:
            a = a[a.dt_msk >= pd.Timestamp(sp.start, tz='Europe/Moscow')]
        if sp.end:
            a = a[a.dt_msk < pd.Timestamp(sp.end, tz='Europe/Moscow')]
        f[name] = a
    f['final_train'] = pd.concat([f['dev_train'], f['validation']])
    return {k: {'n': len(a), 'age': a.age_days.quantile([0, 0.25, 0.5, 0.75, 1]).tolist(), 'median_views': a.views.median()} for k, a in f.items()}

def readability():
    """Rejoin frozen sample with hub labels; compare ranks within author×corpus."""
    d = importlib.import_module('07_readability').load_sample()
    out = {}
    assert len(d) == 1516
    for label, s in [('all', d), ('ai', d[d.corpus == 'ai']), ('wide', d[d.corpus == 'wide'])]:
        s = s.copy()
        res = {}
        z = design(s, hubs=20, density=False, weekend=False)
        z['ln_words'] = np.log1p(s.words)
        z['log_age'] = np.log1p(s.age_days)
        z['corpus_ai'] = (s.corpus == 'ai').astype(float)
        z = independent(sm.add_constant(z)).to_numpy()
        for met in ('oborneva', 'long_word_share'):
            xx = stats.rankdata(s[met])
            yy = stats.rankdata(s.vpd)
            rx = xx - z @ np.linalg.lstsq(z, xx, rcond=None)[0]
            ry = yy - z @ np.linalg.lstsq(z, yy, rcond=None)[0]
            r = {'raw': rho(xx, yy), 'partial': rho(rx, ry), 'within_hubs': {}, 'within_author': {}}
            for h in pd.Series([h for hs in s.hubs for h in hs]).value_counts().head(20).index:
                a = s[s.hubs.map(lambda hs: h in hs)]
                if len(a) >= 50:
                    r['within_hubs'][str(h)] = corr(a[met], a.vpd)
            for nmin in (2, 3):
                t = s.copy()
                t['group'] = t.author_key + '|' + t.corpus
                t = t[t.groupby('group').group.transform('size') >= nmin].copy()
                aa = t.groupby('corpus')[met].rank(pct=True)
                bb = t.groupby('corpus').vpd.rank(pct=True)

                def dm(v):
                    return v - v.groupby(t.group).transform('mean')
                a = design(t, hubs=20, calendar='season', density=False, weekend=False)
                a['ln_words'] = np.log1p(t.words)
                a['log_age'] = np.log1p(t.age_days)
                a = independent(dm(a))
                x0 = dm(aa)
                y0 = dm(bb)
                groups = t.group.nunique()
                authors = t.author_key.nunique()
                df_resid = len(t) - groups - a.shape[1] - 1
                item = {'n': len(t), 'authors': authors, 'author_corpus_groups': groups, 'raw_within_rank': rho(x0, y0), 'n_controls': a.shape[1], 'residual_df': df_resid}
                if df_resid >= 10 and authors >= 20:
                    v = a.to_numpy()
                    u = x0 - v @ np.linalg.lstsq(v, x0, rcond=None)[0]
                    w = y0 - v @ np.linalg.lstsq(v, y0, rcond=None)[0]
                    fit = sm.OLS(y0, pd.concat([x0.rename('metric'), a], axis=1)).fit(cov_type='cluster', cov_kwds={'groups': t.author_key, 'use_correction': False})
                    b = fit.params.metric
                    se = fit.bse.metric * np.sqrt(authors / (authors - 1) * (len(t) - 1) / df_resid)
                    cv = stats.t.ppf(0.975, authors - 1)
                    item.update(adjusted=rho(u, w), slope=b, slope_ci95=[b - cv * se, b + cv * se])
                else:
                    item.update(adjusted=None, slope=None, slope_ci95=None, warning='Too few residual degrees of freedom or author clusters for this specification')
                per = []
                for _, g in t.groupby('group', sort=True):
                    v = rho(stats.rankdata(g[met]), stats.rankdata(g.vpd))
                    if np.isfinite(v):
                        per.append((g.author_key.iloc[0], v))
                vals = pd.DataFrame(per, columns=['author', 'rho']).groupby('author').rho.mean().to_numpy()
                rng = np.random.default_rng(SEED)
                draws = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(N_BOOT)]
                item['equal_author'] = {'n_authors': len(vals), 'mean_rho': vals.mean(), 'ci95': np.quantile(draws, [0.025, 0.975])}
                r['within_author'][str(nmin)] = item
            res[met] = r
        out[label] = res
    return out

def compare_results(expected, actual, path=''):
    """Compare statistical outputs; solver diagnostics and environment are not metrics."""
    if path == '/environment' or path.endswith('/iterations'):
        return []
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            return [f'{path}: keys differ: {set(expected) ^ set(actual)}']
        return [error for k in expected for error in compare_results(expected[k], actual[k], path + '/' + k)]
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f'{path}: length differs']
        return [error for i, (a, b) in enumerate(zip(expected, actual)) for error in compare_results(a, b, f'{path}/{i}')]
    if isinstance(expected, bool) or isinstance(actual, bool) or expected is None or (actual is None):
        ok = type(expected) is type(actual) and expected == actual
    elif isinstance(expected, int) and isinstance(actual, int):
        ok = abs(expected - actual) <= (1 if '/per_arm/' in path else 0)
    elif isinstance(expected, (float, int)) and isinstance(actual, (float, int)):
        atol = 1e-06
        if '/quantile_' in path:
            atol = 0.0005
        elif '/random_intercept' in path:
            atol = 0.0002
        elif '/ci95_episode_week/' in path:
            atol = 0.15
        elif '/age_windows/' in path and '/ci95/' in path:
            atol = 0.02
        elif '/equal_author/ci95/' in path or '/diff_ci95/' in path:
            atol = 0.01
        ok = math.isclose(expected, actual, abs_tol=atol, rel_tol=1e-05)
    else:
        ok = type(expected) is type(actual) and expected == actual
    return [] if ok else [f'{path}: {expected!r} != {actual!r}']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', default=None, help='Optional output path for a recalculation')
    args = parser.parse_args()
    target = RESULTS / '17_final_sensitivity.json'
    manifest = json.loads((DATA / 'input_manifest.json').read_text())
    hashes = {}
    for k in ('habr_ai_ml_raw.jsonl.gz', 'habr_wide_raw.jsonl.gz', 'production_calendar_2016_2026.csv', 'lex_metrics.jsonl'):
        v = hashlib.sha256((DATA / k).read_bytes()).hexdigest()
        assert v == (manifest['sha256'] | manifest['derived_sha256'])[k]
        hashes[k] = v
    out = {'status': 'retrospective exploratory sensitivity; no independent confirmation', 'seed': SEED, 'n_boot': N_BOOT, 'hashes': hashes, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__}, 'corpora': {}}
    out['readability'] = readability()
    assert abs(out['readability']['all']['oborneva']['partial'] - 0.2000414890073068) < 1e-06
    print('readability complete', flush=True)
    for c in ('ai', 'wide'):
        d = load_corpus(c)
        r = recent(d)
        r = r[r.density_complete].copy()
        assert len(r) == {'ai': 6775, 'wide': 16577}[c]
        out['corpora'][c] = {'holiday': holiday(d, c), 'fold_ages': age_folds(d, c)}
        print(c, 'holidays and age complete', flush=True)
        out['corpora'][c]['calendar'] = calendar(r)
        prior = json.loads((RESULTS / '09_robustness_results.json').read_text())['corpora'][c]['adjusted_weekend']
        assert abs(out['corpora'][c]['calendar']['ols']['ratio'] - prior['ratio_target_plus1']) < 1e-06
        print(c, 'calendar complete', flush=True)
    result = finite_json(out)
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if args.check:
        errors = compare_results(json.loads(target.read_text(encoding='utf-8')), result)
        if errors:
            raise SystemExit('\n'.join(errors[:40]) + f'\n{len(errors)} mismatches')
        print('Sensitivity results match within specified tolerances', flush=True)
    elif not args.output:
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
