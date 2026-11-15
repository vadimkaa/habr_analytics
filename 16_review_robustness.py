"""Дополнительные проверки устойчивости основных наблюдательных сравнений.

Отдельный отчёт, не меняющий базовые оценки. Проверяет оставление одного
праздничного эпизода за пределами выборки и два заранее фиксированных
варианта календарных контролей в регрессии выходных. Публикуется вместе
с данными, чтобы чувствительность выводов можно было проверить повторно.
"""

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from habrpub import RESULTS, DATA
from habrpub.analysis import WIDE_COMMON_START, load_corpus, recent
from habrpub.calendar_analysis import _holiday_days, _with_month_pct, holiday_episodes


def _regression(df, calendar="month", age="log"):
    """Независимый контроль регрессии 09, без повторной оценки моделей ML."""
    if calendar == "quarter":
        names = df.dt_msk.dt.strftime("%Y-").astype(str) + (
            ((df.dt_msk.dt.month - 1) // 3 + 1).astype(str)
        )
        X = pd.get_dummies(names.rename("quarter"), drop_first=True, dtype=float)
    else:
        X = pd.get_dummies(df[["month_id"]], drop_first=True, dtype=float)
    X["is_weekend"] = df.is_weekend.astype(float)
    X["is_corp"] = df.is_corp.astype(float)
    X["dens24"] = df.dens24.astype(float)
    if age == "bands":
        b = pd.cut(
            df.age_days, [0, 90, 180, 270, 365, np.inf], include_lowest=True
        ).astype(str)
        X = X.join(pd.get_dummies(b, prefix="age", drop_first=True, dtype=float))
    else:
        X["log_age"] = np.log(df.age_days)
    counts = pd.Series([h for hubs in df.hubs for h in hubs]).value_counts()
    for hub in counts.head(15).index:
        X["hub_" + hub] = df.hubs.apply(lambda hs: float(hub in hs))
    model = sm.OLS(df.log_views, sm.add_constant(X)).fit(
        cov_type="cluster", cov_kwds={"groups": df.author_key}
    )
    coef = float(model.params["is_weekend"])
    ci = model.conf_int().loc["is_weekend"]
    return {
        "n": int(len(df)),
        "ratio_views_plus1": float(10**coef),
        "ci95_cluster_author": [float(10**x) for x in ci],
        "calendar_controls": calendar,
        "age_controls": age,
        "interpretation": "conditional association, not a causal effect",
    }


def _holiday_sensitivity(df, corpus):
    start = (
        pd.Timestamp("2016-01-01", tz="Europe/Moscow")
        if corpus == "ai"
        else WIDE_COMMON_START
    )
    df = df[df.mature & (df.dt_msk >= start)].copy()
    d = _with_month_pct(df)
    cal = _holiday_days(df)
    hol = d[d.date_msk.isin(cal["hol"])]
    ctrl = d[d.date_msk.isin(cal["ctrl"])]
    episodes = [
        e for e in holiday_episodes(cal["hol"]) if hol.date_msk.isin(e).any()
    ]
    def contrast(a, b):
        return float(100 * (a.month_pct.median() - b.month_pct.median()))
    original = contrast(hol, ctrl)
    omitted, compared = [], []
    for e in episodes:
        removed = hol.date_msk.isin(e)
        if len(hol) - removed.sum() > 0:
            omitted.append(
                {
                    "episode_start": str(e[0]),
                    "episode_end": str(e[-1]),
                    "n_articles_removed": int(removed.sum()),
                    "difference_pp_without_episode": contrast(hol.loc[~removed], ctrl),
                }
            )
        dates = {day + pd.Timedelta(days=offset) for day in e for offset in (-21, -14, -7, 7, 14, 21)}
        matched = ctrl[ctrl.date_msk.isin(dates & cal["ctrl"])]
        if len(matched):
            compared.append(
                {
                    "episode_start": str(e[0]),
                    "n_articles": int(removed.sum()),
                    "n_matched_controls": int(len(matched)),
                    "difference_pp_vs_local_controls": contrast(hol.loc[removed], matched),
                }
            )
    return {
        "n_episodes": len(episodes),
        "n_holiday_articles": len(hol),
        "n_control_articles": len(ctrl),
        "difference_pp": original,
        "leave_one_episode_out": omitted,
        "loo_min_pp": min(x["difference_pp_without_episode"] for x in omitted),
        "loo_max_pp": max(x["difference_pp_without_episode"] for x in omitted),
        "episode_local_comparisons": compared,
        "n_positive_local_comparisons": sum(x["difference_pp_vs_local_controls"] > 0 for x in compared),
        "warning": "Small number of episodes; resampling or individual-article p-values do not establish causal effects.",
    }


def main():
    prior = json.loads((RESULTS / "09_robustness_results.json").read_text())
    out = {"corpora": {}, "status": "retrospective sensitivity checks, exploratory"}
    sample = pd.read_csv(DATA / "readability_analysis.csv")
    for corpus in ("ai", "wide"):
        df = load_corpus(corpus)
        r = recent(df)
        r = r[r.density_complete].copy()
        base = _regression(r)
        reference = prior["corpora"][corpus]["adjusted_weekend"]
        # Контракт: новый независимый контроль не должен расходиться с 09.
        assert abs(base["ratio_views_plus1"] - reference["ratio_target_plus1"]) < 1e-8
        assert np.allclose(base["ci95_cluster_author"], reference["ratio_ci95"], atol=1e-8)
        w, wd = r.loc[r.is_weekend, "age_days"], r.loc[~r.is_weekend, "age_days"]
        q = df.vpd.quantile([.25, .50, .75]).to_numpy()
        bins = [-np.inf, *q, np.inf]
        def share(vals):
            return [
                float(x)
                for x in pd.cut(
                    vals, bins, labels=[1, 2, 3, 4]
                ).value_counts(normalize=True).sort_index().mul(100).to_numpy()
            ]
        text = sample[(sample.corpus == corpus) & (sample.age_days >= 30)]
        out["corpora"][corpus] = {
            "text_sample_quartile_check": {
                "mature_population_pct": share(df.loc[df.mature, "vpd"]),
                "mature_text_sample_pct": share(text.vpd),
                "n_text_sample": len(text),
                "definition": "VPD quartiles from entire original corpus including young articles",
            },
            "regression": {
                "baseline": base,
                "quarter_calendar": _regression(r, calendar="quarter"),
                "age_bands": _regression(r, age="bands"),
            },
            "age_comparison": {
                "median_weekend": float(w.median()),
                "median_weekday": float(wd.median()),
                "iqr_weekend": [float(w.quantile(.25)), float(w.quantile(.75))],
                "iqr_weekday": [float(wd.quantile(.25)), float(wd.quantile(.75))],
            },
            "holidays": _holiday_sensitivity(df, corpus),
        }
    # Округляем диагностические числа до шести знаков: JSON и CI остаются
    # побайтово воспроизводимыми при малых разностях версий BLAS/Statsmodels.
    def round_floats(x):
        if isinstance(x, float):
            return round(x, 6)
        if isinstance(x, dict):
            return {k: round_floats(v) for k, v in x.items()}
        if isinstance(x, list):
            return [round_floats(v) for v in x]
        return x

    out = round_floats(out)
    (RESULTS / "16_review_robustness.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for corpus, item in out["corpora"].items():
        hol = item["holidays"]
        print(corpus, "LOO holidays", hol["loo_min_pp"], hol["loo_max_pp"])
        print(corpus, "regression", [v["ratio_views_plus1"] for v in item["regression"].values()])


if __name__ == "__main__":
    main()
