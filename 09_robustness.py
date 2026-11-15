"""Шаг 09. Чувствительность к возрасту, составу и единице анализа."""

import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy import stats
from habrpub.analysis import (
    author_slot_descriptive,
    bootstrap_cluster,
    compare_weekend,
    corr,
    finite_json,
    load_corpus,
    recent,
    save,
)


def adjusted_weekend(df, use_density=True, target="log_views"):
    """Условная регрессия выходных с контролями и кластеризацией по автору.

    Все цели — log10(1+x) (просмотры, комментарии, закладки),
    поэтому коэффициент переводится в отношение как 10**b. use_density=False
    даёт модель без dens24: если выходные действуют через низкую плотность,
    то плотность среди ковариат — посредник, и разница коэффициентов в двух
    вариантах показывает, сколько эффекта она съедает."""
    d = df.copy()
    X = pd.get_dummies(d[["month_id"]], drop_first=True, dtype=float)
    cols = ["is_weekend", "is_corp"] + (["dens24"] if use_density else [])
    for col in cols:
        X[col] = d[col].astype(float)
    X["log_age"] = np.log(d.age_days)
    counts = pd.Series([h for hs in d.hubs for h in hs]).value_counts()
    for h in counts.head(15).index:
        X["hub_" + h] = d.hubs.apply(lambda hs: float(h in hs))
    X = sm.add_constant(X.astype(float))
    model = sm.OLS(d[target], X).fit(
        cov_type="cluster", cov_kwds={"groups": d.author_key}
    )
    i = list(X.columns).index("is_weekend")
    ci = model.conf_int().iloc[i]
    return {
        "coefficient_log10_target": model.params.iloc[i],
        "ci95": ci.tolist(),
        "p_cluster_author": model.pvalues.iloc[i],
        "ratio_target_plus1": 10 ** model.params.iloc[i],
        "ratio_ci95": (10**ci).tolist(),
        "controls": "publication month, log age, corporate"
        + (", dens24" if use_density else "")
        + ", 15 hub indicators",
        "use_density": use_density,
        "target": target,
        "n": len(d),
        "warning": "conditional association; not intervention effect",
    }


def views_dispersion(r):
    """Разброс просмотров после учёта возраста.

    Возрастные группы 30–89, 90–179, 180–364 дней те же, что в age_bands;
    внутри каждой группы статьи делятся по квартилям/децилям просмотров,
    затем объединённые верхняя и нижняя части сравниваются медианами. Это
    масштаб естественного разброса, на фоне которого читаются приросты R²
    отдельных признаков. Рядом — доля выходных статей выше медианы будней."""
    parts = [
        r[(r.age_days >= lo) & (r.age_days < hi)]
        for lo, hi in [(30, 90), (90, 180), (180, 365)]
    ]
    parts = [p for p in parts if len(p) > 50]
    d = pd.concat(parts)

    wd_median = d.loc[~d.is_weekend, "views"].median()
    # Четверть против четверти и верхние 10% против нижних 10% — отношения
    # квантилей просмотров внутри каждой возрастной группы (медиана по
    # группам): медианы хвостов сильно перекошенного распределения дают
    # куда более резкие числа и плохо читаются.
    q2575 = [float(p.views.quantile(0.75) / p.views.quantile(0.25)) for p in parts]
    q1090 = [float(p.views.quantile(0.90) / p.views.quantile(0.10)) for p in parts]
    return {
        "age_bands": "30–89, 90–179, 180–364 days; quantile ratios within band, median across bands",
        "p75_p25_quantile_ratio": float(np.median(q2575)),
        "p90_p10_quantile_ratio": float(np.median(q1090)),
        "weekend_share_above_weekday_median": float(
            (d.loc[d.is_weekend, "views"] > wd_median).mean()
        ),
        "n": len(d),
        "warning": "descriptive spread of already published articles; not a decomposition of contributions",
    }


def score_weekend(df):
    """Рейтинг публикации может быть отрицательным, поэтому лог-регрессия к нему
    не применяется: медианы, ранговый тест и авторский интервал разницы."""
    w = df.loc[df.is_weekend, "score"]
    d = df.loc[~df.is_weekend, "score"]
    u, p = stats.mannwhitneyu(w, d, alternative="two-sided")

    def diff(z):
        return (
            z.loc[z.is_weekend, "score"].median()
            - z.loc[~z.is_weekend, "score"].median()
        )

    return {
        "n_weekend": len(w),
        "n_weekday": len(d),
        "median_weekend": w.median(),
        "median_weekday": d.median(),
        # Рейтинг дискретен, медиана малоинформативна:
        # рядом — доля статей с рейтингом 10 и выше.
        "share_ge10_weekend": float((w >= 10).mean()),
        "share_ge10_weekday": float((d >= 10).mean()),
        "p_iid": p,
        "rank_biserial": 2 * u / (len(w) * len(d)) - 1,
        "median_diff_ci_author": bootstrap_cluster(df, diff),
    }


def translations_weekend(df, rank_col):
    """Переводы и выходные: доля переводов по слотам и
    чувствительность основного сравнения выходных к их исключению.

    Стиль перевода отражает переводчика, а тема и успех частично заимствованы
    у оригинала. Признак есть только у ИИ/МЛ; для широкого корпуса возвращается
    available=False. Чувствительность — на медианах VPD и на перцентиле
    просмотров внутри месяца публикации; rank_col — колонка с этим перцентилем,
    её заранее считает main()."""
    if not df.is_translation.any():
        return {
            "available": False,
            "reason": "is_translation отсутствует в замороженной выгрузке широкого корпуса",
        }
    d = df.copy()
    by_slot = {
        "weekend" if k else "weekday": {
            "share": float(v.is_translation.mean()),
            "n": int(len(v)),
        }
        for k, v in d.groupby("is_weekend")
    }
    sub = d[~d.is_translation]
    w = sub.loc[sub.is_weekend, "vpd"]
    wd = sub.loc[~sub.is_weekend, "vpd"]
    res = {
        "available": True,
        "share_total": float(d.is_translation.mean()),
        "n_translations": int(d.is_translation.sum()),
        "share_by_slot": by_slot,
        "weekend_ratio_excl_translations": float(w.median() / wd.median()),
        "weekend_ratio_all": float(
            d.loc[d.is_weekend, "vpd"].median() / d.loc[~d.is_weekend, "vpd"].median()
        ),
        "cohort_rank_weekend_excl": float(sub.loc[sub.is_weekend, rank_col].median()),
        "cohort_rank_weekday_excl": float(sub.loc[~sub.is_weekend, rank_col].median()),
    }
    tr = d[d.is_translation]
    if len(tr) >= 10 and tr.is_weekend.nunique() == 2:
        res["weekend_ratio_translations_only"] = float(
            tr.loc[tr.is_weekend, "vpd"].median()
            / tr.loc[~tr.is_weekend, "vpd"].median()
        )
    return res


def main():
    corpora = {c: load_corpus(c) for c in ["ai", "wide"]}
    a = set(corpora["ai"].id)
    b = set(corpora["wide"].id)
    res = {
        "identity": {
            "ai": len(a),
            "wide": len(b),
            "overlap": len(a & b),
            "unique": len(a | b),
            "sum": len(a) + len(b),
        },
        "corpora": {},
    }
    for c, df in corpora.items():
        r = recent(df)
        r = r[r.density_complete]
        bands = []
        for lo, hi in [(30, 90), (90, 180), (180, 365)]:
            s = r[(r.age_days >= lo) & (r.age_days < hi)]
            bands.append(
                {"age_min": lo, "age_max_exclusive": hi, **compare_weekend(s, False)}
            )
        # Ранги внутри месяца публикации ослабляют возраст/эпоху, но не полностью устраняют их.
        r = r.copy()
        r["cohort_rank_views"] = r.groupby("month_id").views.rank(pct=True)
        rw = r.loc[r.is_weekend, "cohort_rank_views"]
        rd = r.loc[~r.is_weekend, "cohort_rank_views"]
        # Дополнительные исходы из собранных счётчиков: рейтинг, комментарии, закладки.
        # Все цели в одной шкале log10(1+x) — обратное преобразование едино.
        r["log_comments"] = np.log10(1 + r.comments.clip(lower=0))
        r["log_favorites"] = np.log10(1 + r.favorites.clip(lower=0))
        outcomes = {
            "comments": adjusted_weekend(r, target="log_comments"),
            "favorites": adjusted_weekend(r, target="log_favorites"),
            "score": score_weekend(r),
        }
        res["corpora"][c] = {
            "age_vpd_full": corr(
                df.loc[df.mature, "age_days"], df.loc[df.mature, "vpd"]
            ),
            "age_vpd_recent": corr(r.age_days, r.vpd),
            "age_bands": bands,
            "cohort_rank_median_weekend": rw.median(),
            "cohort_rank_median_weekday": rd.median(),
            "adjusted_weekend": adjusted_weekend(r),
            "adjusted_weekend_no_density": adjusted_weekend(r, use_density=False),
            "alternative_outcomes": outcomes,
            # Описательное «автор важнее слота»: терцили авторов
            # по их же статьям в окне; описание среза, не прогноз.
            "author_slot": author_slot_descriptive(r),
            "views_dispersion": views_dispersion(r),
            "translations_weekend": translations_weekend(r, "cohort_rank_views"),
        }
    save(res, "09_robustness_results.json")
    # finite_json превращает numpy-скаляры в обычные числа для читаемого вывода.
    print("09", finite_json(res["identity"]))
    for c in res["corpora"]:
        adj = res["corpora"][c]["adjusted_weekend"]
        nd = res["corpora"][c]["adjusted_weekend_no_density"]
        print(
            c,
            "weekend ratio dens:",
            finite_json(adj["ratio_target_plus1"]),
            "no-dens:",
            finite_json(nd["ratio_target_plus1"]),
            "| outcomes:",
            finite_json(
                {
                    k: (v.get("ratio_target_plus1") or v.get("median_weekend"))
                    for k, v in res["corpora"][c]["alternative_outcomes"].items()
                }
            ),
        )


if __name__ == "__main__":
    main()
