"""Шаг 07. Читаемость: корреляции, авторский bootstrap, BH, AUC и контроль состава."""

import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy import stats
from habrpub import DATA, FIGS
from habrpub.analysis import (
    N_BOOT,
    bh,
    bootstrap_cluster,
    corr,
    load_corpus,
    save,
    setup_plot,
)

INDICES = ["oborneva", "flesch_web", "flesch_en", "fk_mod", "oborneva_ru", "fk_mod_ru"]
METRICS = INDICES + [
    "asl",
    "ttr_ru4",
    "mattr_ru4",
    "long_word_share",
    "en_word_share",
    "code_char_share",
    "readingTime",
]
LABELS = {
    "oborneva": "ИУФО 1.52/65.14",
    "flesch_web": "Флеш 1.3/60.1",
    "flesch_en": "Флеш англ.",
    "fk_mod": "ФК МОД",
    "oborneva_ru": "ИУФО рус.",
    "fk_mod_ru": "ФК МОД рус.",
    "asl": "Длина предложения",
    "ttr_ru4": "TTR ≥ 4 буквы",
    "mattr_ru4": "MATTR ≥ 4 буквы, окно 500",
    "long_word_share": "Доля длинных слов",
    "en_word_share": "Доля англ. слов",
    "code_char_share": "Доля кода",
    "readingTime": "Время чтения",
}


def load_sample():
    meta = pd.concat([load_corpus("wide"), load_corpus("ai")], ignore_index=True)
    new = pd.read_json(DATA / "lex_metrics.jsonl", lines=True, dtype={"id": str})
    df = new.merge(
        meta[
            [
                "id",
                "corpus",
                "vpd",
                "views",
                "comments",
                "favorites",
                "score",
                "mature",
                "is_corp",
                "age_days",
                "author_key",
                "month_id",
                "is_translation",
                "hubs",
            ]
        ],
        on=["id", "corpus"],
        validate="many_to_one",
    )
    # При пересечении сохраняем wide, независимо от порядка загрузки в кэше.
    df["priority"] = df.corpus.map({"wide": 0, "ai": 1})
    df = df.sort_values(["priority", "id"]).drop_duplicates("id")
    return df[df.mature & df.oborneva.notna()].reset_index(drop=True)


def adjusted_readability(df):
    """Частная ранговая ассоциация; ковариаты не обеспечивают причинной идентификации."""
    d = df.dropna(subset=["oborneva", "vpd", "words", "age_days"]).copy()
    cats = pd.get_dummies(d[["corpus", "month_id"]], drop_first=True, dtype=float)
    cats["corp"] = d.is_corp.astype(float)
    # Тематические индикаторы для 20 наиболее частых хабов (multi-label).
    counts = pd.Series([h for hs in d.hubs for h in hs]).value_counts()
    for h in counts.head(20).index:
        cats["hub_" + h] = d.hubs.apply(lambda hs: float(h in hs))
    cats["log_words"] = np.log1p(d.words)
    cats["log_age"] = np.log1p(d.age_days)
    X = sm.add_constant(cats.astype(float)).to_numpy()
    x = stats.rankdata(d.oborneva)
    y = stats.rankdata(d.vpd)
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return {
        "n": len(d),
        "partial_rank_correlation": float(np.corrcoef(rx, ry)[0, 1]),
        "controls": "corpus, publication month, corporate, 20 hub indicators, log words, log age",
        "note": "descriptive residual Pearson correlation of ranks; no causal interpretation",
    }


def _partial_rank(d, metric):
    """Частная ранговая корреляция метрики с VPD внутри одного корпуса.

    Ковариаты те же, что в adjusted_readability (месяц, корпоративность,
    20 индикаторов хабов, длина, возраст), кроме самого корпуса: числа по
    корпусам считаются той же процедурой, что и общая оценка, а не отдельно
    взяты из упрощённой проверки."""
    s = d.dropna(subset=[metric, "vpd", "words", "age_days"]).copy()
    cats = pd.get_dummies(s[["month_id"]], drop_first=True, dtype=float)
    cats["corp"] = s.is_corp.astype(float)
    counts = pd.Series([h for hs in s.hubs for h in hs]).value_counts()
    for h in counts.head(20).index:
        cats["hub_" + h] = s.hubs.apply(lambda hs: float(h in hs))
    cats["log_words"] = np.log1p(s.words)
    cats["log_age"] = np.log1p(s.age_days)
    X = sm.add_constant(cats.astype(float)).to_numpy()
    x = stats.rankdata(s[metric])
    y = stats.rankdata(s.vpd)
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return {
        "n": len(s),
        "partial_rank_correlation": float(np.corrcoef(rx, ry)[0, 1]),
        "controls": "publication month, corporate, 20 hub indicators, log words, log age (within corpus)",
    }


def length_analysis(df):
    """Длина текста и просмотры: «чем короче, тем лучше»
    не подтверждается — связь длины с просмотрами слабо положительная,
    а самые длинные тексты набирают не меньше коротких."""
    d = df.dropna(subset=["words", "vpd"]).copy()
    d["word_quartile"] = pd.qcut(d.words, 4, labels=False) + 1
    quartiles = [
        {
            "quartile": int(k),
            "n": int(len(s)),
            "words_median": float(s.words.median()),
            "vpd_median": float(s.vpd.median()),
        }
        for k, s in d.groupby("word_quartile")
    ]
    Z = np.column_stack(
        [
            np.ones(len(d)),
            stats.rankdata(d.age_days),
            d.is_corp.astype(float).to_numpy(),
        ]
    )
    x = stats.rankdata(d.words)
    y = stats.rankdata(d.vpd)
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    short = d[d.words < 500]
    return {
        "rho_words_vpd": float(stats.spearmanr(d.words, d.vpd).statistic),
        "rho_words_vpd_partial_age_corp": float(np.corrcoef(rx, ry)[0, 1]),
        "quartiles": quartiles,
        "n_words_below_500": int(len(short)),
        "rho_below_500": (
            float(stats.spearmanr(short.words, short.vpd).statistic)
            if len(short) > 30
            else None
        ),
        "note": "texts below the 500-word MATTR window behave differently (mostly news notes); MATTR returns plain TTR for them",
    }


def main():
    df = load_sample()
    # Перцентиль просмотров внутри (корпус, месяц): альтернативная метрика
    # для проверки читаемости.
    df["cohort_pct"] = df.groupby(["corpus", "month_id"]).views.rank(pct=True)
    rows = []
    for metric in METRICS:
        d = df.dropna(subset=[metric, "vpd"]).copy()
        row = {"metric": metric, **corr(d[metric], d.vpd)}
        # Корреляция вычисляется в реплике заново по рангам; кластеры — авторы.
        row["ci95_author"] = bootstrap_cluster(
            d, lambda z: stats.spearmanr(z[metric], z.vpd).statistic, n_boot=N_BOOT
        )
        rows.append(row)
    for row, q in zip(rows, bh([r["p"] for r in rows])):
        row["q_bh"] = q
    agreement = df[INDICES].corr(method="spearman")
    # Все сравнения очистки на одних и тех же ID/corpus, включая пропуски.
    comparison = {}
    for name, file in [
        ("original_full", "preprocessing_baseline_full.jsonl"),
        ("original_pre_only", "preprocessing_baseline_pre_only.jsonl"),
    ]:
        old = pd.read_json(DATA / file, lines=True, dtype={"id": str}).drop_duplicates(
            ["id", "corpus"]
        )
        join = df.merge(
            old[["id", "corpus"] + INDICES],
            on=["id", "corpus"],
            suffixes=("", "_old"),
            validate="one_to_one",
        )
        comparison[name] = []
        for m in INDICES:
            j = join.dropna(subset=[m, m + "_old", "vpd"])
            new = corr(j[m], j.vpd)
            oldr = corr(j[m + "_old"], j.vpd)
            comparison[name].append(
                {
                    "metric": m,
                    "n": len(j),
                    "rho_new": new["rho"],
                    "rho_old": oldr["rho"],
                    "delta": new["rho"] - oldr["rho"],
                }
            )
    # Два определения «успеха», всегда явно указываем порог.
    auc = {}
    for quantile in [0.75, 0.9]:
        threshold = df.vpd.quantile(quantile)
        a = df.loc[df.vpd >= threshold, "oborneva"]
        b = df.loc[df.vpd < threshold, "oborneva"]
        auc[str(quantile)] = {
            "threshold": threshold,
            "n_hit": len(a),
            "auc": stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)),
        }
    df["decile"] = pd.qcut(df.oborneva, 10, labels=False, duplicates="drop") + 1
    dec = df.groupby("decile").vpd.agg(["median", "size"]).reset_index()
    rt = df[(df.words > 100) & (df.readingTime > 0)]
    res = {
        "n": len(df),
        "n_corp": int(df.is_corp.sum()),
        "n_personal": int((~df.is_corp).sum()),
        "table": rows,
        "agreement": agreement.to_dict(),
        "preprocessing_comparison": comparison,
        "auc": auc,
        "deciles": dec.to_dict("records"),
        "adjusted": adjusted_readability(df),
        # Частные корреляции по корпусам с теми же поправками, что у общей
        # оценки: числа статьи берутся из этого JSON.
        "adjusted_by_corpus": {
            m: {c: _partial_rank(s, m) for c, s in df.groupby("corpus")}
            for m in ["oborneva", "asl", "long_word_share"]
        },
        # Проверка на перцентиле просмотров внутри месяца.
        "percentile_metric": {
            "all": float(
                stats.spearmanr(
                    df.dropna(subset=["oborneva"]).oborneva,
                    df.dropna(subset=["oborneva"]).cohort_pct,
                ).statistic
            ),
            "by_corpus": {
                c: float(
                    stats.spearmanr(
                        s.dropna(subset=["oborneva"]).oborneva,
                        s.dropna(subset=["oborneva"]).cohort_pct,
                    ).statistic
                )
                for c, s in df.groupby("corpus")
            },
            "definition": "Spearman rho of oborneva index with percentile of views within (corpus, month)",
        },
        # Длина текста и просмотры: квартили, частная корреляция, короткие
        # тексты.
        "length": length_analysis(df),
        "subgroups": {c: corr(s.oborneva, s.vpd) for c, s in df.groupby("corpus")},
        "corp_subgroups": {
            str(c): corr(s.oborneva, s.vpd) for c, s in df.groupby("is_corp")
        },
        # Переводы: стиль переводчика и успех частично заимствованы
        # у оригинала. Признак есть только у ИИ/МЛ; проверка — чувствительность
        # связи ИУФО с VPD при исключении переводов и внутри них.
        "translations": {
            c: {
                "n": int(len(s)),
                "n_translations": int(s.is_translation.sum()),
                "share": float(s.is_translation.mean()),
                "rho_excl_translations": corr(
                    s.loc[~s.is_translation, "oborneva"],
                    s.loc[~s.is_translation, "vpd"],
                )["rho"],
                "rho_translations_only": corr(
                    s.loc[s.is_translation, "oborneva"], s.loc[s.is_translation, "vpd"]
                )["rho"],
            }
            for c, s in [(c, df[(df.corpus == c)]) for c in ["ai", "wide"]]
        },
        "words_reading_time": corr(rt.words, rt.readingTime),
        "words_per_minute": (rt.words / rt.readingTime).median(),
    }
    save(res, "07_readability_results.json")
    df.drop(columns=["hubs"]).to_csv(DATA / "readability_analysis.csv", index=False)
    plt = setup_plot()
    fig, ax = plt.subplots(figsize=(7.8, 4.7), layout="constrained")
    for i, r in enumerate(rows):
        ax.plot(r["ci95_author"], [i, i], color="#2d7ab3")
        ax.scatter(r["rho"], i, color="#2d7ab3")
    ax.set_yticks(range(len(rows)), [LABELS[r["metric"]] for r in rows])
    ax.axvline(0, color="gray", lw=1)
    ax.set(
        xlabel="Спирмен ρ с VPD · 95% ДИ по авторам",
        title=f"Описательные связи · n = {len(df)}",
    )
    fig.savefig(FIGS / "08_readability.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    im = ax.imshow(agreement, vmin=-1, vmax=1, cmap="RdBu_r")
    labs = [LABELS[x] for x in INDICES]
    ax.set_xticks(range(6), labs, rotation=45, ha="right")
    ax.set_yticks(range(6), labs)
    for i in range(6):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{agreement.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(agreement.iloc[i, j]) > 0.65 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="Спирмен ρ")
    fig.savefig(FIGS / "07_agreement.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.bar(dec.decile, dec["median"], color="#2d7ab3")
    ax.set(
        xlabel="Дециль ИУФО: 1 — труднее, 10 — проще",
        ylabel="VPD, медиана",
        title="Описательный градиент; состав групп различается",
    )
    fig.savefig(FIGS / "09_deciles.png")
    plt.close(fig)
    print("07", res["n"], rows[0], res["adjusted"], res["subgroups"])


if __name__ == "__main__":
    main()
