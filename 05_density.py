"""Шаг 05. Плотность корпуса за [t−24ч,t) и [t,t+24ч) и ассоциации. Причинность не оценивается."""

import numpy as np
import pandas as pd
from habrpub import FIGS
from habrpub.analysis import (
    bh,
    bootstrap_cluster,
    compare_weekend,
    corr,
    finite_json,
    load_corpus,
    recent,
    save,
    setup_plot,
)


def main():
    res = {}
    plt = setup_plot()
    for corpus in ["ai", "wide"]:
        all_df = load_corpus(corpus)
        df = recent(all_df)
        df = df[df.density_complete].copy()
        df["q"] = pd.qcut(df.dens24, 5, labels=False, duplicates="drop")
        rows = []
        for q, s in df.groupby("q"):
            if s.is_weekend.nunique() < 2:
                continue
            a = compare_weekend(s, with_ci=False)
            # Интервал по авторам для каждого столбца: статьи одного автора не независимы.
            a["ci95_author"] = bootstrap_cluster(
                s,
                lambda z: (
                    z.loc[z.is_weekend, "vpd"].median()
                    / z.loc[~z.is_weekend, "vpd"].median()
                ),
            )
            rows.append(
                {
                    "q": int(q) + 1,
                    "density_min": s.dens24.min(),
                    "density_max": s.dens24.max(),
                    "n": len(s),
                    # Квинтили равны по числу статей; информативен счёт выходных
                    #: они почти целиком лежат в нижнем квинтиле.
                    "n_weekend": int(s.is_weekend.sum()),
                    **a,
                }
            )
        qs = bh([r["p_iid"] for r in rows])
        for r, q in zip(rows, qs):
            r["q_bh"] = q
        eras = [
            {"month": m, **corr(s.dens24, s.vpd)}
            for m, s in df.groupby("month_id")
            if len(s) >= 50
        ]
        # Прямое окно [t, t+24 ч): в хронологической ленте статью
        # вытесняют и те, кто вышел после неё. Окна у правой границы корпуса
        # неполны, для прямой плотности они отсекаются отдельно.
        df_f = df[df.density_complete_fwd]
        eras_fwd = [
            {"month": m, **corr(s.dens24_fwd, s.vpd)}
            for m, s in df_f.groupby("month_id")
            if len(s) >= 50
        ]
        res[corpus] = {
            "n": len(df),
            "rho": corr(df.dens24, df.vpd),
            "quintiles": rows,
            "within_month": eras,
            "median_rho_within_month": np.median(
                [r["rho"] for r in eras if r["rho"] is not None]
            ),
            "full_history_rho": corr(
                all_df.loc[all_df.mature, "dens24"], all_df.loc[all_df.mature, "vpd"]
            ),
            "n_fwd": len(df_f),
            "rho_fwd": corr(df_f.dens24_fwd, df_f.vpd),
            "within_month_fwd": eras_fwd,
            "median_rho_within_month_fwd": (
                np.median([r["rho"] for r in eras_fwd if r["rho"] is not None])
                if eras_fwd
                else None
            ),
        }
        h = df.groupby("hour").is_corp.agg(["mean", "size"])
        res[corpus]["corp_share_by_hour"] = h.reset_index().to_dict("records")
    save(res, "part4_results.json")
    labels = {"ai": "ИИ/МЛ", "wide": "Пять хабов"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    for c, off, col in [("ai", -0.17, "#2d7ab3"), ("wide", 0.17, "#e07a3f")]:
        a = res[c]["quintiles"]
        xs = np.array([r["q"] for r in a]) + off
        ys = np.array([r["ratio"] for r in a])
        ax.bar(xs, ys, width=0.32, label=labels[c], color=col)
        # Усы - 95% интервал медианного отношения при пересэмплировании авторов.
        lo = np.array([r["ci95_author"][0] for r in a])
        hi = np.array([r["ci95_author"][1] for r in a])
        ax.errorbar(
            xs,
            ys,
            yerr=[np.maximum(ys - lo, 0), np.maximum(hi - ys, 0)],
            fmt="none",
            ecolor="#333333",
            capsize=3,
            lw=1,
        )
    ax.axhline(1, color="gray", lw=1)
    ax.set(
        xlabel="Квинтиль плотности за предыдущие 24 ч · усы — 95% ДИ по авторам",
        ylabel="Отношение медиан VPD",
        title="Выходные к будням внутри квинтилей плотности · 12 месяцев",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(FIGS / "05_density.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    for c in ["ai", "wide"]:
        a = res[c]["corp_share_by_hour"]
        ax.plot([r["hour"] for r in a], [100 * r["mean"] for r in a], label=labels[c])
    ax.set(
        xlabel="Час (МСК)",
        ylabel="Корпоративных статей, %",
        title="Состав публикаций по часам",
    )
    ax.legend()
    fig.savefig(FIGS / "06_corporate.png")
    plt.close(fig)
    print(
        "05",
        finite_json(
            {
                k: (v["rho"], v["median_rho_within_month"], v["rho_fwd"])
                for k, v in res.items()
            }
        ),
    )


if __name__ == "__main__":
    main()
