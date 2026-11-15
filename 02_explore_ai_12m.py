"""Шаг 02. Разведка ИИ/МЛ за последние 12 месяцев, только age_days >= 30."""

from habrpub.analysis import (
    load_corpus,
    recent,
    compare_weekend,
    finite_json,
    save,
    setup_plot,
    DATA,
    FIGS,
)
from scipy import stats
import numpy as np


def main():
    all_df = load_corpus("ai")
    df = recent(all_df)
    hit = df.vpd.quantile(0.9)
    day = df.groupby("dow").agg(n=("id", "size"), vpd=("vpd", "median"))
    hour = df.groupby("hour").agg(n=("id", "size"), vpd=("vpd", "median"))
    # Ячейки «день × час» с порогом n >= 20, как на тепловой карте: самые
    # яркие ячейки и доля выходных ячеек выше медианы будних ячеек.
    cells = df.groupby(["dow", "hour"]).vpd.agg(["median", "size"]).reset_index()
    cells = cells[cells["size"] >= 20]
    wd_cell_median = float(cells.loc[cells.dow < 5, "median"].median())
    we_cells = cells[cells.dow >= 5]
    heatmap = {
        "min_n": 20,
        "n_cells": int(len(cells)),
        "top": [
            {
                "dow": int(r.dow),
                "hour": int(r.hour),
                "vpd": float(r["median"]),
                "n": int(r["size"]),
            }
            for _, r in cells.sort_values("median", ascending=False).head(3).iterrows()
        ],
        "weekday_cell_median": wd_cell_median,
        "weekend_cells": int(len(we_cells)),
        "weekend_cells_above_weekday_median": int(
            (we_cells["median"] > wd_cell_median).sum()
        ),
    }
    H, p = stats.kruskal(*[df.loc[df.dow == d, "vpd"] for d in range(7)])
    res = {
        "window": "publication >= 2025-11-09 MSK; age >= 30 days",
        "n": len(df),
        "n_before_mature": len(all_df[all_df.dt_msk >= "2025-11-09"]),
        "snapshot_age_reference": "SNAP_AI; not a per-record retrieval timestamp",
        "weekend": compare_weekend(df),
        "vpd_median": df.vpd.median(),
        "views_median": df.views.median(),
        "hit_threshold_p90": hit,
        "by_day": day.reset_index().to_dict("records"),
        "by_hour": hour.reset_index().to_dict("records"),
        "heatmap": heatmap,
        "kruskal": {
            "H": H,
            "p_iid": p,
            "rank_effect_size": max((H - 6) / (len(df) - 7), 0),
        },
    }
    save(res, "02_explore_results.json")
    df.to_csv(DATA / "habr_12m.csv", index=False)
    plt = setup_plot()
    for group, name, label in [
        (day, "01_weekday.png", "День публикации (МСК)"),
        (hour, "02_hour.png", "Час публикации (МСК)"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 3.7), layout="constrained")
        ax.bar(group.index, group.vpd, color="#2d7ab3")
        if name.startswith("01"):
            ax.set_xticks(range(7), ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
        ax.set(
            xlabel=label,
            ylabel="VPD, медиана",
            title=f"ИИ/МЛ · зрелые статьи · n = {len(df)}",
        )
        for x, r in group.iterrows():
            ax.text(x, r.vpd, f"{int(r.n)}", ha="center", va="bottom", fontsize=7)
        ax.margins(y=0.18)
        fig.savefig(FIGS / name)
        plt.close(fig)
    mat = df.pivot_table(
        index="dow", columns="hour", values="vpd", aggfunc="median"
    ).reindex(index=range(7), columns=range(24))
    ns = (
        df.groupby(["dow", "hour"])
        .size()
        .unstack()
        .reindex(index=range(7), columns=range(24))
        .fillna(0)
    )
    mat = mat.where(ns >= 20)
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    # Скрытые ячейки — светло-серые, а не белые: на шкале Blues
    # белый цвет отвечает слабым значениям, и отсутствие данных сливалось с ними.
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="Blues")
    im.cmap.set_bad("#d9d9d9")
    ax.set_yticks(range(7), ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
    ax.set_xticks(range(0, 24, 2))
    ax.set(xlabel="Час (МСК)", title="Медиана VPD · ИИ/МЛ · 12 месяцев; серое — n < 20")
    fig.colorbar(im, ax=ax)
    fig.savefig(FIGS / "03_heatmap.png")
    plt.close(fig)
    print("02", res["n"], finite_json(res["weekend"]))


if __name__ == "__main__":
    main()
