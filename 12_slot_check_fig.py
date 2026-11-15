"""Шаг 12. Диагностическое сравнение слотов на двух корпусах.

Генерирует отдельный контрольный результат, не используемый в статье.
Тестовые группы формируются через habrpub.legacy_model.policy_groups
(prepare -> eligible -> split -> merge с grid), поэтому медианы групп совпадают
с 08_model_results.json и статьёй. Правило «выходные/будни» считается по тем же
тестовым статьям здесь же.
Запуск из корня пакета: python 12_slot_check_fig.py"""

import json

import numpy as np
import pandas as pd
from scipy import stats

from habrpub import DATA, FIGS, RESULTS
from habrpub import legacy_model as m08
from habrpub.analysis import N_BOOT, SEED, WIDE_COMMON_START, load_corpus, setup_plot

SPLIT = {"ai": "2026-01-01", "wide": "2026-05-01"}
LABEL = {"ai": "ИИ/МЛ", "wide": "Пять хабов"}
RNG = np.random.default_rng(SEED)

def slot_test(corpus):
    """Точные тестовые группы из 08_model.policy_groups."""
    df = m08.prepare(load_corpus(corpus))
    eligible = df.mature & df.density_complete
    eligible &= (
        (df.year >= 2016) if corpus == "ai" else (df.dt_msk >= WIDE_COMMON_START)
    )
    df = df[eligible].copy()
    split = pd.Timestamp(SPLIT[corpus], tz="Europe/Moscow")
    tr, te = df[df.dt_msk < split], df[df.dt_msk >= split]
    grid = pd.read_csv(DATA / f"model_grid_{corpus}.csv")
    _, te_pg = m08.policy_groups(tr, te, grid)
    return te_pg


def boot_ci(values, authors, n=N_BOOT):
    """95% ДИ медианы группы при пересэмплировании целых авторов."""
    values = np.asarray(values, float)
    authors = np.asarray(authors)
    uniq = np.unique(authors)
    by_author = {a: values[authors == a] for a in uniq}
    meds = np.empty(n)
    for i in range(n):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        meds[i] = np.median(np.concatenate([by_author[a] for a in pick]))
    return np.percentile(meds, [2.5, 97.5])


def ratio_ci(out, n=N_BOOT):
    """95% ДИ отношения верх/низ при пересэмплировании авторов."""
    hi, lo = out[out.group == 2], out[out.group == 0]
    pool = {
        a: (s[s.group == 2].views.to_numpy(), s[s.group == 0].views.to_numpy())
        for a, s in pd.concat([hi, lo]).groupby("author_key")
    }
    keys = list(pool)
    rats = np.empty(n)
    for i in range(n):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        h = np.concatenate([pool[a][0] for a in pick])
        low_values = np.concatenate([pool[a][1] for a in pick])
        rats[i] = np.median(h) / np.median(low_values)
    return np.percentile(rats, [2.5, 97.5])


def mask_ratio_ci(out, hi_mask, n=N_BOOT):
    """95% ДИ отношения медиан «верхняя доля против остальных» по авторам.

    У правила разбиение 19/81%, у сетки — трети; сравнивать их
    можно только на одинаковых разбиениях, поэтому верхняя доля прогноза сетки
    берётся равной доле выходных статей."""
    hi, lo = out[hi_mask], out[~hi_mask]
    pool = {
        a: (
            s[s.index.isin(hi.index)].views.to_numpy(),
            s[s.index.isin(lo.index)].views.to_numpy(),
        )
        for a, s in out.groupby("author_key")
    }
    keys = list(pool)
    rats = np.empty(n)
    for i in range(n):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        h = np.concatenate([pool[a][0] for a in pick])
        low_values = np.concatenate([pool[a][1] for a in pick])
        rats[i] = np.median(h) / np.median(low_values)
    return np.percentile(rats, [2.5, 97.5])


def spearman_ci(out, col, n=N_BOOT):
    """Ранговая корреляция с просмотрами и ДИ по авторам (не зависит от разбиения)."""
    rho = float(stats.spearmanr(out[col], out.views).statistic)
    pool = {a: s[[col, "views"]] for a, s in out.groupby("author_key")}
    keys = list(pool)
    rs = np.empty(n)
    for i in range(n):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        part = pd.concat([pool[a] for a in pick])
        rs[i] = stats.spearmanr(part[col], part.views).statistic
    return rho, np.percentile(rs, [2.5, 97.5])


def diff_rho_ci(out, n=N_BOOT):
    """95% ДИ РАЗНОСТИ корреляций (флаг выходных минус прогноз сетки) на общих
    бутстрап-выборках авторов: у двух корреляций одна выборка,
    поэтому их интервалы по отдельности разность не покрывают. Положительная
    разница — правило ранжирует лучше сетки."""
    pool = {
        a: s[["relative_prediction", "is_weekend", "views"]]
        for a, s in out.groupby("author_key")
    }
    keys = list(pool)
    ds = np.empty(n)
    for i in range(n):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        part = pd.concat([pool[a] for a in pick])
        ds[i] = stats.spearmanr(part.is_weekend, part.views).statistic - (
            stats.spearmanr(part.relative_prediction, part.views).statistic
        )
    return np.percentile(ds, [2.5, 97.5])


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    res = json.loads((RESULTS / "08_model_results.json").read_text())
    # Ранговая корреляция полной модели на тесте (совпадает со статьёй).
    sd_model_rho = {c: res["models"][c]["full"]["rho"] for c in ["ai", "wide"]}
    plt = setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained", sharey=True)
    rule_cmp = {}
    whisker_tops = []
    for ax, corpus in zip(axes, ["ai", "wide"]):
        out = slot_test(corpus)
        sd = res["models"][corpus]["slot_diagnostic"]
        meds, sizes = [], []
        for k in (0, 1, 2):
            g = out[out.group == k]
            assert abs(g.views.median() - sd["groups"][k]["views_median"]) < 1e-6, (
                corpus,
                k,
            )
            assert len(g) == sd["groups"][k]["size"], (corpus, k)
            meds.append(g.views.median())
            sizes.append(len(g))
        cis = [
            boot_ci(out[out.group == k].views, out[out.group == k].author_key)
            for k in (0, 1, 2)
        ]
        whisker_tops.append(max(m + (c[1] - m) for m, c in zip(meds, cis)))
        yerr = [
            [m - c[0] for m, c in zip(meds, cis)],
            [c[1] - m for m, c in zip(meds, cis)],
        ]
        ax.bar(range(3), meds, color="#2d7ab3", width=0.62)
        ax.errorbar(
            range(3), meds, yerr=yerr, fmt="none", ecolor="#333333", capsize=4, lw=1.2
        )
        ax.set_xticks(
            range(3),
            [
                f"Низкий\nn={sizes[0]}",
                f"Средний\nn={sizes[1]}",
                f"Высокий\nn={sizes[2]}",
            ],
        )
        ax.set(
            xlabel="Терциль прогноза слота; границы из train",
            title=LABEL[corpus] + " · личные статьи",
        )
        # Опубликованный интервал берётся из JSON; если его там нет - тем же методом здесь.
        ratio = sd["top_bottom_ratio"]
        rci = sd.get("ci95_author") or ratio_ci(out)
        # Наивное правило «выходные/будни» на тех же тестовых статьях (не хардкод).
        # is_weekend после prepare() - целые 0/1, поэтому сравнение явное.
        naive = (
            out[out.is_weekend == 1].views.median()
            / out[out.is_weekend == 0].views.median()
        )
        wd = out[out.is_weekend == 0]
        wd_ratio = wd[wd.group == 2].views.median() / wd[wd.group == 0].views.median()
        rule_cmp[corpus] = {
            "model_ratio": float(ratio),
            "model_ci95": [float(x) for x in rci],
            "naive_ratio": float(naive),
            "weekday_only_ratio": float(wd_ratio),
            "n_test_personal_supported": sd["n_test_personal_supported"],
        }
        # Сравнение политик на одинаковых разбиениях и ранговые
        # корреляции с просмотрами (те от разбиения не зависят).
        share = float((out.is_weekend == 1).mean())
        thr = np.quantile(out.relative_prediction, 1 - share)
        hi_mask = out.relative_prediction >= thr
        rho_grid, rho_grid_ci = spearman_ci(out, "relative_prediction")
        rho_flag, rho_flag_ci = spearman_ci(out, "is_weekend")
        rule_cmp[corpus].update(
            {
                "weekend_share_test": share,
                "grid_same_split": {
                    "ratio": float(
                        out[hi_mask].views.median() / out[~hi_mask].views.median()
                    ),
                    "ci95_author": [float(x) for x in mask_ratio_ci(out, hi_mask)],
                },
                "rank_corr": {
                    "grid_profile": {
                        "rho": rho_grid,
                        "ci95": [float(x) for x in rho_grid_ci],
                    },
                    "weekend_flag": {
                        "rho": rho_flag,
                        "ci95": [float(x) for x in rho_flag_ci],
                    },
                    # Разность корреляций на общих выборках:
                    # флаг выходных минус прогноз сетки — насколько правило
                    # ранжирует лучше сетки; положительная разница в пяти хабах
                    # означает преимущество правила.
                    "difference": {
                        "rho": float(rho_flag - rho_grid),
                        "ci95": [float(x) for x in diff_rho_ci(out)],
                    },
                    "model_full": {"rho": float(sd_model_rho[corpus])},
                },
            }
        )
        rtxt = f"{fmt(ratio)} [{fmt(rci[0])}; {fmt(rci[1])}]"
        ax.text(
            0.03,
            0.97,
            f"верх/низ: {rtxt}\nправило «выходные/будни»: {fmt(naive)}\n"
            f"сетка на том же разбиении: {fmt(rule_cmp[corpus]['grid_same_split']['ratio'])}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbbbbb"),
        )
        print(
            corpus,
            "медианы",
            [round(float(m), 1) for m in meds],
            "ДИ групп:",
            [[round(float(x), 1) for x in c] for c in cis],
            "верх/низ",
            rtxt,
            "правило",
            fmt(naive),
        )
        # Текст на рисунке обязан совпадать со значениями, которые уйдут в статью.
        exp = f"{fmt(sd['top_bottom_ratio'])} [{fmt(sd['ci95_author'][0])}; {fmt(sd['ci95_author'][1])}]"
        assert rtxt == exp, (rtxt, exp)
    axes[0].set_ylabel("Просмотры на тесте, медиана")
    # Запас сверху, чтобы плашка с подписью не перекрывала усы.
    axes[0].set_ylim(0, max(whisker_tops) * 1.5)
    (FIGS / "legacy").mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "legacy" / "lgbm_slot_check.png", dpi=160)
    (RESULTS / "slot_rule_comparison.json").write_text(
        json.dumps(rule_cmp, ensure_ascii=False, indent=2) + "\n"
    )
    print("сохранено: figs/legacy/lgbm_slot_check.png, slot_rule_comparison.json")


if __name__ == "__main__":
    main()
