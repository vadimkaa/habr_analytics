"""Шаг 15. Рисунки 10 и 11 статьи из сохранённых результатов эксперимента CatBoost.

Модели здесь не обучаются: шаг читает JSON из experiments/catboost/results/
(их получает experiments/catboost/run_experiments.py) и только рисует.

Рисунок 10 — корреляция Спирмена внутри месяцев на валидационной выборке
для LightGBM с пуассоновской функцией потерь, LightGBM на логарифме просмотров
и выбранной модели CatBoost (раздел 9, таблица 17).

Рисунок 11 — прирост корреляции относительно правила выходных (Δρ) в режимах
A, B1 и B2 с 95% интервалами по авторам (раздел 10, таблица 18).
"""

import json

from matplotlib.ticker import FuncFormatter

from habrpub import BASE, FIGS
from habrpub.analysis import setup_plot

ML = BASE / "experiments" / "catboost" / "results"
CORPORA = [("ai", "ИИ/МЛ"), ("wide", "Пять хабов")]
COLORS = ["#a3b6cb", "#5f8fb4", "#1a6496"]


def read(rel):
    return json.loads((ML / rel).read_text(encoding="utf-8"))


def no_negative_zero(x, _pos):
    """Подпись оси без «−0.00»."""
    s = f"{x:.2f}"
    return "0.00" if s in ("-0.00", "0.00") else s.replace("-", "−")


def fig_models(plt, loss, winner):
    labels = ["LightGBM\nPoisson", "LightGBM\nRMSE (log1p)", "CatBoost\nвыбранная"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    top = 0
    for ax, (c, title) in zip(axes, CORPORA):
        models = ["lgb_poisson_v24", "lgb_rmse_log1p", winner[c]]
        vals = [loss[f"{c}:{m}"]["validation"]["rho_within_month_mean"] for m in models]
        top = max(top, max(vals))
        bars = ax.bar(labels, vals, color=COLORS, width=0.62)
        for b, x in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                x + 0.004,
                f"{x:.3f}",
                ha="center",
                va="bottom",
            )
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Спирмен внутри месяцев\n(валидационная выборка)")
    axes[0].set_ylim(0, top * 1.18)
    fig.tight_layout()
    fig.savefig(FIGS / "10_catboost_comparison.png")
    plt.close(fig)


def fig_modes(plt, policy, winner):
    # Сверху вниз в том же порядке, что строки таблицы 18: A, B1, B2.
    rows = [
        ("A: полный\nпрогноз", "A"),
        ("B1: только\nкалендарь", "B1"),
        ("B2: личный\nрейтинг слота", "B2"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, sharex=True)
    for ax, (c, title) in zip(axes, CORPORA):
        rep = policy[c][winner[c] + "|density=none"]
        cells = {
            "A": (
                rep["task_A_full_prediction"]["delta_rho"],
                rep["task_A_full_prediction"]["delta_rho_ci_author"],
            ),
            "B1": (
                rep["task_B1_fixed_profile_time_score"]["novice"]["delta_rho"],
                rep["task_B1_fixed_profile_time_score"]["novice"][
                    "delta_rho_ci_author"
                ],
            ),
            "B2": (
                rep["task_B2_percentile"]["delta_rho_views"],
                rep["task_B2_percentile"]["delta_rho_ci_author_views"],
            ),
        }
        for y, (_, key) in enumerate(rows):
            x, (lo, hi) = cells[key]
            ax.errorbar(
                x,
                y,
                xerr=[[x - lo], [hi - x]],
                fmt="o",
                color=COLORS[2],
                capsize=4,
                markersize=7,
                linewidth=2,
            )
        ax.axvline(0, color="grey", linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Δρ относительно правила выходных")
        ax.grid(axis="x", alpha=0.3)
        ax.xaxis.set_major_formatter(FuncFormatter(no_negative_zero))
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)  # A сверху
    fig.tight_layout()
    fig.savefig(FIGS / "11_time_modes.png")
    plt.close(fig)


def main():
    plt = setup_plot()
    winner = read("selected_policies.json")["winner_by_corpus"]
    fig_models(plt, read("metrics/loss_sweep_summary.json"), winner)
    fig_modes(plt, read("metrics/policy_report.json"), winner)
    print("рисунки 10 и 11 обновлены")


if __name__ == "__main__":
    main()
