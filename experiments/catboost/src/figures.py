"""Иллюстрации эксперимента. Вход - готовые metrics JSON и registry.csv,
никакого переобучения при отрисовке. Цифры в подписях не перебиваются руками.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

CORPUS_LABEL = {"ai": "ИИ/МЛ", "wide": "Пять хабов"}
LOSS_LABEL = {
    "L01_rmse_log1p": "RMSE(log1p)",
    "L02_quantile_views_1000": "Q0.5 views/1000",
    "L03_poisson_views_1000": "Poisson /1000",
    "L04_huber_log1p": "Huber(log1p)",
    "L05_quantile_log1p": "Q0.5 log1p",
    "lgb_poisson_v24": "LightGBM Poisson",
    "lgb_rmse_log1p": "LGB RMSE(log1p)",
}


def _setup():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    return plt


def _load(name: str) -> dict:
    path = config.METRICS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _save(fig, name: str) -> None:
    fig.savefig(config.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {name}")


def fig_loss_corpus(summary: dict) -> None:
    """1. Loss x corpus: rho_within_month и MAE_log всех обязательных целей."""
    plt = _setup()
    keys = [k for k in summary if summary[k].get("validation")]
    # Уникальные потери в порядке появления: ключей 2 x модели, модель одна.
    losses = list(dict.fromkeys(k.split(":", 1)[1] for k in keys))
    corpora = sorted({k.split(":", 1)[0] for k in keys})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    x = np.arange(len(losses))
    width = 0.8 / len(corpora)
    for j, corpus in enumerate(corpora):
        rho, mae = [], []
        for loss_id in losses:
            v = summary.get(f"{corpus}:{loss_id}", {}).get("validation") or {}
            rho.append(v.get("rho_within_month_mean") or np.nan)
            mae.append(v.get("mae_log") or np.nan)
        ax = axes[0]
        ax.bar(x + j * width, rho, width, label=CORPUS_LABEL[corpus])
        ax2 = axes[1]
        ax2.bar(x + j * width, mae, width, label=CORPUS_LABEL[corpus])
    axes[0].set(
        xticks=x + width / 2,
        xticklabels=[LOSS_LABEL.get(loss_id, loss_id) for loss_id in losses],
        ylabel="Спирмен внутри месяцев (validation)",
        title="Ранжирование по функциям потерь",
    )
    axes[0].tick_params(axis="x", rotation=40, labelsize=8)
    axes[1].set(
        xticks=x + width / 2,
        xticklabels=[LOSS_LABEL.get(loss_id, loss_id) for loss_id in losses],
        ylabel="MAE на log1p-шкале",
        title="Абсолютный уровень (меньше - лучше)",
    )
    axes[1].tick_params(axis="x", rotation=40, labelsize=8)
    for ax in axes:
        ax.legend(frameon=False)
    _save(fig, "01_loss_x_corpus.png")


def fig_rolling_folds(rolling: dict) -> None:
    """2. Rolling folds: одна линия на модель по периодам; базовые линии."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    models = sorted({k for k in rolling})
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, key in enumerate(models):
        rows = rolling[key]
        ax.plot(
            [r["cut"][:7] for r in rows],
            [r["rho_within_month"] for r in rows],
            marker="o",
            label=LOSS_LABEL.get(key.split(":", 1)[1], key),
            color=palette[i],
        )
    ax.set(
        ylabel="Спирмен внутри месяцев (rolling-фолд)",
        xlabel="граница фолда",
        title="Устойчивость по периодам: гиперпараметры не пересобираются",
    )
    ax.axhline(0, color="#999", lw=0.8)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "02_rolling_folds.png")


def fig_ablation() -> None:
    """3. Ablation: полная модель, без времени, без плотности, без заголовка/автора."""
    plt = _setup()
    reg = pd.read_csv(config.REGISTRY_CSV)
    reg = reg[(reg.stage == "loss") & (reg.corpus == "ai")]
    reg = reg[reg.status == config.STATUS_AUDIT_EXPOSED]
    reg = reg[reg.split_id.str.endswith("audit_exposed")]
    reg = reg.sort_values("rho_within_month", ascending=False)
    if reg.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    labels = [f"{r.model_family}:{r.feature_set}" for _, r in reg.iterrows()]
    ax.barh(labels[::-1], reg["rho_within_month"][::-1], color="#2d7ab3")
    ax.set(
        xlabel="Спирмен внутри месяцев (audit_exposed, ИИ/МЛ)",
        title="Абляция признаков: что держит качество",
    )
    ax.tick_params(axis="y", labelsize=8)
    _save(fig, "03_ablation.png")


def fig_calibration() -> None:
    """4. Калибровка: предсказанные и фактические медианы по децилям прогноза.

    Файлы предсказаний содержат loss_id: каждая панель - своя модель,
    а не перезаписанный последней моделью файл."""
    plt = _setup()
    preds = sorted(config.PREDICTIONS_DIR.glob("loss_*_audit_exposed_*.csv.gz"))
    if not preds:
        return
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, preds):
        df = pd.read_csv(p)
        df["dec"] = pd.qcut(df["pred_views"], 10, duplicates="drop")
        g = df.groupby("dec", observed=True).agg(
            pred=("pred_views", "median"), fact=("views", "median")
        )
        ax.plot(range(len(g)), g["fact"] / 1000, marker="o", label="факт, медиана")
        ax.plot(range(len(g)), g["pred"] / 1000, marker="s", label="прогноз, медиана")
        ax.set(xlabel="дециль прогноза", ylabel="просмотры, тыс.")
        parts = p.name[:-7].split("_")
        ax.set_title(f"{parts[1]} · {parts[-1]}", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    _save(fig, "04_calibration.png")


def fig_slot_map(policy: dict) -> None:
    """5. Slot map: 168 слотов, маска поддержки, конкретные даты policy."""
    plt = _setup()
    grid = None
    for corpus, models_ in policy.items():
        for key, res in models_.items():
            if "grid_D_months_june" in res:
                grid = res["grid_D_months_june"]
                title = f"{CORPUS_LABEL.get(corpus, corpus)} · {key}"
                break
        if grid:
            break
    if grid is None:
        return
    g = pd.DataFrame(grid["grid"])
    piv = g.pivot(index="dow", columns="hour", values="relative_prediction")
    sup = g.pivot(index="dow", columns="hour", values="supported")
    piv = piv.where(sup)
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    im = ax.imshow(
        np.ma.masked_invalid(piv),
        cmap="RdBu_r",
        aspect="auto",
        norm=TwoSlopeNorm(
            vmin=min(float(np.nanmin(piv)), 0.99),
            vcenter=1.0,
            vmax=max(float(np.nanmax(piv)), 1.01),
        ),
    )
    im.cmap.set_bad("#d9d9d9")
    ax.set_yticks(range(7), ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
    ax.set(
        xlabel="Час (МСК)",
        title=f"Относительный прогноз сетки · {title} · июнь · n train ≥ 20",
    )
    fig.colorbar(im, ax=ax, label="Отношение к медиане прогнозов")
    _save(fig, "05_slot_map.png")


def fig_equal_share(policy: dict) -> None:
    """6. Equal-share policy: CatBoost против weekend при одном q и покрытии."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    labels, vals = [], []
    for corpus, models_ in policy.items():
        for key, res in models_.items():
            es = res.get("equal_share") or {}
            wr = res.get("weekend_rule") or {}
            if es.get("top_q_ratio") is None:
                continue
            labels.append(f"{CORPUS_LABEL[corpus]}\n{key.split('|')[0]}")
            vals.append((es["top_q_ratio"], wr.get("weekend_q_ratio")))
    if not labels:
        plt.close(fig)
        return
    x = np.arange(len(labels))
    model_r = [v[0] for v in vals]
    rule_r = [v[1] for v in vals]
    ax.bar(x - 0.2, model_r, 0.4, label="модель (верхние q%)")
    ax.bar(x + 0.2, rule_r, 0.4, label="weekend-правило (тот же q)")
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="отношение медиан верх/остальные",
        title="Одинаковое покрытие: модель против простого правила",
    )
    ax.axhline(1.0, color="#999", lw=0.8)
    ax.legend(frameon=False)
    ax.tick_params(axis="x", labelsize=7)
    _save(fig, "06_equal_share_policy.png")


def fig_weekday_only(policy: dict) -> None:
    """7. Weekday-only: способность отличать время внутри будней."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    labels, vals = [], []
    for corpus, models_ in policy.items():
        for key, res in models_.items():
            wd = res.get("weekday_only") or {}
            if wd.get("weekday_only_hi_lo") is None:
                continue
            labels.append(f"{CORPUS_LABEL[corpus]}\n{key.split('|')[0]}")
            vals.append(wd["weekday_only_hi_lo"])
    if not labels:
        plt.close(fig)
        return
    ax.bar(range(len(labels)), vals, color="#2d7ab3")
    ax.axhline(1.0, color="#999", lw=0.8)
    ax.set(
        xticks=range(len(labels)),
        xticklabels=labels,
        ylabel="отношение медиан верх/низ терциля (только будни)",
        title="Различает ли модель часы внутри будней",
    )
    ax.tick_params(axis="x", labelsize=7)
    _save(fig, "07_weekday_only.png")


def fig_uncertainty(policy: dict) -> None:
    """8. Неопределённость: bootstrap delta_rho, интервалы по моделям и корпусам."""
    plt = _setup()
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ys, y = [], 0
    yticks, ylabels = [], []
    for corpus, models_ in policy.items():
        for key, res in models_.items():
            ci = res.get("delta_rho_ci_author")
            if not ci or ci[0] is None:
                continue
            ys.append((res.get("delta_rho"), ci[0], ci[1]))
            yticks.append(y)
            ylabels.append(f"{CORPUS_LABEL[corpus]} · {key.split('|')[0]}")
            y += 1
    for i, (point, lo, hi) in enumerate(ys):
        ax.errorbar(
            point,
            i,
            xerr=[[point - lo], [hi - point]],
            fmt="o",
            color="#2d7ab3",
            capsize=3,
        )
    ax.axvline(0, color="#999", lw=1)
    ax.set(
        yticks=yticks,
        yticklabels=ylabels,
        xlabel="delta_rho (модель - weekend), парный авторский бутстреп",
        title="Разность корреляций: где интервал не пересекает нуль",
    )
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, "08_uncertainty.png")


def fig_poisson_diagnostics(diag: dict) -> None:
    """9. Численная диагностика: Poisson/масштаб - loss, дисперсия, % finite."""
    plt = _setup()
    if not diag:
        return
    scales = list(diag.keys())
    rho = [diag[s].get("rho_val") for s in scales]
    var = [diag[s].get("pred_var") for s in scales]
    fin = [diag[s].get("frac_finite") for s in scales]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained")
    axes[0].bar(scales, rho, color="#2d7ab3")
    axes[0].set(title="Спирмен на validation", xlabel="масштаб цели Poisson")
    axes[1].bar(scales, var, color="#2d7ab3")
    axes[1].set(title="Дисперсия прогноза", xlabel="масштаб цели Poisson")
    axes[2].bar(scales, fin, color="#2d7ab3")
    axes[2].set(title="Доля finite", xlabel="масштаб цели Poisson", ylim=(0, 1.05))
    fig.suptitle("Численная устойчивость Poisson к масштабу", y=1.03)
    _save(fig, "09_poisson_diagnostics.png")


def render_all() -> None:
    summary = _load("loss_sweep_summary")
    if summary:
        fig_loss_corpus(summary)
    rolling = _load("rolling_folds")
    if rolling:
        fig_rolling_folds(rolling)
    if config.REGISTRY_CSV.exists():
        fig_ablation()
    fig_calibration()
    policy = _load("policy_report")
    if policy:
        fig_slot_map(policy)
        fig_equal_share(policy)
        fig_weekday_only(policy)
        fig_uncertainty(policy)
    fig_poisson_diagnostics(_load("poisson_scale_check"))
