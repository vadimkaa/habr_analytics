"""Калибровка, отчёты и фиксация контрольных артефактов."""

from __future__ import annotations

import json

import time

import numpy as np

import pandas as pd

from src import (
    config,
    metrics as M,
    registry,
)

from src import selection as SEL

from .common import (
    log,
    protocol_frames,
    train_log1p_median,
)


def stage_calibration() -> dict:
    """Калибровка абсолютного уровня финалистов.

    Для каждой модели: MAE/RMSE(log1p), ошибка относительно базового медианного
    прогноза, медианное отношение прогноз/факт и таблица децилей (данные для
    графика калибровки). Прогнозы берутся из финальных прогонов loss-stage
    (train+validation, деревья из заморозки)."""
    log("Калибровка финалистов: уровень прогноза против факта")
    sel = SEL.load_selection()
    out = {
        "models": {},
        "figure_note": (
            "Рисунок «прогнозные и фактические медианы по децилям прогноза» строится "
            "для всех loss-прогонов финальных моделей (loss_*_audit_exposed.csv.gz); "
            "предсказанные медианы в верхних децилях систематически выше фактических."
        ),
    }
    dec_frames = []
    for corpus in ("ai", "wide"):
        winner = sel["winner_by_corpus"][corpus]
        models_list = [winner, "L02_quantile_views_1000"]
        if winner == "L02_quantile_views_1000":
            models_list = [winner, "L01_rmse_log1p"]
        models_list.append("lgb_poisson_v24")
        frames = protocol_frames(corpus, "F1_meta")
        base = train_log1p_median(frames["train"])
        for loss_id in dict.fromkeys(models_list):
            p = (
                config.PREDICTIONS_DIR
                / f"loss_{corpus}_protocol_audit_exposed_{loss_id}.csv.gz"
            )
            if not p.exists():
                continue
            df = pd.read_csv(p)
            pred, fact = df["pred_views"].to_numpy(), df["views"].to_numpy()
            summ = M.level_metrics(pred, fact, base)
            ratio = pred[fact > 0] / fact[fact > 0]
            dec = pd.qcut(pd.Series(pred), 10, labels=False, duplicates="drop")
            g = (
                pd.DataFrame({"dec": dec, "pred": pred, "fact": fact})
                .groupby("dec")
                .agg(
                    pred_median=("pred", "median"),
                    fact_median=("fact", "median"),
                    n=("pred", "size"),
                )
                .reset_index()
            )
            g["pred_over_fact"] = g["pred_median"] / g["fact_median"]
            g["corpus"], g["model_id"] = corpus, f"{corpus}:{loss_id}"
            dec_frames.append(g)
            entry = dict(
                loss_id=loss_id,
                n=int(len(df)),
                train_log1p_median=base,
                baseline_mae_log=float(summ["baseline_mae_log"]),
                mae_log=summ["mae_log"],
                rmse_log=summ["rmse_log"],
                bias_log=summ["bias_log"],
                r2_log=summ["r2_log"],
                mae_vs_baseline_share=(
                    float(summ["mae_log"] / summ["baseline_mae_log"])
                ),
                median_pred_fact_ratio=float(np.median(ratio)),
                share_pred_gt_fact=float(np.mean(pred > fact)),
                deciles=g[
                    ["dec", "pred_median", "fact_median", "pred_over_fact"]
                ].to_dict("records"),
            )
            out["models"][f"{corpus}:{loss_id}"] = entry
            log(
                f"  {corpus}:{loss_id}: MAE_log={summ['mae_log']:.3f} "
                f"(MAE базовой медианы {summ['baseline_mae_log']:.3f}), медиана pred/fact={np.median(ratio):.3f}, "
                f"bias_log={summ['bias_log']:+.3f}"
            )
    registry.save_metrics("calibration", out)
    if dec_frames:
        dec_all = pd.concat(dec_frames, ignore_index=True)
        dec_all.to_csv(config.METRICS_DIR / "calibration_deciles.csv", index=False)
        fig_calibration_final(dec_all)
    return out


def fig_calibration_final(dec_all: pd.DataFrame) -> None:
    """Децильная калибровка для замороженных победителей двух корпусов."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    for fp in (
        "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans SC"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150

    corpus_name = {"ai": "ИИ/МЛ", "wide": "Пять хабов"}
    sel = json.loads(config.SELECTED_POLICIES_JSON.read_text())
    winners = {f"{c}:{w}" for c, w in sel["winner_by_corpus"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    for ax, corpus in zip(axes, ("ai", "wide")):
        w_model = sel["winner_by_corpus"].get(corpus)
        # Только победитель корпуса: у каждой модели свой прогноз, смешивать нельзя
        sub = dec_all[
            (dec_all.corpus == corpus) & (dec_all.model_id == f"{corpus}:{w_model}")
        ]
        ax.plot(
            sub.dec + 0.5, sub.fact_median / 1000, marker="o", label="факт, медиана"
        )
        ax.plot(
            sub.dec + 0.5, sub.pred_median / 1000, marker="s", label="прогноз, медиана"
        )
        ax.set_title(
            f"{corpus_name.get(corpus, corpus)} · победитель {w_model}", fontsize=9
        )
        ax.set(xlabel="дециль прогноза", ylabel="просмотры, тыс.")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(config.FIGURES_DIR / "10_calibration_final.png")
    plt.close(fig)
    print("  figure: 10_calibration_final.png")


def stage_report() -> None:
    from src.figures import render_all
    from src.report import build_results_md

    render_all()
    build_results_md()
    log("Этап 7: рисунки и results.md готовы")


def stage_freeze() -> None:
    """Сохранить показатели контрольной LightGBM-модели и хеши входов.

    Контрольные результаты читаются из data/results/ без изменения файлов
    основного описательного анализа."""
    log("Этап 0: фиксация контрольных результатов и входных данных")
    src = config.REPO / "data" / "results"
    frozen = {}
    for name in (
        "08_model_results.json",
        "slot_rule_comparison.json",
        "11_leakage_diagnostic.json",
        "run_environment.json",
    ):
        p = src / name
        if p.exists():
            frozen[name] = json.loads(p.read_text())
        else:
            frozen[name] = None
            log(f"  предупреждение: {name} не найден")
    # Контрольные показатели служат для сверки, а не подбора параметров.
    reference = {}
    for corpus, key in (("ai", "ai"), ("wide", "wide")):
        block = (
            (frozen.get("08_model_results.json") or {}).get("models", {}).get(key, {})
        )
        full = block.get("full", {})
        reference[corpus] = dict(
            v24_split=block.get("split_msk"),
            rho=full.get("rho"),
            rho_within_month=full.get("rho_within_month_mean"),
            mae_log=full.get("mae_log"),
            baseline_mae_log=full.get("baseline_mae_log"),
        )
    slot = frozen.get("slot_rule_comparison.json") or {}
    slot_ref = {
        c: dict(
            model_ratio=(slot.get(c) or {}).get("model_ratio"),
            naive_ratio=(slot.get(c) or {}).get("naive_ratio"),
            rho_grid=(slot.get(c, {}).get("rank_corr", {}) or {})
            .get("grid_profile", {})
            .get("rho"),
            rho_weekend=(slot.get(c, {}).get("rank_corr", {}) or {})
            .get("weekend_flag", {})
            .get("rho"),
            delta_rho=(slot.get(c, {}).get("rank_corr", {}) or {})
            .get("difference", {})
            .get("rho"),
        )
        for c in ("ai", "wide")
    }
    manifest = dict(
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        protocol=config.protocol()["protocol"],
        input_sha256=config.input_hashes(),
        frozen_artifacts=list(frozen.keys()),
        v24_reference_models=reference,
        v24_reference_slot_rules=slot_ref,
        lightgbm_version="4.5.0",
        note=(
            "Опорные значения скопированы из замороженных артефактов исходной модели; "
            "этап 2 сверяет с ними воспроизведение, но не подгоняет параметры."
        ),
    )
    registry.save_metrics("baseline_manifest", manifest)
    log("baseline_manifest.json готов")


