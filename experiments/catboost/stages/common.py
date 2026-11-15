"""Общие помощники стадий: разбиения, метрики и реестр запусков."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src import (
    config,
    features as feats,
    metrics as M,
    models,
    registry,
    splits,
)

RUN_COUNTER = {"n": 0}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def next_run_id(prefix: str) -> str:
    RUN_COUNTER["n"] += 1
    return f"{prefix}{RUN_COUNTER['n']:04d}"


def protocol_frames(corpus: str, feature_set: str) -> dict[str, pd.DataFrame]:
    """Фонды протокола с плотностью expected из train (если она в наборе)."""
    df = splits.materialize_corpus(corpus)
    frames = splits.split_frame(df, corpus)
    frames = {("train" if k == "dev_train" else k): v for k, v in frames.items()}
    need_density = feature_set in ("F3_dens_expected",)
    if need_density:
        frames = splits.fit_expected_density_column(frames["train"], frames)
    if feature_set == "F5_hubs":
        for name in frames:
            frames[name] = feats.add_hub_flags(frames[name], frames["train"])
    return frames


def features_for(feature_set: str, df: pd.DataFrame | None = None) -> list[str]:
    if feature_set == "F5_hubs":
        base = list(feats.FEATURE_SETS["F5_hubs"]["features"])
        cols = (df.attrs.get("hub_columns") if df is not None else None) or []
        return base + list(cols)
    return list(feats.FEATURE_SETS[feature_set]["features"])


def train_log1p_median(train: pd.DataFrame) -> float:
    return float(np.median(np.log1p(train["views"].to_numpy())))


def registry_row_for(
    stage: str,
    corpus: str,
    split_id: str,
    fit_meta: dict,
    n_train: int,
    n_val: int,
    n_test: int,
    status: str,
    extra: dict | None = None,
) -> dict:
    row = dict(
        stage=stage,
        corpus=corpus,
        split_id=split_id,
        train_start=split_id.split("__")[1] if "__" in split_id else "",
        loss_function=fit_meta.get("loss_id"),
        target_transform=fit_meta.get("target_kind"),
        model_family=f"{fit_meta.get('library')}:{fit_meta.get('config_id')}",
        cat_mode=fit_meta.get("cat_mode", "numeric"),
        feature_set=fit_meta.get("feature_set"),
        density_mode=fit_meta.get("density_mode", "none"),
        seed=fit_meta.get("seed", config.SEED),
        params_json=json.dumps(fit_meta.get("params", {}), default=str)[:2000],
        fit_time_sec=fit_meta.get("fit_time_sec"),
        best_iteration=fit_meta.get("best_iteration"),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        status=status,
    )
    if extra:
        row.update(extra)
    return row


def score_phase(
    stage: str,
    corpus: str,
    split_id: str,
    fit: models.FitResult,
    frames: dict[str, pd.DataFrame],
    features: list[str],
    phases: list[str],
    statuses: dict[str, str],
) -> dict:
    """Метрики по фолдам + строки реестра + сохранение предсказаний."""
    med_train = train_log1p_median(frames["train"])
    out = {}
    for phase in phases:
        frame = frames.get(phase)
        if frame is None or len(frame) == 0:
            continue
        pred = fit.pred_views.get(phase)
        if pred is None:
            pred = fit.target.decode(fit.model.predict(frame[features]))
        summ = M.summarize_A(frame, pred, med_train)
        out[phase] = summ
        row = registry_row_for(
            stage,
            corpus,
            f"{corpus}__{split_id}__{phase}",
            fit.metadata,
            len(frames["train"]),
            len(frames.get("validation", [])),
            len(frame),
            statuses.get(phase, config.STATUS_EXPLORATORY),
            extra=dict(
                rho_all=summ["rho_all"],
                rho_within_month=summ["rho_within_month_mean"],
                mae_log=summ["mae_log"],
                bias_log=summ["bias_log"],
                r2_log=summ["r2_log"],
                run_id=next_run_id(stage[:3]),
            ),
        )
        registry.append_registry_row(row)
        _save_predictions(stage, corpus, split_id, phase, frame, pred, fit)
        log(
            f"  {corpus}/{split_id}/{phase}: rho_m={summ['rho_within_month_mean']}, "
            f"mae_log={summ['mae_log']:.3f} (base {summ['baseline_mae_log']:.3f})"
        )
    return out


def _save_predictions(
    stage: str,
    corpus: str,
    split_id: str,
    phase: str,
    frame: pd.DataFrame,
    pred: np.ndarray,
    fit: models.FitResult,
) -> None:
    out = pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "views": frame["views"].to_numpy(),
            "pred_views": pred,
            "score": np.asarray(fit.score.get(phase, pred), float)[: len(frame)],
            "month_id": frame["month_id"].to_numpy(),
            "author_key": frame["author_key"].to_numpy(),
            "is_weekend": frame["is_weekend"].to_numpy(),
            "is_corp": frame["is_corp"].to_numpy(),
            "age_days": frame["age_days"].to_numpy(),
        }
    )
    # В имени учтена функция потерь, чтобы результаты моделей не пересекались.
    loss_id = fit.metadata.get("loss_id") or "model"
    name = f"{stage}_{corpus}_{split_id}_{phase}_{loss_id}"
    out.to_csv(
        config.PREDICTIONS_DIR / f"{name}.csv.gz", index=False, compression="gzip"
    )


def _fold_bounds(corpus: str, frames: dict) -> dict:
    def b(name):
        f = frames.get(name)
        if f is None or len(f) == 0:
            return ("", "")
        return (str(f.dt_msk.min()), str(f.dt_msk.max()))

    out = {}
    for name in frames:
        s, e = b(name)
        out[f"{name}_start"] = s
        out[f"{name}_end"] = e
    return out


