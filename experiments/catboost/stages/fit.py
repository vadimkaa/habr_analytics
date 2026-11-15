"""Валидация функций потерь, контрольных моделей и выбор конфигураций."""

from __future__ import annotations

import json

import numpy as np

import pandas as pd

from src import (
    baselines,
    config,
    features as feats,
    metrics as M,
    models,
    registry,
)

from src.targets import TargetSpec, VIEWS_PER_UNIT, VIEWS_PER_UNIT_ALT

from src import selection as SEL

from .common import (
    log,
    next_run_id,
    protocol_frames,
    features_for,
    train_log1p_median,
    score_phase,
)


def stage_control(corpora: list[str]) -> None:
    log("Этап 2: контрольные модели и базовые ориентиры")
    for corpus in corpora:
        frames = protocol_frames(corpus, "F1_meta")
        train, val = frames["train"], frames["validation"]
        phases = (
            ["validation"]
            + (["extra_check"] if "extra_check" in frames else [])
            + ["audit_exposed"]
        )
        statuses = {
            "validation": config.STATUS_VALIDATION,
            "extra_check": config.STATUS_VALIDATION,
            "audit_exposed": config.STATUS_AUDIT_EXPOSED,
        }
        metric_frames = {p: frames[p] for p in phases if p in frames}

        # B0-B3: простые правила без обучения на тесте
        base = baselines.run_baselines(train, metric_frames)
        med_train = train_log1p_median(train)
        for bname, b in base.items():
            if "error" in b:
                log(f"  {corpus} {bname}: ошибка базлайна {b['error']}")
                continue
            for phase, frame in metric_frames.items():
                summ = M.summarize_A(frame, b["pred_views"][phase], med_train)
                registry.append_registry_row(
                    dict(
                        stage="control",
                        corpus=corpus,
                        split_id=f"{corpus}__protocol__{phase}",
                        loss_function="none",
                        target_transform="log1p",
                        model_family=f"baseline:{b['name']}",
                        feature_set="none",
                        n_train=len(train),
                        n_test=len(frame),
                        status=statuses[phase],
                        rho_all=summ["rho_all"],
                        rho_within_month=summ["rho_within_month_mean"],
                        mae_log=summ["mae_log"],
                        bias_log=summ["bias_log"],
                        run_id=next_run_id("ctl"),
                    )
                )
                log(
                    f"  {corpus} {b['name']} {phase}: rho_m={summ['rho_within_month_mean']}, mae_log={summ['mae_log']:.3f}"
                )

        # Контроль LightGBM на зафиксированных временных границах.
        # Правило отбора воспроизводится 1-в-1 из 08_model.main (WIDE_COMMON_START,
        # без сдвига на день: расхождение окон - источник расхождения чисел).
        from src.data_contract import load_corpus as _load_raw
        from habrpub.analysis import WIDE_COMMON_START as _WCS

        for lgb_loss, fset in (
            ("lgb_poisson_v24", "V24_full"),
            ("lgb_rmse_log1p", "V24_full"),
        ):
            control_split = "2026-01-01" if corpus == "ai" else "2026-05-01"
            raw = _load_raw(corpus)
            df = feats.prepare(raw)
            eligible = df.mature & df.density_complete
            eligible &= (df.year >= 2016) if corpus == "ai" else (df.dt_msk >= _WCS)
            df = df[eligible].copy()
            tr = df[df.dt_msk < pd.Timestamp(control_split, tz="Europe/Moscow")]
            te = df[df.dt_msk >= pd.Timestamp(control_split, tz="Europe/Moscow")]
            control_features = features_for("V24_full")
            fit = models.fit_lightgbm(tr, None, control_features, [], lgb_loss)
            summ = M.summarize_A(
                te,
                fit.target.decode(fit.model.predict(te[control_features])),
                train_log1p_median(tr),
            )
            registry.append_registry_row(
                dict(
                    stage="control",
                    corpus=corpus,
                    split_id=f"{corpus}__v24split__audit",
                    loss_function=lgb_loss,
                    target_transform=fit.metadata["target_kind"],
                    model_family="lightgbm:v24",
                    feature_set=fset,
                    density_mode="actual",
                    n_train=len(tr),
                    n_test=len(te),
                    status=config.STATUS_AUDIT_EXPOSED,
                    rho_all=summ["rho_all"],
                    rho_within_month=summ["rho_within_month_mean"],
                    mae_log=summ["mae_log"],
                    bias_log=summ["bias_log"],
                    r2_log=summ["r2_log"],
                    run_id=next_run_id("ctl"),
                )
            )
            log(
                f"  {corpus} {lgb_loss} (контрольное разбиение): rho={summ['rho_all']}, rho_m={summ['rho_within_month_mean']}, mae_log={summ['mae_log']:.3f}"
            )

        # C0: дословное приближение LightGBM на CatBoost (Poisson /1000 - 8.3)
        feats_f1 = features_for("F1_meta")
        fit = models.fit_catboost(
            train,
            val,
            feats_f1,
            [],
            "L03_poisson_views_1000",
            config_id="C0_lgb_analog",
        )
        fit.metadata.update(
            feature_set="F1_meta", cat_mode="numeric", density_mode="none"
        )
        fit.metadata["deviation_note"] = (
            "C0: цель views/1000 вместо сырых просмотров (численная устойчивость, 8.3)"
        )
        score_phase(
            "control", corpus, "protocol", fit, frames, feats_f1, phases, statuses
        )


def poisson_scale_check() -> None:
    """Чувствительность масштаба Poisson: /1000 против /10000."""
    log("Poisson: проверка масштаба на ИИ/МЛ")
    corpus = "ai"
    frames = protocol_frames(corpus, "F1_meta")
    feats_f1 = features_for("F1_meta")
    res = {}
    for scale_name, scale in (("1000", VIEWS_PER_UNIT), ("10000", VIEWS_PER_UNIT_ALT)):
        tr = frames["train"].copy()
        va = frames["validation"].copy()
        target = TargetSpec("log1p")
        from catboost import CatBoostRegressor, Pool

        params = dict(models.CATBOOST_CONFIGS["C1_symmetric"])
        params.update(
            loss_function="Poisson",
            eval_metric="Poisson",
            iterations=min(600, params["iterations"]),
        )
        y = tr["views"].to_numpy() / scale
        yv = va["views"].to_numpy() / scale
        m = CatBoostRegressor(**params)
        m.fit(
            Pool(tr[feats_f1], label=y),
            eval_set=Pool(va[feats_f1], label=yv),
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            use_best_model=True,
            verbose=False,
        )
        raw = m.predict(va[feats_f1])
        pv = raw * scale
        health = M.prediction_health(pv)
        from scipy import stats as st

        rho = float(st.spearmanr(pv, va["views"].to_numpy()).statistic)
        loss_val = float(
            np.mean(
                (np.log1p(np.maximum(pv, 0)) - np.log1p(va["views"].to_numpy())) ** 2
            )
        )
        bi = m.get_best_iteration()
        # CatBoost возвращает индекс с нуля; для отчёта нужны оба значения.
        res[scale_name] = dict(
            rho_val=rho,
            rmse_log_val=loss_val**0.5,
            best_iteration_index=int(bi) if bi is not None else None,
            best_iteration_trees=(int(bi) + 1)
            if bi is not None
            else int(params["iterations"]),
            **health,
        )
        log(
            f"  /{scale_name}: rho={rho:.3f}, rmse_log={res[scale_name]['rmse_log_val']:.3f}, "
            f"finite={health['frac_finite']}"
        )
    registry.save_metrics("poisson_scale_check", res)


def stage_loss(corpora: list[str], losses: list[str] | None = None) -> dict:
    log("Этап 3: сравнение функций потерь на неизменном F1 (C1_symmetric)")
    losses = losses or [
        "L01_rmse_log1p",
        "L02_quantile_views_1000",
        "L03_poisson_views_1000",
    ]
    # Дополнительные функции потерь дополняют общую сводку: стадия
    # должна сохранить первую.
    try:
        summary = registry.load_metrics("loss_sweep_summary")
    except Exception:
        summary = {}
    for corpus in corpora:
        frames = protocol_frames(corpus, "F1_meta")
        feats_f1 = features_for("F1_meta")
        phases = (
            ["validation"]
            + (["extra_check"] if "extra_check" in frames else [])
            + ["audit_exposed"]
        )
        statuses = {
            "validation": config.STATUS_VALIDATION,
            "extra_check": config.STATUS_VALIDATION,
            "audit_exposed": config.STATUS_AUDIT_EXPOSED,
        }
        for loss_id in losses:
            fit = models.fit_catboost(
                frames["train"],
                frames["validation"],
                feats_f1,
                [],
                loss_id,
                config_id="C1_symmetric",
            )
            fit.metadata.update(
                feature_set="F1_meta", cat_mode="numeric", density_mode="none"
            )
            best_iter = fit.best_iteration
            log(f"  {corpus} {loss_id}: best_iter={best_iter}, {fit.fit_time_sec}s")
            res = score_phase(
                "loss", corpus, "protocol", fit, frames, feats_f1, phases, statuses
            )
            # Финальный кандидат: train+validation, фиксированное число деревьев (9.3)
            final = models.refit_final(
                frames["train"],
                frames["validation"],
                feats_f1,
                [],
                loss_id,
                "C1_symmetric",
                best_iter,
                "catboost",
            )
            final.metadata.update(
                feature_set="F1_meta", cat_mode="numeric", density_mode="none"
            )
            # Дамб финального кандидата
            try:
                final.model.save_model(
                    config.MODELS_DIR / f"{corpus}_{loss_id}_C1_final.cbm", format="cbm"
                )
            except Exception as exc:
                log(f"  предупреждение: дамба {loss_id} не сохранена: {exc}")
            log(
                f"  {corpus} {loss_id}: финальный рефит (train+val, {best_iter} деревьев)"
            )
            res_final = score_phase(
                "loss",
                corpus,
                "protocol",
                final,
                frames,
                feats_f1,
                ["extra_check", "audit_exposed"],
                {
                    "extra_check": config.STATUS_AUDIT_EXPOSED,
                    "audit_exposed": config.STATUS_AUDIT_EXPOSED,
                },
            )
            # Квантильные диагностики на аудите
            if "Quantile" in fit.metadata["objective"]:
                te = frames["audit_exposed"]
                raw = final.model.predict(te[feats_f1])
                scale = 1000.0 if final.target.kind == "divide_1000" else 1.0
                qd = M.pinball_and_coverage(
                    raw, te["views"].to_numpy() / scale, 0.5, scale
                )
                res_final.setdefault("audit_exposed", {}).update(quantile_diag=qd)
            summary[f"{corpus}:{loss_id}"] = dict(
                validation=res.get("validation"),
                audit=res.get("audit_exposed"),
                final_audit=res_final.get("audit_exposed"),
                extra_check=res_final.get("extra_check"),
            )
            registry.save_metrics(
                f"loss_{corpus}_{loss_id}",
                dict(
                    selection=res,
                    final=res_final,
                    metadata=dict(
                        fit.metadata, params={k: str(v) for k, v in fit.params.items()}
                    ),
                ),
            )
        # LightGBM-контроль на тех же фолдах и признаках (A2)
        for lgb_loss in ("lgb_poisson_v24", "lgb_rmse_log1p"):
            fit = models.fit_lightgbm(
                frames["train"], frames["validation"], feats_f1, [], lgb_loss
            )
            fit.metadata.update(
                feature_set="F1_meta", cat_mode="numeric", density_mode="none"
            )
            log(f"  {corpus} {lgb_loss} (F1, protocol splits)")
            res = score_phase(
                "loss", corpus, "protocol", fit, frames, feats_f1, phases, statuses
            )
            summary[f"{corpus}:{lgb_loss}"] = dict(
                validation=res.get("validation"), audit=res.get("audit_exposed")
            )
    registry.save_metrics("loss_sweep_summary", summary)
    return summary


def stage_tune() -> dict:
    log("Этап 4: конфигурации и кодирование календаря (выбор только на validation)")
    corpus = "ai"
    frames = protocol_frames(corpus, "F1_meta")
    feats_f1 = features_for("F1_meta")
    results = {}
    variants = [
        ("C0_lgb_analog", "L01_rmse_log1p", "numeric", False, feats_f1, "C0 x L01"),
        (
            "C0_lgb_analog",
            "L02_quantile_views_1000",
            "numeric",
            False,
            feats_f1,
            "C0 x L02",
        ),
        ("C2_conservative", "L01_rmse_log1p", "numeric", False, feats_f1, "C2 x L01"),
        (
            "C2_conservative",
            "L02_quantile_views_1000",
            "numeric",
            False,
            feats_f1,
            "C2 x L02",
        ),
        (
            "C1_symmetric",
            "L01_rmse_log1p",
            "categorical",
            False,
            feats_f1,
            "cat dow/hour/month",
        ),
        (
            "C1_symmetric",
            "L01_rmse_log1p",
            "one_hot",
            False,
            feats_f1,
            "one-hot dow/hour (max 24)",
        ),
        ("C1_symmetric", "L01_rmse_log1p", "numeric", True, feats_f1, "has_time=True"),
        (
            "C1_symmetric",
            "L01_rmse_log1p",
            "cyclic",
            False,
            feats.cyclic_only_rename(feats_f1),
            "cyclic sin/cos без сырых dow/hour",
        ),
    ]
    for config_id, loss_id, cat_mode, has_time, feats_used, label in variants:
        cat_cols = feats.cat_features_for(feats_used, cat_mode)
        if cat_mode == "cyclic":
            feats_used = [f for f in feats_used if f not in ("dow", "hour")]
        extra = dict(one_hot_max_size=24) if cat_mode == "one_hot" else None
        fit = models.fit_catboost(
            frames["train"],
            frames["validation"],
            feats_used,
            cat_cols,
            loss_id,
            config_id=config_id,
            has_time=has_time,
            extra_params=extra,
        )
        fit.metadata.update(
            feature_set="F1_meta",
            cat_mode=cat_mode,
            density_mode="none",
            has_time=has_time,
        )
        raw = fit.model.predict(frames["validation"][feats_used])
        pv = fit.target.decode(raw)
        summ = M.summarize_A(
            frames["validation"], pv, train_log1p_median(frames["train"])
        )
        results[label] = dict(
            rho_within_month=summ["rho_within_month_mean"],
            mae_log=summ["mae_log"],
            rho_all=summ["rho_all"],
            best_iteration=fit.best_iteration,
            fit_time_sec=fit.fit_time_sec,
            n_features=len(feats_used),
        )
        registry.append_registry_row(
            dict(
                stage="tune",
                corpus=corpus,
                split_id=f"{corpus}__protocol__validation",
                loss_function=loss_id,
                target_transform=fit.metadata["target_kind"],
                model_family=f"catboost:{config_id}",
                cat_mode=cat_mode,
                feature_set="F1_meta",
                n_train=len(frames["train"]),
                n_val=len(frames["validation"]),
                status=config.STATUS_VALIDATION,
                rho_all=summ["rho_all"],
                rho_within_month=summ["rho_within_month_mean"],
                mae_log=summ["mae_log"],
                best_iteration=fit.best_iteration,
                params_json=json.dumps({k: str(v) for k, v in fit.params.items()}),
                fit_time_sec=fit.fit_time_sec,
                run_id=next_run_id("tun"),
            )
        )
        log(
            f"  {label}: rho_m={summ['rho_within_month_mean']}, mae_log={summ['mae_log']:.3f}"
        )
    registry.save_metrics("tuning_report", results)
    return results


def stage_select(corpora: list[str]) -> dict:
    """Финалисты и победитель фиксируются ТОЛЬКО по validation и
    записываются в selected_policies.json ДО прогона аудита."""
    doc = SEL.write_selected_policies(corpora)
    log("Этап 4.5: выбор зафиксирован в selected_policies.json (без аудита)")
    for corpus in corpora:
        fin = doc["finalists_by_corpus"].get(corpus, [])
        win = doc["winner_by_corpus"].get(corpus)
        log(
            f"  {corpus}: финалисты {fin}, победитель {win} "
            f"(деревьев {doc['frozen_model_configs'].get(f'{corpus}:{win}', {}).get('n_trees_fixed_from_validation')})"
        )
    return doc


