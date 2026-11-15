"""Проверки устойчивости результатов на периодах и подвыборках."""

from __future__ import annotations

import numpy as np

import pandas as pd

from src import (
    config,
    features as feats,
    metrics as M,
    models,
    registry,
    slot_policy as SP,
    splits,
)

from src import selection as SEL

from .common import (
    log,
    next_run_id,
    protocol_frames,
    features_for,
    train_log1p_median,
)


def stage_robust(corpora: list[str], finalist_loss: str | None = None) -> dict:
    """Устойчивость для ЗАМОРОЗКИ из selected_policies.json (победитель
    корпуса по validation). Явный finalist_loss выбирает конкретного кандидата."""
    log("Этап 6: устойчивость и негативные контроли")
    sel = SEL.load_selection()
    out = {}
    for corpus in corpora:
        loss_id = (
            finalist_loss or sel["winner_by_corpus"].get(corpus) or "L01_rmse_log1p"
        )
        frames = protocol_frames(corpus, "F1_meta")
        feats_f1 = features_for("F1_meta")
        res = dict(finalist_loss=loss_id)
        # A11: 3-5 фиксированных seeds для финалиста (audit, без валидационного
        # пересбора: число деревьев из основного прогона)
        base = models.fit_catboost(
            frames["train"], frames["validation"], feats_f1, [], loss_id, "C1_symmetric"
        )
        seed_res = {}
        for seed in (config.SEED, 20261112, 20261113):
            final = models.refit_final(
                frames["train"],
                frames["validation"],
                feats_f1,
                [],
                loss_id,
                "C1_symmetric",
                base.best_iteration,
                "catboost",
                seed=seed,
            )
            te = frames["audit_exposed"]
            pv = final.target.decode(final.model.predict(te[feats_f1]))
            summ = M.summarize_A(te, pv, train_log1p_median(frames["train"]))
            seed_res[str(seed)] = dict(
                rho_within_month=summ["rho_within_month_mean"],
                rho_all=summ["rho_all"],
                mae_log=summ["mae_log"],
            )
        res["seeds"] = seed_res
        log(
            f"  {corpus} seeds: { {k: round(v['rho_within_month'], 4) for k, v in seed_res.items()} }"
        )

        # A12: холодный старт авторов, отсутствующих в train
        final = models.refit_final(
            frames["train"],
            frames["validation"],
            feats_f1,
            [],
            loss_id,
            "C1_symmetric",
            base.best_iteration,
            "catboost",
        )
        te = frames["audit_exposed"].copy()
        pv = final.target.decode(final.model.predict(te[feats_f1]))
        te["pred"] = pv
        train_authors = set(frames["train"]["author_key"])
        cold = te[~te.author_key.isin(train_authors)]
        warm = te[te.author_key.isin(train_authors)]
        res["cold_start"] = dict(
            n_cold=int(len(cold)),
            share_cold=round(len(cold) / len(te), 3),
            rho_within_month_cold=M.rho_within_month(cold.assign(score=cold["pred"]))[
                "rho_within_month_mean"
            ],
            rho_within_month_warm=M.rho_within_month(warm.assign(score=warm["pred"]))[
                "rho_within_month_mean"
            ],
        )
        # A10: хвост - чувствительность к крупнейшим статьям (только анализ)
        thr = float(np.quantile(te["views"], 0.99))
        tail_cut = te[te.views <= thr]
        res["tail_1pct_cut"] = dict(
            threshold_views=thr,
            rho_all_full=M.rho_all(te["pred"], te["views"]),
            rho_all_cut=M.rho_all(tail_cut["pred"], tail_cut["views"]),
            mae_log_cut=float(
                np.mean(
                    np.abs(
                        np.log1p(np.maximum(tail_cut["pred"], 0))
                        - np.log1p(tail_cut["views"])
                    )
                )
            )
            if len(tail_cut)
            else None,
        )
        # A9: эпохи - метрики по когортам аудита (полугодия)
        eras = {}
        for name, sub in te.groupby(
            te.dt_msk.dt.year.astype(str)
            + "H"
            + ((te.dt_msk.dt.month > 6).astype(int) + 1).astype(str)
        ):
            if len(sub) >= 100:
                eras[name] = dict(
                    n=int(len(sub)),
                    rho_within_month=M.rho_within_month(sub.assign(score=sub["pred"]))[
                        "rho_within_month_mean"
                    ],
                )
        res["eras"] = eras
        out[corpus] = res
    registry.save_metrics("robustness", out)
    return out


def robustness_nonintersect() -> dict:
    """A13: непересекающиеся части корпусов (доп. проверка, не независимая реплика)."""
    log("A13: широкий корпус без пересечения с ИИ/МЛ")
    from src.data_contract import corpus_overlap

    inter = set()
    ov = corpus_overlap()
    ai_ids = set(splits.materialize_corpus("ai")["id"])
    wide = splits.materialize_corpus("wide")
    wide_only = wide[~wide["id"].isin(ai_ids)]
    frames = protocol_frames("wide", "F1_meta")
    te = frames["audit_exposed"]
    te_only = te[te["id"].isin(set(wide_only["id"]))]
    feats_f1 = features_for("F1_meta")
    fit = models.fit_catboost(
        frames["train"],
        frames["validation"],
        feats_f1,
        [],
        "L01_rmse_log1p",
        "C1_symmetric",
    )
    final = models.refit_final(
        frames["train"],
        frames["validation"],
        feats_f1,
        [],
        "L01_rmse_log1p",
        "C1_symmetric",
        fit.best_iteration,
        "catboost",
    )
    pv = final.target.decode(final.model.predict(te[feats_f1]))
    full = M.rho_within_month(te.assign(score=pv))
    sub = te_only.assign(
        score=pv[np.isin(te["id"].to_numpy(), te_only["id"].to_numpy())]
    )
    cut = M.rho_within_month(sub)
    res = dict(
        n_intersect_total=ov["n_intersect"],
        n_wide_audit=int(len(te)),
        n_wide_audit_nonintersect=int(len(sub)),
        rho_within_month_full=full["rho_within_month_mean"],
        rho_within_month_nonintersect=cut["rho_within_month_mean"],
    )
    registry.save_metrics("robustness_nonintersect", res)
    log(
        f"  wide audit: full={res['rho_within_month_full']}, non-intersect={res['rho_within_month_nonintersect']}"
    )
    return res


def rolling_folds_for(corpora: list[str], losses: list[str] | None = None) -> dict:
    """Rolling-origin фолды для победителей (гиперпараметры не пересобираются)."""
    if losses is None:
        sel = SEL.load_selection()
        losses = sorted(set(sel["winner_by_corpus"].values()))
    log(f"Rolling-origin фолды для: {', '.join(losses)}")
    out = {}
    for corpus in corpora:
        df = splits.materialize_corpus(corpus)
        folds = splits.rolling_splits(df, corpus)
        for loss_id in losses:
            rows = []
            for f in folds:
                tr, te = f["train"], f["test"]
                fit = models.fit_catboost(
                    tr,
                    None,
                    features_for("F1_meta"),
                    [],
                    loss_id,
                    "C1_symmetric",
                    iterations_fixed=min(600, config.ITERATIONS_CAP),
                )
                pv = fit.target.decode(fit.model.predict(te[features_for("F1_meta")]))
                summ = M.summarize_A(te, pv, train_log1p_median(tr))
                registry.append_registry_row(
                    dict(
                        stage="rolling",
                        corpus=corpus,
                        split_id=f"{corpus}__rolling__{f['cut']}__{loss_id}",
                        loss_function=loss_id,
                        target_transform=fit.metadata["target_kind"],
                        model_family="catboost:C1_symmetric",
                        feature_set="F1_meta",
                        n_train=len(tr),
                        n_test=len(te),
                        status=config.STATUS_VALIDATION,
                        rho_all=summ["rho_all"],
                        rho_within_month=summ["rho_within_month_mean"],
                        mae_log=summ["mae_log"],
                        run_id=next_run_id("rol"),
                    )
                )
                rows.append(
                    dict(
                        cut=f["cut"],
                        n_train=len(tr),
                        n_test=len(te),
                        rho_within_month=summ["rho_within_month_mean"],
                        rho_all=summ["rho_all"],
                        mae_log=summ["mae_log"],
                    )
                )
                log(
                    f"  {corpus}/{loss_id} cut={f['cut']}: rho_m={summ['rho_within_month_mean']}, n_train={len(tr)}"
                )
            out[f"{corpus}:{loss_id}"] = rows
    registry.save_metrics("rolling_folds", out)
    return out


def _te_with_support(te: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    slot_n = train.groupby(["dow", "hour"]).size()
    te = te.copy()
    te["slot_n_train"] = [
        int(slot_n.get((int(d), int(h)), 0)) for d, h in zip(te.dow, te.hour)
    ]
    return te


def _b1_light_metrics(
    final: models.FitResult,
    train: pd.DataFrame,
    te: pd.DataFrame,
    feats: list[str],
    profile: dict,
    n_boot: int | None = None,
) -> dict:
    """Компактные B1-метрики для проверок устойчивости: границы только из train."""
    tr_b1 = SP.b1_fixed_profile_scores(final, train, feats, profile)
    te_b1 = SP.b1_fixed_profile_scores(final, te, feats, profile)
    supp = SP.personal_supported(te_b1)
    th = SP.train_thresholds(tr_b1)
    r = M.rule_metrics(supp, "model_score", n_boot=n_boot)
    r["age_controlled"] = M.b_metrics_age(supp, "model_score", n_boot=n_boot)
    r["equal_share"] = M.equal_share_ratio(
        supp, "model_score", th["q_fixed"], n_boot=n_boot
    )
    r["weekday_only"] = M.weekday_only_metrics(
        supp, "model_score", np.array(th["terciles"])
    )
    r["thresholds_train"] = {
        "threshold_q": th["threshold_q"],
        "terciles": list(th["terciles"]),
        "q_fixed": th["q_fixed"],
    }
    r["n_test"] = int(len(supp))
    return r


def stage_final_checks(corpora: list[str]) -> dict:
    """Сохраняются ли выводы B1 (и B2 на фолдах) на разных
    срезах. Конфигурация НЕ перевыбирается: победитель корпуса и число деревьев
    берутся из selected_policies.json."""
    log("Устойчивость B1/B2: фолды, авторы, top-1%, seeds, пересечения корпусов")
    sel = SEL.load_selection()
    out = {}
    for corpus in corpora:
        loss_id = sel["winner_by_corpus"][corpus]
        trees = sel["frozen_model_configs"][f"{corpus}:{loss_id}"][
            "n_trees_fixed_from_validation"
        ]
        frames = protocol_frames(corpus, "F1_meta")
        feats = features_for("F1_meta")
        prof = (
            (sel.get("b1_profiles_frozen") or {}).get(corpus)
            or SP.fixed_profiles(frames["train"])
        )["novice"]
        te = _te_with_support(frames["audit_exposed"], frames["train"])
        res_c: dict = {"loss_id": loss_id, "trees_frozen": trees}

        # 1) Чувствительность к seed: финальные модели на train+validation,
        #    деревья заморожены, seeds варьируются.
        seeds = {}
        for seed in (config.SEED, 20261112, 20261113):
            final = models.refit_final(
                frames["train"],
                frames["validation"],
                feats,
                [],
                loss_id,
                "C1_symmetric",
                trees,
                "catboost",
                seed=seed,
            )
            m = _b1_light_metrics(
                final, frames["train"], te, feats, prof, n_boot=config.N_BOOT
            )
            seeds[str(seed)] = {
                "delta_rho": m["delta_rho"],
                "delta_rho_ci_author": m["delta_rho_ci_author"],
                "delta_rho_pct": m["age_controlled"]["delta_rho_pct"],
                "delta_rho_ci_author_pct": m["age_controlled"][
                    "delta_rho_ci_author_pct"
                ],
            }
            log(
                f"  {corpus}/{loss_id} seed={seed}: B1 Δρ={m['delta_rho']:+.3f}, "
                f"Δρ(pct месяца)={m['age_controlled']['delta_rho_pct']:+.3f}"
            )
        res_c["seeds"] = seeds

        # Базовая финальная модель (seed по умолчанию) для срезов состава.
        final = models.refit_final(
            frames["train"],
            frames["validation"],
            feats,
            [],
            loss_id,
            "C1_symmetric",
            trees,
            "catboost",
        )
        te_b1 = SP.b1_fixed_profile_scores(final, te, feats, prof)
        supp = SP.personal_supported(te_b1)

        # 2) Известные (has_prior=1) против новых авторов.
        res_c["authors_known"] = M.b_metrics_age(
            supp[supp.has_prior == 1], "model_score", n_boot=config.N_BOOT
        )
        res_c["authors_new"] = M.b_metrics_age(
            supp[supp.has_prior == 0], "model_score", n_boot=config.N_BOOT
        )
        res_c["authors_known"]["n"] = int((supp.has_prior == 1).sum())
        res_c["authors_new"]["n"] = int((supp.has_prior == 0).sum())

        # 3) Без верхнего 1% наиболее популярных публикаций (порог - перцентиль аудита).
        p99 = float(np.quantile(supp["views"], 0.99))
        res_c["no_top1pct"] = M.b_metrics_age(
            supp[supp["views"] <= p99], "model_score", n_boot=config.N_BOOT
        )
        res_c["no_top1pct"]["views_p99"] = p99
        res_c["no_top1pct"]["n"] = int((supp["views"] <= p99).sum())
        log(
            f"  {corpus}: known Δρ={res_c['authors_known']['delta_rho_views']:+.3f} "
            f"(n={res_c['authors_known']['n']}), new Δρ={res_c['authors_new']['delta_rho_views']:+.3f} "
            f"(n={res_c['authors_new']['n']}), без top-1% Δρ={res_c['no_top1pct']['delta_rho_views']:+.3f}"
        )

        # 4) Rolling-фолды: B1 на каждом периоде (деревья фиксированы 600,
        #    гиперпараметры не пересматриваются).
        fold_rows = []
        for f in splits.rolling_splits(splits.materialize_corpus(corpus), corpus)[:4]:
            fte = _te_with_support(f["test"], f["train"])
            ffit = models.fit_catboost(
                f["train"],
                None,
                feats,
                [],
                loss_id,
                "C1_symmetric",
                iterations_fixed=min(600, config.ITERATIONS_CAP),
            )
            fm = _b1_light_metrics(
                ffit, f["train"], fte, feats, prof, n_boot=config.N_BOOT
            )
            fold_rows.append(
                dict(
                    cut=f["cut"],
                    n_train=int(len(f["train"])),
                    n_test=fm["n_test"],
                    delta_rho=fm["delta_rho"],
                    delta_rho_pct=fm["age_controlled"]["delta_rho_pct"],
                    rho_slot=fm["rho_slot"],
                    rho_slot_pct=fm["age_controlled"]["rho_slot_pct"],
                )
            )
            log(
                f"  {corpus}/{loss_id} fold {f['cut']}: B1 Δρ={fm['delta_rho']:+.3f}, "
                f"Δρ(pct)={fm['age_controlled']['delta_rho_pct']:+.3f}"
            )
        res_c["folds_b1"] = fold_rows
        out[corpus] = res_c

    # 5) Широкий корпус без статей, пересекающихся с ИИ/МЛ.
    if {"wide", "ai"} <= set(corpora):
        ai_ids = set(splits.materialize_corpus("ai")["id"].astype(str))
        dfw = splits.materialize_corpus("wide")
        mask_overlap = dfw["id"].astype(str).isin(ai_ids)
        dfw_f = dfw[~mask_overlap].copy()
        frames = splits.split_frame(dfw_f, "wide")
        frames = {("train" if k == "dev_train" else k): v for k, v in frames.items()}
        loss_id = sel["winner_by_corpus"]["wide"]
        trees = sel["frozen_model_configs"][f"wide:{loss_id}"][
            "n_trees_fixed_from_validation"
        ]
        feats = features_for("F1_meta")
        final = models.refit_final(
            frames["train"],
            frames["validation"],
            feats,
            [],
            loss_id,
            "C1_symmetric",
            trees,
            "catboost",
        )
        prof = SP.fixed_profiles(frames["train"])["novice"]
        te = _te_with_support(frames["audit_exposed"], frames["train"])
        m = _b1_light_metrics(
            final, frames["train"], te, feats, prof, n_boot=config.N_BOOT
        )
        out["wide_no_overlap"] = {
            "loss_id": loss_id,
            "n_removed_overlap": int(mask_overlap.sum()),
            "n_train": int(len(frames["train"])),
            "n_test": m["n_test"],
            "delta_rho": m["delta_rho"],
            "delta_rho_ci_author": m["delta_rho_ci_author"],
            "delta_rho_pct": m["age_controlled"]["delta_rho_pct"],
            "delta_rho_ci_author_pct": m["age_controlled"]["delta_rho_ci_author_pct"],
            "rho_slot": m["rho_slot"],
        }
        log(
            f"  wide без пересечений: удалено {int(mask_overlap.sum())} статей, "
            f"B1 Δρ={m['delta_rho']:+.3f}, Δρ(pct)={m['age_controlled']['delta_rho_pct']:+.3f}"
        )
    registry.save_metrics("final_checks", out)
    return out


