"""Сравнение календарных рейтингов A/B1/B2 и фиксация политик."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import pandas as pd

from src import (
    config,
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
)


def stage_policy(corpora: list[str]) -> dict:
    """Сетка D и policy P для ЗАМОРОЗКИ из selected_policies.json.

    density-режимы: none (F1) - все финалисты; expected (F3) и actual
    (F2, диагностическая верхняя граница) - победитель корпуса.
    Задачи разделяются: task A (full_prediction), task B1 (фиксированные
    профили - только календарь), task B2 (персональное планирование).
    LightGBM-контроль - исследовательское сравнение, в отборе не участвует."""
    log("Этап 5: policy/слоты - протоколы D и P (выбор заморожен до аудита)")
    sel = SEL.load_selection()
    report = {}
    for corpus in corpora:
        corpus_report = {}
        winner = sel["winner_by_corpus"].get(corpus)
        finalists = sel["finalists_by_corpus"].get(corpus, [])
        if not finalists:
            raise SystemExit(f"{corpus}: в selected_policies.json нет финалистов")
        frozen_trees = {
            loss: (sel["frozen_model_configs"].get(f"{corpus}:{loss}") or {}).get(
                "n_trees_fixed_from_validation"
            )
            for loss in finalists
        }
        # --- density-free (F1) финалисты: задача A + B1 + B2
        for loss_id in finalists:
            model_kind = "lgb" if loss_id.startswith("lgb") else "cb"
            frames = protocol_frames(corpus, "F1_meta")
            feats_used = features_for("F1_meta")
            if model_kind == "cb":
                fit = models.fit_catboost(
                    frames["train"],
                    frames["validation"],
                    feats_used,
                    [],
                    loss_id,
                    "C1_symmetric",
                )
                final = models.refit_final(
                    frames["train"],
                    frames["validation"],
                    feats_used,
                    [],
                    loss_id,
                    "C1_symmetric",
                    frozen_trees[loss_id],
                    "catboost",
                )
            else:
                fit = models.fit_lightgbm(
                    frames["train"], frames["validation"], feats_used, [], loss_id
                )
                # Согласованная процедура: LightGBM-кандидат тоже переобучается
                # на train+validation с числом деревьев из конфигурации.
                final = models.fit_lightgbm(
                    pd.concat(
                        [frames["train"], frames["validation"]], ignore_index=True
                    ),
                    None,
                    feats_used,
                    [],
                    loss_id,
                )
            tr_pol = frames["train"]
            frames_audit = {"audit_exposed": frames["audit_exposed"]}
            # поддержка слотов из train (личные статьи)
            slot_n = tr_pol.groupby(["dow", "hour"]).size()
            res = _policy_for_fit(
                corpus,
                loss_id,
                final,
                tr_pol,
                frames_audit,
                feats_used,
                None,
                slot_n,
                "none",
                sel,
                is_control=model_kind == "lgb",
                save_articles=True,
                save_slot_table=(loss_id == winner or model_kind == "lgb"),
            )
            res["is_finalist"] = loss_id in finalists
            res["is_locked_winner"] = loss_id == winner
            corpus_report[f"{loss_id}|density=none"] = res

        # --- expected (F3) и actual (F2, диагностика) для победителя корпуса
        if winner and not winner.startswith("lgb"):
            for fset, dmode in (
                ("F3_dens_expected", "expected"),
                ("F2_dens_actual", "actual_diagnostic"),
            ):
                frames = protocol_frames(corpus, fset)
                feats_used = features_for(fset)
                fit = models.fit_catboost(
                    frames["train"],
                    frames["validation"],
                    feats_used,
                    [],
                    winner,
                    "C1_symmetric",
                )
                final = models.refit_final(
                    frames["train"],
                    frames["validation"],
                    feats_used,
                    [],
                    winner,
                    "C1_symmetric",
                    frozen_trees.get(winner),
                    "catboost",
                )
                slot_n = frames["train"].groupby(["dow", "hour"]).size()
                density_col = (
                    "dens24_expected" if fset == "F3_dens_expected" else "dens24"
                )
                res = _policy_for_fit(
                    corpus,
                    winner,
                    final,
                    frames["train"],
                    {"audit_exposed": frames["audit_exposed"]},
                    feats_used,
                    density_col,
                    slot_n,
                    dmode,
                    sel,
                    is_control=False,
                )
                # Сетка D для expected-варианта (диагностическая)
                if fset == "F3_dens_expected":
                    grid = SP.score_grid(
                        final, frames["train"], feats_used, density_col, [1, 4, 7, 10]
                    )
                    res["grid_D"] = {k: v for k, v in grid.items() if k != "months"}
                    res["grid_D_months_june"] = grid["months"][7]
                corpus_report[f"{winner}|density={dmode}"] = res
        report[corpus] = corpus_report
    # Артефакты политики сохраняются ДО проверки fold-стабильности: сбой
    # дополнительной диагностики не должен терять основной отчёт.
    registry.save_metrics("policy_report", report)
    write_policy_json(report, sel)
    # B2: устойчивость рекомендаций к периоду обучения (rolling-фолды победителя)
    _b2_fold_stability(corpora, sel["winner_by_corpus"])
    return report


def _b2_fold_stability(corpora: list[str], winner_by_corpus: dict) -> None:
    """B2 на rolling-фолдах: согласованность рекомендаций между периодами
    обучения. Деревья фиксированы; выбор гиперпараметров не
    пересматривается."""
    log("B2: устойчивость рекомендаций к периоду обучения (rolling-фолды)")
    out = {}
    for corpus in corpora:
        loss_id = winner_by_corpus.get(corpus)
        if not loss_id:
            continue
        df = splits.materialize_corpus(corpus)
        folds = splits.rolling_splits(df, corpus)[:3]
        feats_f1 = features_for("F1_meta")
        vectors, top_recos, scale_metrics = {}, {}, {}
        for f in folds:
            fit = models.fit_catboost(
                f["train"],
                None,
                feats_f1,
                [],
                loss_id,
                "C1_symmetric",
                iterations_fixed=min(600, config.ITERATIONS_CAP),
            )
            res = SP.personalized_scheduling(fit, f["test"], f["train"], feats_f1, None)
            cut = f["cut"]
            mean_vec = None
            vecs = res.get("month_rank_vectors") or {}
            if vecs:
                mean_vec = np.mean([np.asarray(v) for v in vecs.values()], axis=0)
                vectors[cut] = [float(x) for x in mean_vec]
            recos = res.get("reco_per_article") or {}
            if recos:
                c = Counter((v[2], v[0]) for v in recos.values())
                top_recos[cut] = c.most_common(3)
            # Три шкалы B2 на фолде без бутстрепа (устойчивость точечных значений).
            # per_article не содержит month_id/author_key — берём их из фолда теста.
            pa = res.get("per_article")
            fold_scales = {}
            if pa is not None and len(pa):
                pa2 = f["test"][
                    ["id", "author_key", "week_start", "month_id", "is_weekend"]
                ].merge(
                    pa[
                        [
                            "article_id",
                            "personalized_slot_score",
                            "personalized_slot_rel",
                            "personalized_slot_pct",
                            "views",
                        ]
                    ],
                    left_on="id",
                    right_on="article_id",
                    how="inner",
                )
                for tag, col in (
                    ("absolute", "personalized_slot_score"),
                    ("relative", "personalized_slot_rel"),
                    ("percentile", "personalized_slot_pct"),
                ):
                    sub = pa2.dropna(subset=[col])
                    if len(sub) >= 100:
                        m = M.b_metrics_age(sub, col)
                        fold_scales[tag] = {
                            "delta_rho_views": m.get("delta_rho_views"),
                            "delta_rho_pct": m.get("delta_rho_pct"),
                            "n": m.get("n"),
                        }
            log(
                f"  {corpus}/{loss_id} fold {cut}: n_art={res.get('n_articles')}, "
                f"diff_weekend={res.get('share_reco_differs_weekend_rule')}, "
                f"scales={ {k: round((v.get('delta_rho_views') or 0), 3) for k, v in fold_scales.items()} }"
            )
            scale_metrics[cut] = fold_scales
        # Согласованность средних рейтингов слотов между фолдами.
        agree = {}
        keys = list(vectors)
        from src.metrics import rho_all

        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                agree[f"{a}:{b}"] = rho_all(vectors[a], vectors[b])
        # Сохраняем запись даже при нуле подходящих векторов фолдов.
        out[f"{corpus}:{loss_id}"] = dict(
            folds=keys,
            mean_rank_agreement=agree,
            top_recommended=top_recos,
            scale_metrics=scale_metrics,
            note="рейтинг слотов усреднён по тестовым месяцам фолда; гиперпараметры не пересматриваются; просмотры измерены одним срезом 11.2026 - исторического прогноза без утечки поздней динамики это не даёт",
        )
    registry.save_metrics("b2_stability_folds", out)


def _policy_for_fit(
    corpus: str,
    loss_id: str,
    final: models.FitResult,
    train: pd.DataFrame,
    frames: dict,
    feats_used: list[str],
    density_col: str | None,
    slot_n: pd.Series,
    dmode: str,
    selection: dict | None = None,
    is_control: bool = False,
    save_articles: bool = False,
    save_slot_table: bool = False,
) -> dict:
    te = frames["audit_exposed"].copy()
    te["slot_n_train"] = [
        int(slot_n.to_dict().get((int(d), int(h)), 0)) for d, h in zip(te.dow, te.hour)
    ]
    # Задача A (режим A, full_prediction): score фактического слота при
    # train-only плотности; границы терцилей и порог q - по train (12.2.5).
    tr_scored = SP.assign_scores(final, train, train, feats_used, density_col)
    scored = SP.assign_scores(final, te, train, feats_used, density_col)
    supp = SP.personal_supported(scored)
    log(f"  {corpus} {loss_id} [{dmode}]: личных поддержанных {len(supp)} из {len(te)}")
    res = SP.policy_report(supp, tr_scored, n_boot=config.N_BOOT)
    res["task_A_full_prediction"] = {
        k: res[k]
        for k in (
            "rho_slot",
            "rho_weekend",
            "delta_rho",
            "delta_rho_ci_author",
            "delta_rho_ci_week",
            "equal_share",
            "weekend_rule",
            "fixed_q",
            "weekday_only",
            "hi_lo_tercile",
            "hi_lo_tercile_ci_author",
            "coverage",
            "random_control",
            "shuffle_control_rho",
            "real_rho_within_month",
        )
        if k in res
    }
    res["task"] = "A_full_prediction"
    # Реестр: policy-прогон тоже одна строка
    registry.append_registry_row(
        dict(
            stage="policy",
            corpus=corpus,
            split_id=f"{corpus}__protocol__audit_exposed",
            loss_function=loss_id,
            target_transform=final.metadata.get("target_kind"),
            model_family=f"{final.metadata.get('library')}:{final.metadata.get('config_id')}",
            feature_set="+".join(feats_used[:6])
            + ("..." if len(feats_used) > 6 else ""),
            density_mode=dmode,
            n_train=len(train),
            n_test=len(supp),
            status=config.STATUS_AUDIT_EXPOSED,
            rho_all=res.get("rho_slot"),
            rho_within_month=res.get("real_rho_within_month"),
            delta_rho=res.get("delta_rho"),
            delta_rho_ci95=json.dumps(res.get("delta_rho_ci_author") or []),
            top_q_ratio=(res.get("equal_share") or {}).get("top_q_ratio"),
            coverage=res.get("coverage"),
            ci_method="paired_author_bootstrap",
            run_id=next_run_id("pol"),
        )
    )
    res["n_test_personal_total"] = int((te.is_corp == 0).sum())
    if res.get("n_test_personal_total"):
        res["coverage"] = round(len(supp) / res["n_test_personal_total"], 4)
    res["feature_set"] = feats_used
    res["density_mode"] = dmode
    res["is_control"] = is_control

    # Контроль возраста: те же наблюдения, три цели -
    # накопленные просмотры, перцентиль внутри месяца, внутри-месячные корреляции.
    te["views_pct_month"] = M.month_pct_views(te)
    res["age_controlled"] = M.b_metrics_age(supp, "model_score", n_boot=config.N_BOOT)
    # Операционная метрика: порог, зафиксированный на train, без пересчёта на тесте;
    # диагностика остаётся в equal_share.
    th_q = (res.get("thresholds_train") or {}).get("threshold_q")
    if th_q is not None:
        res["fixed_threshold"] = M.fixed_threshold_ratio(
            supp,
            "model_score",
            th_q,
            n_boot=config.N_BOOT,
        )

    # --- Задача B1: фиксированные профили, меняется только календарь.
    b1_frames = {}
    if density_col in (None, "dens24_expected"):
        profiles_frozen = (selection or {}).get("b1_profiles_frozen", {}).get(
            corpus
        ) or {}
        profiles = profiles_frozen or SP.fixed_profiles(train)
        b1 = {}
        for pname, prof in profiles.items():
            tr_b1 = SP.b1_fixed_profile_scores(final, train, feats_used, prof)
            te_b1 = SP.b1_fixed_profile_scores(final, te, feats_used, prof)
            b1_frames[pname] = te_b1
            supp_b1 = SP.personal_supported(te_b1)
            r = SP.policy_report(supp_b1, tr_b1, n_boot=config.N_BOOT)
            r["profile_frozen"] = prof
            r["mode"] = "B1_fixed_profile_time_score"
            r["age_controlled"] = M.b_metrics_age(
                supp_b1, "model_score", n_boot=config.N_BOOT
            )
            th_b1 = (r.get("thresholds_train") or {}).get("threshold_q")
            if th_b1 is not None:
                r["fixed_threshold"] = M.fixed_threshold_ratio(
                    supp_b1,
                    "model_score",
                    th_b1,
                    n_boot=config.N_BOOT,
                )
            b1[pname] = r
        res["task_B1_fixed_profile_time_score"] = b1
        log(
            f"  {corpus} {loss_id} [{dmode}]: B1 профили: "
            + ", ".join(
                f"{p}: Δρ={_fmt_delta(b.get('delta_rho'))}" for p, b in b1.items()
            )
        )

        # --- Задача B2: персональное планирование (фактическая плотность
        # будущей ленты неизвестна; для F2 - dens24_actual - B2 не выполняется).
        if density_col in (None, "dens24_expected"):
            b2_full = SP.personalized_scheduling(
                final,
                te,
                train,
                feats_used,
                density_col,
                return_slot_table=save_slot_table,
            )
            # Порартикульные рамки в JSON не пишутся: отдельные CSV.
            res["task_B2_personalized_scheduling"] = {
                k: v
                for k, v in b2_full.items()
                if k not in ("per_article", "slot_table")
            }
            pa = b2_full.get("per_article")
            if pa is not None and len(pa):
                # Основная совокупность B2 = те же личные статьи поддержанных
                # фактических слотов, что и в A/B1: показатели
                # на разных совокупностях не представляются сопоставимыми.
                supp_ids = set(supp["id"].astype(str))
                pa_main = pa[pa["article_id"].isin(supp_ids)].copy()
                res["task_B2_personalized_scheduling"]["main_population"] = {
                    "n": int(len(pa_main)),
                    "n_authors": int(pa_main["author_id"].nunique()),
                    "definition": (
                        "личные статьи в поддержанных слотах - "
                        "та же совокупность, что в режимах A и B1"
                    ),
                }
                te2 = te.merge(
                    pa_main[
                        [
                            "article_id",
                            "personalized_slot_score",
                            "personalized_slot_rel",
                            "personalized_slot_pct",
                            "eligible_slots_count",
                        ]
                    ],
                    left_on="id",
                    right_on="article_id",
                    how="left",
                )
                # Три шкалы персонализированной оценки.
                # Δρ везде = ρ(score, цель) - ρ(is_weekend, цель) на одних строках:
                # сравнение с двоичным признаком выходных, а не с другим score.
                for tag, col in (
                    ("absolute", "personalized_slot_score"),
                    ("relative", "personalized_slot_rel"),
                    ("percentile", "personalized_slot_pct"),
                ):
                    covered = te2.dropna(subset=[col])
                    blk = M.b_metrics_age(covered, col, n_boot=config.N_BOOT)
                    blk["n_covered"] = int(len(covered))
                    blk["scale"] = {
                        "absolute": "pred(факт) - median(pred поддержанных альтернатив)",
                        "relative": "log1p(pred(факт)) - log1p(median(pred поддержанных альтернатив))",
                        "percentile": "доля поддержанных альтернатив с прогнозом не выше прогноза фактического слота",
                    }[tag]
                    res[f"task_B2_{tag}"] = blk
                # Расширенная совокупность (все личные статьи) - доп. проверка.
                te3 = te.merge(
                    pa[["article_id", "personalized_slot_score"]],
                    left_on="id",
                    right_on="article_id",
                    how="left",
                )
                cov3 = te3.dropna(subset=["personalized_slot_score"])
                res["task_B2_extended_absolute"] = M.b_metrics_age(
                    cov3,
                    "personalized_slot_score",
                    n_boot=config.N_BOOT,
                )
                res["task_B2_extended_absolute"]["n_covered"] = int(len(cov3))
                res["task_B2_extended_absolute"]["population"] = (
                    "extended: все личные статьи"
                )
            log(
                f"  {corpus} {loss_id} [{dmode}]: B2 статей {res['task_B2_personalized_scheduling'].get('n_articles')} "
                f"(основная {(res['task_B2_personalized_scheduling'].get('main_population') or {}).get('n')}), "
                f"отличий от правила выходных "
                f"{res['task_B2_personalized_scheduling'].get('share_reco_differs_weekend_rule')}"
            )

        # Поартикульная таблица A/B1/B2.
        if save_articles:
            art = pd.DataFrame(
                {
                    "article_id": te["id"].astype(str),
                    "author_id": te["author_key"].astype(str),
                    "published_at": te["dt_msk"].astype(str),
                    "views": te["views"].astype(float),
                    "month": te["month_id"],
                    "actual_slot": [
                        f"{int(d)}-{int(h)}" for d, h in zip(te.dow, te.hour)
                    ],
                    "weekend": te["is_weekend"].astype(int),
                }
            )
            art["full_prediction_score"] = scored["model_score"].to_numpy()
            art["fixed_profile_time_score"] = (
                b1_frames["novice"]["model_score"].to_numpy()
                if "novice" in b1_frames
                else np.nan
            )
            if pa is not None and len(pa):
                art = art.merge(
                    pa[
                        [
                            "article_id",
                            "personalized_slot_score",
                            "personalized_slot_rel",
                            "personalized_slot_pct",
                            "eligible_slots_count",
                            "actual_slot_rank_pct_grid",
                            "recommended_slot",
                        ]
                    ],
                    left_on="article_id",
                    right_on="article_id",
                    how="left",
                )
            art["fold_id"] = "final"
            art["model_id"] = f"{corpus}:{loss_id}"
            out_dir = config.PREDICTIONS_DIR / "article_level"
            out_dir.mkdir(exist_ok=True)
            art.to_csv(
                out_dir / f"{corpus}_{loss_id}.csv.gz", index=False, compression="gzip"
            )
            if (
                save_slot_table
                and b2_full.get("slot_table") is not None
                and len(b2_full["slot_table"])
            ):
                st_dir = config.PREDICTIONS_DIR / "b2_slots"
                st_dir.mkdir(exist_ok=True)
                b2_full["slot_table"].to_csv(
                    st_dir / f"{corpus}_{loss_id}.csv.gz",
                    index=False,
                    compression="gzip",
                )
    return res


def _fmt_delta(v) -> str:
    return "— " if v is None else f"{v:+.3f}"


def write_policy_json(report: dict, selection: dict) -> None:
    """Сериализованное правило.

    Победитель берётся ИЗ ЗАМОРОЗКИ (selected_policies.json, выбран по
    validation до аудита); аудит только оценивает зафиксированные политики.
    Изменение результатов аудита не может поменять победителя."""
    policy = {
        "protocol": config.protocol()["protocol"],
        "selected_by": "validation-only (selected_policies.json, заморожено до аудита)",
        "winner_locked_before_audit": True,
        "decision_rule": (
            "score слота из финальной модели; threshold - квантиль train; "
            "при n_train(slot) < 20 - fallback: правило выходных, "
            "вывод «недостаточно наблюдений»"
        ),
        "note": (
            "delta_rho победителя на аудите показан для сведения и в выборе "
            "не участвует; остальные модели - исследовательское сравнение"
        ),
        "corpora": {},
    }
    for corpus, models_ in report.items():
        best_key = f"{selection['winner_by_corpus'].get(corpus)}|density=none"
        res = models_.get(best_key)
        if not res:
            continue
        policy["corpora"][corpus] = dict(
            model=best_key,
            winner_locked=True,
            thresholds_train=res.get("thresholds_train"),
            q_fixed=res.get("thresholds_train", {}).get("q_fixed"),
            coverage=res.get("coverage"),
            fallback="weekend_rule",
            audit_delta_rho_informational=res.get("delta_rho"),
        )
    config.POLICY_JSON.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


