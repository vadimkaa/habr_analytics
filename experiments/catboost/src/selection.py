"""Отбор моделей по валидационной выборке.

Правило отбора задаётся в configs/protocol_v1.yaml: финалисты находятся в
пределах tie_gap от лучшей модели, их число ограничено finalists_max.
Выбор сохраняется в selected_policies.json до оценки audit_exposed.
Контрольные модели LightGBM участвуют в сравнении, но не в конкурсе CatBoost.
"""

from __future__ import annotations

import json
import time

from . import config, registry

CONTROL_LOSSES = ("lgb_poisson_v24", "lgb_rmse_log1p")

# Порядок разрешения равенства метрик: сначала группа конфигурации,
# затем идентификатор функции потерь.
LOSS_WAVE = {
    "L01_rmse_log1p": 1,
    "L02_quantile_views_1000": 1,
    "L03_poisson_views_1000": 1,
    "L04_huber_log1p": 2,
    "L05_quantile_log1p": 2,
}

SELECTION_RULE = (
    "Кандидаты CatBoost L01-L05 (контроль LightGBM не участвует в выборе), "
    "основная метрика задачи A — rho_within_month_mean на validation; "
    "к лучшему добавляются кандидаты в пределах tie_gap=0.02; не более "
    "finalists_max. Ничья: более простая модель (волна конфигурации, затем "
    "имя loss). Победитель по корпусу: лучший primary на validation среди "
    "финалистов того корпуса. Всё фиксируется до аудита; аудит только "
    "оценивает зафиксированное."
)


def _candidate_metrics(summary: dict) -> dict:
    """validation-метрики кандидатов из loss_sweep_summary (без контроля LGB)."""
    out = {}
    for key, res in summary.items():
        corpus, loss_id = key.split(":", 1)
        if loss_id in CONTROL_LOSSES:
            continue  # замороженный контроль в отборе не участвует
        val = res.get("validation") or {}
        if val.get("rho_within_month_mean") is None:
            continue
        out.setdefault(corpus, {})[loss_id] = {
            "rho_within_month_mean": val.get("rho_within_month_mean"),
            "rho_all": val.get("rho_all"),
            "mae_log": val.get("mae_log"),
            "baseline_mae_log": val.get("baseline_mae_log"),
        }
    return out


def derive_selection(summary: dict | None = None) -> dict:
    """Финалисты и победители строго по validation-метрикам."""
    sel_cfg = config.protocol()["selection"]
    tie_gap = float(sel_cfg["tie_gap"])
    finalists_max = int(sel_cfg["finalists_max"])
    summary = (
        summary if summary is not None else registry.load_metrics("loss_sweep_summary")
    )
    cand = _candidate_metrics(summary)

    finalists_by_corpus, winner_by_corpus = {}, {}
    for corpus, models_ in cand.items():
        ordered = sorted(
            models_.items(),
            key=lambda kv: (
                -(kv[1]["rho_within_month_mean"] or -9),
                LOSS_WAVE.get(kv[0], 9),
                kv[0],
            ),
        )
        best_id, best_m = ordered[0]
        group = [best_id]
        for loss_id, m in ordered[1:]:
            if (best_m["rho_within_month_mean"] - m["rho_within_month_mean"]) < tie_gap:
                group.append(loss_id)
        finalists_by_corpus[corpus] = group
        winner_by_corpus[corpus] = best_id

    # Глобальный список финалистов (порядок: лучший по validation первым),
    # не больше finalists_max; каждый корпус представлен своими финалистами.
    global_order: list[str] = []
    for corpus in sorted(cand):
        for loss_id in finalists_by_corpus[corpus]:
            if loss_id not in global_order:
                global_order.append(loss_id)
    finalists = global_order[:finalists_max]

    return {
        "selection_rule": SELECTION_RULE,
        "tie_gap": tie_gap,
        "finalists_max": finalists_max,
        "candidates_validation": cand,
        "finalists_by_corpus": finalists_by_corpus,
        "finalists": finalists,
        "winner_by_corpus": winner_by_corpus,
        "control_models": list(CONTROL_LOSSES),
    }


def _frozen_model_config(corpus: str, loss_id: str) -> dict | None:
    """Конфигурация/параметры/деревья кандидата из validation-артефактов loss."""
    meta = registry.load_metrics(f"loss_{corpus}_{loss_id}").get("metadata") or {}
    if not meta:
        return None
    return {
        "loss_id": loss_id,
        "config_id": meta.get("config_id"),
        "target_kind": meta.get("target_kind"),
        "objective": meta.get("objective"),
        "n_trees_fixed_from_validation": meta.get("best_iteration"),
        "seed": meta.get("seed", config.SEED),
        "feature_set": meta.get("feature_set"),
        "params": {k: str(v) for k, v in (meta.get("params") or {}).items()},
    }


def freeze_policy_config() -> dict:
    """Политика, единая для всех кандидатов, фиксируется до аудита."""
    pol = config.protocol()["policy"]
    return {
        "thresholds_source": "train_only",
        "score_definition": "model_score = декодированный прогноз финальной модели на фактическом слоте статьи (режим A, full_prediction)",
        "q_fixed": pol["q_fixed"],
        "q_sensitivity": pol["q_sensitivity"],
        "support_min_train": pol["support_min_train"],
        "fallback": pol["fallback"],
        "density_mode": "none (F1_meta); dens24_actual в политике запрещён; F3 expected - отдельная диагностическая ветка",
        "b2_recommendation_rule": (
            "аргмакс прогноза среди поддержанных слотов месяца статьи "
            "(канонические реальные даты месяца); слоты с n_train < "
            f"{pol['support_min_train']} не рекомендуются"
        ),
        "weekend_rule_definition": (
            "argmax (is_weekend, hour) лексикографически по тем же "
            "поддержанным слотам (выходной важнее, поздний час важнее)"
        ),
        "n_boot": config.N_BOOT,
        "bootstrap_seed": config.SEED,
    }


def build_selection_doc(corpora: list[str] | None = None) -> dict:
    """Полный документ selected_policies.json (заморозка ДО аудита)."""
    corpora = corpora or ["ai", "wide"]
    sel = derive_selection()
    frozen = {}
    b1_profiles = {}
    for corpus in corpora:
        for loss_id in [sel["winner_by_corpus"].get(corpus)] + [
            x
            for x in sel["finalists_by_corpus"].get(corpus, [])
            if x != sel["winner_by_corpus"].get(corpus)
        ]:
            cfgm = _frozen_model_config(corpus, loss_id)
            if cfgm:
                frozen[f"{corpus}:{loss_id}"] = cfgm
        # Профили B1 фиксируются численно по train ДО аудита.
        from . import slot_policy as SP
        from . import splits

        df = splits.materialize_corpus(corpus)
        frames = splits.split_frame(df, corpus)
        frames = {("train" if k == "dev_train" else k): v for k, v in frames.items()}
        profiles = SP.fixed_profiles(frames["train"])
        b1_profiles[corpus] = {
            name: {
                k: (float(v) if not isinstance(v, (int, str)) else v)
                for k, v in prof.items()
            }
            for name, prof in profiles.items()
        }
    doc = {
        "protocol": config.protocol()["protocol"],
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_hash": registry.code_hash(),
        "input_sha256": config.input_hashes(),
        "selection_rule": sel["selection_rule"],
        "tie_gap": sel["tie_gap"],
        "finalists_max": sel["finalists_max"],
        "candidates_validation": sel["candidates_validation"],
        "finalists_by_corpus": sel["finalists_by_corpus"],
        "finalists": sel["finalists"],
        "winner_by_corpus": sel["winner_by_corpus"],
        "control_models": sel["control_models"],
        "frozen_model_configs": frozen,
        "policy_freeze": freeze_policy_config(),
        "b1_profiles_frozen": b1_profiles,
        "audit_note": config.protocol()["selection"]["audit_note"],
        "winner_locked_before_audit": True,
        "note": (
            "Файл фиксирует выбор ДО прогона audit_exposed; стадия policy "
            "только оценивает зафиксированные политики и не меняет победителя."
        ),
    }
    return doc


def write_selected_policies(corpora: list[str] | None = None) -> dict:
    doc = build_selection_doc(corpora)
    path = config.SELECTED_POLICIES_JSON
    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return doc


def load_selection() -> dict:
    """Прочитать заморозку; отсутствие файла - ошибка (аудит без фиксации запрещён)."""
    path = config.SELECTED_POLICIES_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"{path} не найден: сначала выполните стадию select - финалисты и "
            "победитель обязаны быть зафиксированы до запуска аудита"
        )
    return json.loads(path.read_text(encoding="utf-8"))
