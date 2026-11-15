"""Сборка results.md из сохранённых метрик (этап 7).

Числа не перебиваются руками: всё читается из metrics/*.json и registry.csv.
Таблица моделей обновляется из JSON автоматически.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from . import config, registry

CORPUS_NAME = {"ai": "ИИ/МЛ", "wide": "Пять хабов"}

LOSS_LABEL = {
    "L01_rmse_log1p": "RMSE(log1p) — CatBoost C1",
    "L02_quantile_views_1000": "Quantile α=0.5 (views/1000) — CatBoost C1",
    "L03_poisson_views_1000": "Poisson (views/1000) — CatBoost C1",
    "L04_huber_log1p": "Huber δ=1.0 (log1p) — CatBoost C1",
    "L05_quantile_log1p": "Quantile α=0.5 (log1p) — CatBoost C1",
    "lgb_poisson_v24": "LightGBM Poisson (контроль)",
    "lgb_rmse_log1p": "LightGBM RMSE(log1p) (контроль)",
}


def _load(name):
    path = config.METRICS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def selection_table() -> pd.DataFrame:
    """Сводная таблица A-метрик: validation -> выбор; audit_exposed -> отчёт."""
    summary = _load("loss_sweep_summary")
    rows = []
    for key, res in summary.items():
        corpus, loss_id = key.split(":", 1)
        val = res.get("validation") or {}
        aud = res.get("final_audit") or res.get("audit") or {}
        rows.append(
            {
                "corpus": corpus,
                "loss_id": loss_id,
                "label": LOSS_LABEL.get(loss_id, loss_id),
                "rho_wm_val": val.get("rho_within_month_mean"),
                "rho_all_val": val.get("rho_all"),
                "mae_log_val": val.get("mae_log"),
                "base_mae_val": val.get("baseline_mae_log"),
                "rho_wm_audit": aud.get("rho_within_month_mean"),
                "rho_all_audit": aud.get("rho_all"),
                "mae_log_audit": aud.get("mae_log"),
                "bias_log_audit": aud.get("bias_log"),
            }
        )
    return pd.DataFrame(rows)


def pick_finalists(table: pd.DataFrame, tie_gap: float = 0.02) -> list[str]:
    """Процедура 13.3: сравнение primary A на validation; почти равные ->
    более простая/устойчивая модель. Финалистов максимум 3."""
    out = []
    for corpus in table["corpus"].unique():
        sub = table[table.corpus == corpus]
        sub = sub.dropna(subset=["rho_wm_val"])
        if sub.empty:
            continue
        sub = sub.sort_values("rho_wm_val", ascending=False)
        best = sub.iloc[0]
        out.append(best["loss_id"])
        for _, r in sub.iloc[1:].iterrows():
            if best["rho_wm_val"] - r["rho_wm_val"] < tie_gap:
                out.append(r["loss_id"])
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen[:3]


def build_results_md() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    hashes = config.input_hashes()
    table = selection_table()
    tuning = _load("tuning_report")
    policy = _load("policy_report")
    robust = _load("robustness")
    nonint = _load("robustness_nonintersect")
    rolling = _load("rolling_folds")
    diag = _load("poisson_scale_check")
    reg = (
        pd.read_csv(config.REGISTRY_CSV)
        if config.REGISTRY_CSV.exists()
        else pd.DataFrame()
    )

    parts = [
        "# Результаты перезапуска исследования на CatBoost (cb-research-v1)",
        "",
        f"Сборка отчёта: {now}. Версия протокола: `{config.protocol()['protocol']}`;",
        f"хеш кода: `{registry.code_hash()}`.",
        "",
        "> Тест 2026 года просматривался на предыдущих этапах работы. Все числа на нём",
        "> помечены статусом `audit_exposed` и не являются независимым",
        "> подтверждением. Задачи A и B разделены: B — ассоциативная ретроспективная",
        "> проверка, а не измерение эффекта переноса публикации.",
        "",
        "## 1. Задача A: сравнение функций потерь",
        "",
        "Одинаковые фолды (train → validation выбор), одинаковый набор F1_meta,",
        "конфигурация C1_symmetric. Выбор финалистов — только на validation.",
        "",
    ]
    if not table.empty:
        parts.append(
            "| Корпус | Модель | ρ внутр. мес (val) | ρ all (val) | MAE log (val) | ρ внутр. мес (audit) | ρ all (audit) | MAE log (audit) |"
        )
        parts.append("|---|---|---|---|---|---|---|---|")
        for _, r in table.iterrows():
            parts.append(
                f"| {r.corpus} | {r.label} | {_fmt(r.rho_wm_val)} | {_fmt(r.rho_all_val)} | "
                f"{_fmt(r.mae_log_val)} | {_fmt(r.rho_wm_audit)} | {_fmt(r.rho_all_audit)} | "
                f"{_fmt(r.mae_log_audit)} |"
            )
        parts.append("")
        sel_path = config.SELECTED_POLICIES_JSON
        sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
        if sel.get("finalists") or sel.get("finalists_by_corpus"):
            fin = list(
                dict.fromkeys(
                    sel.get("finalists")
                    or [x for v in sel["finalists_by_corpus"].values() for x in v]
                )
            )[:3]
            parts.append(
                "**Финалисты: только по validation, зафиксированы в "
                f"selected_policies.json ({sel.get('frozen_at', '')}) до аудита:** "
                f"{', '.join(fin) or '—'}. Победители: "
                + ", ".join(
                    f"{c}: {w}" for c, w in (sel.get("winner_by_corpus") or {}).items()
                )
                + "."
            )
        else:
            fin = pick_finalists(
                table, float(config.protocol()["selection"]["tie_gap"])
            )
            parts.append(
                f"**Финалисты по заранее заданной процедуре:** {', '.join(fin) or '—'}."
            )
        parts.append("")

    if tuning:
        parts += [
            "## 2. Кодирование календаря и ограниченный подбор (validation, ИИ/МЛ)",
            "",
        ]
        for label, v in tuning.items():
            parts.append(
                f"- {label}: ρ внутр. мес = {_fmt(v.get('rho_within_month'))}, "
                f"MAE log = {_fmt(v.get('mae_log'))}, лучшая итерация (счёт с нуля) = {v.get('best_iteration')}"
            )
        parts.append("")

    if policy:
        parts += [
            "## 3. Задача B (audit_exposed, личные статьи): A / B1 / B2 раздельно",
            "",
            "Режим A (full_prediction) - прогноз на фактических признаках статьи:",
            "контрольная точка задачи A, не доказательство выбора времени.",
            "B1 (fixed_profile_time_score) - фиксированные профили статьи, меняется",
            "только календарь: календарный сигнал сверх выходных. B2",
            "(personalized_scheduling) - персональные альтернативные слоты,",
            "диагностика политики (наблюдаемый прирост переноса не измеряется).",
            "Победитель зафиксирован в selected_policies.json ДО аудита.",
            "",
        ]
        sel = _load("selected_policies") or {}
        for corpus, models_ in policy.items():
            parts.append(f"### Корпус {CORPUS_NAME.get(corpus, corpus)}")
            parts.append("")
            for key, res in models_.items():
                if res.get("status") == "too_few_test_articles":
                    parts.append(f"- {key}: мало статей, вывод не формируется")
                    continue
                marker = (
                    " (замороженный победитель)"
                    if res.get("is_locked_winner")
                    else (" (контроль)" if res.get("is_control") else "")
                )
                dr = res.get("delta_rho")
                ci = res.get("delta_rho_ci_author") or [None, None]
                es = res.get("equal_share") or {}
                wr = res.get("weekend_rule") or {}
                wd = res.get("weekday_only") or {}
                parts.append(
                    f"- **A / {key}**{marker}: ρ_slot={_fmt(res.get('rho_slot'))}, "
                    f"ρ_weekend={_fmt(res.get('rho_weekend'))}, "
                    f"Δρ={_fmt(dr)} [авторы {_fmt(ci[0])}; {_fmt(ci[1])}], "
                    f"q-покрытие: модель {_fmt(es.get('top_q_ratio'))} vs правило {_fmt(wr.get('weekend_q_ratio'))}, "
                    f"будни-only верх/низ = {_fmt(wd.get('weekday_only_hi_lo'))}, "
                    f"coverage = {_fmt(res.get('coverage'))}"
                )
                for pname, b in (
                    res.get("task_B1_fixed_profile_time_score") or {}
                ).items():
                    ci_b = b.get("delta_rho_ci_author") or [None, None]
                    parts.append(
                        f"  - B1 / профиль {pname}: Δρ={_fmt(b.get('delta_rho'))} "
                        f"[{_fmt(ci_b[0])}; {_fmt(ci_b[1])}], "
                        f"ρ_slot={_fmt(b.get('rho_slot'))}, "
                        f"будни-only = {_fmt((b.get('weekday_only') or {}).get('weekday_only_hi_lo'))}"
                    )
                b2 = res.get("task_B2_personalized_scheduling") or {}
                if b2:
                    season_vals = [
                        v
                        for v in (b2.get("season_rank_agreement") or {}).values()
                        if v is not None
                    ]
                    parts.append(
                        f"  - B2: статей {b2.get('n_articles')}, доля рекомендаций, "
                        f"отличающихся от правила выходных = {_fmt(b2.get('share_reco_differs_weekend_rule'))}, "
                        f"согласованность месяцев (Спирмен, min-max) = "
                        f"{_fmt(min(season_vals))}…{_fmt(max(season_vals))}"
                        if season_vals
                        else f"  - B2: статей {b2.get('n_articles')}, доля рекомендаций, "
                        f"отличающихся от правила выходных = {_fmt(b2.get('share_reco_differs_weekend_rule'))}"
                    )
            parts.append("")

    if robust:
        parts += ["## 4. Устойчивость", ""]
        for corpus, res in robust.items():
            seeds = res.get("seeds") or {}
            vals = ", ".join(
                f"{s}: {_fmt(v.get('rho_within_month'))}" for s, v in seeds.items()
            )
            parts.append(f"- **{corpus}**, seeds (ρ внутр. мес, audit): {vals}")
            cs = res.get("cold_start") or {}
            parts.append(
                f"  - холодный старт авторов: {cs.get('n_cold')} статей "
                f"({cs.get('share_cold')}), ρ внутр. мес = {_fmt(cs.get('rho_within_month_cold'))} "
                f"(у известных авторов {_fmt(cs.get('rho_within_month_warm'))})"
            )
            tail = res.get("tail_1pct_cut") or {}
            parts.append(
                f"  - чувствительность к хвосту (обрезка верхнего 1%): "
                f"ρ all {_fmt(tail.get('rho_all_full'))} → {_fmt(tail.get('rho_all_cut'))}"
            )
            eras = res.get("eras") or {}
            if eras:
                parts.append(
                    "  - по когортам: "
                    + "; ".join(
                        f"{k}: {_fmt(v.get('rho_within_month'))} (n={v.get('n')})"
                        for k, v in eras.items()
                    )
                )
        parts.append("")
    if nonint:
        parts += [
            "## 5. Непересекающиеся части корпусов (A13)",
            "",
            f"- широкий корпус, аудит без статей, пересекающихся с ИИ/МЛ "
            f"(n={nonint.get('n_wide_audit_nonintersect')} из {nonint.get('n_wide_audit')}): "
            f"ρ внутр. мес {_fmt(nonint.get('rho_within_month_full'))} → "
            f"{_fmt(nonint.get('rho_within_month_nonintersect'))}.",
            "",
        ]
    if rolling:
        parts += ["## 6. Rolling-origin фолды (гиперпараметры не пересобираются)", ""]
        for key, rows in rolling.items():
            for r in rows:
                parts.append(
                    f"- {key}, граница {r['cut']}: ρ внутр. мес = {_fmt(r['rho_within_month'])}, "
                    f"MAE log = {_fmt(r['mae_log'])}, n_train={r['n_train']}, n_test={r['n_test']}"
                )
        parts.append("")
    if diag:
        parts += ["## 7. Численная диагностика Poisson (масштаб цели)", ""]
        for scale, v in diag.items():
            parts.append(
                f"- /{scale}: ρ val = {_fmt(v.get('rho_val'))}, "
                f"RMSE log = {_fmt(v.get('rmse_log_val'))}, "
                f"finite = {_fmt(v.get('frac_finite'))}, "
                f"дисперсия прогноза = {_fmt(v.get('pred_var'))}"
            )
        parts.append("")
    if not reg.empty:
        parts += [
            "## 8. Реестр экспериментов",
            "",
            f"Строк в registry.csv: {len(reg)}; код: `{reg.code_hash.iloc[-1] if 'code_hash' in reg else ''}`;",
            f"хеши входов: ai `{hashes.get('ai', '')[:12]}…`, wide `{hashes.get('wide', '')[:12]}…`.",
            "Каждый запуск — отдельный run_id; повторный запуск не стирает историю.",
            "",
            "## 9. Ограничения",
            "",
            "- Тест 2026 года — audit_exposed: просматривался на предыдущих этапах работы;",
            "  окончательное подтверждение — только перспективные данные (collect_panel.py).",
            "- Задача B измеряет ассоциативную связь score слота с исходами;",
            "  даже положительный out-of-time тест не доказывает переносимый прирост.",
            "- Rolling-фолды не пересчитывают полный подбор гиперпараметров;",
            "  интервалы — парный авторский бутстреп, недельная зависимость не моделируется.",
            "- Спирмен общий в растущем корпусе частично измеряет возраст;",
            "  основной критерий — ранжирование внутри месяцев.",
            "",
        ]
    text = "\n".join(parts)
    out = config.RESULTS / "results.md"
    out.write_text(text, encoding="utf-8")
    print(f"results.md записан: {out}")
    return text
