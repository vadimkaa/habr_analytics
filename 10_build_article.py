"""Шаг 10. Сборка article.md из шаблона и сохранённых результатов.

В article_template.md числа стоят как заполнители вида {{имя}}. Скрипт читает
JSON из data/results/, форматирует числа и таблицы и подставляет их в шаблон.
Каждая функция values_* готовит заполнители одного раздела статьи, поэтому по
имени заполнителя легко найти, откуда берётся число:

    values_data         раздел 4: состав корпусов, переводы, текстовая выборка
    values_weekend      раздел 6: выходные, часы, тепловая карта, проверки
    values_authors      раздел 6: автор против самого себя, «когда» или «кто»
    values_density      раздел 7: плотность ленты и квинтили
    values_holidays     раздел 7: праздники
    values_readability  раздел 8: читаемость текста
    values_model        разделы 9–11: модель, сетка слотов, утечка

Результат: article.md и data/results/article_values.json — все подставленные
значения. Основные расчётные показатели подставляются из JSON; поясняющие числа в тексте также допустимы.
"""

import json
import re
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from habrpub import BASE, DATA, RESULTS

LABELS = {"ai": "ИИ/МЛ", "wide": "Пять хабов"}
CORPORA = ("ai", "wide")


# --- Форматирование ----------------------------------------------------------


def get(name):
    return json.loads((RESULTS / name).read_text())


def f(x, n=2):
    """Число с n знаками после десятичной точки.

    Округление половины вверх (а не банковское, как у round): иначе соседние
    значения вроде 18.5 и 19.5 печатались бы как «18 и 20». Минус — знак U+2212.
    """
    q = Decimal(str(float(x))).quantize(Decimal("1." + "0" * n), rounding=ROUND_HALF_UP)
    return f"{q:.{n}f}".replace("-", "−")


def count(x):
    """Целое без разделителя разрядов: 46845."""
    return str(int(x))


def count_k(x):
    """Просмотры в тысячах: «2.0 тыс.». Голое «2026.5» читалось бы как год."""
    return f(x / 1000, 1) + " тыс."


def plural(n, one, few, many):
    """Согласование существительного с числом: 1 статья, 2 статьи, 5 статей."""
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many


def interval(x, n=2):
    return "[" + f(x[0], n) + "; " + f(x[1], n) + "]"


def p_text(p, threshold=0.001):
    return f"p < {threshold}" if p < threshold else "p = " + f(p, 2)


def table(headers, rows):
    """Таблица в markdown. Число ячеек каждой строки сверяется с заголовком:
    лишняя ячейка молча сдвинула бы столбцы."""
    for r in rows:
        assert len(r) == len(headers), (
            f"строка из {len(r)} ячеек против {len(headers)} в заголовке: {r[:2]}"
        )
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def fmt_dates(dates):
    """Даты праздничного блока: ежегодный блок сворачивается в диапазон лет,
    длинный список — в первые три даты и «ещё N»."""
    if not dates:
        return ""
    years = sorted({d[:4] for d in dates})
    if len(years) > 2 and int(years[-1]) - int(years[0]) == len(years) - 1:
        return f"{years[0]}–{years[-1]}, ежегодно"
    short = [d[8:10] + "." + d[5:7] + "." + d[:4] for d in dates]
    if len(short) <= 4:
        return ", ".join(short)
    return ", ".join(short[:3]) + f" и ещё {len(short) - 3}"


# --- Исходные результаты -------------------------------------------------------


def load_results():
    """Все JSON-результаты, которые использует статья, под короткими именами."""
    return {
        "explore": get("02_explore_results.json"),  # ИИ/МЛ, последний год
        "ai": get("part2_ai_results.json"),  # календарь ИИ/МЛ (шаг 04)
        "wide": get("part2_wide_results.json"),  # календарь пяти хабов (шаг 03)
        "density": get("part4_results.json"),  # шаг 05
        "text": get("06_text_results.json"),
        "read": get("07_readability_results.json"),
        "rob": get("09_robustness_results.json"),
        "leak": get("11_leakage_diagnostic.json"),
        "extra": get("14_extra_checks.json"),  # шаг 14
        "ml": load_ml(),  # разделы 9–10: эксперимент experiments/catboost
    }


ML = BASE / "experiments" / "catboost" / "results"


def load_ml():
    """Сохранённые результаты эксперимента CatBoost (сам он здесь не переобучается)."""

    def read(rel):
        return json.loads((ML / rel).read_text(encoding="utf-8"))

    return {
        "loss": read("metrics/loss_sweep_summary.json"),
        "policy": read("metrics/policy_report.json"),
        "calibration": read("metrics/calibration.json"),
        "final": read("metrics/final_checks.json"),
        "selected": read("selected_policies.json"),
    }


# --- Раздел 4. Данные ----------------------------------------------------------


def values_data(R):
    a, w, rob, r = R["ai"], R["wide"], R["rob"], R["read"]
    v = {
        "unique": count(rob["identity"]["unique"]),
        "overlap": count(rob["identity"]["overlap"]),
        "ai_recent_n": count(R["explore"]["n"]),
        "wide_recent_n": count(w["recent_n"]),
    }
    v["corpus_table"] = table(
        ["Параметр", "ИИ/МЛ", "Пять хабов"],
        [
            ["Записей после фильтров", count(a["n_loaded"]), count(w["n_loaded"])],
            [
                "Фактический период",
                "Март 2008 — ноябрь 2026",
                "Ноябрь 2024 — ноябрь 2026, неполное покрытие",
            ],
            [
                "Зрелых статей во всей выгрузке",
                count(a["n_mature"]),
                count(w["n_mature"]),
            ],
        ],
    )
    # Доля переводов во всей выгрузке ИИ/МЛ; в широком корпусе признака нет.
    tr = pd.read_csv(DATA / "part2_ai.csv", usecols=["is_translation"])
    n_tr = int(tr.is_translation.sum())
    v["tr_corpus_n_ai"] = (
        count(n_tr) + " " + plural(n_tr, "запись", "записи", "записей")
    )
    v["tr_corpus_share_ai"] = f(100 * tr.is_translation.mean(), 1) + "%"
    tr_we = rob["corpora"]["ai"]["translations_weekend"]
    if tr_we.get("available"):
        v["tr_share_ai"] = f(100 * tr_we["share_total"], 1) + "%"
    # Порог зрелости (шаг 14): диапазон отношения «выходные / будни» при
    # порогах 7–90 дней.
    for c in CORPORA:
        ratios = [x["weekend_ratio"] for x in R["extra"]["maturity_thresholds"][c]]
        v["mat_" + c] = f(min(ratios)) + "–" + f(max(ratios))
    # Крупные источники публикаций (шаг 14): диапазон отношения без самых
    # активных авторов и блогов, отдельно личные и корпоративные блоги.
    for c in CORPORA:
        tp = R["extra"]["top_producers"][c]
        cut = [
            tp[k]
            for k in (
                "without_top10_authors",
                "without_top1pct_authors",
                "without_top10_blogs",
            )
        ]
        assert all(x["ci95_author"][0] > 1 for x in cut), "интервал задел единицу"
        assert tp["personal_only"]["ci95_author"][0] > 1, "интервал задел единицу"
        # Шаблон говорит, что у корпоративных разница больше, а интервал шире.
        pers, corp = tp["personal_only"], tp["corporate_only"]
        assert corp["weekend_ratio"] > pers["weekend_ratio"]
        assert np.diff(corp["ci95_author"])[0] > np.diff(pers["ci95_author"])[0]
        ratios = [x["weekend_ratio"] for x in cut]
        v["top_range_" + c] = f(min(ratios)) + "–" + f(max(ratios))
        v["top1pct_n_" + c] = count(tp["n_top1pct_authors"])
        v["top_pers_" + c] = (
            f(tp["personal_only"]["weekend_ratio"])
            + " "
            + interval(tp["personal_only"]["ci95_author"])
        )
        v["top_corp_" + c] = (
            f(tp["corporate_only"]["weekend_ratio"])
            + " "
            + interval(tp["corporate_only"]["ci95_author"])
        )
    # Текстовая выборка: 1600 записей → разные статьи → статьи в анализе.
    # Разбивку исключённых (77 моложе 30 дней + 2 текста короче 30 слов)
    # шаблон называет явно, поэтому она сверяется с итогами шагов 06 и 07.
    tx = R["text"]
    by_c = r["adjusted_by_corpus"]["oborneva"]
    assert tx["n_unique_ids"] - r["n"] == 77 + 2, "изменилась цепочка текстовой выборки"
    v.update(
        {
            "read_cache_n": str(tx["n_cache_records"]),
            "read_unique_n": str(tx["n_unique_ids"]),
            "read_dup_n": str(tx["n_cache_records"] - tx["n_unique_ids"]),
            "read_n": count(r["n"]),
            "read_n_ai": str(by_c["ai"]["n"]),
            "read_n_wide": str(by_c["wide"]["n"]),
        }
    )
    return v


# --- Раздел 6. Выходные, часы и проверки ----------------------------------------


def values_weekend(R):
    a, w, rob, e = R["ai"], R["wide"], R["rob"], R["explore"]
    v = {
        "age_ai": f(rob["corpora"]["ai"]["age_vpd_full"]["rho"]),
        "age_wide": f(rob["corpora"]["wide"]["age_vpd_full"]["rho"]),
    }
    v["weekend_table"] = table(
        [
            "Корпус",
            "Медиана в выходные",
            "Медиана в будни",
            "Отношение",
            "95% ДИ по авторам",
            "95% ДИ по неделям",
        ],
        [
            [
                LABELS[c],
                f(s["weekend"]["median_weekend"], 1),
                f(s["weekend"]["median_weekday"], 1),
                f(s["weekend"]["ratio"]),
                interval(s["weekend"]["ratio_ci_author"]),
                interval(s["weekend"]["ratio_ci_week"]),
            ]
            for c, s in [("ai", a), ("wide", w)]
        ],
    )

    # Часы (рисунок 2): дневные 9–15 и вечерние 16–22 часа, самый высокий столбец.
    by_hour = {x["hour"]: x for x in e["by_hour"]}
    day = [by_hour[h]["vpd"] for h in range(9, 16)]
    eve = [by_hour[h]["vpd"] for h in range(16, 23)]
    top = max(e["by_hour"], key=lambda x: x["vpd"])
    v.update(
        {
            "hour_day_min": f(min(day), 0),
            "hour_day_max": f(max(day), 0),
            "hour_eve_min": f(min(eve), 0),
            "hour_eve_max": f(max(eve), 0),
            "hour_top": str(top["hour"]),
            "hour_top_n": str(top["n"]),
        }
    )

    # Тепловая карта (рисунок 3): две самые яркие ячейки и доля выходных
    # ячеек выше медианы будних ячеек.
    heat = e["heatmap"]
    dow_names = [
        "понедельник",
        "вторник",
        "среда",
        "четверг",
        "пятница",
        "суббота",
        "воскресенье",
    ]

    def heat_cell(c):
        return (
            f"{dow_names[c['dow']]}, {c['hour']}:00 ({c['n']} "
            + plural(c["n"], "статья", "статьи", "статей")
            + ")"
        )

    v.update(
        {
            "heat_top1": heat_cell(heat["top"][0]),
            "heat_top2": heat_cell(heat["top"][1]),
            "heat_cells": str(heat["n_cells"]),
            "heat_we_cells": str(heat["weekend_cells"]),
            "heat_we_above": str(heat["weekend_cells_above_weekday_median"]),
        }
    )

    # Автор против самого себя (таблица 5).
    v["paired_table"] = table(
        ["Корпус", "Авторов", "Медиана авторских отношений", "95% ДИ"],
        [
            [
                LABELS[c],
                s["paired_authors"]["n_authors"],
                f(s["paired_authors"]["median_ratio"]),
                interval(s["paired_authors"]["ci95"]),
            ]
            for c, s in [("ai", a), ("wide", w)]
        ],
    )

    # Возрастные группы (таблица 7). Границы включительные: 30–89, 90–179,
    # 180–364 дня.
    bands = {c: rob["corpora"][c]["age_bands"] for c in CORPORA}
    v["age_table"] = table(
        ["Возраст", "ИИ/МЛ, выходные/будни", "Пять хабов, выходные/будни"],
        [
            [
                f"{x['age_min']}–{x['age_max_exclusive'] - 1} "
                + plural(x["age_max_exclusive"] - 1, "день", "дня", "дней"),
                f(x["ratio"]),
                f(y["ratio"]),
            ]
            for x, y in zip(bands["ai"], bands["wide"])
        ],
    )
    for c in CORPORA:
        v["rank_we_" + c] = f(rob["corpora"][c]["cohort_rank_median_weekend"], 3)
        v["rank_wd_" + c] = f(rob["corpora"][c]["cohort_rank_median_weekday"], 3)

    # Переводы и выходные: только ИИ/МЛ.
    tr = rob["corpora"]["ai"]["translations_weekend"]
    if tr.get("available"):
        v["tr_ratio_all_ai"] = f(tr["weekend_ratio_all"])
        v["tr_ratio_excl_ai"] = f(tr["weekend_ratio_excl_translations"])
        v["tr_rank_we_ai"] = f(tr["cohort_rank_weekend_excl"], 2)
        v["tr_rank_wd_ai"] = f(tr["cohort_rank_weekday_excl"], 2)
        v["tr_ratio_tr_only_ai"] = (
            f(tr["weekend_ratio_translations_only"])
            if "weekend_ratio_translations_only" in tr
            else "—"
        )

    # Регрессия (таблица 8): с плотностью в ковариатах и без неё.
    def adj_text(x):
        return (
            f(x["ratio_target_plus1"]) + " (95% ДИ " + interval(x["ratio_ci95"]) + ")"
        )

    v["adjusted_table"] = table(
        ["Корпус", "С плотностью в ковариатах", "Без плотности"],
        [
            [
                LABELS[c],
                adj_text(rob["corpora"][c]["adjusted_weekend"]),
                adj_text(rob["corpora"][c]["adjusted_weekend_no_density"]),
            ]
            for c in CORPORA
        ],
    )

    # Другие счётчики (таблица 9) и рейтинг.
    oc = {c: rob["corpora"][c]["alternative_outcomes"] for c in CORPORA}

    def adj_cell(x):
        return f(x["ratio_target_plus1"]) + " " + interval(x["ratio_ci95"])

    v["outcomes_table"] = table(
        [
            "Исход",
            "ИИ/МЛ: условное отношение (95% ДИ)",
            "Пять хабов: условное отношение (95% ДИ)",
        ],
        [
            ["Просмотры (log10(1 + views))"]
            + [adj_cell(rob["corpora"][c]["adjusted_weekend"]) for c in CORPORA],
            ["Комментарии (log10(1 + comments))"]
            + [adj_cell(oc[c]["comments"]) for c in CORPORA],
            ["Закладки (log10(1 + favorites))"]
            + [adj_cell(oc[c]["favorites"]) for c in CORPORA],
        ],
    )
    # Рейтинг дискретен и его медианы в выходные и будни совпадают, поэтому
    # медиана печатается одна и целым числом (если она одинакова в корпусах —
    # одним числом), а сравниваются доли рейтинга ≥ 10.
    med = {c: f(oc[c]["score"]["median_weekend"], 0) for c in CORPORA}
    v["score_median"] = (
        med["ai"] + " в обоих корпусах"
        if med["ai"] == med["wide"]
        else med["ai"] + " в ИИ/МЛ и " + med["wide"] + " в пяти хабах"
    )
    for c in CORPORA:
        s = oc[c]["score"]
        v["score_ge10_" + c] = (
            f(100 * s["share_ge10_weekend"], 1)
            + "%"
            + " против "
            + f(100 * s["share_ge10_weekday"], 1)
            + "%"
        )
    return v


# --- Раздел 6. Автор: «когда» или «кто» -------------------------------------------


def values_authors(R):
    rob = R["rob"]
    sl = {c: rob["corpora"][c]["author_slot"] for c in CORPORA}
    cells = {
        c: {(x["tercile"], x["weekend"]): x for x in sl[c].get("cells", [])}
        for c in CORPORA
    }

    def median_k(c, t, weekend):
        x = cells[c].get((t, weekend))
        if x and x.get("views_median") is not None:
            return f(x["views_median"] / 1000, 1)
        return "—"

    v = {}
    v["author_slot_table"] = table(
        [
            "Группа статей по результативности других публикаций автора",
            "ИИ/МЛ: будни",
            "ИИ/МЛ: выходные",
            "Пять хабов: будни",
            "Пять хабов: выходные",
        ],
        [
            [lab] + [median_k(c, t, wk) for c in CORPORA for wk in (False, True)]
            for t, lab in [(0, "Нижняя"), (1, "Средняя"), (2, "Верхняя")]
        ],
    )

    def ratios_text(s, key):
        x = s.get("ratios", {}).get(key)
        if not x:
            return "—"
        if isinstance(x, dict):
            return ", ".join(f"{k} {f(val)}" for k, val in x.items())
        return f(x)

    for c in CORPORA:
        s = sl[c]
        disp = rob["corpora"][c]["views_dispersion"]
        shares = s.get("variance_shares", {})
        v.update(
            {
                "author_slot_weekend_" + c: ratios_text(s, "weekend_vs_weekday"),
                "author_slot_effect_" + c: ratios_text(s, "strong_vs_weak_weekday"),
                "author_slot_authors_" + c: count(s.get("n_authors", 0)),
                "authors_span_" + c: count(s.get("n_authors_spanning_groups", 0)),
                # Приросты R² сверх возраста: флаг выходных и сила автора.
                "slot_var_" + c: f(100 * shares.get("weekend_over_age", 0.0), 1),
                "author_var_" + c: f(100 * shares.get("author_loo_over_age", 0.0), 0),
                # Разброс просмотров после учёта возраста.
                "spread_q_" + c: f(disp["p75_p25_quantile_ratio"], 1),
                "spread_d_" + c: f(disp["p90_p10_quantile_ratio"], 1),
            }
        )
    # Доля выходных статей выше медианы будних — в тексте только для ИИ/МЛ.
    disp_ai = rob["corpora"]["ai"]["views_dispersion"]
    v["we_above_wd_ai"] = (
        f(100 * disp_ai["weekend_share_above_weekday_median"], 0) + "%"
    )
    return v


# --- Раздел 7. Плотность ленты ----------------------------------------------------


def values_density(R):
    a, w, d = R["ai"], R["wide"], R["density"]
    v = {}
    v["density_table"] = table(
        ["Уровень сравнения", "ИИ/МЛ, ρ", "Пять хабов, ρ"],
        [
            [
                "Семь агрегатов по дням недели",
                f(a["competition_day"]["rho"]),
                f(w["competition_day"]["rho"]),
            ],
            [
                "Ячейки день × час, n ≥ 20",
                f(a["competition_cell"]["rho"]),
                f(w["competition_cell"]["rho"]),
            ],
            [
                "Статьи в последнем годовом окне",
                f(d["ai"]["rho"]["rho"]),
                f(d["wide"]["rho"]["rho"]),
            ],
            [
                "Медиана корреляций внутри месяцев",
                f(d["ai"]["median_rho_within_month"]),
                f(d["wide"]["median_rho_within_month"]),
            ],
            [
                "Плотность следующих 24 часов (t, t+24 ч), статьи года",
                f(d["ai"]["rho_fwd"]["rho"]),
                f(d["wide"]["rho_fwd"]["rho"]),
            ],
        ],
    )
    for c in CORPORA:
        v["rho_fwd_" + c] = f(d[c]["rho_fwd"]["rho"])

    # Разброс внутри будней: почему корреляция по семи дням почти вся
    # держится на разнице будней и выходных.
    wsp = a.get("weekday_day_spread", {})
    v.update(
        {
            "wd_n_min": count(wsp.get("n_min", 0)),
            "wd_n_max": count(wsp.get("n_max", 0)),
            "wd_n_span": f(wsp.get("n_span_pct", 0), 0),
            "wd_vpd_min": f(wsp.get("vpd_min", 0), 1),
            "wd_vpd_max": f(wsp.get("vpd_max", 0), 1),
        }
    )

    # Квинтили плотности (рисунок 5): в скольких нижних квинтилях интервал
    # целиком выше единицы, сколько выходных статей в двух верхних и отношения
    # в самом плотном.
    num_loc = {0: "нуле", 1: "одном", 2: "двух", 3: "трёх", 4: "четырёх", 5: "пяти"}

    def n_sure(c):
        return sum(1 for q in d[c]["quintiles"] if q["ci95_author"][0] > 1)

    if n_sure("ai") == n_sure("wide"):
        v["q_sure_text"] = f"в {num_loc[n_sure('ai')]} нижних квинтилях обоих корпусов"
    else:
        v["q_sure_text"] = (
            f"в {num_loc[n_sure('ai')]} нижних квинтилях ИИ/МЛ "
            f"и в {num_loc[n_sure('wide')]} — пяти хабов"
        )
    top_we = [q["n_weekend"] for c in CORPORA for q in d[c]["quintiles"] if q["q"] >= 4]
    v["q_top_we_min"] = str(min(top_we))
    v["q_top_we_max"] = str(max(top_we))
    v["q5_ratio_ai"] = f(d["ai"]["quintiles"][4]["ratio"])
    v["q5_ratio_wide"] = f(d["wide"]["quintiles"][4]["ratio"])
    we_by_q = {x["q"]: x.get("n_weekend", 0) for x in d["ai"]["quintiles"]}
    v["we_q_all_ai"] = ", ".join(str(we_by_q.get(q, 0)) for q in range(1, 6))
    return v


# --- Раздел 7. Праздники -------------------------------------------------------------


def values_holidays(R):
    a, w = R["ai"], R["wide"]
    hol = {"ai": a["holiday"], "wide": w["holiday"]}
    v = {}

    # Сводные таблицы 11 и 12. Рядом с числом праздничных будней — число
    # эпизодов и контрольных недель, по которым считается интервал.
    def lead_cells(label, hb):
        eb = hb["episode_bootstrap"]
        return [
            label,
            str(hb["holiday_days"]),
            f"{eb['n_episodes_with_articles']} / {eb['n_control_weeks']}",
        ]

    def pct_cells(hb):
        return [
            f(hb["pct_holiday"], 2) + " / " + f(hb["pct_control"], 2),
            f(hb["pct_diff_pp"], 1) + " " + interval(hb["episode_bootstrap"]["ci95"], 1),
            f(hb["pct_superiority"], 2),
        ]

    def flow_cells(hb):
        return [
            f(hb["vpd_holiday"], 1) + " / " + f(hb["vpd_control"], 1),
            f(hb["volume_holiday"], 1) + " / " + f(hb["volume_control"], 1),
            f(100 * hb["corp_share_holiday"], 1)
            + " против "
            + f(100 * hb["corp_share_control"], 1),
        ]

    row_label = {"ai": "ИИ/МЛ с 2016 года", "wide": "Пять хабов в общем окне"}
    v["holiday_table_pct"] = table(
        [
            "Корпус",
            "Нерабочих будней",
            "Эпизодов / недель контроля",
            "Перцентиль праздник / обычный день",
            "Разница медиан, п.п. (95% ДИ)",
            "Вероятность превосходства",
        ],
        [lead_cells(row_label[c], hol[c]) + pct_cells(hol[c]) for c in CORPORA],
    )
    v["holiday_table_flow"] = table(
        [
            "Корпус",
            "VPD праздник / обычный день",
            "Статей в день праздник / обычный день",
            "Корпоративных в праздники, %",
        ],
        [[row_label[c]] + flow_cells(hol[c]) for c in CORPORA],
    )

    # Отношение VPD и падение числа статей в праздники — диапазоном по корпусам.
    vpd_r = [hol[c]["vpd_ratio"] for c in CORPORA]
    vol_r = [hol[c]["volume_ratio"] for c in CORPORA]
    v["tldr_holiday_vpd"] = f(min(vpd_r), 1) + "–" + f(max(vpd_r), 1)
    v["tldr_holiday_volume"] = f(1 / max(vol_r), 1) + "–" + f(1 / min(vol_r), 1)

    # Значимость сводного сравнения: одна строка, если оба p < 0.001.
    p_ai, p_wide = hol["ai"]["pct_p_iid"], hol["wide"]["pct_p_iid"]
    if p_ai < 0.001 and p_wide < 0.001:
        v["holiday_p_summarize"] = "p < 0.001"
    else:
        v["holiday_p_summarize"] = p_text(p_ai) + " и " + p_text(p_wide)

    for c in CORPORA:
        hb = hol[c]
        eb = hb["episode_bootstrap"]
        sen = hb["summary_excl_ny"]
        mp = hb["month_position"]
        # Интервал по перевыборке праздничных эпизодов и контрольных недель.
        # Новый год и сравнение без него.
        v["holiday_diff_" + c] = f(hb["pct_diff_pp"], 1)
        v["holiday_excl_ny_" + c] = f(sen["pct_diff_pp"], 1)
        v["ny_share_" + c] = f(100 * sen["new_year_share_of_holiday_articles"], 1) + "%"
        v["ny_tr_share_" + c] = f(100 * sen["new_year_episode_share"], 1) + "%"
        # Положение в месяце: праздники в начале месяцев, поэтому контроль —
        # будни в окне ±10 дней от праздников.
        v["pos10_holiday_" + c] = f(100 * mp["first_third_share_holiday"], 0) + "%"
        if mp["window10_pct_diff_pp"] is not None:
            v["holiday_w10_" + c] = (
                f(mp["window10_pct_diff_pp"], 1)
                + " п.п. ("
                + p_text(mp["window10_p_iid"], threshold=0.01)
                + ")"
            )
        else:
            v["holiday_w10_" + c] = "—"
        # Возраст праздничных и контрольных статей.
        v["hol_age_" + c] = f(hb["age_median_holiday"], 0)
        v["hol_age_ctrl_" + c] = f(hb["age_median_control"], 0)

    # Доля контрольных статей в первые десять дней месяца: одна цифра, если
    # в корпусах она совпадает после округления.
    ca = f(100 * hol["ai"]["month_position"]["first_third_share_control"], 0)
    cw = f(100 * hol["wide"]["month_position"]["first_third_share_control"], 0)
    v["pos10_control"] = ca + "%" if ca == cw else ca + "% против " + cw + "%"

    # Корпоративные и личные статьи по отдельности.
    def by_type(hb):
        corp, pers = hb["by_type_corporate"], hb["by_type_personal"]
        return (
            f"корпоративные {f(corp['pct_holiday'], 2)} против {f(corp['pct_control'], 2)} "
            f"({count(corp['n_holiday'])} "
            f"{plural(corp['n_holiday'], 'статья', 'статьи', 'статей')}), "
            f"личные {f(pers['pct_holiday'], 2)} против "
            f"{f(pers['pct_control'], 2)} ({count(pers['n_holiday'])} "
            f"{plural(pers['n_holiday'], 'статья', 'статьи', 'статей')})"
        )

    for c in CORPORA:
        v["holiday_by_type_" + c] = by_type(hol[c])

    # Дни указов 2020–2021 годов: отдельная проверка со своими контролями.
    dec = hol["ai"].get("decree_days_check")
    v["decree_days_ai"] = count(dec["n_days"]) if dec else "—"
    v["decree_n_ai"] = count(dec["n_articles"]) if dec else "—"
    v["decree_pct_ai"] = (
        f(dec["pct_median"], 2) + " против " + f(dec["pct_control"], 2) if dec else "—"
    )
    v["decree_p_ai"] = ("p = " + f(dec["p_iid"], 2)) if dec else "—"

    # Праздничные блоки (таблица 13).
    v["holiday_blocks_table"] = blocks_table(a, w)

    def block_pp(name):
        b = next((x for x in a["holiday_blocks"] if x["block"] == name), None)
        if b is None or b.get("diff_pp") is None:
            return "—"
        return f(b["diff_pp"], 1) + " п.п."

    v["block_transfer_ai"] = block_pp("Переносы")
    v["block_ny_ai"] = block_pp("Новый год")
    # Крайние значения среди отдельных праздников (без двух крупных блоков).
    single = [
        x
        for x in a["holiday_blocks"]
        if x["block"] not in ("Переносы", "Новый год") and x.get("diff_pp") is not None
    ]
    lo = min(single, key=lambda x: x["diff_pp"])
    hi = max(single, key=lambda x: x["diff_pp"])
    assert lo["diff_pp"] < 0 < hi["diff_pp"], "шаблон говорит о разбросе через ноль"
    v["block_min_ai"] = f(lo["diff_pp"], 1) + " п.п."
    v["block_min_name_ai"] = lo["block"]
    v["block_max_ai"] = f(hi["diff_pp"], 1) + " п.п."
    v["block_max_name_ai"] = hi["block"]
    return v


def blocks_table(a, w):
    """Одна таблица праздничных блоков на оба корпуса: строки — блоки, пары
    столбцов — корпуса. Даты — по полному окну ИИ/МЛ (окно пяти хабов — его
    часть). Для блока из одного календарного события интервал не показывается:
    он держался бы на одном событии."""
    names = sorted(
        {b["block"] for b in a["holiday_blocks"]}
        | {b["block"] for b in w["holiday_blocks"]}
    )

    def cells(b):
        if b is None:
            return ["—", "—"]
        ci = b.get("diff_pp_ci_author")
        single = b.get("n_events", 99) <= 1
        ci_text = (
            "—" if single else (interval(ci, 1) if ci and ci[0] is not None else "—")
        )
        diff = (
            f(b["diff_pp"], 1) + " " + ci_text if b.get("diff_pp") is not None else "—"
        )
        return [count(b["n_articles"]), diff]

    rows = []
    for name in names:
        b1 = next((x for x in a["holiday_blocks"] if x["block"] == name), None)
        b2 = next((x for x in w["holiday_blocks"] if x["block"] == name), None)
        rows.append([name, fmt_dates((b1 or b2)["dates"])] + cells(b1) + cells(b2))
    return table(
        [
            "Блок",
            "Даты (окно зрелых статей ИИ/МЛ)",
            "Статей ИИ/МЛ",
            "Разница медиан, п.п. ИИ/МЛ (95% ДИ)",
            "Статей 5 хабов",
            "Разница медиан, п.п. 5 хабов (95% ДИ)",
        ],
        rows,
    )


# --- Раздел 8. Читаемость --------------------------------------------------------------


def values_readability(R):
    r = R["read"]
    metrics = {x["metric"]: x for x in r["table"]}
    names = {
        "oborneva": "ИУФО",
        "flesch_web": "Флеш 1.3/60.1",
        "flesch_en": "Флеш англ.",
        "fk_mod": "ФК МОД",
        "asl": "Длина предложения",
        "long_word_share": "Доля длинных русских слов",
        "readingTime": "Время чтения",
        "code_char_share": "Доля кода",
    }
    v = {
        "read_rho": f(metrics["oborneva"]["rho"]),
        "long_ws_rho": f(metrics["long_word_share"]["rho"], 2),
        "read_ai_rho": f(r["subgroups"]["ai"]["rho"]),
        "read_wide_rho": f(r["subgroups"]["wide"]["rho"]),
        "read_adjusted": f(r["adjusted"]["partial_rank_correlation"]),
        "auc75": f(r["auc"]["0.75"]["auc"], 3),
        "auc90": f(r["auc"]["0.9"]["auc"], 3),
        "decile_ratio": f(r["deciles"][-1]["median"] / r["deciles"][0]["median"], 2),
        "words_rt_rho": f(r["words_reading_time"]["rho"]),
        "words_per_minute": f(r["words_per_minute"], 1),
        # Та же связь, но с перцентилем просмотров внутри месяца вместо VPD.
        "read_pct_all": f(r["percentile_metric"]["all"], 3),
        "read_pct_ai": f(r["percentile_metric"]["by_corpus"]["ai"], 3),
        "read_pct_wide": f(r["percentile_metric"]["by_corpus"]["wide"], 3),
    }
    v["readability_table"] = table(
        ["Метрика", "ρ с VPD", "95% ДИ по авторам", "n"],
        [
            [
                name,
                f(metrics[k]["rho"], 3),
                interval(metrics[k]["ci95_author"]),
                metrics[k]["n"],
            ]
            for k, name in names.items()
        ],
    )
    v["preprocessing_table"] = table(
        [
            "Индекс",
            "ρ без односимвольных слов, regex-сегментация",
            "ρ основной подготовки",
            "Изменение (основная − упрощённая)",
        ],
        [
            [
                names[x["metric"]],
                f(x["rho_old"], 3),
                f(x["rho_new"], 3),
                f(x["delta"], 3),
            ]
            for x in r["preprocessing_comparison"]["original_full"]
            if x["metric"] in names
        ],
    )
    # Частные корреляции по корпусам с теми же поправками, что и общая.
    for metric, tag in [("oborneva", "read_adj"), ("asl", "read_adj_asl")]:
        for c in CORPORA:
            v[tag + "_" + c] = f(
                r["adjusted_by_corpus"][metric][c]["partial_rank_correlation"], 3
            )
    # Календарные поправки (шаг 14): год и месяц года вместо отдельного
    # индикатора на каждый месяц, и вариант без календаря.
    rc = R["extra"]["readability_calendar"]["oborneva"]
    assert (
        abs(
            rc["ai"]["month"]["partial_rank_correlation"]
            - r["adjusted_by_corpus"]["oborneva"]["ai"]["partial_rank_correlation"]
        )
        < 1e-9
    ), "шаг 14 не воспроизводит поправку шага 07"
    for c in CORPORA:
        ym = rc[c]["year_and_month_of_year"]
        v["readcal_" + c] = (
            f(ym["partial_rank_correlation"]) + " " + interval(ym["ci95_author"])
        )
        v["readcal_none_" + c] = f(rc[c]["none"]["partial_rank_correlation"])
    # Шаблон говорит, что интервалы корпусов частично перекрываются.
    lo_ai, hi_ai = rc["ai"]["year_and_month_of_year"]["ci95_author"]
    lo_w, hi_w = rc["wide"]["year_and_month_of_year"]["ci95_author"]
    assert lo_ai < hi_w and hi_ai > hi_w and lo_ai > lo_w, "изменилось перекрытие"
    v["readcal_months_ai"] = count(rc["ai"]["n_months"])
    v["readcal_params_ai"] = count(rc["ai"]["month"]["n_parameters"])
    # Иллюстрации (шаг 14): число картинок на тысячу слов и VPD.
    im = R["extra"]["images"]
    v.update(
        {
            "img_median": f(im["median_images"], 0),
            "img_none_share": f(100 * im["share_without_images"], 0) + "%",
            "img_rho": f(im["rho_per_1000_words"], 2),
            "img_ci": interval(im["rho_per_1000_words_ci_author"]),
            "img_many_n": count(im["n_more_than_20"]),
            "img_many_vpd": f(im["vpd_median_more_than_20"], 1),
            "img_rest_vpd": f(im["vpd_median_rest"], 1),
            "img_many_words": f(im["words_median_more_than_20"], 0),
            "img_rest_words": f(im["words_median_rest"], 0),
        }
    )
    # Переводы (только ИИ/МЛ).
    tr = r["translations"]["ai"]
    v["read_rho_excl_ai"] = f(tr["rho_excl_translations"], 2)
    v["read_rho_tr_only_ai"] = (
        f(tr["rho_translations_only"], 2)
        if tr["rho_translations_only"] is not None
        else "—"
    )
    # Длина текста и просмотры.
    ln = r["length"]
    v["len_rho"] = f(ln["rho_words_vpd"], 2)
    v["len_rho_adj"] = f(ln["rho_words_vpd_partial_age_corp"], 2)
    v["len_quartiles"] = " → ".join(f(x["vpd_median"], 1) for x in ln["quartiles"])
    v["len_n_short"] = count(ln["n_words_below_500"])
    v["len_rho_short"] = (
        f(ln["rho_below_500"], 2) if ln["rho_below_500"] is not None else "—"
    )
    return v


# --- Разделы 9–11. Модель, сетка слотов, утечка ----------------------------------------


LOSS_LABELS = {
    "L01_rmse_log1p": "RMSE от log1p",
    "L02_quantile_views_1000": "Quantile(0.5) от просмотров",
    "L03_poisson_views_1000": "Poisson",
    "L04_huber_log1p": "Huber от log1p",
    "L05_quantile_log1p": "Quantile(0.5) от log1p",
}


def signed(x, n=3):
    return ("+" if x > 0 else "") + f(x, n)


def delta_cell(x, ci):
    return signed(x) + " " + interval(ci, 3)


def values_model(R):
    """Разделы 9–10 (эксперимент CatBoost) и таблица 19 (утечка, шаг 11)."""
    ml, leak = R["ml"], R["leak"]
    winner = ml["selected"]["winner_by_corpus"]
    v = {}

    # Таблица 17: корреляция внутри месяцев на валидационной выборке.
    def val_rho(c, model):
        return ml["loss"][f"{c}:{model}"]["validation"]["rho_within_month_mean"]

    rows = []
    for c in CORPORA:
        pois, log = val_rho(c, "lgb_poisson_v24"), val_rho(c, "lgb_rmse_log1p")
        cb = val_rho(c, winner[c])
        assert cb > log > pois, "текст раздела 9 опирается на этот порядок"
        if c == "wide":  # в тексте разбирается только пример пяти хабов
            v["ml_lgb_pois_" + c] = f(pois, 3)
            v["ml_lgb_log_" + c] = f(log, 3)
        v["ml_cb_" + c] = f(cb, 3)
        rows.append(
            [LABELS[c], f(pois, 3), f(log, 3), f(cb, 3), LOSS_LABELS[winner[c]]]
        )
    v["ml_val_table"] = table(
        [
            "Корпус",
            "LightGBM, Poisson",
            "LightGBM, RMSE от log1p",
            "Выбранная CatBoost",
            "Функция потерь выбранной CatBoost",
        ],
        rows,
    )

    # Калибровка уровня на итоговой проверке и число деревьев.
    for c in CORPORA:
        cal = ml["calibration"]["models"][f"{c}:{winner[c]}"]
        assert cal["mae_log"] < cal["baseline_mae_log"]
        v["ml_overpred_" + c] = f(cal["median_pred_fact_ratio"])
        v["ml_mae_" + c] = f(cal["mae_log"], 3)
        v["ml_mae_base_" + c] = f(cal["baseline_mae_log"], 3)
        cfg = ml["selected"]["frozen_model_configs"][f"{c}:{winner[c]}"]
        v["ml_trees_" + c] = count(cfg["n_trees_fixed_from_validation"])
    meta = ml["loss"]["ai:" + winner["ai"]]
    v["ml_ntrain_ai"] = count(
        meta["validation"]["n_total"]
        + json.loads(
            (ML / "metrics" / f"loss_ai_{winner['ai']}.json").read_text(
                encoding="utf-8"
            )
        )["metadata"]["n_train"]
    )

    # Таблица 18: режимы A / B1 / B2 на одной и той же выборке.
    modes_rows = {"A": [], "B1": [], "B2": []}
    for c in CORPORA:
        rep = ml["policy"][c][winner[c] + "|density=none"]
        a = rep["task_A_full_prediction"]
        b1 = rep["task_B1_fixed_profile_time_score"]["novice"]
        b2 = rep["task_B2_percentile"]
        assert rep["n"] == b1["n"] == b2["n"], "режимы считаются на разных выборках"
        v["ml_n_" + c] = count(rep["n"])
        modes_rows["A"].append(delta_cell(a["delta_rho"], a["delta_rho_ci_author"]))
        modes_rows["B1"].append(delta_cell(b1["delta_rho"], b1["delta_rho_ci_author"]))
        modes_rows["B2"].append(
            delta_cell(b2["delta_rho_views"], b2["delta_rho_ci_author_views"])
        )
        v["ml_b1_" + c] = signed(b1["delta_rho"])
        v["ml_b1_pct_" + c] = signed(b1["age_controlled"]["delta_rho_pct"])
        v["ml_b2_" + c] = signed(b2["delta_rho_views"])
        for scale in ("abs", "rel"):
            s = rep["task_B2_absolute" if scale == "abs" else "task_B2_relative"]
            v[f"ml_b2_{scale}_{c}"] = delta_cell(
                s["delta_rho_views"], s["delta_rho_ci_author_views"]
            )
        wd = b1["weekday_only"]
        v["ml_wd_rho_" + c] = f(wd["weekday_only_rho"])
        v["ml_wd_ratio_" + c] = f(wd["weekday_only_hi_lo"])
        assert wd["weekday_only_rho"] > 0.05, "текст: рейтинг различает будние часы"
    v["ml_modes_table"] = table(
        ["Режим", "ИИ/МЛ: Δρ (95% ДИ)", "Пять хабов: Δρ (95% ДИ)"],
        [
            ["A. Полный прогноз популярности"] + modes_rows["A"],
            ["B1. Фиксированный профиль статьи"] + modes_rows["B1"],
            ["B2. Личный рейтинг времени статьи"] + modes_rows["B2"],
        ],
    )

    # Утверждения текста о знаках интервалов.
    pr = {c: ml["policy"][c][winner[c] + "|density=none"] for c in CORPORA}
    b1_ai = pr["ai"]["task_B1_fixed_profile_time_score"]["novice"]
    b1_wide = pr["wide"]["task_B1_fixed_profile_time_score"]["novice"]
    assert b1_ai["delta_rho_ci_author"][0] < 0 < b1_ai["delta_rho_ci_author"][1]
    assert b1_wide["delta_rho_ci_author"][0] > 0
    b2_wide = pr["wide"]["task_B2_percentile"]["delta_rho_ci_author_views"]
    assert b2_wide[0] > 0
    assert pr["ai"]["task_B2_absolute"]["delta_rho_ci_author_views"][1] < 0

    # Устойчивость B1: другие сиды и ранние временные разбиения.
    fin = ml["final"]
    seeds_ai = fin["ai"]["seeds"]
    main_seed = str(
        ml["selected"]["frozen_model_configs"]["ai:" + winner["ai"]]["seed"]
    )
    other = [s for k, s in seeds_ai.items() if k != main_seed]
    assert abs(seeds_ai[main_seed]["delta_rho"] - b1_ai["delta_rho"]) < 1e-9
    assert all(s["delta_rho_ci_author"][0] > 0 for s in other), (
        "текст: интервалы выше нуля"
    )
    v["ml_b1_seeds_other_ai"] = " и ".join(
        signed(s["delta_rho"]) for s in sorted(other, key=lambda s: s["delta_rho"])
    )
    assert all(s["delta_rho_ci_author"][0] > 0 for s in fin["wide"]["seeds"].values())
    folds = [x["delta_rho"] for x in fin["ai"]["folds_b1"]]
    assert min(folds) < 0, "текст: на ранних разбиениях разница отрицательная"
    v["ml_b1_folds_min_ai"] = signed(min(folds))
    v["ml_b1_folds_max_ai"] = signed(max(folds))

    # Утечка через историю автора (таблица 19).
    variants = leak["variants"]
    v["leak_n_train"] = count(leak["n_train"])
    v["leak_n_test"] = count(leak["n_test"])
    v["leak_orig"] = f(variants["original_future_outcomes"]["rho"], 3)
    v["leak_count"] = f(variants["past_count_only"]["rho"], 3)
    v["leak_none"] = f(variants["no_author"]["rho"], 3)
    v["leakage_table"] = table(
        [
            "Авторский блок при одинаковых остальных настройках",
            "Корреляция Спирмена на тесте",
        ],
        [
            [
                "Поздние результаты предыдущих статей (утечка)",
                f(variants["original_future_outcomes"]["rho"], 3),
            ],
            [
                "Только число предыдущих статей",
                f(variants["past_count_only"]["rho"], 3),
            ],
            ["Без авторского блока", f(variants["no_author"]["rho"], 3)],
        ],
    )
    return v


def values_review_robustness(R):
    """Дополнительная чувствительность из самостоятельного offline-контроля 16."""
    checks = get("16_review_robustness.json")["corpora"]
    v = {}
    for c in CORPORA:
        item = checks[c]
        age = item["age_comparison"]
        v["age_median_we_" + c] = f(age["median_weekend"], 1)
        v["age_median_wd_" + c] = f(age["median_weekday"], 1)
        v["reg_quarter_" + c] = f(item["regression"]["quarter_calendar"]["ratio_views_plus1"], 3)
        v["reg_agebands_" + c] = f(item["regression"]["age_bands"]["ratio_views_plus1"], 3)
        holiday = item["holidays"]
        v["hol_loo_min_" + c] = f(holiday["loo_min_pp"], 1)
        v["hol_loo_max_" + c] = f(holiday["loo_max_pp"], 1)
        authors = R[c]["holiday"]["pct_diff_pp_ci_author"]
        v["holiday_author_ci_" + c] = interval(authors, 1)
    return v


# --- Сборка ---------------------------------------------------------------------------


def main():
    R = load_results()
    vals = {}
    for build in (
        values_data,
        values_weekend,
        values_authors,
        values_density,
        values_holidays,
        values_readability,
        values_model,
        values_review_robustness,
    ):
        part = build(R)
        clash = vals.keys() & part.keys()
        assert not clash, f"заполнители заданы дважды: {sorted(clash)}"
        vals.update(part)

    template = (BASE / "article_template.md").read_text()
    used = set(re.findall(r"\{\{(\w+)\}\}", template))
    missing = used - vals.keys()
    if missing:
        raise KeyError(f"в шаблоне есть заполнители без значений: {sorted(missing)}")
    unused = vals.keys() - used
    if unused:
        raise KeyError(f"значения, которых нет в шаблоне: {sorted(unused)}")
    article = re.sub(r"\{\{(\w+)\}\}", lambda x: str(vals[x[1]]), template)
    # Удвоенный процент — след неудачного форматирования строки.
    if "%%" in article:
        raise ValueError("в article.md есть удвоенный знак процента «%%»")
    (BASE / "article.md").write_text(article)
    # Подписи дублируются в манифесте иллюстраций для удобства редактора.
    # Исходный текст подписей — только шаблон статьи.
    captions = ["# Подписи к рисункам", "", "Генерируется из article.md; править подписи нужно в article_template.md.", ""]
    for image, caption in re.findall(
        r"!\[[^\]]*\]\((figs/[^)]+)\)\s*\n\s*\*([^\n*]+)\*", article
    ):
        captions.append(f"{image} — {caption.strip()}")
        captions.append("")
    (BASE / "figs/captions.md").write_text("\n".join(captions).rstrip() + "\n")
    (RESULTS / "article_values.json").write_text(
        json.dumps(vals, ensure_ascii=False, indent=2) + "\n"
    )
    flat = article.replace(" ", "").replace("\n", "")
    print(
        "Article built:",
        len(article),
        "characters (с пробелами);",
        len(flat),
        "без пробелов;",
        len(used),
        "substitutions",
    )


if __name__ == "__main__":
    main()
