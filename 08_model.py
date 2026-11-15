"""Шаг 08. Ретроспективный временной holdout, не исторический backtest.

Все счётчики-цели измерены одним поздним срезом. Даже после удаления утечки
из признаков обучающие labels не были доступны в дату разделения.
Прогнозы слотов — ассоциации модели, не оценка эффекта переноса публикации.

Цель — накопленные просмотры напрямую с лог-связью (Пуассон), а не
log10(VPD+1): предсказание мультипликативно и не проходит через деление
просмотров на возраст. Возраст статьи признаком не является, поэтому уровень
просмотров в тесте модель систематически смещает; это видно в MAE. По той же
причине общий Спирмен на тесте частично измеряет возраст (в тесте ИИ/МЛ
возраст от 30 до 313 дней, и один возраст даёт с просмотрами больше, чем вся
модель), поэтому рядом публикуется Спирмен внутри месяцев теста.
"""

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from habrpub import DATA, FIGS
from habrpub.analysis import (
    WIDE_COMMON_START,
    corr,
    finite_json,
    load_corpus,
    save,
    setup_plot,
)

from habrpub.legacy_model import (
    AUTHOR,
    FULL,
    PARAMS,
    TIME,
    TIME_NO_EPOCH,
    TITLE,
    fit,
    make_grid,
    policy_groups,
    prepare,
)

# Диагностические рисунки контрольной модели не входят в текст статьи.
LEGACY = FIGS / "legacy"


def main():
    res = {
        "status": "retrospective chronological holdout; labels observed after split",
        "target": "views, Poisson log-link; metrics on log1p scale",
        "features": FULL,
        "parameters": PARAMS,
        "models": {},
        "folds": {},
    }
    plt = setup_plot()
    LEGACY.mkdir(exist_ok=True)
    for corpus, split in [("ai", "2026-01-01"), ("wide", "2026-05-01")]:
        df = prepare(load_corpus(corpus))
        eligible = df.mature & df.density_complete
        if corpus == "ai":
            eligible &= df.year >= 2016
        else:
            eligible &= df.dt_msk >= WIDE_COMMON_START
        df = df[eligible].copy()
        tr = df[df.dt_msk < split]
        te = df[df.dt_msk >= split]
        out = {
            "split_msk": split,
            "train_max": str(tr.dt_msk.max()),
            "test_max": str(te.dt_msk.max()),
        }
        # Диагностика возрастной подмены: сколько с просмотрами
        # объясняет один возраст и номер месяца публикации в тесте.
        out["rho_age_only"] = corr(te.age_days, te.views)
        out["rho_month_views"] = corr(te.month, te.views)
        out["test_age_min"] = float(te.age_days.min())
        out["test_age_max"] = float(te.age_days.max())
        out["test_age_median"] = float(te.age_days.median())
        configs = [
            ("full", FULL),
            ("no_dens_month", TIME_NO_EPOCH + TITLE + AUTHOR + ["is_corp"]),
            ("no_author", TIME + TITLE + ["is_corp"]),
            ("time_only", TIME),
        ]
        for name, features in configs:
            m, metrics = fit(tr, te, features)
            out[name] = metrics
            if name == "full":
                model = m
        grid = make_grid(model, tr)
        out["slot_diagnostic"], test_groups = policy_groups(tr, te, grid)
        grid.to_csv(DATA / f"model_grid_{corpus}.csv", index=False)
        res["models"][corpus] = out
        # Дополнительные временные границы — проверка стабильности, без подбора параметров.
        folds = []
        for cut in ["2024-01-01", "2025-01-01"] if corpus == "ai" else ["2026-01-01"]:
            c = pd.Timestamp(cut, tz="Europe/Moscow")
            end = c + pd.DateOffset(years=1)
            ftr = df[df.dt_msk < c]
            fte = df[(df.dt_msk >= c) & (df.dt_msk < end)]
            if len(ftr) > 500 and len(fte) > 100:
                _, met = fit(ftr, fte, FULL)
                folds.append({"split": cut, **met})
        res["folds"][corpus] = folds
        if corpus == "ai":
            imp = out["full"]["importance_gain"]
            items = sorted(imp.items(), key=lambda t: t[1])
            # Служебные имена признаков на рисунке заменены понятными подписями.
            ru = {
                "dow": "День недели",
                "hour": "Час",
                "hour_sin": "Час, sin",
                "hour_cos": "Час, cos",
                "month": "Месяц",
                "is_weekend": "Суббота/воскресенье",
                "is_holiday": "Нерабочий будень",
                "dens24": "Плотность за 24 ч",
                "t_len": "Длина заголовка",
                "t_words": "Слов в заголовке",
                "t_digit": "Цифра в заголовке",
                "t_qmark": "Вопрос в заголовке",
                "t_colon": "Двоеточие/тире",
                "prior_count": "Пред. публикаций (log)",
                "has_prior": "Есть история",
                "is_corp": "Корпоративная",
            }
            fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
            ax.barh(
                [ru.get(x[0], x[0]) for x in items],
                [x[1] for x in items],
                color="#2d7ab3",
            )
            ax.set(
                xlabel="Суммарный gain",
                title="Важность для модели · не причинный вклад",
            )
            fig.savefig(LEGACY / "lgbm_importance.png")
            plt.close(fig)
            pivot = grid.pivot(
                index="dow", columns="hour", values="relative_prediction"
            ).where(grid.pivot(index="dow", columns="hour", values="supported"))
            fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
            # Нейтральный цвет шкалы обязан стоять на 1,0, а не на середине
            # диапазона данных: иначе слоты чуть выше медианы
            # выглядят «холодными», то есть хуже среднего.
            im = ax.imshow(
                np.ma.masked_invalid(pivot),
                cmap="RdBu_r",
                aspect="auto",
                norm=TwoSlopeNorm(
                    vmin=min(float(np.nanmin(pivot)), 0.99),
                    vcenter=1.0,
                    vmax=max(float(np.nanmax(pivot)), 1.01),
                ),
            )
            im.cmap.set_bad("#d9d9d9")
            ax.set_yticks(range(7), ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
            ax.set(
                xlabel="Час (МСК)",
                # Корпус в заголовке и серый цвет скрытых ячеек:
                # нейтральный белый совпадал с уровнем 1,0 расходящейся шкалы.
                title="Относительный прогноз · ИИ/МЛ · n train ≥ 20; серое — нет данных",
            )
            fig.colorbar(im, ax=ax, label="Отношение к медиане прогнозов")
            fig.savefig(LEGACY / "lgbm_grid.png")
            plt.close(fig)
            g = out["slot_diagnostic"]["groups"]
            fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
            ax.bar(range(3), [r["views_median"] for r in g], color="#2d7ab3")
            ax.set_xticks(range(3), ["Низкий", "Средний", "Высокий"])
            ax.set(
                xlabel="Терциль прогноза слота; границы из train",
                ylabel="Просмотры на тесте, медиана",
                title="Наблюдательная проверка · личные статьи",
            )
            fig.savefig(LEGACY / "lgbm_slot_check.png")
            plt.close(fig)
        print(
            corpus,
            finite_json(
                {
                    k: (v.get("rho"), v.get("r2_log"))
                    for k, v in out.items()
                    if isinstance(v, dict) and "rho" in v
                }
            ),
            finite_json(out["slot_diagnostic"]),
        )
    save(res, "08_model_results.json")


if __name__ == "__main__":
    main()
