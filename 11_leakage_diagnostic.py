"""Шаг 11. Диагностика утечки через историю автора.

Воспроизводит исходную модель в трёх вариантах авторского блока, включая
вариант с утечкой (поздние результаты прошлых статей автора) — только для
сравнения, в основной модели его нет.

Гиперпараметры, выборка, split и цель одинаковы; меняется авторский блок.
Все варианты всё равно ретроспективны из-за поздно измеренных целей.
"""

import numpy as np
import pandas as pd
from habrpub.legacy_model import TITLE, prepare
from habrpub.analysis import (
    corr,
    fixed_holidays,
    load_corpus,
    save,
)
import lightgbm as lgb


def main():
    d = prepare(load_corpus("ai"))
    history = {}
    counts = []
    medians = []
    flags = []
    global_median = d.vpd.median()
    # Воспроизводим ошибочную постановку: история сортируется по времени
    # публикации, но содержит поздно измеренные счётчики просмотров.
    for row in d.itertuples():
        author = row.author if pd.notna(row.author) else "?"
        h = history.setdefault(author, [])
        past = [v for t, v in h if t < row.dt_utc]
        counts.append(np.log10(len(past) + 1))
        flags.append(int(bool(past)))
        medians.append(np.log10((np.median(past) if past else global_median) + 1))
        h.append((row.dt_utc, row.vpd))
    d["prior_cnt"] = counts
    d["prior_med_vpd"] = medians
    d["has_prior"] = flags
    d["t_len"] = d.t_len.clip(upper=200)
    d["t_words"] = d.t_words.clip(upper=40)
    # Сохраняем календарную конфигурацию контрольного сравнения: при диагностике
    # авторских признаков календарные признаки должны быть одинаковыми.
    off = fixed_holidays(d.year.unique()) | {
        pd.Timestamp(s).date()
        for s in [
            "2025-05-02",
            "2025-12-31",
            "2025-05-08",
            "2025-06-13",
            "2025-11-03",
            "2026-01-09",
            "2026-12-31",  # Контрольный список содержит также даты вне среза.
        ]
    }
    d["is_holiday"] = d.date_msk.isin(off).astype(int)
    d = d[d.mature & (d.year >= 2015)]
    tr = d[d.dt_msk < "2026-01-01"]
    te = d[d.dt_msk >= "2026-01-01"]
    time = [
        "dow",
        "hour",
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "month",
        "is_holiday",
        "dens24",
        "is_corp",
    ]
    title = TITLE
    base = time + title
    result = {
        "n_train": len(tr),
        "n_test": len(te),
        "warning": "not a prospective backtest",
        "variants": {},
    }
    for name, feats in [
        (
            "original_future_outcomes",
            time + ["prior_cnt", "prior_med_vpd", "has_prior"] + title,
        ),
        ("past_count_only", time + ["prior_cnt", "has_prior"] + title),
        ("no_author", base),
    ]:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=48,
            min_child_samples=40,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            # Сид исходной модели: шаг воспроизводит её как есть, поэтому
            # здесь не общий SEED.
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(tr[feats], tr.log_vpd, categorical_feature=["dow", "hour", "month"])
        pred = model.predict(te[feats])
        y = te.log_vpd.to_numpy()
        result["variants"][name] = {
            "rho": corr(pred, y)["rho"],
            "r2": 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2),
        }
    save(result, "11_leakage_diagnostic.json")
    print(result)


if __name__ == "__main__":
    main()
