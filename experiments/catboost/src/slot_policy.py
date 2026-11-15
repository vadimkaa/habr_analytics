"""Slot policy: протоколы D (диагностическая сетка 168) и P (policy против
weekend-правила). Режимы задачи B:
 - A full_prediction - прогноз на фактических признаках статьи (assign_scores):
   контрольная точка задачи A, НЕ доказательство выбора времени;
 - B1 fixed_profile_time_score - у фиксированных профилей статьи меняются
   только календарные признаки: рейтинг слотов отражает календарную часть
   модели сверх выходного дня (b1_fixed_profile_scores);
 - B2 personalized_scheduling - для каждой статьи свой профиль и альтернативные
   КАЛЕНДАРНО СОГЛАСОВАННЫЕ слоты реального месяца; наблюдаемого «прироста от
   переноса» не существует, поэтому B2 - диагностика рекомендательной политики
   (personalized_scheduling).

Ключевые ограничения:
 - границы терцилей, порог q и профиль сетки берутся ТОЛЬКО из train
   (test outcomes не участвуют ни в выборе границ, ни в выборе победителей);
 - dens24_actual в policy не используется: только dens24_expected из train
   (режим actual - отдельная диагностическая верхняя граница);
 - у слотов с низкой поддержкой (n_train < 20) - fallback «недостаточно
   наблюдений», а не точечный прогноз с точностью до процента;
 - P/B1/B2 - ассоциативные ретроспективные проверки, а не измерение эффекта
   переноса конкретной статьи (неизмеренные темы и качество частично входят
   в признаки статьи).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data_contract import CorpusData  # noqa: F401 (тип) — используется в аннотациях

# Признаки, приходящие из календаря публикации (слота). Всё остальное -
# характеристики статьи/автора, которые в B1 фиксируются профилем.
CALENDAR_FEATURES = {
    "dow",
    "hour",
    "hour_sin",
    "hour_cos",
    "is_weekend",
    "is_holiday",
    "month",
    "dens24_expected",
}


# ---------------------------------------------------------------------------
# Календарь кандидатов: дата-час -> признаки слота
# ---------------------------------------------------------------------------


def calendar_flags(dates, years=None) -> pd.DataFrame:
    """ЕДИНЫЙ источник календарных флагов.

    Семантика идентична признакам обучения (features.prepare, weekday_holidays):
    - is_holiday - нерабочие праздничные и перенесённые дни, выпавшие на БУДНИ
      (обычные выходные исключены: их эффект несёт is_weekend, иначе праздник
      поглотил бы эффект выходных; одноразовые нерабочие дни указов 2020-2021
      исключены так же, как при обучении);
    - is_weekend - суббота/воскресенье, не являющиеся рабочими днями по
      производственному календарю (переносы учитываются).

"""
    from habrpub.analysis import weekday_holidays, offday_sets

    d = pd.Series(list(dates)).reset_index(drop=True)
    if years is None:
        years = sorted({x.year for x in d})
    hol = weekday_holidays(years)
    _, workweekend = offday_sets(years)
    dow = d.map(lambda x: x.weekday()).astype(int)
    return pd.DataFrame(
        {
            "date": d,
            "dow": dow,
            "is_weekend": ((dow >= 5) & ~d.isin(workweekend)).astype(int),
            "is_holiday": d.isin(hol).astype(int),
        }
    )


def slot_calendar_rows(
    timestamps: pd.Series,
) -> pd.DataFrame:
    """Признаки будущих слотов из КОНКРЕТНОЙ календарной даты.

    Рабочая суббота производственного календаря - будни; перенесённый выходной
    (нерабочий будень) - праздничный флаг; месяц - целое 1-12. Медианные
    дробные месяцы не используются: календарная корректность важнее медианы.
    Флаги - через calendar_flags."""
    ts = pd.Series(pd.to_datetime(timestamps)).dt.tz_convert("Europe/Moscow")
    dates = pd.Series(ts.dt.normalize().dt.tz_localize(None).dt.date, index=ts.index)
    flags = calendar_flags(dates.tolist())
    flags.index = ts.index
    rows = pd.DataFrame(index=ts.index)
    rows["dow"] = ts.dt.dayofweek.astype(int)
    rows["hour"] = ts.dt.hour.astype(int)
    rows["month"] = ts.dt.month.astype(int)
    rows["year"] = ts.dt.year.astype(int)
    rows["is_weekend"] = flags["is_weekend"].astype(int)
    rows["is_holiday"] = flags["is_holiday"].astype(int)
    rows["date"] = dates
    return rows


def build_grid_frame(
    train: pd.DataFrame,
    month: int,
    density_col: str | None,
) -> pd.DataFrame:
    """168 слотов для фиксированного реального месяца.

    Профиль: личная статья, автор без истории, медианная длина заголовка,
    праздничный флаг выключен. Плотность - из train-only слотовых медиан."""
    p = config.protocol()["policy"]["grid_profile"]
    profile = dict(p)
    profile["t_len"] = float(train["t_len"].median())
    profile["t_words"] = float(train["t_words"].median())
    density = (
        train.groupby(["dow", "hour"])[density_col].median() if density_col else None
    )
    count = train.groupby(["dow", "hour"]).size()
    rows = []
    for d in range(7):
        for h in range(24):
            r = dict(
                profile,
                dow=d,
                hour=h,
                is_weekend=int(d >= 5),
                hour_sin=float(np.sin(2 * np.pi * h / 24)),
                hour_cos=float(np.cos(2 * np.pi * h / 24)),
                month=int(month),
            )
            if density_col:
                r[density_col] = float(density.get((d, h), np.nan))
            r["n_train"] = int(count.get((d, h), 0))
            rows.append(r)
    grid = pd.DataFrame(rows)
    if density_col:
        grid[density_col] = grid[density_col].fillna(float(train[density_col].median()))
    return grid


def score_grid(
    fitresult,
    train: pd.DataFrame,
    features: list[str],
    density_col: str | None,
    months: list[int],
) -> dict:
    """Протокол D: прогноз в 168 слотах x 4 сезона; support и относительные значения."""
    from .metrics import rho_all

    out = {"months": {}}
    rank_maps = {}
    for month in months:
        grid = build_grid_frame(train, month, density_col)
        pred = fitresult.predict_views_for(grid, features)
        grid["pred_views"] = pred
        supported = grid.n_train >= config.protocol()["policy"]["support_min_train"]
        grid["supported"] = supported
        med = float(grid.loc[supported, "pred_views"].median())
        grid["relative_prediction"] = grid["pred_views"] / med
        # Ранжирование слотов (Спирмен между сезонами) - устойчивость сетки.
        rank_maps[month] = grid.loc[supported].relative_prediction.rank()
        out["months"][int(month)] = {
            "grid": grid[
                [
                    "dow",
                    "hour",
                    "n_train",
                    "supported",
                    "pred_views",
                    "relative_prediction",
                ]
            ].to_dict("records"),
            "median_supported_pred": med,
        }
    # Согласованность ранжирования слотов между сезонами.
    keys = list(rank_maps)
    agree = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            agree[f"{a}:{b}"] = rho_all(rank_maps[a], rank_maps[b])
    out["season_rank_agreement"] = agree
    # Чувствительность порога поддержки.
    cov = {}
    for thr in (10, 20, 30, 50):
        cov[thr] = float((train.groupby(["dow", "hour"]).size() >= thr).mean())
    out["slot_coverage_by_threshold"] = cov
    return out


# ---------------------------------------------------------------------------
# Протокол P: score фактических слотов test-статей и сравнение с правилами
# ---------------------------------------------------------------------------


def assign_scores(
    fitresult,
    test: pd.DataFrame,
    train: pd.DataFrame,
    features: list[str],
    density_col: str | None,
    mode: str = "full_prediction",
) -> pd.DataFrame:
    """Режим A (full_prediction): score слота, где статья РЕАЛЬНО вышла.

    Прогноз делается на фактических признаках статьи (заголовок/автор/блог +
    её календарь); dens24_actual подменена expected (или исключена из набора) -
    будущая плотность в момент решения неизвестна. Результат - контрольная
    точка задачи A: высокое качество ранжирования может обеспечиваться
    характеристиками СТАТЬИ, а не выбором времени, поэтому как доказательство
    эффективности выбора времени не используется. Для календарной
    части используйте b1_fixed_profile_scores / personalized_scheduling."""
    if mode != "full_prediction":
        raise ValueError(
            f"неизвестный режим {mode!r}: assign_scores - только full_prediction "
            "(режим A); календарные оценки - b1_fixed_profile_scores (B1) и "
            "personalized_scheduling (B2)"
        )
    te = test.copy()
    if density_col is None and "dens24" in features:
        raise ValueError("density-free политика не может использовать dens24")
    raw = fitresult.model.predict(te[features])
    from .targets import TargetSpec

    losses_cfg = config.losses_cfg()
    loss_id = fitresult.metadata["loss_id"]
    spec = losses_cfg["losses"].get(loss_id) or losses_cfg["lightgbm_controls"][loss_id]
    target = TargetSpec(spec["target"])
    # Колонка "score" в данных Хабра - рейтинг статьи; score модели живёт
    # в отдельном поле model_score, чтобы не пересекаться.
    te["model_score"] = target.decode(raw)
    return te


# ---------------------------------------------------------------------------
# Режим B1: фиксированные профили статьи, меняется только календарь
# ---------------------------------------------------------------------------


def fixed_profiles(train: pd.DataFrame) -> dict[str, dict]:
    """Фиксированные профили статьи для B1 (значения замораживаются в
    selected_policies.json до аудита).

    Заголовочные признаки - медианы личных статей train; история автора -
    ноль (новичок), верхний квартиль личной истории (опытный) и медиана
    корпоративной истории (корп). Некалендарные признаки одинаковые внутри
    профиля, поэтому различия прогнозов между статьями могут идти только
    от календаря."""
    pers = train[train.is_corp == 0]
    t_len = float(pers["t_len"].median())
    t_words = float(pers["t_words"].median())
    exp_prior = (
        float(pers.loc[pers.has_prior == 1, "prior_count"].quantile(0.75))
        if (pers.has_prior == 1).any()
        else 0.0
    )
    corp_prior = train.loc[(train.is_corp == 1) & (train.has_prior == 1), "prior_count"]
    corp_prior = float(corp_prior.median()) if len(corp_prior) else exp_prior
    return {
        "novice": dict(
            t_len=t_len,
            t_words=t_words,
            t_digit=0,
            t_qmark=0,
            t_colon=0,
            is_corp=0,
            has_prior=0,
            prior_count=0.0,
        ),
        "experienced": dict(
            t_len=t_len,
            t_words=t_words,
            t_digit=0,
            t_qmark=0,
            t_colon=0,
            is_corp=0,
            has_prior=1,
            prior_count=exp_prior,
        ),
        "corp": dict(
            t_len=t_len,
            t_words=t_words,
            t_digit=0,
            t_qmark=0,
            t_colon=0,
            is_corp=1,
            has_prior=1,
            prior_count=corp_prior,
        ),
    }


def _profile_frame(
    frame: pd.DataFrame, features: list[str], profile: dict
) -> pd.DataFrame:
    """Календарные признаки - из строк frame; некалендарные - из профиля."""
    out = frame[list(features)].copy()
    for c in features:
        if c in CALENDAR_FEATURES:
            continue
        if c not in profile:
            raise KeyError(f"профиль B1 не фиксирует некалендарный признак {c!r}")
        out[c] = profile[c]
    return out


def b1_fixed_profile_scores(
    fitresult,
    frame: pd.DataFrame,
    features: list[str],
    profile: dict,
    score_col: str = "model_score",
) -> pd.DataFrame:
    """Режим B1: прогноз профиля в КАЛЕНДАРЕ каждой строки.

    Некалендарные признаки зафиксированы профилем, поэтому различия score между
    строками могут идти только от календарных признаков (dow/hour/weekend/
    holiday/expected-плотность). Возвращаемая рамка = frame + score_col."""
    if "dens24" in features:
        raise ValueError(
            "B1 не может использовать dens24_actual: только "
            "календарь и dens24_expected из train"
        )
    from .targets import TargetSpec

    losses_cfg = config.losses_cfg()
    loss_id = fitresult.metadata["loss_id"]
    spec = losses_cfg["losses"].get(loss_id) or losses_cfg["lightgbm_controls"][loss_id]
    target = TargetSpec(spec["target"])
    raw = fitresult.model.predict(_profile_frame(frame, features, profile))
    out = frame.copy()
    out[score_col] = target.decode(raw)
    return out


# ---------------------------------------------------------------------------
# Режим B2: персональное планирование на календарно согласованных слотах
# ---------------------------------------------------------------------------


def _canonical_month_dates(dates: pd.Series) -> dict:
    """Первые реальные даты каждого дня недели внутри месяца (ключ - (year, month)).

    Календарная согласованность B2: dow/hour слота всегда опираются на
    РЕАЛЬНУЮ дату месяца статьи, is_weekend/is_holiday берутся из
    производственного календаря этой даты. Сетка строится ИЗ КАЛЕНДАРЯ
    месяца (не из дат тестовых строк): частичная подвыборка месяца не
    может «потерять» день недели и слоты."""
    import calendar as _calendar

    out = {}
    for year, month in sorted(
        set(zip(dates.dt.year.tolist(), dates.dt.month.tolist()))
    ):
        ndays = _calendar.monthrange(int(year), int(month))[1]
        d = pd.date_range(f"{int(year)}-{int(month):02d}-01", periods=ndays, freq="D")
        first = {}
        for dd in d:
            # datetime.date (не Timestamp): множества offday_sets содержат date
            dd_date = dd.date()
            first.setdefault(dd_date.weekday(), dd_date)
        out[(int(year), int(month))] = first
    return out


def b2_month_slot_frame(
    train: pd.DataFrame,
    year: int,
    month: int,
    density_col: str | None,
    dates_by_dow: dict,
) -> pd.DataFrame:
    """168 слотов месяца на реальных датах (по канонической дате каждого dow).

    Флаги is_weekend/is_holiday - через calendar_flags: та же семантика, что
    и при обучении. Каноническая дата используется только для
    АЛЬТЕРНАТИВ; фактический слот статьи всегда получает свою реальную дату
    (см. personalized_scheduling)."""
    flags = calendar_flags(list(dates_by_dow.values()), years=[year])
    flag_by_dow = dict(zip(flags["dow"], zip(flags["is_weekend"], flags["is_holiday"])))
    density = (
        train.groupby(["dow", "hour"])[density_col].median() if density_col else None
    )
    count = train.groupby(["dow", "hour"]).size()
    rows = []
    for d in range(7):
        real = dates_by_dow.get(d)
        if real is None:
            continue
        is_weekend, is_holiday = flag_by_dow[d]
        for h in range(24):
            r = dict(
                dow=d,
                hour=h,
                month=int(month),
                year=int(year),
                is_weekend=int(is_weekend),
                is_holiday=int(is_holiday),
                hour_sin=float(np.sin(2 * np.pi * h / 24)),
                hour_cos=float(np.cos(2 * np.pi * h / 24)),
                date=str(real),
                n_train=int(count.get((d, h), 0)),
            )
            if density_col:
                r[density_col] = float(density.get((d, h), np.nan))
            rows.append(r)
    grid = pd.DataFrame(rows)
    if density_col:
        grid[density_col] = grid[density_col].fillna(float(train[density_col].median()))
    return grid


def _alternatives_frame(
    arts: pd.DataFrame,
    sf: pd.DataFrame,
    cal_cols: list[str],
    art_cols: list[str],
    profile_override: dict | None = None,
) -> pd.DataFrame:
    """Кросс-джойн статей месяца и слотов (единичная проверка инвариантности).

    Невременные признаки статьи повторяются БЕЗ ИЗМЕНЕНИЙ во всех альтернативных
    слотах (проверяется тестом test_b2_invariance), календарные признаки берутся
    только из сетки реального месяца."""
    n_slots = len(sf)
    big = pd.DataFrame(index=range(len(arts) * n_slots))
    for c in cal_cols:
        big[c] = np.tile(sf[c].to_numpy(), len(arts))
    for c in art_cols:
        big[c] = np.repeat(arts[c].to_numpy(), n_slots)
    if profile_override:
        for c, v in profile_override.items():
            if c in big.columns and c not in CALENDAR_FEATURES:
                big[c] = v
    big["_art"] = np.repeat(arts["id"].to_numpy(), n_slots)
    big["_slot"] = np.tile(np.arange(n_slots), len(arts))
    return big


def _eligible_slot_statistics(
    predictions: np.ndarray,
    actual_predictions: np.ndarray,
    supported_positions: np.ndarray,
    actual_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Сравнить фактический слот с поддержанными альтернативами каждой статьи.

    Фактическая комбинация дня и часа исключается из альтернатив, чтобы не
    учитывать её второй раз. Возвращаются медиана, логарифмическое отношение,
    доля альтернатив с прогнозом не выше фактического и размер пула.
    """
    n_articles = len(actual_predictions)
    medians = np.full(n_articles, np.nan)
    relative = np.full(n_articles, np.nan)
    percentiles = np.full(n_articles, np.nan)
    counts = np.zeros(n_articles, dtype=int)
    for i, actual in enumerate(actual_predictions):
        candidates = supported_positions[supported_positions != actual_positions[i]]
        pool = predictions[i, candidates]
        counts[i] = len(pool)
        if len(pool) == 0:
            continue
        median = float(np.median(pool))
        medians[i] = median
        relative[i] = float(np.log1p(max(actual, 0.0)) - np.log1p(max(median, 0.0)))
        percentiles[i] = float(np.mean(pool <= actual))
    return medians, relative, percentiles, counts


def _article_slot_results(
    articles: pd.DataFrame,
    slot_frame: pd.DataFrame,
    predicted_grid: np.ndarray,
    actual_predictions: np.ndarray,
    medians: np.ndarray,
    relative: np.ndarray,
    percentiles: np.ndarray,
    eligible_counts: np.ndarray,
    recommended_slots: np.ndarray,
    rule_slot: int,
    month_label: str,
) -> pd.DataFrame:
    """Сформировать одну строку результата на фактическую публикацию."""
    n_slots = predicted_grid.shape[1]
    return pd.DataFrame(
        {
            "article_id": articles["id"].to_numpy(),
            "author_id": articles["author_key"].astype(str).to_numpy(),
            "published_at": articles["dt_msk"].astype(str).to_numpy(),
            "views": articles["views"].to_numpy(),
            "month": month_label,
            "actual_slot": [
                f"{int(d)}-{int(h)}"
                for d, h in zip(articles["dow"].to_numpy(), articles["hour"].to_numpy())
            ],
            "weekend": articles["is_weekend"].astype(int).to_numpy(),
            "actual_pred": actual_predictions,
            "median_eligible_pred": medians,
            "personalized_slot_score": actual_predictions - medians,
            "personalized_slot_rel": relative,
            "personalized_slot_pct": percentiles,
            "eligible_slots_count": eligible_counts,
            "actual_slot_rank_pct_grid": (
                (predicted_grid <= actual_predictions[:, None]).sum(axis=1) / n_slots
            ),
            "recommended_slot": [
                f"{int(slot_frame.iloc[p]['dow'])}-{int(slot_frame.iloc[p]['hour'])}"
                for p in recommended_slots
            ],
            "rule_slot": (
                f"{int(slot_frame.iloc[rule_slot]['dow'])}-"
                f"{int(slot_frame.iloc[rule_slot]['hour'])}"
            ),
        }
    )


def personalized_scheduling(
    fitresult,
    test: pd.DataFrame,
    train: pd.DataFrame,
    features: list[str],
    density_col: str | None,
    profile_override: dict | None = None,
    min_slots: int = 24,
    return_slot_table: bool = False,
) -> dict:
    """Режим B2: рейтинг альтернативных слотов для каждой тестовой статьи.

    Для статьи фиксируются СВОИ невременные признаки (или profile_override -
    проверка устойчивости к профилю); альтернативы - 168 слотов её реального
    месяца на канонических датах (месяц, dow, holiday согласованы). Будущая
    плотность неизвестна - только dens24_expected из train; dens24_actual
    запрещён. Персонализированная оценка фактического слота: personalized_slot_score = прогноз фактического слота минус медиана
    прогнозов ДОПУСТИМЫХ (поддержанных) слотов того же месяца. Наблюдаемый
    «прирост от переноса» не существует, поэтому метрики B2 диагностические."""
    from .targets import TargetSpec

    losses_cfg = config.losses_cfg()
    loss_id = fitresult.metadata["loss_id"]
    spec = losses_cfg["losses"].get(loss_id) or losses_cfg["lightgbm_controls"][loss_id]
    target = TargetSpec(spec["target"])
    if density_col is None and "dens24" in features:
        raise ValueError("density-free политика не может использовать dens24")
    support_min = config.protocol()["policy"]["support_min_train"]
    te = test.copy()
    te["_m"] = list(zip(te.dt_msk.dt.year, te.dt_msk.dt.month))
    dates_by = _canonical_month_dates(te.dt_msk)
    month_keys = sorted(dates_by)
    slot_frames = {
        key: b2_month_slot_frame(train, key[0], key[1], density_col, dates_by[key])
        for key in month_keys
    }
    cal_cols = [
        c
        for c in (
            "dow",
            "hour",
            "month",
            "is_weekend",
            "is_holiday",
            "hour_sin",
            "hour_cos",
            density_col,
        )
        if c and c in features
    ]
    art_cols = [c for c in features if c not in CALENDAR_FEATURES]

    frames, owners = [], []
    for key in month_keys:
        arts = te[te["_m"] == key]
        if len(arts) == 0:
            continue
        sf = slot_frames[key]
        if len(sf) == 0:
            continue
        alt = _alternatives_frame(arts, sf, cal_cols, art_cols, profile_override)
        # Фактический слот оценивается по исходным признакам статьи,
        # а не по канонической дате альтернативного слота.
        act = arts[features].copy().reset_index(drop=True)
        act["_art"] = arts["id"].to_numpy()
        act["_slot"] = -1
        frames.append(pd.concat([alt, act], ignore_index=True))
        owners.append((key, sf, arts))
    if not frames:
        return {"status": "no_test_articles"}
    big = pd.concat(frames, ignore_index=True)
    # Прогноз батчами: декодирование ровно один раз.
    raw = np.concatenate(
        [
            fitresult.model.predict(big.iloc[i : i + 500_000][features])
            for i in range(0, len(big), 500_000)
        ]
    )
    big["pred"] = target.decode(raw)
    del raw

    # Сводка по месяцам: support, рекомендация модели и правила выходных.
    reco_by_art = {}
    month_rank_vec = {}
    out_months = {}
    pct_all = []
    art_rows, slot_tables = [], []
    for key, sf, arts in owners:
        sf_mask = big["_art"].isin(arts["id"])
        part = big[sf_mask]
        alt_part = part[part["_slot"] >= 0]
        act_part = (
            part[part["_slot"] == -1].set_index("_art").loc[arts["id"].to_numpy()]
        )
        n_slots = int(alt_part["_slot"].max()) + 1
        supp = sf["n_train"].to_numpy() >= support_min
        if supp.sum() < min_slots:
            out_months[f"{key[0]}-{key[1]:02d}"] = {
                "status": "too_few_supported_slots",
                "n_supported": int(supp.sum()),
            }
            continue
        preds = alt_part["pred"].to_numpy().reshape(len(arts), n_slots)
        slot_idx = sf["n_train"].to_numpy() >= support_min
        supp_positions = np.flatnonzero(slot_idx)
        best = preds[:, supp_positions].argmax(axis=1)
        # Правило выходных на тех же поддержанных слотах: argmax(is_weekend, hour).
        order = sorted(
            range(len(supp_positions)),
            key=lambda j: (
                -sf.iloc[supp_positions[j]]["is_weekend"],
                -sf.iloc[supp_positions[j]]["hour"],
                -sf.iloc[supp_positions[j]]["dow"],
            ),
        )
        rule_slot = supp_positions[order[0]]
        art_slot_pos = np.full(len(arts), -1)
        sf_dow, sf_hour = sf["dow"].to_numpy(), sf["hour"].to_numpy()
        for i, (d, h) in enumerate(
            zip(arts["dow"].to_numpy(), arts["hour"].to_numpy())
        ):
            hit = np.flatnonzero((sf_dow == d) & (sf_hour == h))
            if len(hit):
                art_slot_pos[i] = int(hit[0])
        known = art_slot_pos >= 0
        reco_slots = supp_positions[best]
        reco_by_art.update(
            {
                aid: (int(pos), int(rule_slot), int(n_slots), key)
                for aid, pos in zip(arts["id"].to_numpy(), reco_slots)
            }
        )
        mean_vec = preds.mean(axis=0)
        month_rank_vec[key] = pd.Series(mean_vec).rank()
        # Прогноз фактического слота - из строки статьи на её реальных признаках.
        own = act_part["pred"].to_numpy().astype(float)
        # Для сравнения исключаем фактический слот из множества альтернатив.
        med_elig, rel_score, pct_pool, n_elig = _eligible_slot_statistics(
            preds, own, supp_positions, art_slot_pos
        )
        pct_all.append(pct_pool[known])
        # Таблица альтернатив для независимой проверки персонализированной оценки.
        if return_slot_table:
            slot_tables.append(
                pd.DataFrame(
                    {
                        "article_id": alt_part["_art"].to_numpy(),
                        "month": f"{key[0]}-{key[1]:02d}",
                        "dow": np.tile(sf["dow"].to_numpy(), len(arts)),
                        "hour": np.tile(sf["hour"].to_numpy(), len(arts)),
                        "date": np.tile(sf["date"].astype(str).to_numpy(), len(arts)),
                        "is_weekend": np.tile(sf["is_weekend"].to_numpy(), len(arts)),
                        "is_holiday": np.tile(sf["is_holiday"].to_numpy(), len(arts)),
                        "n_train": np.tile(sf["n_train"].to_numpy(), len(arts)),
                        "supported": np.tile(slot_idx.astype(int), len(arts)),
                        "pred": alt_part["pred"].to_numpy(),
                    }
                )
            )
        month_label = f"{key[0]}-{key[1]:02d}"
        art_rows.append(
            _article_slot_results(
                arts, sf, preds, own, med_elig, rel_score, pct_pool,
                n_elig, reco_slots, int(rule_slot), month_label,
            )
        )
        out_months[month_label] = {
            "n_articles": int(len(arts)),
            "n_supported_slots": int(supp.sum()),
            "recommended_slot_top10": (
                pd.Series(
                    [
                        f"{sf.iloc[p]['dow']}-{int(sf.iloc[p]['hour'])}"
                        for p in reco_slots
                    ]
                )
                .value_counts()
                .head(10)
                .to_dict()
            ),
            "weekend_rule_slot": (
                f"{int(sf.iloc[rule_slot]['dow'])}-{int(sf.iloc[rule_slot]['hour'])}"
            ),
            "actual_slot_pct_pool_mean": (
                float(np.mean(pct_pool[known])) if known.any() else None
            ),
            "actual_slot_pct_pool_median": (
                float(np.median(pct_pool[known])) if known.any() else None
            ),
        }
    # Согласованность рейтингов слотов между месяцами (сезоны).
    mkeys = list(month_rank_vec)
    season_agree = {}
    from .metrics import rho_all

    for i, a in enumerate(mkeys):
        for b in mkeys[i + 1 :]:
            season_agree[f"{a}:{b}"] = rho_all(month_rank_vec[a], month_rank_vec[b])
    reco = pd.DataFrame(
        [(aid, v[0], v[1], v[3]) for aid, v in reco_by_art.items()],
        columns=["id", "reco_slot", "rule_slot", "month"],
    )
    diff_share = float((reco.reco_slot != reco.rule_slot).mean()) if len(reco) else None
    per_article = pd.concat(art_rows, ignore_index=True) if art_rows else pd.DataFrame()
    result = {
        "months": out_months,
        "season_rank_agreement": season_agree,
        "n_articles": int(len(reco)),
        "share_reco_differs_weekend_rule": diff_share,
        "actual_slot_pct_pool_mean": (
            float(np.concatenate(pct_all).mean()) if pct_all else None
        ),
        "actual_slot_pct_pool_median": (
            float(np.median(np.concatenate(pct_all))) if pct_all else None
        ),
        # Для межпрофильной/межмодельной согласованности (сравнение на вызывающей стороне)
        "month_rank_vectors": {
            f"{k[0]}-{k[1]:02d}": [float(x) for x in month_rank_vec[k].to_numpy()]
            for k in mkeys
        },
        "reco_per_article": {
            str(r.id): [int(r.reco_slot), int(r.rule_slot), str(r.month)]
            for r in reco.itertuples()
        },
        "per_article": per_article,
        "profile_override": {k: str(v) for k, v in (profile_override or {}).items()},
    }
    if return_slot_table:
        result["slot_table"] = (
            pd.concat(slot_tables, ignore_index=True) if slot_tables else pd.DataFrame()
        )
    return result


def train_thresholds(train: pd.DataFrame, score_col: str = "model_score") -> dict:
    """Границы из train: терцили и доля рекомендуемых q.

    train_scored: статьи train со score их фактического слота (та же процедура,
    что и для теста). Границы зависят только от train."""
    q_fixed = config.protocol()["policy"]["q_fixed"]
    qs = config.protocol()["policy"]["q_sensitivity"]
    s = train[score_col].to_numpy()
    return {
        "terciles": np.quantile(s, [1 / 3, 2 / 3]).tolist(),
        "threshold_q": float(np.quantile(s, 1 - q_fixed)),
        "q_fixed": q_fixed,
        "threshold_sensitivity": {str(q): float(np.quantile(s, 1 - q)) for q in qs},
    }


def personal_supported(te: pd.DataFrame) -> pd.DataFrame:
    """Личные статьи со слотами, представленными в обучающей выборке."""
    if "slot_n_train" not in te.columns:
        te = te.assign(slot_n_train=np.nan)
    ok = (te.is_corp == 0) & (
        te.slot_n_train >= config.protocol()["policy"]["support_min_train"]
    )
    return te[ok.fillna(False)].copy()


def policy_report(
    te_scored: pd.DataFrame,
    train: pd.DataFrame,
    n_boot: int = 1000,
) -> dict:
    """Метрики выбора времени для одной модели на проверочной выборке.

    te_scored: личные статьи поддержанных слотов с колонками score/views/
    is_weekend/month_id/slot_n_train. Границы - из train (train_thresholds)."""
    from . import metrics as M
    from . import statistics as ST

    th = train_thresholds(train)
    te = te_scored.copy()
    out = {
        "thresholds_train": th,
        "n_test_personal_supported": int(len(te)),
        "weekend_share_test": float(te.is_weekend.mean()) if len(te) else None,
        "coverage": None,
    }
    if len(te) < 200:
        out["status"] = "too_few_test_articles"
        return out
    # Терцили score заданы train-границами: test не участвует.
    te = te.assign(
        tercile=np.searchsorted(
            np.asarray(th["terciles"]), te["model_score"].to_numpy(), side="right"
        )
    )
    out.update(M.rule_metrics(te, "model_score", n_boot=n_boot))
    # Сенситивность: недельная кластеризация вместо авторской -
    # авторский бутстреп не учитывает общий новостной фон недели.
    out["delta_rho_ci_week"] = ST.paired_delta_rho_ci(
        te, "model_score", cluster="week_start", n_boot=n_boot
    )
    # Равное покрытие: q из train и вариант q = доля выходных (как в исходной проверке слотов).
    q = th["q_fixed"]
    out["fixed_q"] = M.equal_share_ratio(te, "model_score", q, n_boot=n_boot)
    out["weekend_rule"] = M.weekend_rule_ratio(te, q, n_boot=n_boot)
    q_we = float(te.is_weekend.mean())
    out["equal_share"] = M.equal_share_ratio(te, "model_score", q_we, n_boot=n_boot)
    out["weekday_only"] = M.weekday_only_metrics(
        te, "model_score", np.array(th["terciles"])
    )
    # Негативный контроль: случайный допустимый слот (P3).
    rng = np.random.default_rng(config.SEED)
    te_rand = te.assign(model_score_rand=rng.random(len(te)))
    out["random_control"] = {
        "rho": M.rho_all(te_rand["model_score_rand"], te_rand["views"]),
        "delta_rho_vs_weekend": (
            (
                M.rho_all(te_rand["model_score_rand"], te_rand["views"])
                - M.rho_all(te_rand["is_weekend"].astype(float), te_rand["views"])
            )
            if M.rho_all(te_rand["model_score_rand"], te_rand["views"]) is not None
            else None
        ),
    }
    # Негативный контроль: перемешивание исходов внутри месяца.
    sh = M.rho_within_month(te, "model_score")
    from .statistics import shuffle_labels_within_month

    te_sh = shuffle_labels_within_month(te)
    shuffled = M.rho_all(te_sh["model_score"], te_sh["views"])
    out["shuffle_control_rho"] = shuffled
    out["real_rho_within_month"] = sh["rho_within_month_mean"]
    return out
