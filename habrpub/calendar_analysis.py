"""Календарные сравнения с ограничением фактического окна наблюдения."""

from datetime import timedelta
import numpy as np
import pandas as pd
from scipy import stats

from . import DATA, FIGS
from .analysis import (
    N_BOOT,
    SEED,
    WIDE_COMMON_START,
    bootstrap_cluster,
    compare_weekend,
    corr,
    load_corpus,
    offday_sets,
    paired_authors,
    production_calendar,
    recent,
    save,
    setup_plot,
)

# Блоки праздников по названию из производственного календаря; дни без названия -
# перенесённые выходные (перенос делается постановлением Правительства РФ).
BLOCK_RULES = [
    ("Новогодние каникулы", "Новый год"),
    ("Рождество", "Новый год"),
    ("защитника", "23 февраля"),
    ("женский", "8 марта"),
    ("Весны и Труда", "1 мая"),
    ("Победы", "9 мая"),
    ("День России", "12 июня"),
    ("единства", "4 ноября"),
]


def block_label(title):
    for key, label in BLOCK_RULES:
        if isinstance(title, str) and key in title:
            return label
    return "Переносы"


def _with_month_pct(df):
    """Перцентиль просмотров статьи внутри месяца публикации (0..1].

    Снижает влияние различий между месяцами публикации и уменьшает риск
    смешения с возрастом: внутри месяца возраст статей
    всё ещё различается почти на месяц, а праздники сосредоточены в начале
    месяцев, поэтому он не заменяет возрастные проверки, а дополняет их.
    Для праздников перцентиль — основная метрика, VPD — вторичная, в скобках."""
    d = df.copy()
    d["month_pct"] = d.groupby("month_id").views.rank(pct=True)
    return d


def holiday_episodes(dates):
    """Праздничные эпизоды: максимальные цепочки дат с разрывом не более 3 дней.

    Обычные выходные или нерабочий перенос внутри цепочки её не рвут
    (пятница -> понедельник = 3 дня), это одна сцена «праздничной ленты».
    Зависимость статей внутри одной даты и эпизода бутстрап по авторам
    не воспроизводит — для неё нужен бутстрап по эпизодам."""
    ds = sorted(dates)
    out = []
    for d in ds:
        if out and (d - out[-1][-1]).days <= 3:
            out[-1].append(d)
        else:
            out.append([d])
    return out


def _calendar_titles():
    """Словарь «дата -> название праздника» из производственного календаря.

    Пустые (NaN) названия - перенесённые выходные без праздника; одноразовые
    нерабочие дни указов Президента (карантин 2020, ноябрь 2021, голосование)
    несут служебное название и праздниками не являются - это определение
    weekday_holidays и текста статьи, и здесь оно соблюдается."""
    cal = production_calendar()
    titles = dict(zip(cal.date, cal.holiday_title.fillna("")))
    one_off = {d for d, t in titles.items() if ("Нерабоч" in t) or ("голосован" in t)}
    return titles, one_off


def _superiority(a, b):
    """Вероятность превосходства: P(праздничная выше контрольной), ничьи — 0.5.
    Тот же AUC, что объясняется в разделе 8 статьи."""
    u = stats.mannwhitneyu(a, b).statistic
    return float(u / (len(a) * len(b)))


def _pct_pair(z):
    """Разница медианных перцентилей (п. п.) праздничных (hol=1) и контрольных статей."""
    return 100 * (
        z.loc[z.hol == 1, "month_pct"].median()
        - z.loc[z.hol == 0, "month_pct"].median()
    )


def _holiday_days(df):
    """Праздничные будни и их контрольные даты в окне данных.

    Возвращает словарь: границы окна, все дни окна, нерабочие дни и рабочие
    выходные календаря, названия праздников, одноразовые дни указов,
    праздничные будни hol и контрольные даты ctrl."""
    # Первые и последние неполные календарные дни не включаем в экспозицию.
    lo = df.dt_msk.min().date() + timedelta(days=1)
    hi = df.dt_msk.max().date() - timedelta(days=1)
    days = set(pd.date_range(lo, hi).date)
    off, workweekend = offday_sets()
    titles, one_off = _calendar_titles()
    hol = {d for d in off if d.weekday() < 5 and d not in one_off} & days
    ctrl = {
        d + timedelta(days=k) for d in hol for k in [-21, -14, -7, 7, 14, 21]
    } & days
    # Контроли — только рабочие дни календаря (рабочие субботы не исключаются).
    ctrl -= off
    return {
        "lo": lo,
        "hi": hi,
        "days": days,
        "off": off,
        "workweekend": workweekend,
        "titles": titles,
        "one_off": one_off,
        "hol": hol,
        "ctrl": ctrl,
    }


def _holiday_summary(df, h, c, cal, transferred):
    """Сводное сравнение праздничных статей с контрольными: перцентиль внутри
    месяца (основная метрика), VPD, число статей в день, доля корпоративных."""
    hol, ctrl, days = cal["hol"], cal["ctrl"], cal["days"]
    vol = df.groupby("date_msk").size()
    hv = np.array([vol.get(dd, 0) for dd in sorted(hol)])
    cv = np.array([vol.get(dd, 0) for dd in sorted(ctrl)])

    def corp_share(z):
        return float(z.is_corp.mean()) if len(z) else None

    pair_frame = pd.concat([h.assign(hol=1), c.assign(hol=0)])
    has_authors = "author_key" in df.columns
    return {
        "definition": "production calendar RF (xmlcalendar.ru): weekday off-days incl. transfers; controls = same weekday, working days",
        "primary_metric": "percentile of views within publication month; VPD secondary",
        "n_holiday": len(h),
        "n_control": len(c),
        "holiday_days": len(hol),
        "control_days": len(ctrl),
        "start": str(cal["lo"]),
        "end": str(cal["hi"]),
        "holiday_dates": sorted(str(d) for d in hol),
        "transferred_days": transferred,
        "pct_holiday": h.month_pct.median(),
        "pct_control": c.month_pct.median(),
        # Возрастные составы групп различаются, поэтому VPD сравнивается
        # осторожно, а основной метрикой служит перцентиль.
        "age_median_holiday": float(h.age_days.median())
        if "age_days" in h.columns
        else None,
        "age_median_control": float(c.age_days.median())
        if "age_days" in c.columns
        else None,
        "pct_ratio": h.month_pct.median() / c.month_pct.median(),
        "pct_p_iid": stats.mannwhitneyu(h.month_pct, c.month_pct).pvalue,
        # Разница медиан в процентных пунктах с авторским интервалом и вероятность
        # превосходства: отношение перцентилей плохо читается.
        "pct_diff_pp": float(_pct_pair(pair_frame)),
        "pct_diff_pp_ci_author": (
            bootstrap_cluster(pair_frame, _pct_pair, cluster="author_key")
            if has_authors
            else [None, None]
        ),
        "pct_superiority": _superiority(h.month_pct, c.month_pct),
        # Интервал отношения перцентилей: перевыборка целых авторов.
        "pct_ratio_ci_author": (
            bootstrap_cluster(
                pair_frame,
                lambda z: (
                    z.loc[z.hol == 1, "month_pct"].median()
                    / z.loc[z.hol == 0, "month_pct"].median()
                ),
                cluster="author_key",
            )
            if has_authors
            else [None, None]
        ),
        "vpd_holiday": h.vpd.median(),
        "vpd_control": c.vpd.median(),
        "vpd_ratio": h.vpd.median() / c.vpd.median(),
        "volume_holiday": hv.mean(),
        "volume_control": cv.mean(),
        "volume_ratio": hv.mean() / cv.mean(),
        "p_iid": stats.mannwhitneyu(h.vpd, c.vpd).pvalue,
        "corp_share_holiday": corp_share(h),
        "corp_share_control": corp_share(c),
        "working_weekend_days_in_window": sorted(
            str(d) for d in cal["workweekend"] & days
        ),
    }


def _holiday_by_type(h, c):
    """Сравнение внутри корпоративных и внутри личных статей: в праздники
    состав потока другой, и это отделяет смену состава от смены успеха."""
    out = {}
    for flag, label in [(True, "corporate"), (False, "personal")]:
        hh = h[h.is_corp == flag]
        cc = c[c.is_corp == flag]
        both = len(hh) and len(cc)
        out["by_type_" + label] = {
            "n_holiday": len(hh),
            "n_control": len(cc),
            "pct_holiday": hh.month_pct.median() if len(hh) else None,
            "pct_control": cc.month_pct.median() if len(cc) else None,
            "pct_ratio": hh.month_pct.median() / cc.month_pct.median()
            if both
            else None,
            "vpd_holiday": hh.vpd.median() if len(hh) else None,
            "vpd_control": cc.vpd.median() if len(cc) else None,
            "ratio": hh.vpd.median() / cc.vpd.median() if both else None,
        }
    return out


def _holiday_by_kind(h, c, cal, transferred):
    """Названные праздники против перенесённых выходных: в ИИ/МЛ переносов
    заметно много, и сводное число усредняет разные явления."""
    titled_dates = {str(x) for x in cal["hol"] if cal["titles"].get(x, "")}
    titled = h[h.date_msk.map(str).isin(titled_dates)]
    moved = h[h.date_msk.map(str).isin(transferred)]
    out = {}
    for kind, subset in [("titled", titled), ("transferred", moved)]:
        out["by_kind_" + kind] = {
            "n_articles": len(subset),
            "pct_median": subset.month_pct.median() if len(subset) else None,
            "pct_ratio_vs_control": (
                subset.month_pct.median() / c.month_pct.median()
                if len(subset)
                else None
            ),
        }
    out["pct_p_titled_vs_transferred"] = (
        stats.mannwhitneyu(titled.month_pct, moved.month_pct).pvalue
        if titled_dates and transferred
        else None
    )
    return out


def _episode_bootstrap(h, c, hol, n=N_BOOT):
    """Интервал разницы перцентилей перевыборкой праздничных эпизодов
    (подряд идущих праздничных дат) и контрольных недель.

    Статьи одной даты и одного эпизода зависимы, авторский бутстреп этой
    зависимости не воспроизводит. Эпизоды без статей в перевыборку не
    попадают: в медианах они всё равно не участвуют."""
    episodes = holiday_episodes(hol)
    ep_id = {dd: i for i, e in enumerate(episodes) for dd in e}
    h_e = h.assign(episode=h.date_msk.map(lambda x: ep_id.get(x, -1)))

    def diff_pair(hz, cz):
        return 100 * (hz.month_pct.median() - cz.month_pct.median())

    epg = [x.index.to_numpy() for _, x in h_e.groupby("episode")]
    # week_start есть в корпусных данных; в синтетических фреймах тестов
    # его может не быть — тогда каждая статья свой кластер.
    if "week_start" in c.columns:
        wks = [x.index.to_numpy() for _, x in c.groupby("week_start")]
    else:
        wks = [np.array([i]) for i in c.index]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        pick_e = rng.integers(len(epg), size=len(epg))
        pick_w = rng.integers(len(wks), size=len(wks))
        hh = h_e.loc[np.concatenate([epg[i] for i in pick_e])]
        cc = c.loc[np.concatenate([wks[i] for i in pick_w])]
        v = diff_pair(hh, cc)
        if np.isfinite(v):
            vals.append(v)
    return {
        "n_episodes_calendar": len(episodes),
        "n_episodes_with_articles": int(h_e.episode.nunique()),
        "n_control_weeks": int(c.week_start.nunique())
        if "week_start" in c.columns
        else len(c),
        "pct_diff_pp": float(diff_pair(h, c)),
        "ci95": np.quantile(vals, [0.025, 0.975]).tolist() if vals else [None, None],
        "note": "resampling holiday episodes (holiday side) and control weeks (control side)",
    }


def _month_position(d, h, c, cal):
    """Положение в месяце. Праздники сосредоточены в начале месяцев, и
    перцентиль внутри месяца сам по себе это смешение не снимает. Считаются
    доли в первой трети месяца и сравнение с будними статьями в окне ±10 дней
    от праздников: контроль из тех же дней месяца, разница — в обычных
    процентных пунктах перцентиля."""
    hol, days, off = cal["hol"], cal["days"], cal["off"]

    def first_third(z):
        return float((z.date_msk.map(lambda x: x.day) <= 10).mean())

    # Будние рабочие дни в пределах ±10 дней от праздников.
    near = {
        dd + timedelta(days=k)
        for dd in hol
        for k in range(-10, 11)
        if dd + timedelta(days=k) in days
        and (dd + timedelta(days=k)) not in off
        and (dd + timedelta(days=k)).weekday() < 5
    }
    c10 = d[d.date_msk.isin(near - hol)]
    enough = len(c10) > 30
    return {
        "first_third_share_holiday": first_third(h),
        "first_third_share_control": first_third(c),
        "window10_pct_diff_pp": (
            float(100 * (h.month_pct.median() - c10.month_pct.median()))
            if enough
            else None
        ),
        "window10_p_iid": (
            stats.mannwhitneyu(h.month_pct, c10.month_pct).pvalue if enough else None
        ),
        "window10_n_control": len(c10),
        "note": "percentile of views within month; holidays cluster at the start of months, so day-of-month is a confounder",
    }


def _summary_excl_ny(h, c, cal):
    """Сводное сравнение без новогодних каникул и доля новогодних статей:
    проверка, что эффект не держится на одном Новом годе."""
    hol, titles = cal["hol"], cal["titles"]
    ny_dates = {dd for dd in hol if block_label(titles.get(dd, "")) == "Новый год"}
    h2 = h[~h.date_msk.isin(ny_dates)]
    # Доля «Новый год с переносами» — статьи на датах НОВОГОДНЕГО ЭПИЗОДА
    # целиком (перенос 31 декабря и 9 января принадлежит той же цепочке
    # праздничных дней), а не все переносы года.
    ny_eps_dates = {
        dd
        for e in holiday_episodes(hol)
        if any(block_label(titles.get(dd, "")) == "Новый год" for dd in e)
        for dd in e
    }
    return {
        "n_articles_holiday": len(h2),
        "pct_holiday": h2.month_pct.median() if len(h2) else None,
        "pct_control": c.month_pct.median(),
        "pct_diff_pp": (
            float(100 * (h2.month_pct.median() - c.month_pct.median()))
            if len(h2)
            else None
        ),
        "new_year_share_of_holiday_articles": (
            float((len(h) - len(h2)) / len(h)) if len(h) else None
        ),
        "new_year_episode_share": float(h.date_msk.isin(ny_eps_dates).mean())
        if len(h)
        else None,
        "n_holiday_articles": len(h),
    }


def _decree_days_check(d, cal):
    """Одноразовые дни указов (карантин 2020, ноябрь 2021, голосование):
    исключены из праздников и проверяются отдельно, с собственными
    контрольными датами — тот же день недели ±7/14/21 день, только рабочие.
    Общий пул контролей всех праздников подмешал бы сезонность чужих месяцев."""
    days, off = cal["days"], cal["off"]
    u_days = {dd for dd in cal["one_off"] if dd.weekday() < 5} & days
    if not u_days:
        return None
    u = d[d.date_msk.isin(u_days)]
    u_ctrl = {
        dd + timedelta(days=k) for dd in u_days for k in [-21, -14, -7, 7, 14, 21]
    } & days
    u_ctrl -= off
    uc = d[d.date_msk.isin(u_ctrl)]
    return {
        "n_days": len(u_days),
        "n_articles": len(u),
        "years": sorted({dd.year for dd in u_days}),
        "pct_median": u.month_pct.median(),
        "pct_control": uc.month_pct.median() if len(uc) else None,
        "p_iid": (
            stats.mannwhitneyu(u.month_pct, uc.month_pct).pvalue
            if len(uc) > 10
            else None
        ),
        "n_control_articles": len(uc),
        "control_rule": "own controls: same weekday +-7/14/21 days, working calendar days",
    }


def holiday_analysis(df):
    """Праздничные нерабочие дни на буднях против рабочих дней того же дня недели.

    Календарь — производственный календарь РФ (праздники и переносы, без обычных
    суббот и воскресений и без одноразовых дней указов Президента). Контроли —
    тот же день недели в пределах 3 недель, только рабочие дни календаря;
    рабочие субботы контролем быть могут, нерабочие дни — нет. Обе группы
    ограничены окном данных. Основная метрика — перцентиль просмотров внутри
    месяца публикации; VPD приводится рядом, потому что возрастные составы
    групп различаются.

    Части результата считают отдельные функции: сводное сравнение, разбивки
    по типу блога и виду праздника, бутстреп эпизодов, положение в месяце,
    сравнение без Нового года и проверка дней указов."""
    cal = _holiday_days(df)
    d = _with_month_pct(df)
    h = d[d.date_msk.isin(cal["hol"])]
    c = d[d.date_msk.isin(cal["ctrl"])]
    # Перенесённые выходные — нерабочие дни без праздничного названия.
    transferred = [
        str(dd) for dd in sorted(cal["hol"]) if not cal["titles"].get(dd, "")
    ]
    res = _holiday_summary(df, h, c, cal, transferred)
    res.update(_holiday_by_type(h, c))
    res.update(_holiday_by_kind(h, c, cal, transferred))
    res["episode_bootstrap"] = _episode_bootstrap(h, c, cal["hol"])
    res["month_position"] = _month_position(d, h, c, cal)
    res["summary_excl_ny"] = _summary_excl_ny(h, c, cal)
    decree = _decree_days_check(d, cal)
    if decree is not None:
        res["decree_days_check"] = decree
    return res


def holiday_blocks(df):
    """Разбивка праздничных дней по блокам: даты блока берутся из КАЛЕНДАРЯ,
    а не из найденных статей (праздничная дата без публикаций
    в ИИ/МЛ — 30 из 158 — исчезала вместе со своими контрольными датами, и
    определение сравнения зависело от числа публикаций). Порядок: полный набор
    дат блока по календарю, затем его контрольные даты, и только потом
    присоединяются статьи. У блока без статей в окне медианы пусты.
    Контроли собственные для каждого блока: +-7/14/21 день от его дат за все
    годы наблюдения, минус нерабочие дни (общий пул контролей всех
    праздников подмешивал блоку сезонность и состав чужих месяцев)."""
    lo = df.dt_msk.min().date() + timedelta(days=1)
    hi = df.dt_msk.max().date() - timedelta(days=1)
    days = set(pd.date_range(lo, hi).date)
    off, _ = offday_sets()
    titles, one_off = _calendar_titles()
    hol = {d for d in off if d.weekday() < 5 and d not in one_off} & days
    df = _with_month_pct(df)
    labels = {}
    for d in hol:
        labels.setdefault(block_label(titles.get(d, "")), []).append(d)
    rows = []
    for label in sorted(labels):
        dates = sorted(labels[label])
        own_ctrl = {
            d + timedelta(days=k) for d in dates for k in [-21, -14, -7, 7, 14, 21]
        } & days
        own_ctrl -= off
        # Контрольная дата обязана лежать в пределах 21 дня от какой-то даты блока.
        assert all(any(abs((cd - d).days) <= 21 for d in dates) for cd in own_ctrl), (
            label
        )
        h = df[df.date_msk.isin(dates)]
        c = df[df.date_msk.isin(own_ctrl)]
        ratio = (
            h.month_pct.median() / c.month_pct.median() if len(h) and len(c) else None
        )
        vpd_ratio = h.vpd.median() / c.vpd.median() if len(h) and len(c) else None
        # Разница медиан в п. п. вместо отношения + вероятность
        # превосходства; интервал — той же перевыборкой целых авторов.
        if len(h) and len(c) and len(h) >= 10:
            pair = pd.concat([h.assign(hol=1), c.assign(hol=0)])
            diff = float(100 * (h.month_pct.median() - c.month_pct.median()))
            dci = bootstrap_cluster(
                pair,
                lambda z: (
                    100
                    * (
                        z.loc[z.hol == 1, "month_pct"].median()
                        - z.loc[z.hol == 0, "month_pct"].median()
                    )
                ),
                cluster="author_key",
            )
            supv = float(
                stats.mannwhitneyu(h.month_pct, c.month_pct).statistic
                / (len(h) * len(c))
            )
        else:
            diff, dci, supv = None, [None, None], None
        # Интервал отношения перцентилей: перевыборка целых авторов внутри
        # праздника и его собственных контролей.
        if ratio is not None and len(h) >= 10:
            ci = bootstrap_cluster(
                pd.concat([h.assign(hol=1), c.assign(hol=0)]),
                lambda z: (
                    z.loc[z.hol == 1, "month_pct"].median()
                    / z.loc[z.hol == 0, "month_pct"].median()
                ),
                cluster="author_key",
            )
        else:
            ci = [None, None]
        rows.append(
            {
                "block": label,
                "dates": [str(d) for d in dates],
                "control_dates": sorted(str(d) for d in own_ctrl),
                "n_days": len(dates),
                # Число календарных эпизодов блока в окне: строка с одним
                # эпизодом держится на одном событии, интервал для неё
                # условен.
                "n_events": len(holiday_episodes(dates)),
                "n_articles": len(h),
                "n_control_articles": len(c),
                "pct_holiday": h.month_pct.median() if len(h) else None,
                "pct_control": c.month_pct.median() if len(c) else None,
                "ratio": ratio,
                "diff_pp": diff,
                "diff_pp_ci_author": dci,
                "superiority": supv,
                "vpd_holiday": h.vpd.median() if len(h) else None,
                "vpd_control": c.vpd.median() if len(c) else None,
                "vpd_ratio": vpd_ratio,
                "ci95_author": ci,
                "corp_share": float(h.is_corp.mean()) if len(h) else None,
            }
        )
    return sorted(rows, key=lambda r: r["block"])


def run(corpus):
    df = load_corpus(corpus)
    df.to_csv(DATA / f"part2_{corpus}.csv", index=False)
    r = recent(df)
    # У AI сохраняем всю историю в CSV, календарь с 2016; wide — общее покрытие хабов.
    start = (
        pd.Timestamp("2016-01-01", tz="Europe/Moscow")
        if corpus == "ai"
        else WIDE_COMMON_START
    )
    cal = df[df.mature & (df.dt_msk >= start)].copy()
    day = r.groupby("dow").agg(n=("id", "size"), vpd=("vpd", "median"))
    cell = r.groupby(["dow", "hour"]).agg(n=("id", "size"), vpd=("vpd", "median"))
    cell = cell[cell.n >= 20]
    dw = day[day.index < 5]
    res = {
        "n_loaded": len(df),
        "n_mature": int(df.mature.sum()),
        "date_min": str(df.dt_msk.min()),
        "date_max": str(df.dt_msk.max()),
        "n_calendar": len(cal),
        "calendar_start": str(start),
        "recent_n": len(r),
        "weekend": compare_weekend(r),
        "paired_authors": paired_authors(r),
        "competition_day": corr(day.n, day.vpd),
        "competition_cell": corr(cell.n, cell.vpd),
        # Разброс внутри будней: корреляция по семи дням
        # недели почти целиком состоит из разницы будней и выходных.
        "weekday_day_spread": {
            "n_min": int(dw.n.min()),
            "n_max": int(dw.n.max()),
            "vpd_min": float(dw.vpd.min()),
            "vpd_max": float(dw.vpd.max()),
            "n_span_pct": float(100 * (dw.n.max() / dw.n.min() - 1)),
        },
        "holiday": holiday_analysis(cal),
        "holiday_blocks": holiday_blocks(cal),
        "subgroups": {},
    }
    for flag, label in [(False, "personal"), (True, "corporate")]:
        res["subgroups"][label] = compare_weekend(r[r.is_corp == flag], with_ci=False)
    res["hub_coverage"] = {
        h: {
            "n": len(s),
            "min": str(s.dt_msk.min().date()),
            "max": str(s.dt_msk.max().date()),
        }
        for h in sorted({h for hs in df.hub_query for h in hs})
        if len(s := df[df.hub_query.apply(lambda hs: h in hs)])
    }
    # Сезонный объём: только полные календарные годы в реально покрытом интервале.
    lo = cal.dt_msk.min().date()
    hi = cal.dt_msk.max().date()
    years = [
        y
        for y in sorted(cal.year.unique())
        if lo <= pd.Timestamp(f"{y}-01-01").date()
        and hi >= pd.Timestamp(f"{y}-12-31").date()
    ]
    # Границы начала сбора не позволяют считать первое 1 января доказанно полным.
    years = [y for y in years if y > start.year or start.month == 1 and start.day == 1]
    months = []
    for y in years:
        s = cal[cal.year == y]
        counts = s.groupby("month").size().reindex(range(1, 13), fill_value=0)
        for m, n in counts.items():
            nd = pd.Period(f"{y}-{m:02d}").days_in_month
            months.append({"year": y, "month": m, "per_day": n / nd})
    res["season_full_years"] = years
    res["season_volume"] = months
    res["weather"] = {
        "status": "not_analyzed",
        "reason": "moscow_weather.csv absent from supplied archive",
    }
    save(res, f"part2_{corpus}_results.json")
    plt = setup_plot()
    fig, ax = plt.subplots(figsize=(6.8, 3.7), layout="constrained")
    ax.scatter(cell.n, cell.vpd, alpha=0.6, s=24, color="#2d7ab3")
    label = {"ai": "ИИ/МЛ", "wide": "Пять хабов"}[corpus]
    ax.set(
        xlabel="Статей в ячейке день × час",
        ylabel="VPD, медиана ячейки",
        title=f"{label} · 12 месяцев · n ячейки ≥ 20 · ρ = {res['competition_cell']['rho']:.2f}",
    )
    fig.savefig(FIGS / f"04_competition_{corpus}.png")
    plt.close(fig)
    # Короткая сводка в консоль; полный словарь — в part2_{corpus}_results.json.
    hb = res["holiday"]
    print(
        f"{corpus}: статей за год {len(r)}; "
        f"выходные/будни {res['weekend']['ratio']:.2f}; "
        f"пары авторов {res['paired_authors']['n_authors']}, "
        f"медиана отношений {res['paired_authors']['median_ratio']:.2f}; "
        f"праздники: {hb['holiday_days']} будних дней, {hb['n_holiday']} статей, "
        f"разница перцентилей {hb['pct_diff_pp']:+.1f} п. п."
    )
