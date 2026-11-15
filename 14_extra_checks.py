"""Шаг 14. Дополнительные проверки к статье.

1. Порог зрелости. Статья сравнивает статьи старше 30 дней. Порог выбран из
   общих соображений (основная масса просмотров приходится на первые дни), а не
   подобран, поэтому здесь отношение медиан VPD «выходные / будни» за последний
   год пересчитывается при порогах 7, 14, 30, 60 и 90 дней.

2. Иллюстрации. Число изображений в тексте статьи (без формул-картинок) и его
   связь с VPD на текстовой выборке раздела 8. Счёт изображений берётся из
   полного HTML-кэша, если он есть (полная версия пакета), и сохраняется в
   data/image_counts.csv: там только номер статьи, корпус и числа, без текста.
   В публичном пакете HTML-кэша нет, и шаг читает готовый CSV.

3. Крупные источники публикаций. Авторский бутстреп учитывает, что статьи
   одного автора зависимы, но не отвечает, не держится ли эффект выходных на
   нескольких самых активных авторах или блогах. Здесь отношение медиан VPD
   за последний год (статьи старше 30 дней) пересчитывается без 10 самых
   активных авторов, без 1% самых активных, без 10 крупнейших корпоративных
   блогов и отдельно для личных и корпоративных блогов.

4. Календарные поправки в читаемости. Шаг 07 учитывает месяц публикации
   отдельным индикатором на каждый месяц. В ИИ/МЛ текстовая выборка
   разбросана по 116 месяцам, и это 139 параметров на 385 статей. Здесь та же
   частная ранговая корреляция считается с поправкой на год и месяц года и
   совсем без календаря; остальные поправки те же, что у шага 07.

Результат: data/results/14_extra_checks.json.
"""

import gzip
import json

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from scipy import stats

from habrpub import DATA
from habrpub.analysis import RECENT_START, bootstrap_cluster, corr, load_corpus, save

THRESHOLDS = [7, 14, 30, 60, 90]
IMAGE_COUNTS = DATA / "image_counts.csv"
RAW = DATA / "lex_texts_raw.jsonl.gz"
# ----------------------------------------------------------------------
# 1. Порог зрелости
# ----------------------------------------------------------------------


def maturity_thresholds():
    out = {}
    for corpus in ("ai", "wide"):
        df = load_corpus(corpus)
        window = df[df.dt_msk >= RECENT_START]
        rows = []
        for t in THRESHOLDS:
            z = window[window.age_days >= t]
            ratio = (
                z.loc[z.is_weekend, "vpd"].median()
                / z.loc[~z.is_weekend, "vpd"].median()
            )
            rows.append({"min_age_days": t, "n": len(z), "weekend_ratio": float(ratio)})
        out[corpus] = rows
    return out


# ----------------------------------------------------------------------
# 2. Иллюстрации
# ----------------------------------------------------------------------


# Формулы-картинки — те же признаки, что у шага 06 при очистке текста.
FORMULA_SELECTOR = "img.formula, img[data-tex]"


def count_images(html):
    """Число изображений в HTML статьи, не считая формул-картинок."""
    soup = BeautifulSoup(html or "", "html.parser")
    formulas = set(map(id, soup.select(FORMULA_SELECTOR)))
    return sum(1 for img in soup.find_all("img") if id(img) not in formulas)


def image_counts():
    """Счёт изображений: из HTML-кэша, если он есть, иначе из сохранённого CSV."""
    if RAW.exists():
        rows = []
        with gzip.open(RAW, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append(
                    {
                        "id": str(r["id"]),
                        "corpus": r["corpus"],
                        "n_images": count_images(r["textHtml"]),
                    }
                )
        counts = pd.DataFrame(rows).sort_values(["id", "corpus"]).reset_index(drop=True)
        counts.to_csv(IMAGE_COUNTS, index=False)
        return counts
    return pd.read_csv(IMAGE_COUNTS, dtype={"id": str})


def images():
    # Та же выборка, что в разделе 8: статьи, прошедшие фильтры шага 07.
    read = pd.read_csv(DATA / "readability_analysis.csv", dtype={"id": str})
    counts = image_counts().drop_duplicates("id")
    d = read.merge(counts[["id", "n_images"]], on="id", how="inner").reset_index(
        drop=True
    )
    d["images_per_1000_words"] = d.n_images / d.words * 1000

    def rho_per_k(z):
        return corr(z.images_per_1000_words, z.vpd)["rho"]

    many = d.n_images > 20
    return {
        "n": len(d),
        "median_images": float(d.n_images.median()),
        "share_without_images": float((d.n_images == 0).mean()),
        "rho_count": corr(d.n_images, d.vpd)["rho"],
        "rho_per_1000_words": rho_per_k(d),
        "rho_per_1000_words_ci_author": bootstrap_cluster(d, rho_per_k),
        "rho_per_1000_words_by_corpus": {
            c: corr(g.images_per_1000_words, g.vpd)["rho"]
            for c, g in d.groupby("corpus")
        },
        "n_more_than_20": int(many.sum()),
        "vpd_median_more_than_20": float(d.loc[many, "vpd"].median()),
        "vpd_median_rest": float(d.loc[~many, "vpd"].median()),
        "words_median_more_than_20": float(d.loc[many, "words"].median()),
        "words_median_rest": float(d.loc[~many, "words"].median()),
    }


# ----------------------------------------------------------------------
# 3. Крупные источники публикаций
# ----------------------------------------------------------------------


def weekend_ratio(z):
    return z.loc[z.is_weekend, "vpd"].median() / z.loc[~z.is_weekend, "vpd"].median()


def company_blog(hubs):
    """Корпоративный блог статьи: у корпоративных статей это первый хаб."""
    return hubs[0] if len(hubs) else None


def top_producers():
    out = {}
    for corpus in ("ai", "wide"):
        df = load_corpus(corpus)
        w = df[(df.dt_msk >= RECENT_START) & (df.age_days >= 30)].copy()
        w["blog"] = np.where(w.is_corp, w.hubs.map(company_blog), None)
        authors = w.author_key.value_counts()
        blogs = w.loc[w.is_corp, "blog"].value_counts()
        top1 = max(1, len(authors) // 100)
        variants = {
            "all": w,
            "without_top10_authors": w[~w.author_key.isin(authors.index[:10])],
            "without_top1pct_authors": w[~w.author_key.isin(authors.index[:top1])],
            "without_top10_blogs": w[~w.blog.isin(blogs.index[:10])],
            "personal_only": w[~w.is_corp],
            "corporate_only": w[w.is_corp],
        }
        rows = {}
        for name, z in variants.items():
            z = z.reset_index(drop=True)
            rows[name] = {
                "n": len(z),
                "share_of_window": float(len(z) / len(w)),
                "weekend_ratio": float(weekend_ratio(z)),
                "ci95_author": bootstrap_cluster(z, weekend_ratio),
            }
        rows["n_top1pct_authors"] = int(top1)
        rows["top10_blogs"] = blogs.index[:10].tolist()
        out[corpus] = rows
    return out


# ----------------------------------------------------------------------
# 4. Календарные поправки в читаемости
# ----------------------------------------------------------------------


def _calendar_dummies(s, calendar):
    if calendar == "month":
        return pd.get_dummies(s[["month_id"]].astype(str), drop_first=True, dtype=float)
    if calendar == "year_and_month_of_year":
        ym = pd.DataFrame(
            {
                "year": s.month_id.astype(str).str[:4],
                "month_of_year": s.month_id.astype(str).str[5:7],
            },
            index=s.index,
        )
        return pd.get_dummies(ym, drop_first=True, dtype=float)
    return pd.DataFrame(index=s.index)


def partial_rank(s, metric, calendar):
    """Частная ранговая корреляция метрики с VPD внутри корпуса.

    Поправки те же, что у _partial_rank шага 07 (корпоративность, 20 самых
    частых хабов, логарифмы длины и возраста), меняется только календарь.
    При calendar="month" результат совпадает с числом шага 07."""
    s = s.dropna(subset=[metric, "vpd", "words", "age_days"]).reset_index(drop=True)
    cats = _calendar_dummies(s, calendar)
    cats["corp"] = s.is_corp.astype(float)
    counts = pd.Series([h for hs in s.hubs for h in hs]).value_counts()
    for h in counts.head(20).index:
        cats["hub_" + h] = s.hubs.apply(lambda hs: float(h in hs))
    cats["log_words"] = np.log1p(s.words)
    cats["log_age"] = np.log1p(s.age_days)
    X = np.column_stack([np.ones(len(s)), cats.to_numpy(dtype=float)])
    x = stats.rankdata(s[metric])
    y = stats.rankdata(s.vpd)
    rx = x - X @ np.linalg.lstsq(X, x, rcond=None)[0]
    ry = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), int(X.shape[1])


def readability_calendar():
    read = pd.read_csv(DATA / "readability_analysis.csv", dtype={"id": str})
    meta = pd.concat([load_corpus("wide"), load_corpus("ai")], ignore_index=True)
    d = read.merge(
        meta[["id", "corpus", "hubs"]], on=["id", "corpus"], validate="one_to_one"
    )
    out = {}
    for metric in ("oborneva", "asl"):
        out[metric] = {}
        for corpus, s in d.groupby("corpus"):
            s = s.reset_index(drop=True)
            row = {"n": len(s), "n_months": int(s.month_id.nunique())}
            for calendar in ("month", "year_and_month_of_year", "none"):
                rho, k = partial_rank(s, metric, calendar)
                row[calendar] = {"partial_rank_correlation": rho, "n_parameters": k}
            row["year_and_month_of_year"]["ci95_author"] = bootstrap_cluster(
                s, lambda z: partial_rank(z, metric, "year_and_month_of_year")[0]
            )
            out[metric][corpus] = row
    return out


def main():
    res = {
        "maturity_thresholds": maturity_thresholds(),
        "images": images(),
        "top_producers": top_producers(),
        "readability_calendar": readability_calendar(),
    }
    save(res, "14_extra_checks.json")
    im = res["images"]
    print(
        f"иллюстрации: n={im['n']}, ρ на 1000 слов {im['rho_per_1000_words']:.3f}; "
        f"порог зрелости: ИИ/МЛ "
        + ", ".join(
            f"{x['min_age_days']}д {x['weekend_ratio']:.2f}"
            for x in res["maturity_thresholds"]["ai"]
        )
    )


if __name__ == "__main__":
    main()
