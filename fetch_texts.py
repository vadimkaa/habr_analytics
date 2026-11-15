"""Докачка текстов выборки по манифесту — внешняя проверка извлечения метрик.

Канонический анализ (run_analysis.py) работает offline от замороженного кэша
lex_texts_raw.jsonl.gz. Этот скрипт — опциональный сетевой шаг для читателей
пакета: он скачивает живые страницы статей из data/text_sample_manifest.json
(1600 ID), извлекает HTML тела статьи и считает метрики тем же кодом
06_text_sample.py, затем сравнивает с приложенными lex_metrics.jsonl.

Ожидания честно ограничены:
- живой текст мог быть отредактирован после снапшота 9–10 ноября 2026 года;
- разметка страницы меняется, селектор тела статьи — с запасными вариантами;
- цель — проверка методики извлечения, а не побайтовое воспроизведение кэша.

Вежливость к сайту: последовательные запросы, пауза между ними, повтор после
429 с учётом Retry-After. Запуск: python fetch_texts.py --limit 20 (проба),
python fetch_texts.py (полная выгрузка, ~20 минут при паузе 0.7 c).
"""

import argparse
import gzip
import importlib.util
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent

# Селектор тела статьи: основной вариант и запасные для изменившейся разметки.
BODY_SELECTORS = [
    "div.tm-article-body",
    "#post-content-body",
    "div.article-body",
    "div.post__body",
    "article",
]


def load_06():
    spec = importlib.util.spec_from_file_location("habr_06", ROOT / "06_text_sample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_url(article_id):
    return f"https://habr.com/ru/articles/{article_id}/"


def extract_article_html(page_html):
    """HTML тела статьи из живой страницы; None, если ни один селектор не сработал."""
    soup = BeautifulSoup(page_html, "html.parser")
    for sel in BODY_SELECTORS:
        node = soup.select_one(sel)
        if node is not None and node.get_text(strip=True):
            return str(node)
    return None


def get_with_retry(session, url, attempts=4):
    for attempt in range(attempts):
        try:
            r = session.get(url, timeout=40)
            if r.status_code == 429:
                delay = r.headers.get("Retry-After", "30")
                time.sleep(min(float(delay) if delay.isdigit() else 30, 60))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt, 30))
    return None


def make_record(article_id, corpus, html, fetched_at, meta, corpora=None):
    return {
        "id": str(article_id),
        "corpus": corpus,
        # Полный список корпусов, в которых id встречается в манифесте: при
        # пересечении корпусов одна страница скачивается один раз.
        "corpora": sorted(corpora or [corpus]),
        "textHtml": html,
        "fetched_at": fetched_at,
        "source_url": build_url(article_id),
        **meta,
    }


def load_metadata():
    """readingTime/complexity/isCorporative из метаданных корпусов, если они рядом."""
    meta = {}
    for name in ["habr_ai_ml_raw.jsonl.gz", "habr_wide_raw.jsonl.gz"]:
        path = ROOT / "data" / name
        if not path.exists():
            continue
        import gzip as gz

        with gz.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                meta[str(r["id"])] = {
                    k: r.get(k) for k in ["readingTime", "complexity", "isCorporative"]
                }
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="скачать только N первых записей (проба)")
    ap.add_argument("--sleep", type=float, default=0.7, help="пауза между запросами, c")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/lex_texts_refetched.jsonl.gz",
        help="выгрузка текстов (jsonl.gz, схема совместима с 06)",
    )
    ap.add_argument(
        "--skip-metrics",
        action="store_true",
        help="только скачать тексты, без пересчёта метрик и сравнения",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data/refetch_report.json",
    )
    args = ap.parse_args()

    manifest = json.loads((ROOT / "data/text_sample_manifest.json").read_text())
    if args.limit:
        manifest = manifest[: args.limit]

    # Манифест хранит пары (id, корпус). Статья из обоих корпусов скачивается
    # один раз, а оба корпуса перечисляются в записи и в отдельной карте: при
    # возобновлении по одному id вторая принадлежность иначе терялась бы.
    corpora_by_id = {}
    for rec in manifest:
        corpora_by_id.setdefault(str(rec["id"]), set()).add(rec["corpus"])

    # Возобновление: ID, уже попавшие в выгрузку, пропускаются; множество
    # пополняется внутри запуска после каждой записи, а карта корпусов
    # хранится рядом и обновляется после каждой строки. Повреждённая
    # последняя строка не блокирует возобновление.
    done = set()
    if args.out.exists():
        with gzip.open(args.out, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    continue

    corpus_map_path = ROOT / "data/refetch_corpus_map.json"
    corpus_map = (
        json.loads(corpus_map_path.read_text(encoding="utf-8"))
        if corpus_map_path.exists()
        else {}
    )

    def save_corpus_map():
        corpus_map_path.write_text(
            json.dumps(corpus_map, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    meta_by_id = load_metadata()
    fetched = skipped = failed = 0
    session = requests.Session()
    session.headers["User-Agent"] = (
        "habr-analytics-research/3.0 (text extraction verification; sequential)"
    )
    with gzip.open(args.out, "at", encoding="utf-8") as out:
        for rec in manifest:
            aid = str(rec["id"])
            if aid in done:
                # id скачан, но принадлежность могла появиться позже: карта
                # обновляется без повторной загрузки страницы.
                if set(corpus_map.get(aid, [])) != corpora_by_id[aid]:
                    corpus_map[aid] = sorted(corpora_by_id[aid])
                    save_corpus_map()
                skipped += 1
                continue
            page = get_with_retry(session, build_url(aid))
            if page is None:
                failed += 1
                print(f"{aid}: страница недоступна (404 или ошибка)")
                continue
            html = extract_article_html(page)
            if html is None:
                failed += 1
                print(f"{aid}: тело статьи не найдено ни одним селектором")
                continue
            now = datetime.now(UTC).isoformat()
            out.write(
                json.dumps(
                    make_record(
                        aid,
                        rec["corpus"],
                        html,
                        now,
                        meta_by_id.get(aid, {}),
                        corpora=sorted(corpora_by_id[aid]),
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()  # каждая запись сразу на диске: сбой не теряет её
            done.add(aid)
            corpus_map[aid] = sorted(corpora_by_id[aid])
            save_corpus_map()
            fetched += 1
            time.sleep(args.sleep)
    print(
        f"скачано {fetched}, пропущено (уже в выгрузке) {skipped}, не удалось {failed}"
    )

    if args.skip_metrics:
        return

    mod_06 = load_06()
    new_metrics = []
    with gzip.open(args.out, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prose, counters = mod_06.prepare_prose(r["textHtml"])
            meta = {
                k: r.get(k) for k in ["id", "corpus", "readingTime", "isCorporative"]
            }
            meta["id"] = str(meta["id"])
            meta["fetched_at"] = r["fetched_at"]
            new_metrics.append(mod_06.compute_metrics(prose, {**meta, **counters}))
    shipped = {}
    shipped_path = ROOT / "data/lex_metrics.jsonl"
    if shipped_path.exists():
        with open(shipped_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                shipped[str(r["id"])] = r

    from scipy import stats

    common = [m for m in new_metrics if m["id"] in shipped]
    # Сравнение только числа слов не ловит ошибки сегментации,
    # подсчёта слогов и индексов. Рядом — предложения, ASL, ASW и индексы;
    # каждый показатель: медиана абсолютной разницы, ранговая корреляция
    # и доля совпадений в пределах 10% (для ненулевых значений).
    METRICS_CHECK = ["words", "sentences", "asl", "asw_all", "oborneva", "fk_mod"]

    def metric_report(key):
        pairs = [
            (m[key], shipped[m["id"]][key])
            for m in common
            if key in m and key in shipped[m["id"]]
        ]
        if len(pairs) < 10:
            return None
        a = [x for x, _ in pairs]
        b = [y for _, y in pairs]
        nz = [(x, y) for x, y in pairs if y]
        return {
            "n_pairs": len(pairs),
            "median_abs_diff": sorted(abs(x - y) for x, y in pairs)[len(pairs) // 2],
            "rho_fresh_vs_snapshot": float(stats.spearmanr(a, b)[0]),
            "share_within_10pct": (
                sum(1 for x, y in nz if abs(x / y - 1) <= 0.10) / len(nz)
                if nz
                else None
            ),
        }

    report = {
        "n_refetched": len(new_metrics),
        "n_common_with_snapshot": len(common),
        "snapshot_date": "2026-11-09..10; live pages may have been edited since",
        "consistency": {k: metric_report(k) for k in METRICS_CHECK},
        "conclusion": "проверка СОГЛАСОВАННОСТИ извлечения: тот же алгоритм на живых страницах, а не независимая валидация; расхождения ожидаются из-за правок и разметки",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
