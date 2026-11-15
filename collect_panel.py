"""Панельный сбор счётчиков свежих статей; входит в публичный пакет.

Зачем: в замороженном срезе все счётчики сняты один поздним срезом, и «успех»
приходится определять как накопленные просмотры, делённые на возраст. Повторные
наблюдения одних и тех же статей в раннем возрасте дают честную короткую цель —
например «просмотры к 7 дням», — и позволяют измерить, сколько просмотров
статья набирает в первые сутки/неделю без возрастного артефакта.

Как запускать (на своей машине):
    python collect_panel.py --days 14            # собрать окно последних 14 дней
    python collect_panel.py --summary            # покрытие панели без сети
Рекомендуется запуск раз в сутки (cron/systemd timer, команды в README).
Каждый запуск добавляет строки в data/panel/panel.jsonl: одна строка =
(id, момент наблюдения). Файл append-only, прерывание не портит записанное.
Вежливость к сайту — последовательные запросы с паузами и повторами после 429
(логика 01_fetch_metadata) и явный User-Agent; сборщик читает только публичные
счётчики, чужие тексты не сохраняет.

Ограничение. Каждый запуск обходит свежие страницы хабов и обновляет только
статьи, которые на них нашлись. Реестр (id → достигнутый возраст) помнит
статьи, но отдельно по идентификатору их не запрашивает: статья, выпавшая из
обхода раньше горизонта, остаётся в панели с пропусками. --summary показывает,
сколько статей дошло до каждого горизонта. Повторный запрос по id для всех
статей реестра — ближайшее улучшение сборщика.
"""

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent

PANEL_FIELDS = [
    "id",
    "hub_query",
    "time_published",
    "observed_at",
    "age_hours",
    "views",
    "score",
    "comments",
    "favorites",
    "is_corp",
    "title",
]


def load_01():
    spec = importlib.util.spec_from_file_location(
        "habr_01", ROOT / "01_fetch_metadata.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def to_utc(ts):
    return datetime.fromisoformat(ts) if ts else None


def panel_record(row, now):
    """Строка панели из записи ленты 01: только счётчики и возраст, без текста.

    now — datetime (UTC); в JSON уходит строкой ISO, иначе dumps упадёт."""
    tp = to_utc(row.get("timePublished"))
    return {
        "id": row["id"],
        "hub_query": row["hub_query"],
        "time_published": row.get("timePublished"),
        "observed_at": now.isoformat() if isinstance(now, datetime) else now,
        "age_hours": round((now - tp).total_seconds() / 3600, 2) if tp else None,
        "views": row.get("readingCount"),
        "score": row.get("score"),
        "comments": row.get("commentsCount"),
        "favorites": row.get("favoritesCount"),
        "is_corp": row.get("isCorporative"),
        "title": row.get("title"),
    }


def collect_once(fetch_page, corpus, days, now, max_pages=200, sleep=0.0, clock=None):
    """Один проход по ленте хабов корпуса. fetch_page(hub, page) -> dict | None.

    Страницы отсортированы по дате: после первой страницы, где все публикации
    старше окна, сбор прекращается. Повторы внутри прохода схлопываются по id.
    Время наблюдения записывается отдельно для КАЖДОГО ответа сайта: один timestamp на весь обход занижал бы точность возраста.
    clock - инъекция часов для тестов; по умолчанию datetime.now(UTC).
    Возвращает (records, limit_reached): limit_reached - список хабов,
    где окно закончилось по лимиту страниц, а не по естественному выходу
    из окна (признак того, что проход мог что-то не увидеть)."""
    from time import sleep as do_sleep

    if clock is None:
        def clock():
            return datetime.now(UTC)

    m01 = load_01()
    cutoff = now - timedelta(days=days)
    records = {}
    limit_reached = []
    for hub in m01.HUBS[corpus]:
        done_natural = False
        for page in range(1, max_pages + 1):
            data = fetch_page(hub, page)
            # Момент ответа сайта: один на страницу, а не на весь обход.
            resp = clock()
            if not data:
                done_natural = True
                break
            refs = data.get("publicationRefs") or {}
            if not refs:
                done_natural = True
                break
            times = []
            iso_now = resp.isoformat()
            for ref in refs.values():
                row = m01.extract(ref, hub, iso_now)
                tp = to_utc(row.get("timePublished"))
                if tp:
                    times.append(tp)
                if tp and tp >= cutoff:
                    rec = panel_record(row, resp)
                    old = records.get(rec["id"])
                    if old:
                        rec["hub_query"] = sorted(set(old["hub_query"] + [hub]))
                    records[rec["id"]] = rec
            if times and max(times) < cutoff:
                done_natural = True
                break
            if sleep:
                do_sleep(sleep)
        if not done_natural:
            limit_reached.append(hub)
    return list(records.values()), limit_reached


def summarize(panel_lines, target_age_days=7.0, window_days=1.0):
    """Покрытие панели: наблюдений на статью, возраст, наличие цели-окна.

    Цель «просмотры к N дням» считается по наблюдению внутри окна
    [N - window/2, N + window/2] суток; из попавших берётся наблюдение,
    БЛИЖАЙШЕЕ к горизонту, а не самое позднее. Повреждённая
    строка JSON пропускается с подсчётом, а не останавливает сводку; статьи
    без просмотров в окне исключаются из медианы, но не падают StatisticsError."""
    by_id = defaultdict(list)
    damaged = 0
    for line in panel_lines:
        try:
            r = json.loads(line)
        except (ValueError, TypeError):
            damaged += 1
            continue
        if r.get("age_hours") is not None:
            by_id[r["id"]].append(r)
    lo, hi = (
        (target_age_days - window_days / 2) * 24,
        (target_age_days + window_days / 2) * 24,
    )
    in_window = {}
    for aid, rows in by_id.items():
        hit = [
            r for r in rows if r["age_hours"] is not None and lo <= r["age_hours"] <= hi
        ]
        if hit:
            in_window[aid] = min(
                hit, key=lambda r: abs(r["age_hours"] - target_age_days * 24)
            )
    ages = sorted(r["age_hours"] for rows in by_id.values() for r in rows)
    views_at_target = [
        r["views"] for r in in_window.values() if r.get("views") is not None
    ]
    return {
        "n_articles": len(by_id),
        "n_observations": sum(len(v) for v in by_id.values()),
        "median_observations_per_article": (
            median(sorted(len(v) for v in by_id.values())) if by_id else None
        ),
        "age_hours_min": ages[0] if ages else None,
        "age_hours_max": ages[-1] if ages else None,
        "target_age_days": target_age_days,
        "n_articles_with_target": len(in_window),
        "median_views_at_target": (
            median(sorted(views_at_target)) if views_at_target else None
        ),
        "damaged_lines_skipped": damaged,
        "note": "цель по наблюдению, ближайшему к горизонту внутри окна; без интерполяции и экстраполяции",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["all", "ai", "wide"], default="all")
    ap.add_argument(
        "--days",
        type=float,
        default=35.0,
        help="окно когорты, суток; с запасом относительно горизонта 30 дней",
    )
    ap.add_argument(
        "--horizon-days",
        type=float,
        default=30.0,
        help="последний горизонт наблюдения; статьи старше него исключаются из реестра",
    )
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument(
        "--sleep", type=float, default=1.0, help="пауза между страницами, c"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/panel/panel.jsonl",
        help="append-only файл панели",
    )
    ap.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "data/panel/registry.json",
        help="реестр наблюдаемых id и их последнего возраста",
    )
    ap.add_argument(
        "--summary",
        action="store_true",
        help="не собирать: посчитать покрытие по существующему файлу",
    )
    ap.add_argument("--target-age-days", type=float, default=7.0)
    args = ap.parse_args()

    if args.summary:
        lines = (
            args.out.read_text(encoding="utf-8").splitlines()
            if args.out.exists()
            else []
        )
        s = summarize(lines, args.target_age_days)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / "panel_summary.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(s, ensure_ascii=False, indent=2))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    m01 = load_01()

    def fetch_page(hub, page):
        return m01.request_page(requests_session, hub, page)

    import requests

    requests_session = requests.Session()
    requests_session.headers["User-Agent"] = (
        "habr-analytics-research/3.0 (panel counters; sequential requests)"
    )
    # Реестр наблюдаемых: статью ведём до последнего горизонта (по умолчанию
    # 30 дней), а не бросаем после первого прохода.
    registry = {}
    if args.registry.exists():
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
    corpora = list(m01.HUBS) if args.corpus == "all" else [args.corpus]
    horizon_hours = args.horizon_days * 24
    total = 0
    limit_reached_all = []
    for corpus in corpora:
        now = datetime.now(UTC)
        records, limit_reached = collect_once(
            fetch_page, corpus, args.days, now, args.max_pages, args.sleep
        )
        limit_reached_all += limit_reached
        total += len(records)
        # Порционная запись: после каждого корпуса данные уже на диске,
        # сбой дальше не теряет этот проход.
        with args.out.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
        for r in records:
            if r["age_hours"] is not None:
                registry[r["id"]] = max(
                    registry.get(r["id"], 0.0), float(r["age_hours"])
                )
        registry = {k: v for k, v in registry.items() if v < horizon_hours}
        print(corpus, len(records), "записей в панели")
    args.registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Признак завершённости обхода и достижения лимита страниц.
    status = {
        "completed": not limit_reached_all,
        "limit_reached_hubs": limit_reached_all,
        "corpora": corpora,
        "days_window": args.days,
        "horizon_days": args.horizon_days,
        "n_observations_added": total,
        "n_tracked": len(registry),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (args.out.parent / "run_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"итого добавлено {total} наблюдений -> {args.out}")


if __name__ == "__main__":
    main()
