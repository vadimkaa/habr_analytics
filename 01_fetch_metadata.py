"""Дополнительный сбор метаданных в новый каталог, отдельно от замороженного среза.

JSON endpoint Хабра не является гарантированным публичным контрактом.
Новый сбор создаёт новый датасет: заменять им архив при старых SNAP_* нельзя.
После прерывания записи сохраняются; повторный запуск идёт с первой страницы,
поскольку в живой ленте номера страниц сдвигаются. Это не транзакционный snapshot.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import requests

HUBS = {
    "ai": ["artificial_intelligence", "machine_learning"],
    "wide": ["infosecurity", "programming", "popular_science", "webdev", "python"],
}


def extract(ref, hub, observed_at):
    st = ref.get("statistics") or {}
    return {
        "id": str(ref["id"]),
        "hub_query": [hub],
        "timePublished": ref.get("timePublished"),
        "observed_at": observed_at,
        "title": ref.get("titleHtml") or ref.get("title"),
        "postType": ref.get("postType"),
        "lang": ref.get("lang"),
        "author": (ref.get("author") or {}).get("alias"),
        "isCorporative": ref.get("isCorporative"),
        "readingTime": ref.get("readingTime"),
        "complexity": ref.get("complexity"),
        "is_translation": any(
            x.get("type") == "translation" for x in ref.get("postLabels") or []
        ),
        "tags": [
            x.get("titleHtml") or x.get("title") or "" for x in ref.get("tags") or []
        ],
        "hubs": [x.get("alias") for x in ref.get("hubs") or []],
        **{
            k: st.get(k)
            for k in [
                "readingCount",
                "score",
                "commentsCount",
                "favoritesCount",
                "votesCount",
                "reach",
                "readers",
            ]
        },
    }


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def request_page(session, hub, page):
    for attempt in range(6):
        try:
            r = session.get(
                "https://habr.com/kek/v2/articles/",
                params={
                    "fl": "ru",
                    "hl": "ru",
                    "sort": "date",
                    "period": "alltime",
                    "perPage": 100,
                    "hub": hub,
                    "page": page,
                },
                timeout=40,
            )
            if r.status_code in [401, 403]:
                raise PermissionError(f"Сервер отклонил запрос: HTTP {r.status_code}")
            if r.status_code == 429:
                delay = r.headers.get("Retry-After", "30")
                time.sleep(min(float(delay) if delay.isdigit() else 30, 60))
                continue
            r.raise_for_status()
            result = r.json()
            if not isinstance(result.get("publicationRefs"), dict):
                raise ValueError("Изменился формат publicationRefs")
            return result
        except (requests.RequestException, ValueError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("Превышено число повторов после HTTP 429")


def collect(corpus, out, max_pages, cutoff):
    path = out / f"{corpus}_checkpoint.json"
    state = (
        json.loads(path.read_text()) if path.exists() else {"records": {}, "hubs": {}}
    )
    with requests.Session() as session:
        session.headers["User-Agent"] = (
            "habr-analytics-research/2.0 (public metadata; sequential requests)"
        )
        for hub in HUBS[corpus]:
            reason = "page_limit"
            last_ids = None
            state["hubs"][hub] = {"last_page": 0, "stop_reason": "in_progress"}
            try:
                for page in range(1, max_pages + 1):
                    data = request_page(session, hub, page)
                    refs = data["publicationRefs"]
                    ids = tuple(sorted(refs))
                    if not refs:
                        reason = "empty_page"
                        break
                    if ids == last_ids:
                        raise RuntimeError(
                            "Повторилась предыдущая страница; покрытие не подтверждено"
                        )
                    last_ids = ids
                    now = datetime.now(timezone.utc).isoformat()
                    times = []
                    for ref in refs.values():
                        row = extract(ref, hub, now)
                        aid = row["id"]
                        if row["timePublished"]:
                            times.append(datetime.fromisoformat(row["timePublished"]))
                        old = state["records"].get(aid)
                        if old:
                            row["hub_query"] = sorted(set(old["hub_query"] + [hub]))
                        state["records"][aid] = row
                    state["hubs"][hub] = {
                        "last_page": page,
                        "stop_reason": "in_progress",
                        "observed_at": now,
                    }
                    atomic_json(path, state)
                    if cutoff and times and max(times) < cutoff:
                        reason = "cutoff"
                        break
                    if page >= int(data.get("pagesCount") or max_pages):
                        reason = "api_last_page"
                        break
                    time.sleep(1)
                state["hubs"][hub]["stop_reason"] = reason
            except Exception as e:
                state["hubs"][hub] = {"stop_reason": "error", "message": str(e)}
                atomic_json(path, state)
                raise
            atomic_json(path, state)
    dest = out / f"habr_{corpus}_new.jsonl"
    with dest.open("w") as f:
        for row in state["records"].values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(corpus, len(state["records"]), state["hubs"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["all", "ai", "wide"], default="all")
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Новый каталог; не data исходного проекта",
    )
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument(
        "--cutoff",
        help="Нижняя граница YYYY-MM-DD в UTC; страница целиком старше границы останавливает сбор",
    )
    args = ap.parse_args()
    if args.output_dir.resolve() == (Path(__file__).parent / "data").resolve():
        ap.error("Нельзя перезаписывать замороженные входные данные")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = (
        datetime.fromisoformat(args.cutoff).replace(tzinfo=timezone.utc)
        if args.cutoff
        else None
    )
    for c in HUBS if args.corpus == "all" else [args.corpus]:
        collect(c, args.output_dir, args.max_pages, cutoff)


if __name__ == "__main__":
    main()
