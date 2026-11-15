"""Проверки collect_panel без сети: подменный fetch, дедупликация, окно, сводка."""

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ref(aid, iso, views, hub, title="T"):
    return {
        "id": aid,
        "timePublished": iso,
        "titleHtml": title,
        "postType": "article",
        "lang": "ru",
        "author": {"alias": "someone"},
        "isCorporative": False,
        "postLabels": [],
        "tags": [],
        "hubs": [{"alias": hub}],
        "statistics": {
            "readingCount": views,
            "score": 5,
            "commentsCount": 2,
            "favoritesCount": 1,
        },
    }


def fake_page(pages_by_hub):
    return lambda hub, page: pages_by_hub.get((hub, page))


def test_panel_collect_once_dedup_and_cutoff():
    m = module("collect_panel.py")
    now = datetime(2026, 11, 11, tzinfo=UTC)
    fresh = (now - timedelta(days=1)).isoformat()
    older = (now - timedelta(days=3)).isoformat()
    ancient = (now - timedelta(days=10)).isoformat()
    pages = {
        ("artificial_intelligence", 1): {
            "publicationRefs": {
                "100": ref("100", fresh, 500, "artificial_intelligence")
            }
        },
        ("machine_learning", 1): {
            "publicationRefs": {
                "100": ref("100", fresh, 500, "machine_learning"),
                "102": ref("102", older, 50, "machine_learning"),
            }
        },
        ("machine_learning", 2): {
            "publicationRefs": {"999": ref("999", ancient, 1, "machine_learning")}
        },
    }
    rows, limit_reached = m.collect_once(
        fake_page(pages),
        "ai",
        days=7,
        now=now,
        max_pages=5,
        sleep=0,
        clock=lambda: now,
    )
    ids = {r["id"] for r in rows}
    assert ids == {"100", "102"}  # 999 вне окна, дубликат 100 схлопнут
    by_id = {r["id"]: r for r in rows}
    assert by_id["100"]["hub_query"] == ["artificial_intelligence", "machine_learning"]
    assert by_id["100"]["views"] == 500
    assert 23 <= by_id["100"]["age_hours"] <= 25
    assert by_id["100"]["observed_at"] == now.isoformat()
    # Естественный выход из окна на обоих хабах; признак лимита пуст.
    assert limit_reached == []


def test_panel_summarize_nearest_to_horizon_and_damaged():
    """Цель — наблюдение, ближайшее к горизонту; повреждённая строка не
    останавливает сводку, а счётчиком попадает в damaged_lines_skipped."""
    m = module("collect_panel.py")
    lines = []
    for age_h, views in [(163.0, 100), (168.0, 500), (180.0, 200)]:
        lines.append(json.dumps({"id": "A", "age_hours": age_h, "views": views}))
    lines.append("{битая строка json")
    lines.append(json.dumps({"id": "B", "age_hours": 48.0, "views": None}))
    s = m.summarize(lines, target_age_days=7.0, window_days=1.0)
    # Окно [156; 180] ч: в нём все три наблюдения A; ближайшее к 168 ч — 168.
    assert s["n_articles_with_target"] == 1
    assert s["median_views_at_target"] == 500
    assert s["damaged_lines_skipped"] == 1
    assert s["n_articles"] == 2


def test_panel_summarize_target_window():
    m = module("collect_panel.py")
    now = datetime(2026, 11, 11, tzinfo=UTC)
    lines = []
    for age_h, views in [(24, 100), (168, 500)]:
        r = {
            "id": "A",
            "age_hours": age_h,
            "views": views,
            "observed_at": (now - timedelta(hours=0)).isoformat(),
        }
        lines.append(json.dumps(r))
    lines.append(json.dumps({"id": "B", "age_hours": 48.0, "views": 10}))
    s = m.summarize(lines, target_age_days=7.0, window_days=1.0)
    assert s["n_articles"] == 2 and s["n_observations"] == 3
    assert s["n_articles_with_target"] == 1
    assert s["median_views_at_target"] == 500
