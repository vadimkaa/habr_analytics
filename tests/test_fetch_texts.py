"""Проверки fetch_texts без сети: URL, селекторы тела, схема записей."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fetch_texts_url_and_body_extraction():
    m = module("fetch_texts.py")
    assert m.build_url("123") == "https://habr.com/ru/articles/123/"
    page = (
        '<html><body><nav>menu</nav><div class="tm-article-body">'
        "<p>Текст статьи.</p></div><footer>foot</footer></body></html>"
    )
    html = m.extract_article_html(page)
    assert "Текст статьи." in html and "menu" not in html and "foot" not in html
    # запасной селектор, если основная разметка изменилась
    fallback = "<html><body><article><p>Другая разметка</p></article></body></html>"
    assert "Другая разметка" in m.extract_article_html(fallback)
    assert m.extract_article_html("<html><body>пусто</body></html>") is None


def test_fetch_texts_record_schema():
    m = module("fetch_texts.py")
    r = m.make_record(
        "77", "wide", "<p>h</p>", "2026-11-11T00:00:00+00:00", {"readingTime": 4}
    )
    assert r["id"] == "77" and r["corpus"] == "wide"
    assert r["source_url"] == m.build_url("77")
    assert {"id", "corpus", "textHtml", "fetched_at"} <= set(r)
    # В записи перечислены все корпуса манифеста для этого id,
    # страница скачивается один раз, а принадлежности сохраняются полностью.
    r2 = m.make_record(
        "77",
        "wide",
        "<p>h</p>",
        "2026-11-11T00:00:00+00:00",
        {},
        corpora=["ai", "wide"],
    )
    assert r2["corpora"] == ["ai", "wide"] and r2["corpus"] == "wide"
