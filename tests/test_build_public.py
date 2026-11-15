"""Сборщик публичного пакета работает по белому списку.

Прежний список запретов по именам пропускал data/lex_texts_refetched.jsonl.gz
(файл, который пишет fetch_texts.py по умолчанию): .gitignore действует только
на git add, а сборщик копировал каталог целиком."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("build_public", ROOT / "build_public.py")
bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bp)


def test_whitelist_denies_private_artifacts():
    rels = {r.as_posix() for r in bp.collect(bp.ROOT)}
    for bad in [
        "data/lex_texts_raw.jsonl.gz",
        "data/lex_texts_refetched.jsonl.gz",
        "data/lex_metrics_refetched.jsonl",
        "data/refetch_report.json",
        "data/lex_prose.jsonl",
        "data/panel/panel.jsonl",
        "data/sources/art_000.json",
        "docs/verification.md",
        "PUBLISH_TO_GITHUB.md",
        "check_v2.py",
        "article.docx",
        "data/input_manifest.json.bak",
    ]:
        assert bad not in rels, bad
    assert "LICENSE" in rels and "docs/audit.md" in rels
    assert "data/results/08_model_results.json" in rels


def test_planted_refetched_texts_do_not_pass(tmp_path):
    """Синтетический мини-проект с подложенными полными текстами: в список
    публикации не попадает ни один чужой артефакт, даже новый по имени."""
    root = tmp_path
    (root / "habrpub").mkdir()
    (root / "habrpub" / "analysis.py").write_text("x = 1", encoding="utf-8")
    (root / "data").mkdir()
    for name in [
        "lex_texts_refetched.jsonl.gz",
        "lex_texts_raw.jsonl.gz",
        "lex_metrics_refetched.jsonl",
        "refetch_report.json",
    ]:
        (root / "data" / name).write_text(
            '{"html": "<p>чужой текст</p>", "text": "чужой текст"}\n', encoding="utf-8"
        )
    (root / "data" / "lex_metrics.jsonl").write_text('{"id": "1"}\n', encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "verification.md").write_text(
        "внутренний журнал", encoding="utf-8"
    )
    rels = {r.as_posix() for r in bp.collect(root)}
    assert rels == {"habrpub/analysis.py", "data/lex_metrics.jsonl"}


def test_wildcards_do_not_cross_directories(tmp_path):
    """Звёздочка в белом списке действует внутри одного имени: подложенные в
    подкаталоги файлы с «разрешёнными» расширениями в пакет не попадают."""
    root = tmp_path
    planted = [
        "private/secret.py",
        "misc/sub/sensitive.py",
        "figs/draft.png",
        "figs/sub/01_weekday.png",
        "tests/fixtures/key.txt",
        "tests/helper.py",
        "habrpub/draft/test.ipynb",
        "habrpub/notes.txt",
        "data/results/old/02_explore_results.json",
        "scratch.py",
    ]
    good = [
        "01_fetch_metadata.py",
        "habrpub/analysis.py",
        "tests/test_methods.py",
        "figs/01_weekday.png",
        "figs/captions.md",
        "data/results/02_explore_results.json",
    ]
    for rel in planted + good:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    rels = {r.as_posix() for r in bp.collect(root)}
    assert rels == set(good)


def test_every_figs_png_is_referenced_in_article():
    """Белый список пропускает любой рисунок вида figs/NN_*.png: рисунок, на
    который не ссылается article.md, прошёл бы сборку незаметно — например,
    устаревший рисунок со старой нумерацией и числами."""
    article = (ROOT / "article.md").read_text(encoding="utf-8")
    pngs = sorted((ROOT / "figs").glob("*.png"))
    assert pngs, "в figs/ нет рисунков"
    for png in pngs:
        assert png.name in article, f"{png.name} не упомянут в article.md"


def test_no_internal_review_references_in_public_files():
    """Пакет читают пользователи репозитория: ссылки на внутреннюю переписку
    и следы рабочего окружения им ничего не говорят. Проверка держит их на
    нуле во всех текстовых файлах пакета, кроме данных (в заголовках и тегах
    статей Хабра такие слова встречаются законно). Слова собраны из частей,
    чтобы тест не находил сам себя."""
    # Только устойчивые маркеры переписки и рабочего окружения. Обычные
    # технические слова («итерации бустинга», «ревью кода») сюда не входят.
    words = [
        "песоч" + "ниц",
        "по замеча" + r"ни[юям]",
        "в рев" + "ью",
        "редакци" + r"[ияей] ?\d",
        "верси" + r"[ияей] \d+(?![.\d])",  # номер версии, но не десятичная дробь
        r"\bП\. ?\d+[:.]",
        r"\bред ?\d",
        r"\bТ" + r"З\b",  # ссылки на внутренние технические задания
        "pla" + r"n\.pdf",
        "редакци" + r"[ияей] v\d",
        "стать" + r"[яиюей] v\d",
        "as dis" + "cussed",
        "as req" + "uested",
        "per re" + "view",
        "fixed in " + r"v\d",  # только конкретные следы внутренних исправлений
    ]
    pattern = re.compile("|".join(words), re.IGNORECASE)
    text_exts = {".py", ".md", ".cff", ".txt", ".cfg", ".toml", ".yml", ".yaml"}
    for rel in bp.collect(bp.ROOT):
        rel_str = str(rel).replace("\\", "/")
        if rel_str.startswith("data/"):
            continue
        p = bp.ROOT / rel
        if p.suffix not in text_exts:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        m = pattern.search(text)
        assert m is None, f"след рабочего процесса «{m.group(0)}» в {rel_str}"


def test_article_numbers_use_decimal_point_without_grouping():
    """Десятичная точка, как в коде, и никаких разделителей разрядов:
    «1.42», а не «1,42»; «46845», а не «46 845»."""
    import re

    for name in ("article_template.md", "article.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert not re.findall(r"\d,\d", text), f"десятичная запятая в {name}"
        assert not re.findall(r"\d[  ]\d{3}\b", text), f"разряды через пробел в {name}"


def test_table_captions_follow_tables():
    """Подпись таблицы идёт сразу под таблицей (или её подстановкой)."""
    import re

    lines = (ROOT / "article_template.md").read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^\*Таблица \d+\.", line):
            prev = lines[i - 2]
            assert lines[i - 1] == "" and (
                prev.startswith("|") or prev.startswith("{{")
            ), f"подпись не под таблицей: {line[:40]}"


def test_public_package_has_every_manifest_input_for_metrics_only():
    """run_analysis.py --metrics-only проверяет по манифесту все входы, кроме
    полных текстов. Если хотя бы одного нет в белом списке, команда быстрого
    старта из README падает в публичном пакете."""
    import json

    rels = {r.as_posix() for r in bp.collect(bp.ROOT)}
    manifest = json.loads((ROOT / "data/input_manifest.json").read_text())
    needed = [n for n in manifest["sha256"] if n != "lex_texts_raw.jsonl.gz"]
    needed += list(manifest["derived_sha256"])
    missing = [n for n in needed if "data/" + n not in rels]
    assert not missing, f"в публичный пакет не попадут входы манифеста: {missing}"
