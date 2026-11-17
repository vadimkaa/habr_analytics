"""Собрать каталог публичного репозитория из разрешённых файлов.

Разрешённые пути перечислены в ALLOWED и сверяются по сегментам: шаблон
одного уровня не открывает доступ к файлам вложенных директорий. Полные
тексты и временные результаты скачивания не входят в пакет. Сборка работает
локально и не взаимодействует с git или сетью."""

import argparse
import fnmatch
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

# Производные CSV (part2_*, habr_12m, readability_analysis) публикуются как
# готовые промежуточные таблицы; run_analysis.py перезаписывает их при
# пересчёте. По хешам из input_manifest.json проверяются только входы (sha256)
# и подготовленные метрики (derived_sha256: lex_metrics.jsonl, image_counts.csv).
# Все входы, кроме полных текстов lex_texts_raw.jsonl.gz, обязаны быть в
# пакете — иначе --metrics-only остановится на проверке манифеста.
# preprocessing_baseline_*.jsonl — числовые метрики упрощённой предобработки
# (без текстов), по ним шаг 07 строит таблицу «Насколько всё зависит от
# предобработки».
ALLOWED = [
    "[0-9][0-9]_*.py",
    "run_analysis.py",
    "build_public.py",
    "collect_panel.py",
    "fetch_texts.py",
    "conftest.py",
    "habrpub/*.py",
    "tests/test_*.py",
    "figs/[0-9][0-9]_*.png",
    "figs/captions.md",
    "README.md",
    "VERSION",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    "LICENSE",
    "LICENSE-DATA.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "article.md",
    "article_template.md",
    "short_article/article.md",
    "short_article/figures/weekend_density.png",
    "short_article/figures/readability.png",
    "requirements.txt",
    "requirements-docx.txt",
    "requirements-dev.txt",
    "docs/audit.md",
    "docs/reproducibility.md",
    "docs/data_dictionary.md",
    "docs/collection.md",
    "docs/architecture.md",
    "docs/robustness.md",
    "data/habr_ai_ml_raw.jsonl.gz",
    "data/habr_wide_raw.jsonl.gz",
    "data/lex_metrics.jsonl",
    "data/preprocessing_baseline_full.jsonl",
    "data/preprocessing_baseline_pre_only.jsonl",
    "data/image_counts.csv",
    "data/readability_analysis.csv",
    "data/part2_ai.csv",
    "data/part2_wide.csv",
    "data/habr_12m.csv",
    "data/model_grid_ai.csv",
    "data/model_grid_wide.csv",
    "data/input_manifest.json",
    "data/text_sample_manifest.json",
    "data/production_calendar_2016_2026.csv",
    "data/production_calendar_source.json",
    "data/sources/sources_index.csv",
    "data/results/*.json",
    # Модели и протокол сравнения.
    "experiments/catboost/README.md",
    "experiments/catboost/requirements-catboost.txt",
    "experiments/catboost/run_experiments.py",
    "experiments/catboost/src/*.py",
    "experiments/catboost/stages/*.py",
    "experiments/catboost/configs/*.yaml",
    "experiments/catboost/tests/*.py",
    # Итоговые агрегаты, модели и графики.
    "experiments/catboost/results/*.json",
    "experiments/catboost/results/*.md",
    "experiments/catboost/results/selected_policies.json",
    "experiments/catboost/results/policy.json",
    "experiments/catboost/results/registry.csv",
    "experiments/catboost/results/metrics/*.json",
    "experiments/catboost/results/metrics/*.csv",
    "experiments/catboost/results/models/*.cbm",
    "experiments/catboost/results/figures/*.png",
    # Прогнозы для статей: идентификаторы, метаданные и расчёты,
    # без текстов статей. Таблицы альтернативных слотов B2 (predictions/b2_slots,
    # около 33 МБ) в репозиторий не входят: они прикладываются к релизу на GitHub
    # и могут быть воспроизведены стадией policy.
    "experiments/catboost/results/predictions/article_level/*.csv.gz",
    "experiments/catboost/results/predictions/*audit_exposed*.csv.gz",
]

# Правила игнорирования — часть публичного проекта; не держим их копию в коде.
PUBLIC_GITIGNORE = (ROOT / ".gitignore").read_text(encoding="utf-8")


def allowed(rel: Path) -> bool:
    """Совпадение с белым списком по сегментам пути.

    Обычный fnmatch разрешает звёздочке захватывать «/», и шаблон "*.py"
    пропустил бы private/secret.py. Здесь шаблон и путь сравниваются
    покомпонентно: число уровней должно совпасть, а «*» действует только
    внутри одного имени."""
    parts = rel.parts
    for pat in ALLOWED:
        pat_parts = pat.split("/")
        if len(pat_parts) == len(parts) and all(
            fnmatch.fnmatchcase(name, p) for name, p in zip(parts, pat_parts)
        ):
            return True
    return False


def collect(root: Path):
    """Относительные пути файлов к публикации: только белый список.

    Файл попадает в пакет, только если совпадает с шаблоном из ALLOWED.
    Шаблоны с «*» пропустят и новый файл подходящего вида в том же каталоге
    (например, habrpub/new_module.py или data/results/new.json), но не в
    подкаталоге и не в другом месте. Файлы иного вида и вне перечисленных
    каталогов без правки ALLOWED в пакет не попадают."""
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if set(rel.parts) & {"__pycache__", ".pytest_cache", ".git", ".venv"}:
            continue
        if p.suffix == ".pyc":
            continue
        if allowed(rel):
            files.append(rel)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("Выходной каталог уже существует; выберите новый путь.")
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("Выберите каталог вне исходного проекта.")
    out.mkdir(parents=True)
    files = collect(ROOT)
    # Дополнительная защита от включения скачанных текстов публикаций.
    for rel in files:
        assert "lex_texts" not in rel.name and "lex_prose" not in rel.name, rel
    for rel in files:
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dest)
    (out / ".gitignore").write_text(PUBLIC_GITIGNORE)
    print(out, "| файлов:", len(files))


if __name__ == "__main__":
    main()
