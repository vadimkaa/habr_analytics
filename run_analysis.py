"""Единая точка запуска. По умолчанию полный offline-пересчёт из исходного кэша."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distributions, version

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics-only",
        action="store_true",
        help="Использовать готовые lex_metrics.jsonl, не перечитывать полные тексты",
    )
    args = ap.parse_args()
    manifest = json.loads((ROOT / "data/input_manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if args.metrics_only and name == "lex_texts_raw.jsonl.gz":
            continue
        p = ROOT / "data" / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise SystemExit(
                f"Входной файл отсутствует или изменён: {name}. Старые snapshot-константы неприменимы к новой выгрузке."
            )
    if args.metrics_only:
        for name, expected in manifest["derived_sha256"].items():
            if (
                hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
                != expected
            ):
                raise SystemExit(f"Изменились подготовленные метрики: {name}")
    steps = [
        "02_explore_ai_12m.py",
        "03_calendar_wide.py",
        "04_calendar_ai.py",
        "05_density.py",
    ]
    if not args.metrics_only:
        steps += ["06_text_sample.py"]
    steps += [
        "07_readability.py",
        "08_model.py",
        "09_robustness.py",
        "11_leakage_diagnostic.py",
        "12_slot_check_fig.py",
        "14_extra_checks.py",
        "15_ml_figures.py",
        "16_review_robustness.py",
        "17_final_sensitivity.py",
        "18_sensitivity_figures.py",
        "19_editorial_checks.py",
        "10_build_article.py",
    ]
    for step in steps:
        print(f">>> {step}", flush=True)
        command = [sys.executable, str(ROOT / step)]
        env = None
        if step == "17_final_sensitivity.py":
            command.append("--check")
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        subprocess.run(command, cwd=ROOT, check=True, env=env)
    # тесты методики — часть прогона: пересчёт, прошедший тесты, фиксируется целиком
    print(">>> pytest -q", flush=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
    # Записываем версии тех пакетов, которые объявлены в манифестах
    # воспроизведения. Полное окружение также сохраняется ниже.
    deps = set()
    for req_path in (
        ROOT / "requirements.txt",
        ROOT / "experiments/catboost/requirements-catboost.txt",
    ):
        for line in req_path.read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                deps.add(entry.split("==", 1)[0].split(">=", 1)[0].lower())
    def installed_version(package: str) -> str | None:
        try:
            return version(package)
        except PackageNotFoundError:
            # Опциональный пакет может отсутствовать в режиме metrics-only.
            return None

    (ROOT / "data/results/run_environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {p: installed_version(p) for p in sorted(deps)},
                # Полный список установленных пакетов: по нему можно собрать
                # точный lock-файл окружения, в котором шёл пересчёт.
                "all_packages": dict(
                    sorted((d.metadata["Name"], d.version) for d in distributions())
                ),
                "mode": "metrics-only" if args.metrics_only else "full-offline",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
