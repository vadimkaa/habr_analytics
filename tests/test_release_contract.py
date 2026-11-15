"""Проверки публичной сборки, контрольных параметров и шаблона статьи."""

from pathlib import Path
import importlib.util

import yaml

from habrpub import legacy_model

ROOT = Path(__file__).resolve().parents[1]


def test_control_lightgbm_matches_saved_protocol():
    config = yaml.safe_load((ROOT / "experiments/catboost/configs/losses.yaml").read_text())
    assert legacy_model.PARAMS == config["lightgbm_controls"]["lgb_poisson_v24"]["params"]


def test_rendered_article_has_no_unexpanded_variables():
    article = (ROOT / "article.md").read_text()
    assert "{{" not in article and "}}" not in article


def test_public_gitignore_is_single_source():
    spec = importlib.util.spec_from_file_location("public_builder", ROOT / "build_public.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PUBLIC_GITIGNORE == (ROOT / ".gitignore").read_text()


def test_named_lightgbm_reference_has_same_features_as_script():
    spec = importlib.util.spec_from_file_location("legacy_script", ROOT / "08_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.FULL == legacy_model.FULL
    assert module.prepare is legacy_model.prepare


def test_code_hash_includes_stage_modules():
    from experiments.catboost.src import config, registry
    stage_files = set((config.CATBOOST_DIR / "stages").glob("*.py"))
    indexed_files = set(registry.code_sources())
    assert stage_files and stage_files <= indexed_files
    assert len(registry.code_hash()) == 12  # Display length, not source count.


def test_caption_manifest_matches_rendered_article():
    import re
    article = (ROOT / "article.md").read_text()
    captions = (ROOT / "figs/captions.md").read_text()
    for path, label in re.findall(
        r"!\[[^\]]*\]\((figs/[^)]+)\)\s*\n\s*\*([^\n*]+)\*", article
    ):
        assert f"{path} — {label.strip()}" in captions


def test_registry_source_type_annotations_resolve():
    """Prevent missing Path imports hidden by postponed annotations."""
    from typing import get_type_hints
    from experiments.catboost.src import registry
    assert get_type_hints(registry.code_sources)["return"] == list[Path]


def test_catboost_random_seed_matches_frozen_protocol():
    """Statistical defaults and the frozen YAML share the CatBoost seed."""
    import inspect
    from experiments.catboost.src import config, metrics, statistics

    assert config.SEED == config.protocol()["limits"]["catboost_seed"]
    for func in (
        statistics.paired_delta_rho_ci,
        statistics.ratio_ci_author,
        statistics.rho_ci_author,
        statistics.delta_rho_two_scores,
        statistics.shuffle_labels_within_month,
        statistics.bootstrap_median_ci,
        metrics.b_metrics_age,
    ):
        assert inspect.signature(func).parameters["seed"].default == config.SEED
