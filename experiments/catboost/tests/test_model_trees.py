"""Семантика best_iteration и число деревьев финальной модели.

CatBoost get_best_iteration() возвращает ИНДЕКС лучшей итерации (нумерация с
нуля), поэтому фактическое число деревьев = best_iteration + 1. Проверяется:
финальная модель после train+validation имеет ровно столько деревьев, сколько
было в лучшей модели валидации; индекс 0 означает одно дерево; при
iterations_fixed деревья равны зафиксированному числу.
"""

from __future__ import annotations

import pytest

# Эти тесты обучают настоящие модели; без CatBoost они пропускаются.
pytest.importorskip("catboost")

import pytest

from src import models as MO


def test_best_iteration_is_count_not_index(small_fit):
    """fit.best_iteration = get_best_iteration() + 1 = фактическое число деревьев."""
    fit = MO.fit_catboost(
        small_fit["train"],
        small_fit["test"],
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=80,
    )
    # 80 деревьев на синтетике без ранней остановки улучшений... проверяем
    # согласованность: модель обучена до ранней остановки или до 80 деревьев,
    # и best_iteration == реальному числу деревьев модели.
    assert fit.best_iteration == fit.model.tree_count_


def test_refit_final_trees_equal_validation_best(small_fit):
    """Финальный рефит на train+validation: деревьев ровно столько, сколько
    вернула валидационная процедура (индекс + 1)."""
    fit = MO.fit_catboost(
        small_fit["train"],
        small_fit["test"],
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=60,
    )
    final = MO.refit_final(
        small_fit["train"],
        small_fit["test"],
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        "C1_symmetric",
        fit.best_iteration,
        "catboost",
    )
    assert final.model.tree_count_ == fit.best_iteration
    assert final.best_iteration == fit.best_iteration


def test_best_iteration_zero_means_one_tree(small_fit):
    """Индекс 0 = одно дерево: refit с best_iteration=1 строит 1 дерево,
    а не ноль; 0 деревьев быть не может."""
    final = MO.refit_final(
        small_fit["train"],
        small_fit["test"],
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        "C1_symmetric",
        1,
        "catboost",
    )
    assert final.model.tree_count_ == 1


def test_fixed_iterations_without_validation(small_fit):
    """iterations_fixed без eval_set: best_iteration = зафиксированному числу."""
    fit = MO.fit_catboost(
        small_fit["train"],
        None,
        small_fit["features"],
        [],
        "L01_rmse_log1p",
        iterations_fixed=25,
    )
    assert fit.best_iteration == 25
    assert fit.model.tree_count_ == 25
