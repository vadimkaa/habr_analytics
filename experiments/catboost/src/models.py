"""Единый интерфейс обучения CatBoost и контрольных моделей LightGBM.

Единый контракт: fit / predict_views / predict_score / feature_names /
model_metadata одинаковы для обеих библиотек и всех loss. Целевая
трансформация (TargetSpec) отделена от модели; decode вызывается ровно один
раз; прогон численного здоровья обязателен (np.isfinite, дисперсия).

Ранняя остановка - только с настоящей валидацией (train -> val); финальный
кандидат переобучается на train+validation с фиксированным числом деревьев; use_best_model на финальном тесте не включается.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .targets import TargetSpec


@dataclass
class FitResult:
    model: object
    pred_views: dict[str, np.ndarray]  # фонд -> предсказанные просмотры
    score: dict[str, np.ndarray]  # фонд -> ранжирующий score (монотонен по views)
    best_iteration: int | None
    fit_time_sec: float
    params: dict
    metadata: dict = field(default_factory=dict)
    target: TargetSpec | None = None

    def predict_views_for(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Декодированные просмотры для произвольной рамки (grid/policy)."""
        raw = self.model.predict(df[features])
        assert self.target is not None
        return self.target.decode(raw)


# Конфигурации CatBoost зафиксированы до обучения.
CATBOOST_CONFIGS: dict[str, dict] = {
    "C0_lgb_analog": dict(
        iterations=300,
        learning_rate=0.04,
        grow_policy="Lossguide",
        max_leaves=15,
        min_data_in_leaf=60,
        l2_leaf_reg=3.0,
        bootstrap_type="Bernoulli",
        subsample=1.0,
        random_seed=config.SEED,
        thread_count=config.THREAD_COUNT,
        allow_writing_files=False,
    ),
    "C1_symmetric": dict(
        iterations=1000,
        learning_rate=0.04,
        grow_policy="SymmetricTree",
        depth=6,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        random_seed=config.SEED,
        thread_count=config.THREAD_COUNT,
        allow_writing_files=False,
    ),
    "C2_conservative": dict(
        iterations=800,
        learning_rate=0.03,
        grow_policy="SymmetricTree",
        depth=4,
        l2_leaf_reg=10.0,
        random_strength=1.0,
        random_seed=config.SEED,
        thread_count=config.THREAD_COUNT,
        allow_writing_files=False,
    ),
}


def catboost_objective(loss_id: str) -> tuple[str, str]:
    """(objective, eval_metric) из configs/losses.yaml."""
    cfg = config.losses_cfg()["losses"][loss_id]
    return cfg["library_objective"], cfg["eval_metric"]


def assert_finite(pred: np.ndarray, where: str) -> None:
    p = np.asarray(pred, float)
    if not np.isfinite(p).all():
        raise ValueError(
            f"невырожденность нарушена в {where}: {np.isnan(p).sum()} NaN, "
            f"{np.isinf(p).sum()} inf - проверь масштаб цели"
        )
    if np.nanstd(p) == 0:
        raise ValueError(f"вырожденная модель (константный прогноз) в {where}")


def fit_catboost(
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
    features: list[str],
    cat_features: list[str],
    loss_id: str,
    config_id: str = "C1_symmetric",
    target: TargetSpec | None = None,
    early_stopping: bool = True,
    iterations_fixed: int | None = None,
    has_time: bool = False,
    seed: int | None = None,
    timestamp_col: str = "dt_utc",
    loss_scale: float | None = None,
    extra_params: dict | None = None,
) -> FitResult:
    """Обучение CatBoost. При valid=None и iterations_fixed - финальный рефит
    на train+validation без ранней остановки."""
    from catboost import CatBoostRegressor, Pool

    losses = config.losses_cfg()["losses"]
    if loss_id not in losses:
        raise KeyError(loss_id)
    objective, eval_metric = catboost_objective(loss_id)
    target = target or TargetSpec(losses[loss_id]["target"])
    params = dict(CATBOOST_CONFIGS[config_id])
    params["loss_function"] = objective
    params["eval_metric"] = eval_metric
    if seed is not None:
        params["random_seed"] = seed
    # has_time - параметр модели: хронологический порядок строк не перемешивается.
    # Наши фолды отсортированы по dt_utc в prepare().
    params["has_time"] = bool(has_time)
    if extra_params:
        params.update(extra_params)

    t0 = time.time()
    y = target.encode(train["views"].to_numpy())
    ts_train = pd.to_datetime(train[timestamp_col]).astype("int64").to_numpy()
    train_pool = Pool(
        train[features],
        label=y,
        cat_features=cat_features or None,
        timestamp=ts_train,
    )
    fit_kwargs: dict = dict(verbose=False)
    if valid is not None and early_stopping:
        yv = target.encode(valid["views"].to_numpy())
        ts_valid = pd.to_datetime(valid[timestamp_col]).astype("int64").to_numpy()
        valid_pool = Pool(
            valid[features],
            label=yv,
            cat_features=cat_features or None,
            timestamp=ts_valid,
        )
        fit_kwargs.update(
            eval_set=valid_pool,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            use_best_model=True,
        )
    if iterations_fixed is not None:
        params["iterations"] = int(iterations_fixed)
    m = CatBoostRegressor(**params)
    m.fit(train_pool, **fit_kwargs)
    # Число деревьев лучшей модели:
    # - get_best_iteration() возвращает ИНДЕКС лучшей итерации, нумерация С НУЛЯ,
    #   поэтому фактическое число деревьев = best_iteration + 1;
    # - 0 индексов означает ОДНО дерево (первое), а не ноль деревьев;
    # - None (улучшение не найдено / eval_set не задан) -> все деревья params["iterations"].
    # Сама модель с use_best_model=True уже содержит best_iteration+1 деревьев;
    # refit_final использует это же число, а не индекс.
    if valid is not None and early_stopping:
        bi = m.get_best_iteration()
        best_iter = (int(bi) + 1) if bi is not None else int(params["iterations"])
    else:
        best_iter = int(params["iterations"])

    pred_views, score = {}, {}
    frames: dict[str, pd.DataFrame] = {"train": train}
    if valid is not None:
        frames["valid"] = valid
    for name, frame in frames.items():
        raw = m.predict(frame[features])
        assert_finite(raw, f"{loss_id}/{config_id}:{name}")
        pv = target.decode(raw)
        assert_finite(pv, f"{loss_id}/{config_id}:{name}:views")
        pred_views[name] = pv
        # score монотонен по предсказанным вьюхам для всех перечисленных loss:
        # RMSE/log1p - log-число, Poisson/Quantile - медиана/интенсивность.
        score[name] = np.asarray(raw, float)

    meta = dict(
        library="catboost",
        config_id=config_id,
        loss_id=loss_id,
        objective=objective,
        eval_metric=eval_metric,
        target_kind=target.kind,
        loss_scale=loss_scale,
        features=list(features),
        cat_features=list(cat_features),
        has_time=bool(has_time),
        n_train=len(train),
        best_iteration=best_iter,
    )
    return FitResult(
        model=m,
        pred_views=pred_views,
        score=score,
        best_iteration=best_iter,
        fit_time_sec=round(time.time() - t0, 2),
        params=params,
        metadata=meta,
        target=target,
    )


# Параметры контрольной модели заданы в configs/losses.yaml.
def fit_lightgbm(
    train: pd.DataFrame,
    valid: pd.DataFrame | None,
    features: list[str],
    cat_features: list[str],
    loss_id: str,
) -> FitResult:
    """Контрольные LightGBM-модели с фиксированными параметрами.

    Параметры заданы в конфигурации и не оптимизируются на аудите
    (включая категориальные dow/hour/month и отсутствие early stopping)."""
    import lightgbm as lgb

    cfg = config.losses_cfg()["lightgbm_controls"][loss_id]
    target = TargetSpec(cfg["target"])
    params = dict(cfg["params"])
    cat_used = [c for c in ("dow", "hour", "month") if c in features]

    t0 = time.time()
    m = lgb.LGBMRegressor(**params)
    # Оба контрольных LightGBM используют одинаковое число деревьев;
    # сравнивается целевая функция, а не ранняя остановка на validation.
    m.fit(
        train[features],
        target.encode(train["views"].to_numpy()),
        categorical_feature=cat_used,
    )
    pred_views, score = {}, {}
    for name, frame in [("train", train)] + (
        [("valid", valid)] if valid is not None else []
    ):
        raw = m.predict(frame[features])
        assert_finite(raw, f"{loss_id}:{name}")
        pv = target.decode(raw)
        pred_views[name] = pv
        score[name] = np.asarray(raw, float)
    meta = dict(
        library="lightgbm",
        config_id=loss_id,
        loss_id=loss_id,
        objective=params.get("objective"),
        target_kind=target.kind,
        features=list(features),
        cat_features=cat_used,
        n_train=len(train),
        best_iteration=None,
    )
    return FitResult(
        model=m,
        pred_views=pred_views,
        score=score,
        best_iteration=None,
        fit_time_sec=round(time.time() - t0, 2),
        params=params,
        metadata=meta,
        target=target,
    )


def refit_final(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    cat_features: list[str],
    loss_id: str,
    config_id: str,
    best_iteration: int,
    library: str,
    has_time: bool = False,
    seed: int | None = None,
) -> FitResult:
    """Финальный кандидат: train+validation, фиксированное число деревьев.

    Число деревьев выбрано на валидации заранее; валидационные метрики
    повторно не вычисляются (иначе набор становится набором настройки)."""
    if library == "catboost":
        tr = pd.concat([train, valid], ignore_index=True)
        return fit_catboost(
            tr,
            None,
            features,
            cat_features,
            loss_id,
            config_id,
            early_stopping=False,
            iterations_fixed=best_iteration,
            has_time=has_time,
            seed=seed,
        )
    tr = pd.concat([train, valid], ignore_index=True)
    return fit_lightgbm(tr, None, features, cat_features, loss_id)
