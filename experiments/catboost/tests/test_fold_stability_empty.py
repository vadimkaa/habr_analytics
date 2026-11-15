"""Результат устойчивости сохраняется даже без допустимых векторов слотов."""

import pandas as pd

from stages import policy


def test_empty_rank_vectors_still_save_fold_stability(monkeypatch):
    frame = pd.DataFrame({"id": ["a"], "author_key": ["author"], "week_start": ["2026-01-05"], "month_id": ["2026-01"], "is_weekend": [False]})
    monkeypatch.setattr(policy.splits, "materialize_corpus", lambda _: frame)
    monkeypatch.setattr(policy.splits, "rolling_splits", lambda df, corpus: [dict(train=frame, test=frame, cut="2026-01")])
    monkeypatch.setattr(policy, "features_for", lambda _: [])
    monkeypatch.setattr(policy.models, "fit_catboost", lambda *args, **kwargs: object())
    monkeypatch.setattr(policy.SP, "personalized_scheduling", lambda *args, **kwargs: {"n_articles": 0, "month_rank_vectors": {}, "reco_per_article": {}, "per_article": None})
    saved = {}
    monkeypatch.setattr(policy.registry, "save_metrics", lambda name, data: saved.update({name: data}))
    policy._b2_fold_stability(["ai"], {"ai": "L05"})
    item = saved["b2_stability_folds"]["ai:L05"]
    assert item["folds"] == []
    assert item["mean_rank_agreement"] == {}
    assert "2026-01" in item["scale_metrics"]
