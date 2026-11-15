"""Командная строка эксперимента CatBoost.

Этапы запуска и протокол описаны в experiments/catboost/README.md.
Примеры:
  python experiments/catboost/run_experiments.py audit
  python experiments/catboost/run_experiments.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

CATBOOST_DIR = Path(__file__).resolve().parent
if str(CATBOOST_DIR) not in sys.path:
    sys.path.insert(0, str(CATBOOST_DIR))

from src import data_contract
from stages.common import log
from stages.fit import stage_control, poisson_scale_check, stage_loss, stage_tune, stage_select
from stages.policy import stage_policy
from stages.robustness import stage_robust, robustness_nonintersect, rolling_folds_for, stage_final_checks
from stages.reporting import stage_calibration, stage_report, stage_freeze

def main(argv: list[str]) -> None:
    if not argv:
        argv = ["all"]
    stage = argv[0]
    corpora = [a for a in argv[1:] if a in ("ai", "wide")] or ["ai", "wide"]
    if stage == "freeze":
        stage_freeze()
    elif stage == "audit":
        p = data_contract.build_quality_report()
        log(f"аудит записан: {p}")
    elif stage == "control":
        stage_control(corpora)
        poisson_scale_check()
    elif stage == "loss":
        stage_loss(corpora)
    elif stage == "loss_w2":
        stage_loss(corpora, ["L04_huber_log1p", "L05_quantile_log1p"])
    elif stage == "tune":
        stage_tune()
    elif stage == "select":
        stage_select(corpora)
    elif stage == "policy":
        stage_policy(corpora)
    elif stage == "robust":
        stage_robust(corpora)
        robustness_nonintersect()
        rolling_folds_for(corpora)
    elif stage == "final":
        stage_final_checks(corpora)
    elif stage == "calibration":
        stage_calibration()
    elif stage == "report":
        stage_report()
    elif stage == "all":
        stage_freeze()
        data_contract.build_quality_report()
        stage_control(corpora)
        poisson_scale_check()
        stage_loss(corpora)
        stage_loss(corpora, ["L04_huber_log1p", "L05_quantile_log1p"])
        stage_tune()
        stage_select(corpora)
        stage_policy(corpora)
        stage_robust(corpora)
        robustness_nonintersect()
        rolling_folds_for(corpora)
        stage_final_checks(corpora)
        stage_calibration()
        stage_report()
    else:
        raise SystemExit(f"неизвестная стадия: {stage}")



if __name__ == "__main__":
    main(sys.argv[1:])
