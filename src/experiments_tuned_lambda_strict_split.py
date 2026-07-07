"""
experiments_strict_split.py – exchangeability-safe experiment runners.

The functions here fix the data-leakage issue present in experiments_original.py:
when lambda is selected via a validation fold, that fold must be kept strictly
separate from the calibration set used to compute qhat.  Using the same data
for both hyperparameter selection and quantile estimation violates the
exchangeability assumption that underpins conformal prediction coverage
guarantees.

Each function in this module maintains three non-overlapping splits per trial:

    ┌─────────────┬──────────────────────┬────────────────┐
    │  cal (10 %) │  val (10 %)           │  test (80 %)   │
    └─────────────┴──────────────────────┴────────────────┘
         ↑                ↑                     ↑
    qhat estimation   λ selection           evaluation

Public API
----------
run_raps_ma_avg_opt_strict_split(...)
    strict_split MA superclass-mass penalty, λ chosen on val by min avg-set-size.

run_raps_ms_avg_opt_strict_split(...)
    strict_split MS binary-distance penalty, λ chosen on val by min avg-set-size.

run_raps_ma_binary_dist_avg_opt_strict_split(...)
    strict_split MA binary-distance penalty, λ chosen on val by min avg-set-size.
"""

from __future__ import annotations

import numpy as np

from .config import (
    ALPHA, NUM_CLASS, SIZE_SUPER_CLASS,
    CAL_FRAC, VAL_FRAC, T, RAND, DISALLOW_ZERO,
    LAM_REG, K_REG,
)
from .score_functions import (
    build_reg_vec,
    raps_cal_scores_binary_dist,
    raps_cal_scores_superclass,
    build_raps_prediction_sets_binary_dist,
    build_raps_prediction_sets_superclass,
    conformal_quantile,
)
from .metrics import (
    marginal_coverage,
    class_conditional_coverage,
    worst_class_coverage_deviation,
    top_k_worst_coverage_deviation,
    set_size_stats,
    mean_distinct_superclasses,
    print_metrics,
)
from .experiments_original import _aggregate, _format_for_print, _make_config


# ─────────────────────────────────────────────────────────────────────────────
# MA – superclass-mass penalty (strict_split)
# ─────────────────────────────────────────────────────────────────────────────

def run_raps_ma_avg_opt_strict_split(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda_2_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    RAPS + superclass-mass (MA) penalty – exchangeability-safe version.

    Lambda is selected on a dedicated validation fold; qhat is then computed
    solely on the calibration fold.
    """
    cfg = _make_config(config)
    alpha = cfg["alpha"]; num_class = cfg["num_class"]
    size_sc = cfg["size_super_class"]
    T_ = cfg["T"]; rand = cfg["rand"]; disallow_zero = cfg["disallow_zero_sets"]
    reg_vec = build_reg_vec(num_class, cfg["k_reg"], cfg["lam_reg"])

    cov_acc = 0.0; cc_acc = np.zeros(num_class)
    size_acc = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc = 0.0; avg_sizes: list[float] = []

    for _ in range(T_):
        cal_smx, cal_labels, val_smx, val_labels, test_smx, test_labels, n_cal = \
            _three_way_split(probabilities, labels, cfg)

        # ── select lambda on val ──────────────────────────────────────────
        opt_lamda, _ = _select_lambda_by_avg_size_superclass(
            cal_smx, cal_labels, val_smx, val_labels,
            reg_vec, adjacency_matrix, adjacency_matrix_smaller,
            lamda_2_values, n_cal, num_class, size_sc, alpha, rand, disallow_zero,
        )

        # ── refit qhat on cal only ────────────────────────────────────────
        scores = raps_cal_scores_superclass(
            cal_smx, cal_labels, reg_vec,
            adjacency_matrix_smaller, opt_lamda, rand=rand,
        )
        qhat = conformal_quantile(scores, n_cal, alpha)

        prediction_sets = build_raps_prediction_sets_superclass(
            test_smx, reg_vec, adjacency_matrix, adjacency_matrix_smaller,
            opt_lamda, qhat, num_class, size_sc, rand=rand,
            disallow_zero_sets=disallow_zero,
        )

        cov_acc += marginal_coverage(prediction_sets, test_labels)
        cc_acc  += class_conditional_coverage(prediction_sets, test_labels, num_class)
        s = set_size_stats(prediction_sets)
        for k in size_acc: size_acc[k] += s[k]
        avg_sizes.append(s["mean"])
        sc_acc += mean_distinct_superclasses(prediction_sets, adjacency_matrix, num_class)

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MA binary-distance penalty (strict_split)
# ─────────────────────────────────────────────────────────────────────────────

def run_raps_ma_binary_dist_avg_opt_strict_split(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_real: np.ndarray,
    lamda_2_values: list[float],
    *,
    config: dict | None = None,
    fixed_lambda: float | None = None,
) -> dict:
    """
    RAPS + binary-distance penalty (MA) – strict-split version.
    If fixed_lambda is provided, that value is used directly instead of tuning.
    """
    return _run_binary_dist_strict_split(
        probabilities, labels, adjacency_matrix, adjacency_matrix_real,
        lamda_2_values, config=config, fixed_lambda=fixed_lambda,
    )


def run_raps_ms_avg_opt_strict_split(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_ms: np.ndarray,
    lamda_2_values: list[float],
    *,
    config: dict | None = None,
    fixed_lambda: float | None = None,
) -> dict:
    """
    RAPS + binary-distance penalty (MS cosine-similarity) – strict-split version.
    If fixed_lambda is provided, that value is used directly instead of tuning.
    """
    return _run_binary_dist_strict_split(
        probabilities, labels, adjacency_matrix, adjacency_matrix_ms,
        lamda_2_values, config=config, fixed_lambda=fixed_lambda,
    )


def _run_binary_dist_strict_split(
    probabilities, labels, adjacency_matrix, penalty_matrix, lamda_2_values,
    *, config, fixed_lambda: float | None = None,
) -> dict:
    cfg = _make_config(config)
    alpha = cfg["alpha"]; num_class = cfg["num_class"]
    T_ = cfg["T"]; rand = cfg["rand"]; disallow_zero = cfg["disallow_zero_sets"]
    reg_vec = build_reg_vec(num_class, cfg["k_reg"], cfg["lam_reg"])

    cov_acc = 0.0; cc_acc = np.zeros(num_class)
    size_acc = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc = 0.0; avg_sizes: list[float] = []

    for _ in range(T_):
        cal_smx, cal_labels, val_smx, val_labels, test_smx, test_labels, n_cal = \
            _three_way_split(probabilities, labels, cfg)

        if fixed_lambda is not None:
            opt_lamda = float(fixed_lambda)
        else:
            opt_lamda = 0; opt_size = float("inf")
            for lam in lamda_2_values:
                # evaluate on val using qhat from cal
                scores = raps_cal_scores_binary_dist(
                    cal_smx, cal_labels, reg_vec, penalty_matrix, lam, rand=rand,
                )
                qhat = conformal_quantile(scores, n_cal, alpha)
                ps_val = build_raps_prediction_sets_binary_dist(
                    val_smx, reg_vec, penalty_matrix, lam, qhat,
                    rand=rand, disallow_zero_sets=disallow_zero,
                )
                size_val = ps_val.sum(axis=1).mean()
                if lam == 0:
                    opt_size = size_val + 10; opt_lamda = 0
                elif size_val < opt_size:
                    opt_size = size_val; opt_lamda = lam

        # refit qhat on cal only
        scores = raps_cal_scores_binary_dist(
            cal_smx, cal_labels, reg_vec, penalty_matrix, opt_lamda, rand=rand,
        )
        qhat = conformal_quantile(scores, n_cal, alpha)

        prediction_sets = build_raps_prediction_sets_binary_dist(
            test_smx, reg_vec, penalty_matrix, opt_lamda, qhat,
            rand=rand, disallow_zero_sets=disallow_zero,
        )

        cov_acc += marginal_coverage(prediction_sets, test_labels)
        cc_acc  += class_conditional_coverage(prediction_sets, test_labels, num_class)
        s = set_size_stats(prediction_sets)
        for k in size_acc: size_acc[k] += s[k]
        avg_sizes.append(s["mean"])
        sc_acc += mean_distinct_superclasses(prediction_sets, adjacency_matrix, num_class)

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _three_way_split(probabilities, labels, cfg):
    """Return (cal_smx, cal_labels, val_smx, val_labels, test_smx, test_labels, n_cal)."""
    n_total = probabilities.shape[0]
    n_cal   = int(cfg["cal_frac"] * n_total)
    n_val   = int(cfg["val_frac"] * n_total)
    idx     = np.random.permutation(n_total)
    cal_idx, val_idx, test_idx = idx[:n_cal], idx[n_cal:n_cal+n_val], idx[n_cal+n_val:]
    return (
        probabilities[cal_idx], labels[cal_idx],
        probabilities[val_idx], labels[val_idx],
        probabilities[test_idx], labels[test_idx],
        n_cal,
    )


def _select_lambda_by_avg_size_superclass(
    cal_smx, cal_labels, val_smx, val_labels,
    reg_vec, adjacency_matrix, adjacency_matrix_smaller,
    lamda_2_values, n_cal, num_class, size_sc, alpha, rand, disallow_zero,
):
    opt_lamda = 0; opt_size = float("inf")
    for lam in lamda_2_values:
        scores = raps_cal_scores_superclass(
            cal_smx, cal_labels, reg_vec,
            adjacency_matrix_smaller, lam, rand=rand,
        )
        qhat = conformal_quantile(scores, n_cal, alpha)
        ps_val = build_raps_prediction_sets_superclass(
            val_smx, reg_vec, adjacency_matrix, adjacency_matrix_smaller,
            lam, qhat, num_class, size_sc, rand=rand,
            disallow_zero_sets=disallow_zero,
        )
        size_val = ps_val.sum(axis=1).mean()
        if lam == 0:
            opt_size = size_val + 10; opt_lamda = 0
        elif size_val < opt_size:
            opt_size = size_val; opt_lamda = lam
    return opt_lamda, opt_size
