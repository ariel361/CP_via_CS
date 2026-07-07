"""
experiments_original.py – experiment runners that replicate the results from
the original RAPS paper (Angelopoulos et al., 2020).

These functions do NOT satisfy the exchangeability requirement of conformal
prediction when lambda is selected via a validation split, because the same
calibration data is reused for both lambda selection and final quantile
estimation.  They are kept here exactly as in the paper for reproducibility.

See experiments_tuned_lambda_strict_split.py for the exchangeability-safe versions.

Public API
----------
run_raps_baseline(probabilities, labels, *, config)
    Standard RAPS without any class-similarity penalty (lambda_2 = 0).

run_raps_ma_avg_opt(probabilities, labels, adjacency_matrix,
                    adjacency_matrix_smaller, lamda_2_values, *, config)
    RAPS + superclass-mass penalty (MA), lambda chosen by min avg-set-size
    on a held-out validation fold.

run_raps_ms_avg_opt(probabilities, labels, adjacency_matrix,
                    adjacency_matrix_ms, lamda_2_values, *, config)
    RAPS + binary-distance penalty (MS), lambda chosen by min avg-set-size.

run_raps_ma_binary_dist_avg_opt(probabilities, labels, adjacency_matrix,
                                 adjacency_matrix_real, lamda_2_values, *, config)
    RAPS + binary-distance penalty with MA adjacency matrix, lambda chosen
    by min avg-set-size.
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
    raps_cal_scores,
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


def _make_config(config: dict | None) -> dict:
    defaults = dict(
        alpha=ALPHA, num_class=NUM_CLASS, size_super_class=SIZE_SUPER_CLASS,
        cal_frac=CAL_FRAC, val_frac=VAL_FRAC, T=T, rand=RAND,
        disallow_zero_sets=DISALLOW_ZERO, lam_reg=LAM_REG, k_reg=K_REG,
    )
    if config:
        defaults.update(config)
    return defaults


# ─────────────────────────────────────────────────────────────────────────────
# Baseline RAPS (no class-similarity)
# ─────────────────────────────────────────────────────────────────────────────

def run_raps_baseline(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    config: dict | None = None,
) -> dict:
    """
    Standard RAPS (lambda_2 = 0).

    This is the baseline that every other method is compared against.
    Because lambda_2 = 0 there is no validation split needed, so the full
    10 % calibration fold is used for quantile estimation.
    """
    cfg = _make_config(config)
    alpha = cfg["alpha"]; num_class = cfg["num_class"]
    T_ = cfg["T"]; rand = cfg["rand"]
    disallow_zero = cfg["disallow_zero_sets"]
    reg_vec = build_reg_vec(num_class, cfg["k_reg"], cfg["lam_reg"])

    # accumulators
    cov_acc = 0.0
    cc_acc  = np.zeros(num_class)
    size_acc = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc  = 0.0
    avg_sizes: list[float] = []

    for _ in range(T_):
        n_cal = int(cfg["cal_frac"] * probabilities.shape[0])
        idx   = np.random.permutation(probabilities.shape[0])
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        cal_smx, test_smx     = probabilities[cal_idx], probabilities[test_idx]
        cal_labels, test_labels = labels[cal_idx], labels[test_idx]

        scores = raps_cal_scores(cal_smx, cal_labels, reg_vec, rand=rand)
        qhat   = conformal_quantile(scores, n_cal, alpha)

        # prediction sets (binary-dist with zero penalty = plain RAPS)
        pi      = test_smx.argsort(1)[:, ::-1]
        srt     = np.take_along_axis(test_smx, pi, axis=1)
        srt_reg = srt + reg_vec
        n_test  = test_smx.shape[0]
        noise   = np.random.rand(n_test, 1) if rand else np.zeros((n_test, 1))
        indicators = (srt_reg.cumsum(axis=1) - noise * srt_reg) <= qhat
        if disallow_zero:
            indicators[:, 0] = True
        prediction_sets = np.take_along_axis(indicators, pi.argsort(axis=1), axis=1)

        # metrics
        cov_acc += marginal_coverage(prediction_sets, test_labels)
        cc_acc  += class_conditional_coverage(prediction_sets, test_labels, num_class)
        s = set_size_stats(prediction_sets)
        for k in size_acc:
            size_acc[k] += s[k]
        avg_sizes.append(s["mean"])

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MA – superclass-mass penalty, lambda chosen by min avg-set-size
# ─────────────────────────────────────────────────────────────────────────────

def run_raps_ma_avg_opt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda_2_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    RAPS + superclass-mass (MA) penalty.

    Lambda is selected on a separate validation fold by minimising average
    set size.  When only [0] is passed the validation fold is merged into
    calibration (pure baseline mode).

    NOTE: reuses cal set for both lambda selection and qhat estimation –
    this violates exchangeability.  See run_raps_ma_avg_opt_strict_split for
    the fixed version.
    """
    cfg = _make_config(config)
    alpha = cfg["alpha"]; num_class = cfg["num_class"]
    size_sc = cfg["size_super_class"]
    T_ = cfg["T"]; rand = cfg["rand"]; disallow_zero = cfg["disallow_zero_sets"]
    reg_vec = build_reg_vec(num_class, cfg["k_reg"], cfg["lam_reg"])

    baseline_only = (len(lamda_2_values) == 1 and lamda_2_values[0] == 0)

    cov_acc = 0.0; cc_acc = np.zeros(num_class)
    size_acc = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc = 0.0; avg_sizes: list[float] = []

    for _ in range(T_):
        n_total = probabilities.shape[0]
        n_cal   = int(cfg["cal_frac"] * n_total)
        n_val   = int(cfg["val_frac"] * n_total)
        idx     = np.random.permutation(n_total)

        if baseline_only:
            cal_idx  = idx[:n_cal + n_val]
            test_idx = idx[n_cal + n_val:]
            n_cal_eff = n_cal + n_val
            val_smx = val_labels = None
        else:
            cal_idx  = idx[:n_cal]
            val_idx  = idx[n_cal:n_cal + n_val]
            test_idx = idx[n_cal + n_val:]
            n_cal_eff = n_cal
            val_smx    = probabilities[val_idx]
            val_labels = labels[val_idx]

        cal_smx, test_smx     = probabilities[cal_idx], probabilities[test_idx]
        cal_labels, test_labels = labels[cal_idx], labels[test_idx]

        # ── lambda selection ─────────────────────────────────────────────
        opt_lamda = 0
        opt_size  = float("inf")

        for lam in lamda_2_values:
            scores = raps_cal_scores_superclass(
                cal_smx, cal_labels, reg_vec,
                adjacency_matrix_smaller, lam, rand=rand,
            )
            qhat = conformal_quantile(scores, n_cal_eff, alpha)

            if baseline_only:
                break  # no need to evaluate on val

            ps_val = build_raps_prediction_sets_superclass(
                val_smx, reg_vec, adjacency_matrix, adjacency_matrix_smaller,
                lam, qhat, num_class, size_sc, rand=rand,
                disallow_zero_sets=disallow_zero,
            )
            size_val = ps_val.sum(axis=1).mean()
            if lam == 0:
                opt_size  = size_val + 10  # ensure any non-zero lam can win
                opt_lamda = 0
            elif size_val < opt_size:
                opt_size  = size_val
                opt_lamda = lam

        # ── refit on full cal with opt_lamda ─────────────────────────────
        scores = raps_cal_scores_superclass(
            cal_smx, cal_labels, reg_vec,
            adjacency_matrix_smaller, opt_lamda, rand=rand,
        )
        qhat = conformal_quantile(scores, n_cal_eff, alpha)

        prediction_sets = build_raps_prediction_sets_superclass(
            test_smx, reg_vec, adjacency_matrix, adjacency_matrix_smaller,
            opt_lamda, qhat, num_class, size_sc, rand=rand,
            disallow_zero_sets=disallow_zero,
        )

        # ── accumulate ───────────────────────────────────────────────────
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
# MA – binary distance penalty, lambda chosen by min avg-set-size
# ─────────────────────────────────────────────────────────────────────────────

def run_raps_ma_binary_dist_avg_opt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_real: np.ndarray,
    lamda_2_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    RAPS + binary-distance penalty (MA adjacency matrix).

    Lambda selected by min avg-set-size on a validation fold.
    Not exchangeability-safe – see strict_split version.
    """
    return _run_binary_dist_avg_opt(
        probabilities, labels, adjacency_matrix, adjacency_matrix_real,
        lamda_2_values, config=config,
    )


def run_raps_ms_avg_opt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_ms: np.ndarray,
    lamda_2_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    RAPS + binary-distance penalty (MS / cosine-similarity adjacency matrix).

    Lambda selected by min avg-set-size on a validation fold.
    Not exchangeability-safe – see strict_split version.
    """
    return _run_binary_dist_avg_opt(
        probabilities, labels, adjacency_matrix, adjacency_matrix_ms,
        lamda_2_values, config=config,
    )


def _run_binary_dist_avg_opt(
    probabilities, labels, adjacency_matrix, penalty_matrix, lamda_2_values,
    *, config,
) -> dict:
    """Shared implementation for MA/MS binary-distance runners."""
    cfg = _make_config(config)
    alpha = cfg["alpha"]; num_class = cfg["num_class"]
    T_ = cfg["T"]; rand = cfg["rand"]; disallow_zero = cfg["disallow_zero_sets"]
    reg_vec = build_reg_vec(num_class, cfg["k_reg"], cfg["lam_reg"])

    baseline_only = (len(lamda_2_values) == 1 and lamda_2_values[0] == 0)

    cov_acc = 0.0; cc_acc = np.zeros(num_class)
    size_acc = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc = 0.0; avg_sizes: list[float] = []

    for _ in range(T_):
        n_total = probabilities.shape[0]
        n_cal   = int(cfg["cal_frac"] * n_total)
        n_val   = int(cfg["val_frac"] * n_total)
        idx     = np.random.permutation(n_total)

        if baseline_only:
            cal_idx  = idx[:n_cal + n_val]
            test_idx = idx[n_cal + n_val:]
            n_cal_eff = n_cal + n_val
            val_smx = val_labels = None
        else:
            cal_idx  = idx[:n_cal]
            val_idx  = idx[n_cal:n_cal + n_val]
            test_idx = idx[n_cal + n_val:]
            n_cal_eff = n_cal
            val_smx    = probabilities[val_idx]
            val_labels = labels[val_idx]

        cal_smx, test_smx     = probabilities[cal_idx], probabilities[test_idx]
        cal_labels, test_labels = labels[cal_idx], labels[test_idx]

        opt_lamda = 0; opt_size = float("inf")

        for lam in lamda_2_values:
            scores = raps_cal_scores_binary_dist(
                cal_smx, cal_labels, reg_vec, penalty_matrix, lam, rand=rand,
            )
            qhat = conformal_quantile(scores, n_cal_eff, alpha)

            if baseline_only:
                break

            ps_val = build_raps_prediction_sets_binary_dist(
                val_smx, reg_vec, penalty_matrix, lam, qhat,
                rand=rand, disallow_zero_sets=disallow_zero,
            )
            size_val = ps_val.sum(axis=1).mean()
            if lam == 0:
                opt_size = size_val + 10; opt_lamda = 0
            elif size_val < opt_size:
                opt_size = size_val; opt_lamda = lam

        scores = raps_cal_scores_binary_dist(
            cal_smx, cal_labels, reg_vec, penalty_matrix, opt_lamda, rand=rand,
        )
        qhat = conformal_quantile(scores, n_cal_eff, alpha)

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

def _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class):
    cc = cc_acc / T_
    worst_dev, worst_cls = worst_class_coverage_deviation(cc, alpha)
    return {
        "marginal_coverage": cov_acc / T_,
        "mean_class_cc":     float(cc.mean()),
        "class_cc":          cc,
        "worst_cc_deviation": worst_dev,
        "worst_cc_class":    worst_cls,
        "top5_cc_deviation": top_k_worst_coverage_deviation(cc, alpha, k=5),
        "set_sizes": {k: v / T_ for k, v in size_acc.items()},
        "avg_sizes_per_trial": avg_sizes,
        "mean_distinct_superclasses": sc_acc / T_,
    }


def _format_for_print(results: dict) -> dict:
    return {
        "coverage":   results["marginal_coverage"],
        "mean_cc":    results["mean_class_cc"],
        "worst_dev":  results["worst_cc_deviation"],
        "worst_class": results["worst_cc_class"],
        "top5_dev":   results["top5_cc_deviation"],
        "sizes":      results["set_sizes"],
        "mean_sc":    results["mean_distinct_superclasses"],
    }
