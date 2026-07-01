"""
experiments_lac.py
==================
Experiment runners for LAC-based conformal prediction.

LAC (Sadinle et al., 2019) uses score s(x, y) = 1 − p_y and includes class y
whenever that score falls below the conformal threshold q̂.

Three penalty variants are implemented, each in an *original* version
(paper-replication, may merge val back into cal) and a *corrected* version
(exchangeability-safe, strict cal / val / test separation).

Original (not exchangeability-safe when λ > 0):
    run_lac_baseline
    run_lac_ma_binary_dist_avg_opt
    run_lac_ms_binary_dist_avg_opt
    run_lac_ma_superclass_avg_opt

Corrected (exchangeability-safe):
    run_lac_ma_binary_dist_avg_opt_corrected
    run_lac_ms_binary_dist_avg_opt_corrected
    run_lac_ma_superclass_avg_opt_corrected
"""

from __future__ import annotations

import numpy as np

from .config import ALPHA, NUM_CLASS, SIZE_SUPER_CLASS, CAL_FRAC, VAL_FRAC, T
from .score_functions import (
    lac_cal_scores,
    lac_cal_scores_binary_dist,
    lac_cal_scores_superclass,
    build_lac_prediction_sets,
    build_lac_prediction_sets_binary_dist,
    build_lac_prediction_sets_superclass,
    conformal_quantile,
)
from .metrics import (
    marginal_coverage,
    class_conditional_coverage,
    set_size_stats,
    mean_distinct_superclasses,
    print_metrics,
)
from .experiments_original import _make_config, _aggregate, _format_for_print
from .experiments_corrected import _three_way_split


# ─────────────────────────────────────────────────────────────────────────────
# Baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_lac_baseline(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    config: dict | None = None,
) -> dict:
    """Standard LAC with no class-similarity penalty (λ = 0)."""
    cfg       = _make_config(config)
    alpha     = cfg["alpha"];  num_class = cfg["num_class"];  T_ = cfg["T"]
    cov_acc   = 0.0;  cc_acc = np.zeros(num_class)
    size_acc  = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc    = 0.0;  avg_sizes: list[float] = []

    for _ in range(T_):
        n_cal    = int(cfg["cal_frac"] * probabilities.shape[0])
        idx      = np.random.permutation(probabilities.shape[0])
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]
        cal_smx,  test_smx    = probabilities[cal_idx], probabilities[test_idx]
        cal_lbl,  test_lbl    = labels[cal_idx], labels[test_idx]

        scores = lac_cal_scores(cal_smx, cal_lbl)
        qhat   = conformal_quantile(scores, n_cal, alpha)
        ps     = build_lac_prediction_sets(test_smx, qhat)

        cov_acc += marginal_coverage(ps, test_lbl)
        cc_acc  += class_conditional_coverage(ps, test_lbl, num_class)
        s = set_size_stats(ps);  [size_acc.__setitem__(k, size_acc[k] + s[k]) for k in size_acc]
        avg_sizes.append(s["mean"])

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Binary distance penalty – original protocol (MA and MS share one impl)
# ─────────────────────────────────────────────────────────────────────────────

def run_lac_ma_binary_dist_avg_opt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_real: np.ndarray,
    lamda_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    LAC + binary distance penalty, MA adjacency matrix.
    λ selected by min avg-set-size on a validation fold.
    Not exchangeability-safe – see corrected version.
    """
    return _run_lac_binary_dist(
        probabilities, labels, adjacency_matrix, adjacency_matrix_real,
        lamda_values, corrected=False, config=config,
    )


def run_lac_ms_binary_dist_avg_opt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_ms: np.ndarray,
    lamda_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    LAC + binary distance penalty, MS cosine-similarity matrix.
    λ selected by min avg-set-size on a validation fold.
    Not exchangeability-safe – see corrected version.
    """
    return _run_lac_binary_dist(
        probabilities, labels, adjacency_matrix, adjacency_matrix_ms,
        lamda_values, corrected=False, config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Superclass mass penalty (MA) – original protocol
# ─────────────────────────────────────────────────────────────────────────────

def run_lac_ma_superclass_avg_opt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    LAC + within-superclass mass penalty (MA).
    λ selected by min avg-set-size on a validation fold.
    Not exchangeability-safe – see corrected version.
    """
    return _run_lac_superclass(
        probabilities, labels, adjacency_matrix, adjacency_matrix_smaller,
        lamda_values, corrected=False, config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Corrected (exchangeability-safe) versions
# ─────────────────────────────────────────────────────────────────────────────

def run_lac_ma_binary_dist_avg_opt_corrected(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_real: np.ndarray,
    lamda_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """LAC + MA binary distance penalty – exchangeability-safe."""
    return _run_lac_binary_dist(
        probabilities, labels, adjacency_matrix, adjacency_matrix_real,
        lamda_values, corrected=True, config=config,
    )


def run_lac_ms_binary_dist_avg_opt_corrected(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_ms: np.ndarray,
    lamda_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """LAC + MS binary distance penalty – exchangeability-safe."""
    return _run_lac_binary_dist(
        probabilities, labels, adjacency_matrix, adjacency_matrix_ms,
        lamda_values, corrected=True, config=config,
    )


def run_lac_ma_superclass_avg_opt_corrected(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """LAC + superclass mass penalty (MA) – exchangeability-safe."""
    return _run_lac_superclass(
        probabilities, labels, adjacency_matrix, adjacency_matrix_smaller,
        lamda_values, corrected=True, config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared implementations
# ─────────────────────────────────────────────────────────────────────────────

def _run_lac_binary_dist(
    probabilities, labels, adjacency_matrix, penalty_matrix,
    lamda_values, *, corrected: bool, config,
) -> dict:
    cfg       = _make_config(config)
    alpha     = cfg["alpha"];  num_class = cfg["num_class"];  T_ = cfg["T"]
    baseline  = (len(lamda_values) == 1 and lamda_values[0] == 0)

    cov_acc   = 0.0;  cc_acc = np.zeros(num_class)
    size_acc  = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc    = 0.0;  avg_sizes: list[float] = []

    for _ in range(T_):
        if corrected:
            cal_smx, cal_lbl, val_smx, val_lbl, test_smx, test_lbl, n_cal = \
                _three_way_split(probabilities, labels, cfg)
        else:
            n_total  = probabilities.shape[0]
            n_cal_0  = int(cfg["cal_frac"] * n_total)
            n_val    = int(cfg["val_frac"] * n_total)
            idx      = np.random.permutation(n_total)
            if baseline:
                cal_idx, test_idx = idx[:n_cal_0 + n_val], idx[n_cal_0 + n_val:]
                n_cal = n_cal_0 + n_val;  val_smx = val_lbl = None
            else:
                cal_idx  = idx[:n_cal_0]
                val_idx  = idx[n_cal_0:n_cal_0 + n_val]
                test_idx = idx[n_cal_0 + n_val:]
                n_cal    = n_cal_0
                val_smx, val_lbl = probabilities[val_idx], labels[val_idx]
            cal_smx, test_smx = probabilities[cal_idx], probabilities[test_idx]
            cal_lbl, test_lbl = labels[cal_idx], labels[test_idx]

        opt_lam = 0;  opt_sz = float("inf")
        for lam in lamda_values:
            sc = lac_cal_scores_binary_dist(cal_smx, cal_lbl, penalty_matrix, lam)
            qh = conformal_quantile(sc, n_cal, alpha)
            if corrected or not baseline:
                ps_v   = build_lac_prediction_sets_binary_dist(val_smx, penalty_matrix, lam, qh)
                sz_v   = ps_v.sum(axis=1).mean()
                if lam == 0:   opt_sz = sz_v + 10;  opt_lam = 0
                elif sz_v < opt_sz:  opt_sz = sz_v;  opt_lam = lam

        sc   = lac_cal_scores_binary_dist(cal_smx, cal_lbl, penalty_matrix, opt_lam)
        qhat = conformal_quantile(sc, n_cal, alpha)
        ps   = build_lac_prediction_sets_binary_dist(test_smx, penalty_matrix, opt_lam, qhat)

        cov_acc += marginal_coverage(ps, test_lbl)
        cc_acc  += class_conditional_coverage(ps, test_lbl, num_class)
        s = set_size_stats(ps);  [size_acc.__setitem__(k, size_acc[k] + s[k]) for k in size_acc]
        avg_sizes.append(s["mean"])
        sc_acc += mean_distinct_superclasses(ps, adjacency_matrix, num_class)

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results


def _run_lac_superclass(
    probabilities, labels, adjacency_matrix, adjacency_matrix_smaller,
    lamda_values, *, corrected: bool, config,
) -> dict:
    cfg      = _make_config(config)
    alpha    = cfg["alpha"];  num_class = cfg["num_class"]
    size_sc  = cfg["size_super_class"];  T_ = cfg["T"]
    baseline = (len(lamda_values) == 1 and lamda_values[0] == 0)

    cov_acc  = 0.0;  cc_acc = np.zeros(num_class)
    size_acc = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc   = 0.0;  avg_sizes: list[float] = []

    for _ in range(T_):
        if corrected:
            cal_smx, cal_lbl, val_smx, val_lbl, test_smx, test_lbl, n_cal = \
                _three_way_split(probabilities, labels, cfg)
        else:
            n_total = probabilities.shape[0]
            n_cal_0 = int(cfg["cal_frac"] * n_total)
            n_val   = int(cfg["val_frac"] * n_total)
            idx     = np.random.permutation(n_total)
            if baseline:
                cal_idx, test_idx = idx[:n_cal_0 + n_val], idx[n_cal_0 + n_val:]
                n_cal = n_cal_0 + n_val;  val_smx = val_lbl = None
            else:
                cal_idx  = idx[:n_cal_0]
                val_idx  = idx[n_cal_0:n_cal_0 + n_val]
                test_idx = idx[n_cal_0 + n_val:]
                n_cal    = n_cal_0
                val_smx, val_lbl = probabilities[val_idx], labels[val_idx]
            cal_smx, test_smx = probabilities[cal_idx], probabilities[test_idx]
            cal_lbl, test_lbl = labels[cal_idx], labels[test_idx]

        opt_lam = 0;  opt_sz = float("inf")
        for lam in lamda_values:
            sc = lac_cal_scores_superclass(cal_smx, cal_lbl, adjacency_matrix_smaller, lam)
            qh = conformal_quantile(sc, n_cal, alpha)
            if corrected or not baseline:
                ps_v  = build_lac_prediction_sets_superclass(
                    val_smx, adjacency_matrix, adjacency_matrix_smaller,
                    lam, qh, num_class, size_sc,
                )
                sz_v  = ps_v.sum(axis=1).mean()
                if lam == 0:  opt_sz = sz_v + 10;  opt_lam = 0
                elif sz_v < opt_sz:  opt_sz = sz_v;  opt_lam = lam

        sc   = lac_cal_scores_superclass(cal_smx, cal_lbl, adjacency_matrix_smaller, opt_lam)
        qhat = conformal_quantile(sc, n_cal, alpha)
        ps   = build_lac_prediction_sets_superclass(
            test_smx, adjacency_matrix, adjacency_matrix_smaller,
            opt_lam, qhat, num_class, size_sc,
        )

        cov_acc += marginal_coverage(ps, test_lbl)
        cc_acc  += class_conditional_coverage(ps, test_lbl, num_class)
        s = set_size_stats(ps);  [size_acc.__setitem__(k, size_acc[k] + s[k]) for k in size_acc]
        avg_sizes.append(s["mean"])
        sc_acc += mean_distinct_superclasses(ps, adjacency_matrix, num_class)

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results
