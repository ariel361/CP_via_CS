"""
experiments_saps.py
===================
Experiment runners for SAPS-based conformal prediction.

SAPS (CP_via_CS paper) uses a rank-based score:
  s(x, y) = U · p_max               if y is the top-1 class
           = p_max + (L(y)−1+U)·λ₂  otherwise

Two penalty variants are implemented, each in original and corrected forms.

Original (not exchangeability-safe when λ > 0):
    run_saps_baseline
    run_saps_ms_binary_dist_avg_opt

Corrected (exchangeability-safe):
    run_saps_ms_binary_dist_avg_opt_corrected
"""

from __future__ import annotations

import numpy as np

from .config import ALPHA, NUM_CLASS, CAL_FRAC, VAL_FRAC, T, RAND
from .score_functions import (
    saps_cal_scores,
    saps_cal_scores_binary_dist,
    build_saps_prediction_sets,
    build_saps_prediction_sets_binary_dist,
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
# Baseline SAPS
# ─────────────────────────────────────────────────────────────────────────────

def run_saps_baseline(
    probabilities: np.ndarray,
    labels: np.ndarray,
    lamda_2: float,
    *,
    config: dict | None = None,
) -> dict:
    """
    Base SAPS with no class-similarity penalty.

    Parameters
    ----------
    lamda_2 : float
        Rank-spacing weight.  lamda_2 = 0 degenerates to LAC.
        Typical values explored in the paper: 0 – 1 (grid search).
    """
    cfg      = _make_config(config)
    alpha    = cfg["alpha"];  num_class = cfg["num_class"]
    T_       = cfg["T"];      rand      = cfg["rand"]

    cov_acc  = 0.0;  cc_acc = np.zeros(num_class)
    size_acc = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    sc_acc   = 0.0;  avg_sizes: list[float] = []

    for _ in range(T_):
        n_cal    = int(cfg["cal_frac"] * probabilities.shape[0])
        idx      = np.random.permutation(probabilities.shape[0])
        cal_smx, test_smx = probabilities[idx[:n_cal]], probabilities[idx[n_cal:]]
        cal_lbl, test_lbl = labels[idx[:n_cal]], labels[idx[n_cal:]]

        scores = saps_cal_scores(cal_smx, cal_lbl, lamda_2, rand=rand)
        qhat   = conformal_quantile(scores, n_cal, alpha)
        ps     = build_saps_prediction_sets(test_smx, lamda_2, qhat, rand=rand)

        cov_acc += marginal_coverage(ps, test_lbl)
        cc_acc  += class_conditional_coverage(ps, test_lbl, num_class)
        s = set_size_stats(ps);  [size_acc.__setitem__(k, size_acc[k] + s[k]) for k in size_acc]
        avg_sizes.append(s["mean"])

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SAPS + MS binary distance penalty
# ─────────────────────────────────────────────────────────────────────────────

def run_saps_ms_binary_dist_avg_opt(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_ms: np.ndarray,
    lamda_2: float,
    lamda_3_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    SAPS + binary distance penalty (MS cosine-similarity matrix).

    λ₂ is fixed (controls rank spacing); λ₃ is selected on a validation fold
    by minimising average set size.

    Not exchangeability-safe – see corrected version.

    Parameters
    ----------
    lamda_2       : float       fixed rank-spacing weight
    lamda_3_values: list[float] search grid for the distance-penalty weight
    """
    return _run_saps_binary_dist(
        probabilities, labels, adjacency_matrix, adjacency_matrix_ms,
        lamda_2, lamda_3_values, corrected=False, config=config,
    )


def run_saps_ms_binary_dist_avg_opt_corrected(
    probabilities: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_ms: np.ndarray,
    lamda_2: float,
    lamda_3_values: list[float],
    *,
    config: dict | None = None,
) -> dict:
    """
    SAPS + binary distance penalty (MS) – exchangeability-safe.

    λ₂ fixed; λ₃ selected on a dedicated validation fold that is kept
    strictly separate from the calibration set used to compute q̂.
    """
    return _run_saps_binary_dist(
        probabilities, labels, adjacency_matrix, adjacency_matrix_ms,
        lamda_2, lamda_3_values, corrected=True, config=config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared implementation
# ─────────────────────────────────────────────────────────────────────────────

def _run_saps_binary_dist(
    probabilities, labels, adjacency_matrix, penalty_matrix,
    lamda_2, lamda_3_values, *, corrected: bool, config,
) -> dict:
    cfg      = _make_config(config)
    alpha    = cfg["alpha"];  num_class = cfg["num_class"]
    T_       = cfg["T"];      rand      = cfg["rand"]
    baseline = (len(lamda_3_values) == 1 and lamda_3_values[0] == 0)

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

        opt_lam3 = 0;  opt_sz = float("inf")
        for lam3 in lamda_3_values:
            sc = saps_cal_scores_binary_dist(
                cal_smx, cal_lbl, lamda_2, lam3, penalty_matrix, rand=rand
            )
            qh = conformal_quantile(sc, n_cal, alpha)
            if corrected or not baseline:
                ps_v = build_saps_prediction_sets_binary_dist(
                    val_smx, lamda_2, lam3, penalty_matrix, qh, rand=rand
                )
                sz_v = ps_v.sum(axis=1).mean()
                if lam3 == 0:  opt_sz = sz_v + 10;  opt_lam3 = 0
                elif sz_v < opt_sz:  opt_sz = sz_v;  opt_lam3 = lam3

        sc   = saps_cal_scores_binary_dist(
            cal_smx, cal_lbl, lamda_2, opt_lam3, penalty_matrix, rand=rand
        )
        qhat = conformal_quantile(sc, n_cal, alpha)
        ps   = build_saps_prediction_sets_binary_dist(
            test_smx, lamda_2, opt_lam3, penalty_matrix, qhat, rand=rand
        )

        cov_acc += marginal_coverage(ps, test_lbl)
        cc_acc  += class_conditional_coverage(ps, test_lbl, num_class)
        s = set_size_stats(ps);  [size_acc.__setitem__(k, size_acc[k] + s[k]) for k in size_acc]
        avg_sizes.append(s["mean"])
        sc_acc += mean_distinct_superclasses(ps, adjacency_matrix, num_class)

    results = _aggregate(cov_acc, cc_acc, size_acc, sc_acc, avg_sizes, T_, alpha, num_class)
    print_metrics(**_format_for_print(results))
    return results
