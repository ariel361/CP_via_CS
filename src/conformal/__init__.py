"""
conformal – CIFAR-100 conformal prediction experiments.

Public API
----------
Original (paper-replication):
    run_raps_baseline
    run_raps_ma_avg_opt
    run_raps_ms_avg_opt
    run_raps_ma_binary_dist_avg_opt

Corrected (exchangeability-safe):
    run_raps_ma_avg_opt_corrected
    run_raps_ms_avg_opt_corrected
    run_raps_ma_binary_dist_avg_opt_corrected

Data helpers:
    load_logits
    load_adjacency_matrix
    load_similarity_matrix_cosine
"""

from .experiments_original import (
    run_raps_baseline,
    run_raps_ma_avg_opt,
    run_raps_ms_avg_opt,
    run_raps_ma_binary_dist_avg_opt,
)
from .experiments_corrected import (
    run_raps_ma_avg_opt_corrected,
    run_raps_ms_avg_opt_corrected,
    run_raps_ma_binary_dist_avg_opt_corrected,
)
from .data_loader import (
    load_logits,
    load_adjacency_matrix,
    load_similarity_matrix_cosine,
    load_similarity_matrix_gaussian,
)

__all__ = [
    # original
    "run_raps_baseline",
    "run_raps_ma_avg_opt",
    "run_raps_ms_avg_opt",
    "run_raps_ma_binary_dist_avg_opt",
    # corrected
    "run_raps_ma_avg_opt_corrected",
    "run_raps_ms_avg_opt_corrected",
    "run_raps_ma_binary_dist_avg_opt_corrected",
    # data
    "load_logits",
    "load_adjacency_matrix",
    "load_similarity_matrix_cosine",
    "load_similarity_matrix_gaussian",
]
