"""
score_functions.py
==================
Low-level building blocks for conformal score computation and prediction-set
construction.  All functions are pure NumPy with no side-effects so they can
be shared freely across the original (paper-replication) and strict_split
(exchangeability-safe) experiment runners.

Score families
--------------
RAPS  – Regularised Adaptive Prediction Sets (Angelopoulos et al., ICLR 2021)
LAC   – Least Ambiguous set-valued Classifiers (Sadinle et al., JASA 2019)
SAPS  – Sorted Adaptive Prediction Sets / conformal prediction via label
        ranking (CP_via_CS paper)

Penalty variants (applied on top of the base score)
----------------------------------------------------
(none)        – baseline, no class-similarity term
binary dist   – λ · (1 − A[y, ŷ])  where A is an MA or MS adjacency matrix
superclass    – λ · (cumulative outside-superclass softmax mass up to rank of y)
              – MA variant only; applies to RAPS and LAC
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def build_reg_vec(num_class: int, k_reg: int, lam_reg: float) -> np.ndarray:
    """
    RAPS regularisation vector of shape (1, num_class).

    The first k_reg positions are 0; the remaining positions are lam_reg.
    """
    return np.array(k_reg * [0.0] + (num_class - k_reg) * [lam_reg])[None, :]


def conformal_quantile(scores: np.ndarray, n_cal: int, alpha: float) -> float:
    """
    Standard split-conformal quantile.

    Returns the ⌈(n+1)(1−α)⌉/n empirical quantile of *scores*, which
    guarantees marginal coverage ≥ 1−α under exchangeability.
    """
    level = float(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    return float(np.quantile(scores, level, interpolation="higher"))


# ═════════════════════════════════════════════════════════════════════════════
# RAPS
# Angelopoulos et al. (2021) – https://arxiv.org/abs/2009.14193
#
# Score s(x, y) = Σ_{k=1}^{L(y)} (p_{(k)} + r_k) − U · (p_{L(y)} + r_{L(y)})
# where (k) is the k-th largest class, L(y) is the rank of y, and
# r_k = lam_reg · max(k − k_reg, 0) is the regularisation vector entry.
# ═════════════════════════════════════════════════════════════════════════════

def raps_cal_scores(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    reg_vec: np.ndarray,
    rand: bool = True,
) -> np.ndarray:
    """
    Base RAPS calibration scores (no class-similarity penalty).

    Parameters
    ----------
    cal_smx    : (n, C)  softmax probabilities
    cal_labels : (n,)    integer true labels
    reg_vec    : (1, C)  RAPS regularisation vector from build_reg_vec()
    rand       : bool    subtract uniform noise on the score at rank L(y)

    Returns
    -------
    scores : (n,)
    """
    n = cal_smx.shape[0]
    pi      = cal_smx.argsort(1)[:, ::-1]
    srt     = np.take_along_axis(cal_smx, pi, axis=1)
    srt_reg = srt + reg_vec
    L       = np.where(pi == cal_labels[:, None])[1]
    noise   = np.random.rand(n) if rand else np.zeros(n)
    return srt_reg.cumsum(axis=1)[np.arange(n), L] - noise * srt_reg[np.arange(n), L]


def raps_cal_scores_binary_dist(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix: np.ndarray,
    lamda: float,
    rand: bool = True,
) -> np.ndarray:
    """
    RAPS calibration scores + binary distance penalty (MA or MS).

    Penalty term: lamda · (1 − A[y_true, ŷ])   where ŷ = argmax p.

    Parameters
    ----------
    adjacency_matrix : (C, C) or (C, C+1)  – only first C columns used
    lamda            : float  penalty weight
    """
    n       = cal_smx.shape[0]
    pi      = cal_smx.argsort(1)[:, ::-1]
    srt     = np.take_along_axis(cal_smx, pi, axis=1)
    srt_reg = srt + reg_vec
    L       = np.where(pi == cal_labels[:, None])[1]
    yhat    = cal_smx.argmax(1)
    dist    = 1.0 - adjacency_matrix[cal_labels, yhat]
    noise   = np.random.rand(n) if rand else np.zeros(n)
    return (
        srt_reg.cumsum(axis=1)[np.arange(n), L]
        - noise * srt_reg[np.arange(n), L]
        + lamda * dist
    )


def raps_cal_scores_superclass(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda: float,
    rand: bool = True,
) -> np.ndarray:
    """
    RAPS calibration scores + within-superclass probability mass penalty (MA).

    Penalty term: lamda · Σ_{k < L(y)} p_{(k)} · 1[class_{(k)} ∉ SC(y)]
    i.e. cumulative outside-superclass softmax mass at ranks strictly before y.

    Parameters
    ----------
    adjacency_matrix_smaller : (C, C)  binary; entry [i,j]=1 if j is in same
                                        superclass as i
    """
    n  = cal_smx.shape[0]
    pi = cal_smx.argsort(1)[:, ::-1]
    srt_reg = np.take_along_axis(cal_smx, pi, axis=1) + reg_vec
    L  = np.where(pi == cal_labels[:, None])[1]

    super_mask    = adjacency_matrix_smaller[cal_labels, :].astype(bool)
    smx_copy      = cal_smx.copy()
    smx_copy[super_mask] = 0.0
    srt_zeroed    = np.take_along_axis(smx_copy, pi, axis=1)
    # (cumsum − self) gives mass at strictly preceding ranks
    penalty       = (srt_zeroed.cumsum(axis=1) - srt_zeroed)[np.arange(n), L]

    noise = np.random.rand(n) if rand else np.zeros(n)
    return (
        srt_reg.cumsum(axis=1)[np.arange(n), L]
        - noise * srt_reg[np.arange(n), L]
        + lamda * penalty
    )


def build_raps_prediction_sets(
    smx: np.ndarray,
    reg_vec: np.ndarray,
    qhat: float,
    rand: bool = True,
    disallow_zero_sets: bool = False,
) -> np.ndarray:
    """Base RAPS prediction sets (no penalty)."""
    n  = smx.shape[0]
    pi = smx.argsort(1)[:, ::-1]
    srt_reg = np.take_along_axis(smx, pi, axis=1) + reg_vec
    noise   = np.random.rand(n, 1) if rand else np.zeros((n, 1))
    ind     = (srt_reg.cumsum(axis=1) - noise * srt_reg) <= qhat
    if disallow_zero_sets:
        ind[:, 0] = True
    return np.take_along_axis(ind, pi.argsort(axis=1), axis=1)


def build_raps_prediction_sets_binary_dist(
    smx: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix: np.ndarray,
    lamda: float,
    qhat: float,
    rand: bool = True,
    disallow_zero_sets: bool = False,
) -> np.ndarray:
    """
    RAPS prediction sets with binary distance penalty (MA or MS).

    Inclusion rule: cumsum(srt_reg) + lamda·penalty − U·srt_reg[L] ≤ q̂
    """
    n, C = smx.shape
    yhat  = smx.argmax(1)
    pi    = smx.argsort(1)[:, ::-1]

    # penalty[i, :] = 1 − A[:, ŷ_i]  in sorted order
    penalty        = 1.0 - adjacency_matrix[:C, yhat].T          # (n, C)
    penalty_sorted = np.take_along_axis(penalty, pi, axis=1)

    srt_reg = np.take_along_axis(smx, pi, axis=1) + reg_vec
    noise   = np.random.rand(n, 1) if rand else np.zeros((n, 1))
    ind     = (srt_reg.cumsum(axis=1) + lamda * penalty_sorted - noise * srt_reg) <= qhat
    if disallow_zero_sets:
        ind[:, 0] = True
    return np.take_along_axis(ind, pi.argsort(axis=1), axis=1)


def build_raps_prediction_sets_superclass(
    smx: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda: float,
    qhat: float,
    num_class: int,
    size_super_class: int,
    rand: bool = True,
    disallow_zero_sets: bool = False,
) -> np.ndarray:
    """
    RAPS prediction sets with within-superclass probability mass penalty (MA).

    Parameters
    ----------
    adjacency_matrix         : (C, C+1)  last column = integer superclass id
    adjacency_matrix_smaller : (C, C)    binary, 1 if same superclass
    """
    n  = smx.shape[0]
    pi = smx.argsort(1)[:, ::-1]
    srt_reg = np.take_along_axis(smx, pi, axis=1) + reg_vec

    # Build penalty matrix in label space: penalty[i, y] = outside-mass before rank of y
    srt_total = np.zeros((n, num_class))
    seen_sc   = np.zeros(num_class // size_super_class, dtype=bool)

    for y in range(num_class):
        sc_id = int(adjacency_matrix[y, num_class])
        if seen_sc[sc_id]:
            continue
        seen_sc[sc_id] = True
        mask     = adjacency_matrix_smaller[y, :].astype(bool)
        smx_copy = smx.copy()
        smx_copy[:, mask] = 0.0
        srt_z    = np.take_along_axis(smx_copy, pi, axis=1)
        for member in np.where(mask)[0]:
            col = np.where(pi == np.full(n, member)[:, None])[1]
            srt_total[np.arange(n), col] = (srt_z.cumsum(axis=1) - srt_z)[np.arange(n), col]

    noise = np.random.rand(n, 1) if rand else np.zeros((n, 1))
    ind   = (srt_reg.cumsum(axis=1) + lamda * srt_total - noise * srt_reg) <= qhat
    if disallow_zero_sets:
        ind[:, 0] = True
    return np.take_along_axis(ind, pi.argsort(axis=1), axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# LAC
# Sadinle et al. (2019) – https://arxiv.org/abs/1609.00451
#
# Score s(x, y) = 1 − p_y.
# Prediction set C(x) = {y : 1 − p_y ≤ q̂}  (i.e. all classes with
# sufficiently high softmax probability).
# ═════════════════════════════════════════════════════════════════════════════

def lac_cal_scores(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
) -> np.ndarray:
    """
    Base LAC calibration scores: s(x, y) = 1 − p_y.

    Parameters
    ----------
    cal_smx    : (n, C)
    cal_labels : (n,)

    Returns
    -------
    scores : (n,)
    """
    n = cal_smx.shape[0]
    return 1.0 - cal_smx[np.arange(n), cal_labels]


def lac_cal_scores_binary_dist(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    adjacency_matrix: np.ndarray,
    lamda: float,
) -> np.ndarray:
    """
    LAC calibration scores + binary distance penalty (MA or MS).

    Score = (1 − p_y) + lamda · (1 − A[y, ŷ])
    """
    n    = cal_smx.shape[0]
    yhat = cal_smx.argmax(1)
    dist = 1.0 - adjacency_matrix[cal_labels, yhat]
    return (1.0 - cal_smx[np.arange(n), cal_labels]) + lamda * dist


def lac_cal_scores_superclass(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda: float,
) -> np.ndarray:
    """
    LAC calibration scores + within-superclass probability mass penalty (MA).

    Score = (1 − p_y) + lamda · R(x, y)
    where R(x, y) = cumulative outside-superclass softmax mass at ranks
    strictly before y in the sorted list.
    """
    n  = cal_smx.shape[0]
    pi = cal_smx.argsort(1)[:, ::-1]
    L  = np.where(pi == cal_labels[:, None])[1]

    super_mask = adjacency_matrix_smaller[cal_labels, :].astype(bool)
    smx_copy   = cal_smx.copy()
    smx_copy[super_mask] = 0.0
    srt_z      = np.take_along_axis(smx_copy, pi, axis=1)
    penalty    = (srt_z.cumsum(axis=1) - srt_z)[np.arange(n), L]

    return (1.0 - cal_smx[np.arange(n), cal_labels]) + lamda * penalty


def build_lac_prediction_sets(
    smx: np.ndarray,
    qhat: float,
) -> np.ndarray:
    """Base LAC prediction sets: {y : 1 − p_y ≤ q̂}."""
    return (1.0 - smx) <= qhat


def build_lac_prediction_sets_binary_dist(
    smx: np.ndarray,
    adjacency_matrix: np.ndarray,
    lamda: float,
    qhat: float,
) -> np.ndarray:
    """
    LAC prediction sets with binary distance penalty (MA or MS).

    Inclusion rule: (1 − p_y) + lamda · (1 − A[y, ŷ]) ≤ q̂

    Vectorised: penalty[i, y] = 1 − A[y, ŷ_i]  via broadcasting.
    """
    n, C = smx.shape
    yhat    = smx.argmax(1)                              # (n,)
    penalty = 1.0 - adjacency_matrix[:C, yhat].T        # (n, C)
    return ((1.0 - smx) + lamda * penalty) <= qhat


def build_lac_prediction_sets_superclass(
    smx: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda: float,
    qhat: float,
    num_class: int,
    size_super_class: int,
) -> np.ndarray:
    """
    LAC prediction sets with within-superclass probability mass penalty (MA).

    Inclusion rule: (1 − p_y) + lamda · R(x, y) ≤ q̂

    Parameters
    ----------
    adjacency_matrix         : (C, C+1)  last column = integer superclass id
    adjacency_matrix_smaller : (C, C)    binary, 1 if same superclass
    """
    n  = smx.shape[0]
    pi = smx.argsort(1)[:, ::-1]

    # Build outside-superclass cumulative mass matrix in label space
    srt_total = np.zeros((n, num_class))
    seen_sc   = np.zeros(num_class // size_super_class, dtype=bool)

    for y in range(num_class):
        sc_id = int(adjacency_matrix[y, num_class])
        if seen_sc[sc_id]:
            continue
        seen_sc[sc_id] = True
        mask     = adjacency_matrix_smaller[y, :].astype(bool)
        smx_copy = smx.copy()
        smx_copy[:, mask] = 0.0
        srt_z    = np.take_along_axis(smx_copy, pi, axis=1)
        for member in np.where(mask)[0]:
            col = np.where(pi == np.full(n, member)[:, None])[1]
            srt_total[np.arange(n), col] = (srt_z.cumsum(axis=1) - srt_z)[np.arange(n), col]

    return ((1.0 - smx) + lamda * srt_total) <= qhat


# ═════════════════════════════════════════════════════════════════════════════
# SAPS
# Conformal Prediction via Class Similarity (CP_via_CS)
#
# Core score for sample (x, y):
#   rank L(y) = 0  (y is the top-1 class):  s = U · p_max
#   rank L(y) > 0                         :  s = p_max + (L(y) − 1 + U) · λ₂
#
# λ₂ controls rank spacing.  Small λ₂ ≈ LAC; large λ₂ gives uniform spacing,
# yielding smaller average sets when the model is confident.
#
# An optional binary distance penalty (λ₃) can be added on top:
#   s += λ₃ · (1 − A[y, ŷ])
# ═════════════════════════════════════════════════════════════════════════════

def saps_cal_scores(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    lamda_2: float,
    rand: bool = True,
) -> np.ndarray:
    """
    Base SAPS calibration scores.

    Parameters
    ----------
    cal_smx    : (n, C)  softmax probabilities
    cal_labels : (n,)    integer true labels
    lamda_2    : float   rank-spacing penalty weight
    rand       : bool    use randomised (smoothed) score

    Returns
    -------
    scores : (n,)
    """
    n     = cal_smx.shape[0]
    pi    = cal_smx.argsort(1)[:, ::-1]
    p_max = cal_smx.max(axis=1)                     # (n,)
    L     = np.where(pi == cal_labels[:, None])[1]  # 0-based rank of true label
    u     = np.random.rand(n) if rand else np.ones(n)

    return np.where(L == 0, u * p_max, p_max + (L - 1 + u) * lamda_2)


def saps_cal_scores_binary_dist(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    lamda_2: float,
    lamda_3: float,
    adjacency_matrix: np.ndarray,
    rand: bool = True,
) -> np.ndarray:
    """
    SAPS calibration scores + binary distance penalty (MS variant).

    Score = SAPS(x, y) + λ₃ · (1 − A[y, ŷ])

    Parameters
    ----------
    lamda_2          : float   rank-spacing weight
    lamda_3          : float   distance penalty weight
    adjacency_matrix : (C, C)  similarity matrix A[i, j]
    """
    n     = cal_smx.shape[0]
    pi    = cal_smx.argsort(1)[:, ::-1]
    p_max = cal_smx.max(axis=1)
    yhat  = pi[:, 0]
    L     = np.where(pi == cal_labels[:, None])[1]
    u     = np.random.rand(n) if rand else np.ones(n)

    base = np.where(L == 0, u * p_max, p_max + (L - 1 + u) * lamda_2)
    dist = 1.0 - adjacency_matrix[cal_labels, yhat]
    return base + lamda_3 * dist


def build_saps_prediction_sets(
    smx: np.ndarray,
    lamda_2: float,
    qhat: float,
    rand: bool = True,
) -> np.ndarray:
    """
    Base SAPS prediction sets.

    For each sample i and every class y, compute score(i, y) and include y
    iff score(i, y) ≤ q̂.

    Because scores depend on y's rank, we build the full (n, C) score matrix.

    Parameters
    ----------
    smx     : (n, C)  softmax probabilities
    lamda_2 : float   rank-spacing weight
    qhat    : float   conformal threshold
    rand    : bool    per-(sample, class) uniform noise

    Returns
    -------
    prediction_sets : (n, C) boolean array
    """
    n, C  = smx.shape
    pi    = smx.argsort(1)[:, ::-1]
    p_max = smx.max(axis=1, keepdims=True)          # (n, 1)
    ranks = pi.argsort(axis=1)                       # (n, C) 0-based rank of each class
    u     = np.random.rand(n, C) if rand else np.ones((n, C))

    scores = np.where(ranks == 0, u * p_max, p_max + (ranks - 1 + u) * lamda_2)
    return scores <= qhat


def build_saps_prediction_sets_binary_dist(
    smx: np.ndarray,
    lamda_2: float,
    lamda_3: float,
    adjacency_matrix: np.ndarray,
    qhat: float,
    rand: bool = True,
) -> np.ndarray:
    """
    SAPS prediction sets with binary distance penalty (MS variant).

    Inclusion rule: SAPS(x, y) + λ₃ · (1 − A[y, ŷ]) ≤ q̂

    Parameters
    ----------
    smx              : (n, C)
    lamda_2          : float   rank-spacing weight
    lamda_3          : float   distance penalty weight
    adjacency_matrix : (C, C)  similarity matrix A[i, j]
    """
    n, C  = smx.shape
    pi    = smx.argsort(1)[:, ::-1]
    p_max = smx.max(axis=1, keepdims=True)
    yhat  = pi[:, 0]                                 # (n,) top-1 per sample
    ranks = pi.argsort(axis=1)                       # (n, C)
    u     = np.random.rand(n, C) if rand else np.ones((n, C))

    base_scores = np.where(ranks == 0, u * p_max, p_max + (ranks - 1 + u) * lamda_2)
    penalty     = 1.0 - adjacency_matrix[:C, yhat].T  # (n, C)
    return (base_scores + lamda_3 * penalty) <= qhat