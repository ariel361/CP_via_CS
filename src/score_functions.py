"""
score_functions.py – low-level building blocks for conformal score computation
and prediction-set construction.

These are pure NumPy functions with no side-effects so they can be reused
across both the original (paper-replication) and the corrected (exchangeable)
experiment runners.
"""




# ── Helpers ───────────────────────────────────────────────────────────────────

def build_reg_vec(num_class: int, k_reg: int, lam_reg: float) -> np.ndarray:
    """
    RAPS regularization vector of shape (1, num_class).

    The first k_reg positions are 0; the rest are lam_reg.
    """
    return np.array(k_reg * [0.0] + (num_class - k_reg) * [lam_reg])[None, :]


# ── Calibration score computation ─────────────────────────────────────────────

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
    cal_smx   : (n_cal, C) softmax probabilities
    cal_labels: (n_cal,)   integer true labels
    reg_vec   : (1, C)     RAPS regularization vector
    rand      : smooth scores with uniform noise if True

    Returns
    -------
    scores : (n_cal,)
    """
    n_cal = cal_smx.shape[0]
    cal_pi  = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L = np.where(cal_pi == cal_labels[:, None])[1]
    noise = np.random.rand(n_cal) if rand else np.zeros(n_cal)
    return (
        cal_srt_reg.cumsum(axis=1)[np.arange(n_cal), cal_L]
        - noise * cal_srt_reg[np.arange(n_cal), cal_L]
    )


def raps_cal_scores_binary_dist(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix: np.ndarray,
    lamda_2: float,
    rand: bool = True,
) -> np.ndarray:
    """
    RAPS calibration scores with binary distance penalty (MA or MS variant).

    The penalty for sample i is  lamda_2 * (1 - A[y_i, yhat_i])
    where yhat is the top-1 predicted class.
    """
    n_cal = cal_smx.shape[0]
    cal_pi  = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L   = np.where(cal_pi == cal_labels[:, None])[1]
    cal_yhat = cal_smx.argmax(1)
    dist     = 1.0 - adjacency_matrix[cal_labels, cal_yhat]
    noise = np.random.rand(n_cal) if rand else np.zeros(n_cal)
    return (
        cal_srt_reg.cumsum(axis=1)[np.arange(n_cal), cal_L]
        - noise * cal_srt_reg[np.arange(n_cal), cal_L]
        + lamda_2 * dist
    )


def raps_cal_scores_superclass(
    cal_smx: np.ndarray,
    cal_labels: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda_2: float,
    rand: bool = True,
) -> np.ndarray:
    """
    RAPS calibration scores with within-superclass probability mass penalty (MA variant).

    The penalty for sample i is  lamda_2 * (cumulative softmax mass from
    non-superclass classes up to the true class rank).
    """
    n_cal = cal_smx.shape[0]
    cal_pi  = cal_smx.argsort(1)[:, ::-1]
    cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
    cal_srt_reg = cal_srt + reg_vec
    cal_L   = np.where(cal_pi == cal_labels[:, None])[1]

    # Zero out probabilities of same-superclass classes before measuring mass
    super_mask = adjacency_matrix_smaller[cal_labels, :].astype(bool)
    cal_smx_copy = cal_smx.copy()
    cal_smx_copy[super_mask] = 0.0
    cal_srt_zeroed  = np.take_along_axis(cal_smx_copy, cal_pi, axis=1)
    cal_regularization = (cal_srt_zeroed.cumsum(axis=1) - cal_srt_zeroed)[np.arange(n_cal), cal_L]

    noise = np.random.rand(n_cal) if rand else np.zeros(n_cal)
    return (
        cal_srt_reg.cumsum(axis=1)[np.arange(n_cal), cal_L]
        - noise * cal_srt_reg[np.arange(n_cal), cal_L]
        + lamda_2 * cal_regularization
    )


# ── Prediction-set construction ───────────────────────────────────────────────

def build_prediction_sets_binary_dist(
    smx: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix: np.ndarray,
    lamda_2: float,
    qhat: float,
    rand: bool = True,
    disallow_zero_sets: bool = False,
) -> np.ndarray:
    """
    Prediction sets using the binary-distance (MA/MS) penalty at test time.

    Parameters
    ----------
    smx : (N, C) softmax probabilities
    adjacency_matrix : (C, C) or (C, C+1) – only the first C columns are used

    Returns
    -------
    prediction_sets : (N, C) boolean array
    """
    n, num_class = smx.shape
    yhat   = smx.argmax(1)
    pi     = smx.argsort(1)[:, ::-1]

    # Penalty matrix: 1 - A[:, yhat_i] for each sample i
    penalty = np.stack([
        1.0 - adjacency_matrix[:num_class, yhat[i]]
        for i in range(n)
    ])  # (N, C)
    penalty_sorted = np.take_along_axis(penalty, pi, axis=1)

    srt     = np.take_along_axis(smx, pi, axis=1)
    srt_reg = srt + reg_vec
    noise   = np.random.rand(n, 1) if rand else np.zeros((n, 1))

    indicators = (
        srt_reg.cumsum(axis=1) + lamda_2 * penalty_sorted - noise * srt_reg
    ) <= qhat

    if disallow_zero_sets:
        indicators[:, 0] = True

    return np.take_along_axis(indicators, pi.argsort(axis=1), axis=1)


def build_prediction_sets_superclass(
    smx: np.ndarray,
    reg_vec: np.ndarray,
    adjacency_matrix: np.ndarray,
    adjacency_matrix_smaller: np.ndarray,
    lamda_2: float,
    qhat: float,
    num_class: int,
    size_super_class: int,
    rand: bool = True,
    disallow_zero_sets: bool = False,
) -> np.ndarray:
    """
    Prediction sets using the within-superclass probability mass penalty.

    Parameters
    ----------
    adjacency_matrix        : (C, C+1) – last column is superclass id
    adjacency_matrix_smaller: (C, C)   – binary, 1 if same superclass

    Returns
    -------
    prediction_sets : (N, C) boolean array
    """
    n = smx.shape[0]
    pi      = smx.argsort(1)[:, ::-1]
    srt     = np.take_along_axis(smx, pi, axis=1)
    srt_reg = srt + reg_vec

    # Compute per-sample superclass penalty in sorted order
    srt_total    = np.zeros((n, num_class))
    superclasses_seen = np.zeros(num_class // size_super_class, dtype=bool)

    for y in range(num_class):
        sc_id = int(adjacency_matrix[y, num_class])
        if superclasses_seen[sc_id]:
            continue
        superclasses_seen[sc_id] = True

        mask = adjacency_matrix_smaller[y, :].astype(bool)
        smx_copy = smx.copy()
        smx_copy[:, mask] = 0.0
        srt_zeroed   = np.take_along_axis(smx_copy, pi, axis=1)
        members      = np.where(mask)[0]

        for member in members:
            col_in_sorted = np.where(pi == member[:, None] if member.ndim else
                                     np.where(pi == np.full(n, member)[:, None])[1])[1] \
                            if False else None
            # Vectorised lookup for each member class
            arr_y = np.full(n, member)
            val_L = np.where(pi == arr_y[:, None])[1]
            srt_total[np.arange(n), val_L] = srt_zeroed.cumsum(axis=1)[np.arange(n), val_L]

    noise = np.random.rand(n, 1) if rand else np.zeros((n, 1))
    indicators = (
        srt_reg.cumsum(axis=1) + lamda_2 * srt_total - noise * srt_reg
    ) <= qhat

    if disallow_zero_sets:
        indicators[:, 0] = True

    return np.take_along_axis(indicators, pi.argsort(axis=1), axis=1)


# ── Quantile helper ───────────────────────────────────────────────────────────

def conformal_quantile(scores: np.ndarray, n_cal: int, alpha: float) -> float:
    """Standard conformal quantile: ceil((n+1)(1-alpha)) / n empirical quantile."""
    level = float(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    return float(np.quantile(scores, level, interpolation="higher"))
