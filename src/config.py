"""
Global configuration for CIFAR-100 conformal prediction experiments.

Modify these values before running to control dataset size, coverage target,
RAPS regularization, and number of Monte Carlo trials.
"""

# ── Coverage ──────────────────────────────────────────────────────────────────
ALPHA = 0.1          # target miscoverage rate; 1-alpha is the desired coverage

# ── Dataset ───────────────────────────────────────────────────────────────────
NUM_CLASS       = 100
SIZE_SUPER_CLASS = 5   # number of fine-grained classes per CIFAR-100 superclass

# ── Data splits (fractions of total dataset) ──────────────────────────────────
CAL_FRAC = 0.1        # calibration set fraction
VAL_FRAC = 0.1        # validation set fraction (for lambda search); rest = test

# ── RAPS base regularization ──────────────────────────────────────────────────
LAM_REG = 0.01        # RAPS penalty weight for ranks beyond k_reg
K_REG   = 5           # RAPS: top-k classes are unpenalized

# ── Experiment ────────────────────────────────────────────────────────────────
T             = 50    # number of Monte Carlo trials
RAND          = True  # use randomized (smoothed) score; set False for deterministic
DISALLOW_ZERO = False # if True, prediction sets always contain at least one class
