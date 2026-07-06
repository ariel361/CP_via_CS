# CP_via_CS
This is the code accompanying the paper: [Enhancing Conformal Prediction via Class Similarity](https://arxiv.org/abs/2511.19359) by Ariel Fargion, Lahav Dabah and Dr. Tom Tirer

<!-- Banner -->
<div align="center">

# CP via Class Similarity

**Enhancing Conformal Prediction via Class Similarity**

[![arXiv](https://img.shields.io/badge/arXiv-2511.19359-b31b1b.svg)](https://arxiv.org/abs/2511.19359)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Official code for the paper. Modular implementations of RAPS, LAC, and SAPS with class-similarity penalties for CIFAR-100.*

</div>

---

## Overview

Standard conformal prediction treats all misclassifications equally. This work exploits the **hierarchical structure of class labels** — penalising prediction sets that include classes semantically far from the true label — to produce sets that are both *smaller* and *more semantically coherent*.

We introduce two penalty variants on top of established score functions:

| Penalty | Abbreviation | Definition |
|:--------|:------------:|:-----------|
| Superclass mass | **MA** | Cumulative softmax mass from classes outside the true superclass, accumulated up to the rank of the true label |
| Binary distance | **MS** | `λ · (1 − A[y, ŷ])` where `A` is a cosine-similarity matrix between class embeddings and `ŷ` is the top-1 prediction |

Both penalties are applied to three score families — **RAPS**, **LAC**, and **SAPS** — and each experiment comes in an *original* (paper-replication) and a *exchangeability* (exchangeability-safe) version.

---

## Repository layout

```
CP_via_CS/
├── src/
│   ├── config.py                # all hyperparameters in one place
│   ├── data_loader.py           # load logits, adjacency & similarity matrices
│   ├── score_functions.py       # RAPS · LAC · SAPS score + prediction-set builders
│   ├── metrics.py               # coverage and set-size evaluation helpers
│   ├── experiments_original.py  # RAPS paper-replication runners
│   ├── experiments_tuned_lambda_exchangeability.py # RAPS exchangeability-safe runners
│   ├── experiments_lac.py       # LAC runners (original + exchangeability)
│   └── experiments_saps.py      # SAPS runners (original + exchangeability)
├── notebooks/
│   └── demo.ipynb               # end-to-end walkthrough of all variants
├── data/                        # place your .npz data files here (see below)
└── README.md
```

---

## Score families

### RAPS — Regularised Adaptive Prediction Sets
> Angelopoulos et al., ICLR 2021 · [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)

$$s(x,y) = \sum_{k=1}^{L(y)}\bigl(p_{(k)} + r_k\bigr) - U \cdot \bigl(p_{L(y)} + r_{L(y)}\bigr)$$

where $r_k = \lambda_{\text{reg}} \cdot \max(k - k_{\text{reg}},\, 0)$ penalises large sets and $L(y)$ is the rank of the true class.

### LAC — Least Ambiguous set-valued Classifiers
> Sadinle et al., JASA 2019 · [arXiv:1609.00451](https://arxiv.org/abs/1609.00451)

$$s(x,y) = 1 - p_y$$

Prediction set $\mathcal{C}(x) = \{y : 1 - p_y \leq \hat{q}\}$ — all classes with sufficiently high softmax probability.

### SAPS — Sorted Adaptive Prediction Sets
> CP via Class Similarity paper

$$s(x,y) = \begin{cases} U \cdot p_{\max} & \text{if } L(y) = 0 \\ p_{\max} + (L(y)-1+U)\cdot\lambda_2 & \text{otherwise} \end{cases}$$

$\lambda_2$ controls rank spacing: small values approach LAC; large values give uniform spacing and smaller average sets when the model is confident.

---

## Original vs exchangeability protocol

When $\lambda > 0$, selecting it on a held-out validation fold and then reusing that fold in the calibration set violates the exchangeability assumption that underpins conformal coverage guarantees.

| Protocol | Cal / Val / Test | Coverage guarantee |
|:---------|:----------------:|:------------------:|
| `_original` | Val merged back into Cal after λ selection | ✗ when λ > 0 |
| `_exchangeability` | Strict three-way split throughout | ✓ always |

---

## Quick start

```python
import sys
sys.path.insert(0, '.')   # run from repo root

import src.data_loader as data_loader
import src.experiments_original as experiments_original
import src.experiments_tuned_lambda_exchangeability as experiments_tuned_lambda_exchangeability
import src.experiments_lac as experiments_lac
import src.experiments_saps as experiments_saps

# Load data
probabilities, labels = data_loader.load_logits('data/Cifar100_ResNet50_logits_new.npz')
adj, adj_small        = data_loader.load_adjacency_matrix('data/Adjacency_matrix.npz')
adj_ms                = data_loader.load_similarity_matrix_cosine('data/similarity_matrix_cosine_resnet50.npz')

lam_grid = [0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3]

# RAPS baseline
experiments_original.run_raps_baseline(probabilities, labels)

# RAPS + MS binary-distance penalty (exchangeability)
experiments_tuned_lambda_exchangeability.run_raps_ms_avg_opt_exchangeability(
    probabilities, labels, adj, adj_ms, lam_grid
)

# LAC + MS binary-distance penalty (exchangeability)
experiments_lac.run_lac_ms_binary_dist_avg_opt_exchangeability(
    probabilities, labels, adj, adj_ms, lam_grid
)

# SAPS + MS binary-distance penalty (exchangeability)
experiments_saps.run_saps_ms_binary_dist_avg_opt_exchangeability(
    probabilities, labels, adj, adj_ms,
    lamda_2=0.2, lamda_3_values=lam_grid,
)
```

Override any hyperparameter inline:

```python
experiments_original.run_raps_baseline(
    probabilities, labels,
    config={"alpha": 0.1, "T": 100, "lam_reg": 0.02},
)
```

See [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for a full walkthrough with results comparison.

---

## Data

Each experiment expects three `.npz` files placed in `data/`:

| File | Keys | Shape |
|:-----|:-----|:------|
| `Cifar100_ResNet50_logits_new.npz` | `logits`, `labels` | `(N, 100)`, `(N,)` |
| `Adjacency_matrix.npz` | `full_matrix`, `small_matrix` | `(100, 101)`, `(100, 100)` |
| `similarity_matrix_cosine_resnet50.npz` | `cosine_sim` | `(100, 100)` |

The last column of `full_matrix` stores the integer superclass id for each of the 100 CIFAR-100 fine-grained classes.

---

## Metrics reported

Each experiment runner prints and returns:

- **Marginal coverage** — fraction of test samples whose true label is in the set
- **Mean class-conditional coverage** — average per-class coverage rate
- **Worst-class CC deviation** — max |coverage(y) − (1−α)| over all classes
- **Top-5 worst CC deviation** — same, averaged over the 5 most-undercovered classes
- **Avg / std / min / max set size**
- **Avg distinct superclasses per set**

---

## Citation

If you use this code, please cite:

```bibtex
@article{dabah2024enhancing,
  title   = {Enhancing Conformal Prediction via Class Similarity},
  author  = {Dabah, Ariel and Tirer, Tom},
  journal = {arXiv preprint arXiv:2511.19359},
  year    = {2024}
}
```

---

## References

- Angelopoulos, A. N., et al. *Uncertainty Sets for Image Classifiers using Conformal Prediction.* ICLR 2021. [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)
- Sadinle, M., Lei, J., & Wasserman, L. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA 2019. [arXiv:1609.00451](https://arxiv.org/abs/1609.00451)
- Huang, X., et al. *Conformal Prediction for Deep Classifier via Label Ranking.* [arXiv:2310.06430](https://arxiv.org/abs/2310.06430)