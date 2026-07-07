# CP_via_CS
This is the code accompanying the paper: [Enhancing Conformal Prediction via Class Similarity](https://arxiv.org/abs/2511.19359) by Ariel Fargion, Lahav Dabah and Dr. Tom Tirer

<!-- Banner -->
<div align="center">

# CP via Class Similarity

**Enhancing Conformal Prediction via Class Similarity**

[![arXiv](https://img.shields.io/badge/arXiv-2511.19359-b31b1b.svg)](https://arxiv.org/abs/2511.19359)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

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

Both penalties are applied to three score families — **RAPS**, **LAC**, and **SAPS**

---

## Repository layout

```
CP_via_CS/
├── src/
│   ├── config.py     
│   ├── data_loader.py  
│   ├── score_functions.py    
│   ├── metrics.py       
│   ├── experiments_original.py 
│   ├── experiments_tuned_lambda_strict_split.py
│   ├── experiments_lac.py  
│   └── experiments_saps.py   
├── notebooks/
│   └── demo_.ipynb     
├── data/     
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
| `_strict_split` | Strict three-way split throughout | ✓ always |

---

## Example

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
@article{fargion2026enhancing,
  title   = {Enhancing Conformal Prediction via Class Similarity},
  author  = {Fargion, Ariel and Dabah, Lahav and Tirer, Tom},
  journal = {arXiv preprint arXiv:2511.19359},
  year    = {2026}
}
```

---

## References

- Angelopoulos, A. N., et al. *Uncertainty Sets for Image Classifiers using Conformal Prediction.* ICLR 2021. [arXiv:2009.14193](https://arxiv.org/abs/2009.14193)
- Sadinle, M., Lei, J., & Wasserman, L. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA 2019. [arXiv:1609.00451](https://arxiv.org/abs/1609.00451)
- Huang, X., et al. *Conformal Prediction for Deep Classifier via Label Ranking.* ICML 2024. [arXiv:2310.06430](https://arxiv.org/abs/2310.06430)
