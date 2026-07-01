"""
data_loader.py – helpers for loading CIFAR-100 logits, labels, and adjacency matrices.

Expected file layout (paths can be overridden when calling each function):

    data/
        cifar-100-python.tar.gz          # raw CIFAR-100 archive
        Adjacency_matrix.npz             # full adjacency + superclass column
        similarity_matrix_cosine_resnet50.npz
        Cifar100_ResNet50_logits_new.npz
        ...
"""

import os
import pickle
import tarfile
import numpy as np
from scipy.special import softmax


# ── Raw CIFAR-100 archive ─────────────────────────────────────────────────────

def extract_cifar100(archive_path: str, extract_dir: str = "cifar-100-python") -> None:
    """Extract the CIFAR-100 tar.gz archive (no-op if already extracted)."""
    if not os.path.isdir(extract_dir):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)


def load_cifar100_test(extract_dir: str = "cifar-100-python") -> dict:
    """Return the raw CIFAR-100 test pickle (keys are byte strings)."""
    test_file = os.path.join(extract_dir, "cifar-100-python", "test")
    with open(test_file, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar100_meta(extract_dir: str = "cifar-100-python") -> dict:
    """Return the CIFAR-100 meta pickle (fine/coarse label name lists)."""
    meta_file = os.path.join(extract_dir, "cifar-100-python", "meta")
    with open(meta_file, "rb") as f:
        return pickle.load(f, encoding="bytes")


# ── Logits / probabilities ────────────────────────────────────────────────────

def load_logits(logits_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed logits and integer labels from a .npz file.

    Returns
    -------
    probabilities : ndarray of shape (N, C)
        Softmax probabilities.
    labels : ndarray of shape (N,), dtype int
    """
    with np.load(logits_path) as data:
        logits = data["logits"]
        labels = data["labels"].astype(int)
    probabilities = softmax(logits, axis=1)
    return probabilities, labels


# ── Adjacency / similarity matrices ──────────────────────────────────────────

def load_adjacency_matrix(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the full adjacency matrix (shape [C, C+1]) and the smaller version
    (shape [C, C]) without the superclass column.

    The last column (index C) stores the integer superclass id for each class.

    Returns
    -------
    adjacency_matrix : ndarray, shape (C, C+1)
    adjacency_matrix_smaller : ndarray, shape (C, C)
    """
    data = np.load(path)
    adjacency_matrix = data["Adjacency_matrix"]
    adjacency_matrix_smaller = adjacency_matrix[:,:-1]
    return adjacency_matrix, adjacency_matrix_smaller


def load_similarity_matrix_cosine(path: str) -> np.ndarray:
    """Load a cosine-similarity matrix stored under the key 'cosine_sim'."""
    return np.load(path)["cosine_sim"]


def load_similarity_matrix_gaussian(path: str) -> np.ndarray:
    """Load a Gaussian-kernel similarity matrix stored under the key 'gaussian_sim'."""
    return np.load(path)["gaussian_sim"]
