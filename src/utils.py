import numpy as np
from numpy.linalg import eigh

EPS = 1e-10


def update_W(S, base_connectivity, beta, lam):
    """Update self-paced learning weight matrix W."""
    m = len(base_connectivity)
    S_d = S.toarray()
    A = np.zeros_like(S_d)

    for k in range(m):
        diff = S_d - base_connectivity[k].toarray()
        A += (beta[k] ** 2) * (diff ** 2)

    W = np.minimum(lam / (2 * (A + EPS)), 1)
    return W


def update_alpha(v, base_scores):
    """Update ensemble weights alpha."""
    m = len(base_scores)
    arr = np.zeros(m)

    for k in range(m):
        dist = np.linalg.norm(v - base_scores[k]) ** 2
        arr[k] = 1.0 / (dist + EPS)

    alpha = arr / (arr.sum() + EPS)
    return alpha


def update_beta(S, base_connectivity, W):
    """Update ensemble weights beta."""
    m = len(base_connectivity)
    S_d = S.toarray()
    arr = np.zeros(m)

    for k in range(m):
        diff = W * (S_d - base_connectivity[k].toarray())
        val = np.linalg.norm(diff, 'fro') ** 2
        arr[k] = 1.0 / (val + EPS)

    beta = arr / (arr.sum() + EPS)
    return beta