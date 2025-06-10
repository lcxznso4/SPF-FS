import numpy as np
from sklearn.mixture import GaussianMixture

EPS = 1e-10


def compute_hvg_scores(X_subset):
    """
    Compute Highly Variable Gene (HVG) scores based on variance.

    Parameters
    ----------
    X_subset : np.ndarray
        Gene expression subset, shape=(n_genes, n_samples)

    Returns
    -------
    scores : np.ndarray
        HVG scores, normalized to [0, 1]
    """
    var_ = np.var(X_subset, axis=1)
    scores = (var_ - var_.min()) / (var_.max() - var_.min() + EPS)
    return scores


def compute_fano_scores(X_subset):
    """
    Compute Fano factor scores (variance/mean).

    Parameters
    ----------
    X_subset : np.ndarray
        Gene expression subset, shape=(n_genes, n_samples)

    Returns
    -------
    scores : np.ndarray
        Fano factor scores, normalized to [0, 1]
    """
    mean_ = np.mean(X_subset, axis=1) + EPS
    var_ = np.var(X_subset, axis=1)
    fano = var_ / mean_
    scores = (fano - fano.min()) / (fano.max() - fano.min() + EPS)
    return scores


def compute_gmm_scores(X_subset):
    """
    Compute GMM-based bimodality scores using BIC difference.

    Parameters
    ----------
    X_subset : np.ndarray
        Gene expression subset, shape=(n_genes, n_samples)

    Returns
    -------
    scores : np.ndarray
        GMM scores, normalized to [0, 1]
    """
    d, _ = X_subset.shape
    gmm_score = np.zeros(d)

    for i in range(d):
        data_i = X_subset[i, :].reshape(-1, 1)

        gm1 = GaussianMixture(n_components=1, random_state=0).fit(data_i)
        gm2 = GaussianMixture(n_components=2, random_state=0).fit(data_i)
        bic1 = gm1.bic(data_i)
        bic2 = gm2.bic(data_i)
        diff = bic1 - bic2
        # Higher diff means 2-component model is better
        gmm_score[i] = max(diff, 0.0)

    # Normalize
    gmm_score = (gmm_score - gmm_score.min()) / (gmm_score.max() - gmm_score.min() + EPS)
    return gmm_score