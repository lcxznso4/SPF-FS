import numpy as np
import scanpy as sc
import anndata
from scipy.sparse import csr_matrix, lil_matrix


def spearman_distance_matrix(X):
    """
    Compute Spearman distance matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape=(n_features, n_samples)

    Returns
    -------
    dist : np.ndarray
        Distance matrix, shape=(n_samples, n_samples)
    """
    X_t = X.T  # shape=(n_samples, n_features)
    # Compute ranks
    ranks = np.argsort(np.argsort(X_t, axis=1), axis=1).astype(float)
    # Center ranks
    rc = ranks - ranks.mean(axis=1, keepdims=True)
    cov = rc @ rc.T
    norm_ = np.linalg.norm(rc, axis=1, keepdims=True)
    corr = cov / (norm_ * norm_.T + 1e-10)
    dist = 1 - corr
    return dist


def construct_knn_graph(dist, k=10):
    """
    Construct k-nearest neighbor graph from distance matrix.

    Parameters
    ----------
    dist : np.ndarray
        Distance matrix, shape=(n, n)
    k : int, default=10
        Number of nearest neighbors

    Returns
    -------
    adj_sym : np.ndarray
        Symmetric adjacency matrix
    """
    n = dist.shape[0]
    adj = np.zeros((n, n), dtype=float)

    for i in range(n):
        idx_sorted = np.argsort(dist[i, :])
        neighbors = idx_sorted[1:k + 1]
        adj[i, neighbors] = 1

    adj_sym = np.maximum(adj, adj.T)
    return adj_sym


def leiden_clustering(adj, resolution=1.0):
    """
    Perform Leiden clustering on adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix
    resolution : float, default=1.0
        Resolution parameter for Leiden algorithm

    Returns
    -------
    labels : np.ndarray
        Cluster labels
    """
    adata = anndata.AnnData(np.zeros((adj.shape[0], 1)))
    A_csr = csr_matrix(adj)
    adata.obsp["connectivities"] = A_csr
    adata.uns["neighbors"] = {}
    adata.uns["neighbors"]["connectivities_key"] = "connectivities"
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden')
    labels = adata.obs['leiden'].astype(int).values
    return labels