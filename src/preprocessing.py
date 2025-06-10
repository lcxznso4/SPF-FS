import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_data(file, min_samples=3):
    """
    Preprocess single-cell RNA-seq data.

    Parameters
    ----------
    file : str
        Path to the CSV file containing gene expression data
    min_samples : int, default=3
        Minimum number of samples where a gene must be expressed

    Returns
    -------
    X : np.ndarray
        Gene expression matrix, shape=(n_genes, n_samples)
    df_normalized : pd.DataFrame
        Normalized expression dataframe
    """
    df = pd.read_csv(file, index_col=0)
    # Filter genes expressed in at least min_samples
    df_filtered = df.loc[:, (df > 0).sum(axis=0) >= min_samples]
    # Log normalization
    df_normalized = np.log1p(df_filtered)
    # Transpose to get shape=(n_genes, n_samples)
    X = df_normalized.values.T
    return X, df_normalized


def compute_variance_score(X):
    """
    Compute variance score for each gene.

    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix, shape=(n_genes, n_samples)

    Returns
    -------
    scores : np.ndarray
        Variance score for each gene
    """
    return np.var(X, axis=1)