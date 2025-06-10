"""
Example usage of SPF-FS framework.
"""
import pandas as pd
from src.spf_fs import SPFFS
from src.preprocessing import preprocess_data, compute_variance_score
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)


def main():
    # Load data
    data_file = "./data/sample_data/pollen_counts.csv"
    X_full, df_full = preprocess_data(data_file, min_samples=3)

    # Initial feature selection by variance
    var_scores = compute_variance_score(X_full)
    top2000_idx = np.argsort(var_scores)[-2000:]
    X = X_full[top2000_idx, :]
    gene_names = df_full.columns[top2000_idx]

    # Apply SPF-FS
    spf = SPFFS(
        n_features=400,
        n_base_learners=10,
        n_clusters=11,
        max_iter=10,
        gamma=1.0,
        eta=0.2
    )

    # Fit and transform
    X_selected = spf.fit_transform(X, gene_names)
    selected_genes = gene_names[spf.selected_features_]

    # Save results
    result_df = pd.DataFrame(
        X_selected.T,
        columns=selected_genes,
        index=df_full.index
    )
    result_df.to_csv("./output/selected_features.csv")

    print(f"Selected {len(selected_genes)} features")
    print(f"Top 10 genes: {selected_genes[:10].tolist()}")


if __name__ == "__main__":
    main()