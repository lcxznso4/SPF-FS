# SPF-FS: A Self-Paced Framework Fusing Feature Selection and Clustering with Prior Knowledge for scRNA-seq

SPF-FS is a novel two-tier optimization framework for feature selection in single-cell RNA sequencing (scRNA-seq) data. It combines multiple feature selection methods with Gene Ontology Biological Process (GO-BP) priors to jointly reduce high-dimensional noise and improve biological interpretability.

🌟 Key Features


Multi-method Ensemble: Integrates Highly Variable Genes (HVG), Fano factor, and GMM-based bimodality scores

Biological Prior Integration: Incorporates GO-BP enrichment information to guide feature selection

Self-paced Learning: Adaptively selects reliable samples during optimization

Inter-cluster Separation: Enhances cluster separability through structural co-optimization

Basic Usage
```
import pandas as pd
import numpy as np
from spf_fs import preprocess_data, compute_variance_score, SPFFS

# Load data
X, df = preprocess_data("your_data.csv", min_samples=3)

# Initial filtering by variance (optional but recommended)
var_scores = compute_variance_score(X)
top2000_idx = np.argsort(var_scores)[-2000:]
X_filtered = X[top2000_idx, :]
gene_names = df.columns[top2000_idx]

# Apply SPF-FS
spf = SPFFS(
    n_features=400,      # Number of features to select
    n_clusters=5,        # Expected number of clusters
    n_base_learners=10,  # Number of base learners
    max_iter=10          # Maximum iterations
)

# Fit and select features
X_selected = spf.fit_transform(X_filtered, gene_names)
selected_genes = gene_names[spf.selected_features_]

print(f"Selected {len(selected_genes)} features")
print(f"Top 10 genes: {selected_genes[:10].tolist()}")
```
📁 Project Structure
```
SPF-FS/
├── src/                    # Core algorithm implementation
│   ├── __init__.py
│   ├── spf_fs.py          # Main SPF-FS class
│   ├── preprocessing.py    # Data preprocessing utilities
│   ├── feature_scoring.py  # Feature scoring methods
│   ├── clustering.py       # Clustering algorithms
│   ├── enrichment.py       # GO enrichment analysis
│   └── utils.py           # Helper functions
├── data/                  # Data files
│   ├── c5.go.bp.v2024.1.Hs.symbols.gmt  # GO BP database
│   └── sample_data/       # Example datasets
├── examples/              # Usage examples
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── requirements.txt       # Package dependencies
├── setup.py              # Package setup
├── LICENSE               # MIT license
└── README.md            # This file
```


🔧 Advanced Usage
Custom Parameters
```
spf = SPFFS(
    n_features=400,
    n_base_learners=10,
    n_clusters=11,
    max_iter=10,
    gamma=1.0,           # GO prior weight (0-2)
    eta=0.2,             # Inter-cluster separation strength
    lambda_init=0.2,     # Initial self-paced learning parameter
    lambda_growth=0.3,   # Lambda growth rate per iteration
    select_ratio=0.5,    # Ratio of features for base learners
    k=10,                # k-NN graph parameter
    resolution=1.0       # Leiden clustering resolution
)
```
🧬 Input Data Format
SPF-FS expects input data in the following format:

CSV file: Rows = cells/samples, Columns = genes

Expression values: Raw counts or normalized values

Gene names: As column headers

Cell IDs: As row indices (optional)

```
Gene1   Gene2   Gene3   ...
Cell1   0       125     0       ...
Cell2   10      0       45      ...
Cell3   0       200     12      ...
...
```
