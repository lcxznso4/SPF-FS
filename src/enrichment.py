import numpy as np
import gseapy
import os


def run_gobp_enrichment(gene_list, gmt_path=None):
    """
    Run GO Biological Process enrichment analysis.

    Parameters
    ----------
    gene_list : list
        List of gene names
    gmt_path : str, optional
        Path to GMT file. If None, uses default path

    Returns
    -------
    res_df : pd.DataFrame
        Enrichment results
    """
    if gmt_path is None:
        # Default path relative to project root
        gmt_path = os.path.join(os.path.dirname(__file__),
                                "../data/c5.go.bp.v2024.1.Hs.symbols.gmt")

    enr = gseapy.enrichr(
        gene_list=gene_list,
        gene_sets=gmt_path,
        outdir=None,
        cutoff=1.0
    )
    enr.run()
    return enr.res2d


def define_additive_prior_gobp(gene_list, p_cut=0.05, gmt_path=None):
    """
    Define additive prior weights based on GO enrichment p-values.

    Parameters
    ----------
    gene_list : list
        List of gene names
    p_cut : float, default=0.05
        P-value cutoff
    gmt_path : str, optional
        Path to GMT file

    Returns
    -------
    weights : np.ndarray
        Prior weights for genes
    """
    res_df = run_gobp_enrichment(gene_list, gmt_path)
    gene2p = {g: 1.0 for g in gene_list}

    if len(res_df) > 0:
        for _, row in res_df.iterrows():
            p = row["P-value"]
            genes_in_term = row["Genes"].replace(" ", "").split(";")
            for gg in genes_in_term:
                gene2p[gg] = min(gene2p.get(gg, 1.0), p)

    w = []
    for g in gene_list:
        p = gene2p[g]
        if p > p_cut:
            w.append(0.2)
        else:
            ratio = (p_cut - p) / p_cut
            val = 0.2 + 0.6 * ratio
            w.append(val)

    return np.array(w)