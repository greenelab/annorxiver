# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:annorxiver]
#     language: python
#     name: conda-env-annorxiver-py
# ---

# # Visualize BioRxiv Document Embeddings

# This notebook is designed to visualize umap and tsne representations of bioRxiv document embeddings.
# Each document embedding is generated via an average of every word in a given article.

# +
import itertools
from pathlib import Path
import re

import pandas as pd
import plotnine as p9
# -

journal_map_df = pd.read_csv(
    "../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t"
)
journal_map_df.head()

# Keep only current versions of articles
biorxiv_umap_models_latest = {
    "original": (
        pd.read_csv("output/embedding_output/umap/biorxiv_umap_300.tsv", sep="\t")
        .groupby("doi")
        .agg(
            {
                "doi": "last",
                "document": "last",
                "umap1": "last",
                "umap2": "last",
                "category": "last",
            }
        )
        .reset_index(drop=True)
    ),
    "doc2vec": (
        pd.read_csv(
            "output/embedding_output/umap/biorxiv_umap_300_doc2vec.tsv", sep="\t"
        )
        .groupby("doi")
        .agg(
            {
                "doi": "last",
                "document": "last",
                "umap1": "last",
                "umap2": "last",
                "category": "last",
            }
        )
        .reset_index(drop=True)
    ),
}

# Keep only current versions of articles
biorxiv_pca_models_latest = {
    "original": (
        pd.read_csv("output/embedding_output/pca/biorxiv_pca_300.tsv", sep="\t")
        .groupby("doi")
        .agg(
            {
                "doi": "last",
                "document": "last",
                "pca1": "last",
                "pca2": "last",
                "category": "last",
            }
        )
        .reset_index(drop=True)
    ),
    "doc2vec": (
        pd.read_csv("output/embedding_output/pca/biorxiv_pca_300_doc2vec.tsv", sep="\t")
        .groupby("doi")
        .agg(
            {
                "doi": "last",
                "document": "last",
                "pca1": "last",
                "pca2": "last",
                "category": "last",
            }
        )
        .reset_index(drop=True)
    ),
}

# # UMAP of the Documents

g = (
    p9.ggplot(biorxiv_umap_models_latest["original"])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(title="UMAP of BioRxiv (Word dim: 300)", color="Article Category")
)
g.save("output/embedding_output/umap/figures/biorxiv_umap_300.png", dpi=500)
print(g)

g = (
    p9.ggplot(biorxiv_umap_models_latest["doc2vec"])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(title="UMAP of BioRxiv (Doc2vec Word dim: 300)", color="Article Category")
)
g.save("output/embedding_output/umap/figures/biorxiv_umap_300_doc2vec.png", dpi=500)
print(g)

# # PCA of the Documents

g = (
    p9.ggplot(biorxiv_pca_models_latest["original"])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(title="PCA of BioRxiv (Word dim: 300)", color="Article Category")
)
g.save("output/embedding_output/pca/figures/biorxiv_pca_300.png", dpi=500)
print(g)

g = (
    p9.ggplot(biorxiv_pca_models_latest["doc2vec"])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(title="PCA of BioRxiv (Doc2vec Word dim: 300)", color="Article Category")
)
g.save("output/embedding_output/pca/figures/biorxiv_pca_300_doc2vec.png", dpi=500)
print(g)
