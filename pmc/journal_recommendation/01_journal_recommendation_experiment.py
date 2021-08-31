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

# # Journal Recommendation for Preprints

# The goal of this notebook is to help users know which journal would be most appropriate for their preprint. The central idea is to use euclidean distance between documents to gauge which journal similar works have been sent.

# +
# %load_ext autoreload
# %autoreload 2

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm_notebook

from annorxiver_modules.journal_rec_helper import (
    cross_validation,
    dummy_evaluate,
    knn_evaluate,
    knn_centroid_evaluate,
)
# -

# # Load bioRxiv Document Vectors

biorxiv_journal_df = (
    pd.read_csv(
        Path("../..")
        / Path("biorxiv")
        / Path("journal_tracker")
        / Path("output/mapped_published_doi.tsv"),
        sep="\t",
    )
    .groupby("doi")
    .agg(
        {
            "document": "last",
            "category": "first",
            "journal": "first",
            "doi": "last",
            "published_doi": "first",
            "pmcid": "first",
        }
    )
    .reset_index(drop=True)
)
biorxiv_journal_df.head()

# Count number of Non-NaN elements
print(f"Number of Non-NaN entries: {biorxiv_journal_df.pmcid.count()}")
print(f"Total number of entries: {biorxiv_journal_df.shape[0]}")
print(
    f"Percent Covered: {(biorxiv_journal_df.pmcid.count()/biorxiv_journal_df.shape[0])*100:.2f}%"
)

golden_set_df = biorxiv_journal_df.query("pmcid.notnull()")
golden_set_df.head()

# # Load Pubmed Central Document Vectors

pmc_articles_df = pd.read_csv(
    Path("../exploratory_data_analysis")
    / Path("output")
    / Path("pubmed_central_journal_paper_map.tsv.xz"),
    sep="\t",
).query("article_type=='research-article'")
print(pmc_articles_df.shape)
pmc_articles_df.head()

journals = pmc_articles_df.journal.value_counts()
print(journals.shape)
journals

# Filter out low count journals
pmc_articles_df = pmc_articles_df.query(
    f"journal in {journals[journals > 100].index.tolist()}"
)
print(pmc_articles_df.shape)
pmc_articles_df.head()

pmc_embedding_dict = {
    int(path.stem.split(".")[0].split("_")[-1]): pd.read_csv(str(path), sep="\t")
    for path in Path("../word_vector_experiment/output/").rglob("*tsv.xz")
}
pmc_embedding_dict[300].head()

full_training_dataset = {
    dim: (
        pmc_articles_df.query(f"pmcid not in {golden_set_df.pmcid.tolist()}")[
            ["pmcid", "journal"]
        ]
        .merge(pmc_embedding_dict[dim], left_on="pmcid", right_on="document")
        .drop("pmcid", axis=1)
        .set_index("document")
    )
    for dim in pmc_embedding_dict
}

subsampled_training_dataset = {
    dim: (
        pmc_articles_df.query(f"pmcid not in {golden_set_df.pmcid.tolist()}")[
            ["pmcid", "journal"]
        ]
        .merge(pmc_embedding_dict[dim], left_on="pmcid", right_on="document")
        .drop("pmcid", axis=1)
        .groupby("journal", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 100), random_state=100))
        .set_index("document")
    )
    for dim in pmc_embedding_dict
}

# # Train Recommendation System

knn_model = KNeighborsClassifier(n_neighbors=10)

# ## Random Journal Prediction

# The central idea here is to answer the question what is the accuracy when journals are recommended at random?

model = DummyClassifier(strategy="uniform")

_ = cross_validation(
    model,
    subsampled_training_dataset[300],
    dummy_evaluate,
    cv=10,
    random_state=100,
    top_predictions=10,
)

# ## KNearestNeighbors Paper by Paper Comparison - Full Dataset

# Assuming I didn't take mega-journal influence into account, what would the initial recommendation accuracy be?

_ = cross_validation(
    knn_model, full_training_dataset[300], knn_evaluate, cv=10, random_state=100
)

# ## KNearestNeighbors Paper by Paper Comparison Subsampled

# The first idea for a classifier is to compare which papers are similar to other papers. Due to the overflow of PLOS One papers I sub-sampled each journal to have only 100 papers for representation. Then trained a KNearestNeighbors to determine how often does the correct journal appear in the top ten neighbors as well as top twenty neighbors.

result_dict = {}
for dim in subsampled_training_dataset:
    print(dim)

    fold_predictions = cross_validation(
        knn_model,
        subsampled_training_dataset[dim],
        knn_evaluate,
        cv=10,
        random_state=100,
    )

    print()

    result_dict[dim] = fold_predictions

# ## KNearestNeighbors Centroid analysis

# Following up on the original idea, I thought a helpful experiment would be to perform a centroid analysis (i.e. take the average of all papers within each journal). Similar to above I trained a KNearestNeighbors classifier to see if the correct journal will appear in the top 10 neighbors.

_ = cross_validation(
    knn_model,
    subsampled_training_dataset[300],
    knn_centroid_evaluate,
    cv=10,
    random_state=100,
)

# ## KNearestNeighbors Centroid Analysis - Full dataset

# This section I'm using the entire dataset to calculate journal centroids and then evaluate performance on the sub-sampled dataset.

predictions, true_labels = knn_centroid_evaluate(
    knn_model, full_training_dataset[300], subsampled_training_dataset[300]
)

accs = [
    (1 if true_labels[data_idx] in prediction_row else 0)
    for data_idx, prediction_row in enumerate(predictions)
]

print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)*100}% correct")

# ## KNearestNeighbors Manhattan Distance - Paper by Paper

knn_model = KNeighborsClassifier(n_neighbors=10, metric="manhattan")

fold_predictions = cross_validation(
    knn_model, subsampled_training_dataset[300], knn_evaluate, cv=10, random_state=100
)

# ## KNearestNeighbors Manhattan Distance - Centroid

predictions, true_labels = knn_centroid_evaluate(
    knn_model, full_training_dataset[300], subsampled_training_dataset[300]
)

accs = [
    (1 if true_labels[data_idx] in prediction_row else 0)
    for data_idx, prediction_row in enumerate(predictions)
]

print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)*100}% correct")

# ## KNearestNeighbors - PCA'd Paper prediction

knn_model = KNeighborsClassifier(n_neighbors=10)
pca = PCA(n_components=10)

pca_subsampled_training_dataset = pca.fit_transform(
    subsampled_training_dataset[300].drop("journal", axis=1).values
)

new_dataset = pd.DataFrame(
    pca_subsampled_training_dataset, columns=[f"pca_{comp+1}" for comp in range(10)]
).assign(journal=subsampled_training_dataset[300].journal.values.tolist())
new_dataset.head()

fold_predictions = cross_validation(
    knn_model, new_dataset, knn_evaluate, cv=10, random_state=100
)

# # Golden Set Analysis

# +
biorxiv_embeddings_df = pd.read_csv(
    Path("../..")
    / Path("word_vector_experiment")
    / Path("output/word2vec_output")
    / Path("biorxiv_all_articles_300.tsv.xz").resolve(),
    sep="\t",
)

biorxiv_embeddings_df.head()
# -

golden_dataset = (
    golden_set_df[["document", "pmcid"]]
    .merge(pmc_articles_df[["journal", "pmcid"]], on="pmcid")
    .merge(biorxiv_embeddings_df, on="document")
)
golden_dataset.head()

# ## Centroid Analysis

predictions, true_labels = knn_centroid_evaluate(
    knn_model,
    full_training_dataset[300],
    golden_dataset.drop(["pmcid", "document"], axis=1),
)

accs = [
    (1 if true_labels[data_idx] in prediction_row else 0)
    for data_idx, prediction_row in enumerate(predictions)
]

print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)*100}% correct")

# ## Subsampled Paper Analysis

predictions, true_labels = knn_evaluate(
    knn_model,
    subsampled_training_dataset[300],
    golden_dataset.drop(["pmcid", "document"], axis=1),
)

accs = [
    (1 if true_labels[data_idx] in prediction_row else 0)
    for data_idx, prediction_row in enumerate(predictions)
]

print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)*100}% correct")

# # Save Best Dataset

# Save best dataset for future experiments
(
    full_training_dataset[300].to_csv(
        "output/paper_dataset/paper_dataset_full.tsv.xz",
        sep="\t",
        compression="xz",
        index=False,
    )
)

# Conclusions for this notebook:
# 1. Mega-journals cover a wide range of research topics.
# 2. The correct journal only appears in the top ten about 37-39 percent of the time.
# 3. 300 dimensions gives the best performance compared to the other dimensions.
# 4. Reporting a combination of centroid analysis and individual paper predictions will be needed to go forward.
