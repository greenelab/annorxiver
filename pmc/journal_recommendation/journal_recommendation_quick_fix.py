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

# # Re-Run KNearestNeighbors for Journal Recommendation

# This notebook is designed to predict journals based on an updated version of document vector generation. Before I was doing a simple token analysis using spaces, but now I incorporated Spacy with lemma generation. To simplify running the recommendation notebook all over again, I'm just using the 300 dimensions to train a KNN-model and to compare its performance against a random baseline.

# +
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
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

# # Load bioRxiv Papers

biorxiv_journal_df = (
    pd.read_csv(
        Path("../..")
        / Path("biorxiv")
        / Path("journal_tracker")
        / Path("output/mapped_published_doi.tsv"),
        sep="\t",
    )
    .groupby("preprint_doi")
    .agg(
        {
            "document": "last",
            "category": "first",
            "preprint_doi": "last",
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

# # Load PubMed Central Papers

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

pmc_embedding_df = pd.read_csv(
    "../word_vector_experiment/output/pmc_document_vectors_300_replace.tsv.xz", sep="\t"
)
pmc_embedding_df.head()

full_dataset_df = (
    pmc_articles_df.query(f"pmcid not in {golden_set_df.pmcid.tolist()}")[["pmcid"]]
    .merge(pmc_embedding_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
    .set_index("document")
)
full_dataset_df.head()

subsampled_df = (
    pmc_articles_df.query(f"pmcid not in {golden_set_df.pmcid.tolist()}")[["pmcid"]]
    .merge(pmc_embedding_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
    .groupby("journal", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 100), random_state=100))
    .set_index("document")
)
subsampled_df.head()

# # Train Similarity Search System

knn_model = KNeighborsClassifier(n_neighbors=10)

# ## Random Journal Prediction

model = DummyClassifier(strategy="uniform")

_ = cross_validation(
    model, subsampled_df, dummy_evaluate, cv=10, random_state=100, top_predictions=10
)

# ## Centroid Prediction

_ = cross_validation(
    knn_model, subsampled_df, knn_centroid_evaluate, cv=10, random_state=100
)

# ## Paper by Paper prediction

_ = cross_validation(knn_model, subsampled_df, knn_evaluate, cv=10, random_state=100)

# # Gold Set Analysis

biorxiv_embeddings_df = pd.read_csv(
    Path(
        "../../biorxiv/word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz"
    ).resolve(),
    sep="\t",
)
biorxiv_embeddings_df.head()

golden_dataset = (
    golden_set_df[["document", "pmcid"]]
    .merge(pmc_articles_df[["journal", "pmcid"]], on="pmcid")
    .merge(biorxiv_embeddings_df, on="document")
)
golden_dataset.head()

model = DummyClassifier(strategy="uniform")

_ = cross_validation(
    model,
    golden_dataset.drop(["pmcid", "document"], axis=1),
    dummy_evaluate,
    cv=10,
    random_state=100,
    top_predictions=10,
)

# ## Centroid Analysis

predictions, true_labels = knn_centroid_evaluate(
    knn_model, subsampled_df, golden_dataset.drop(["pmcid", "document"], axis=1)
)

accs = [
    (1 if true_labels[data_idx] in prediction_row else 0)
    for data_idx, prediction_row in enumerate(predictions)
]

print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)*100}% correct")

# ## Paper by Paper analysis

predictions, true_labels = knn_evaluate(
    knn_model, subsampled_df, golden_dataset.drop(["pmcid", "document"], axis=1)
)

accs = [
    (1 if true_labels[data_idx] in prediction_row else 0)
    for data_idx, prediction_row in enumerate(predictions)
]

print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)*100}% correct")

# # Save Entire Dataset

(
    pmc_articles_df[["pmcid"]]
    .merge(pmc_embedding_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
    .to_csv(
        "output/paper_dataset/paper_dataset_full.tsv.xz",
        sep="\t",
        compression="xz",
        index=False,
    )
)

# +
cols = dict(document="first")
cols.update({col: "mean" for col in pmc_embedding_df if "feat" in col})

(
    pmc_articles_df[["pmcid"]]
    .merge(pmc_embedding_df, left_on="pmcid", right_on="document")
    .drop(["pmcid"], axis=1)
    .groupby("journal")
    .agg(cols)
    .reset_index()
    .to_csv("output/paper_dataset/centroid.tsv", sep="\t", index=False)
)
