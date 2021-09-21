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

# # Find published articles missing from bioRxiv using abstracts alone

# +
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
from scipy.spatial.distance import cdist
import scipy.stats
import seaborn as sns
from sklearn.metrics import roc_curve, auc, recall_score, precision_score
import tqdm

import svgutils.transform as sg
from svgutils.compose import Unit
from cairosvg import svg2png
from IPython.display import Image
from lxml import etree
# -

# # Load Embeddings

# ## bioRxiv

biorxiv_journal_df = (
    pd.read_csv(
        "../journal_tracker/output/mapped_published_doi_before_update.tsv", sep="\t"
    )
    .rename(index=str, columns={"doi": "preprint_doi"})
    .groupby("preprint_doi")
    .agg(
        {
            "document": "last",
            "category": "first",
            "preprint_doi": "last",
            "published_doi": "first",
            "pmcid": "first",
            "pmcoa": "first",
        }
    )
    .reset_index(drop=True)
)
biorxiv_journal_df.head()

biorxiv_embed_df = pd.read_csv(
    Path("../word_vector_experiment/output/")
    / "word2vec_output/"
    / "biorxiv_all_articles_300.tsv.xz",
    sep="\t",
)
biorxiv_embed_df = biorxiv_embed_df.dropna()
biorxiv_embed_df.head()

biorxiv_journal_mapped_df = biorxiv_journal_df[
    ["document", "preprint_doi", "published_doi", "pmcid", "pmcoa"]
].merge(biorxiv_embed_df, on="document")
print(biorxiv_journal_mapped_df.shape)
biorxiv_journal_mapped_df.head()

biorxiv_embed_abstract_only_df = pd.read_csv(
    Path("../word_vector_experiment/output/")
    / "word2vec_output/"
    / "biorxiv_all_articles_300_abstract_only_delete_me.tsv.xz",
    sep="\t",
)
biorxiv_embed_abstract_only_df = biorxiv_embed_abstract_only_df.dropna()
biorxiv_embed_abstract_only_df.head()

# ### Remove preprints with malformed abstracts

missing_abstracts = set(biorxiv_embed_df.document.tolist()).difference(
    set(biorxiv_embed_abstract_only_df.document.tolist())
)
print(len(missing_abstracts))

biorxiv_journal_mapped_df = biorxiv_journal_mapped_df.query(
    f"document not in {list(missing_abstracts)}"
)
print(biorxiv_journal_mapped_df.shape)
biorxiv_journal_mapped_df.head()

biorxiv_journal_mapped_abstract_df = biorxiv_journal_df[
    ["document", "preprint_doi", "published_doi", "pmcid", "pmcoa"]
].merge(biorxiv_embed_abstract_only_df, on="document")
print(biorxiv_journal_mapped_df.shape)
biorxiv_journal_mapped_abstract_df.head()

# ## Pubmed Central

pmc_articles_df = pd.read_csv(
    Path("../../pmc/exploratory_data_analysis/")
    / "output/pubmed_central_journal_paper_map.tsv.xz",
    sep="\t",
).query("article_type=='research-article'")
pmc_articles_df.head()

pmc_embed_df = pd.read_csv(
    Path("../../pmc/word_vector_experiment/output")
    / Path("pmc_document_vectors_300_replace.tsv.xz"),
    sep="\t",
)
pmc_embed_df.head()

pmc_journal_mapped_df = (
    pmc_articles_df[["doi", "pmcid"]]
    .merge(pmc_embed_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
)
pmc_journal_mapped_df.head()

pmc_embed_abstract_only_df = pd.read_csv(
    Path("../../pmc/word_vector_experiment")
    / "output"
    / "pmc_document_vectors_300_abstract_only.tsv.xz",
    sep="\t",
)
pmc_embed_abstract_only_df = pmc_embed_abstract_only_df.dropna()
pmc_embed_abstract_only_df.head()

pmc_journal_mapped_abstract_df = (
    pmc_articles_df[["doi", "pmcid"]]
    .merge(pmc_embed_abstract_only_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
)
pmc_journal_mapped_abstract_df.head()

# ### Remove Published articles with Malformed Abstracts

pmc_full_text = set(pmc_journal_mapped_df.document.tolist())
pmc_abstract = set(pmc_journal_mapped_abstract_df.document.tolist())
missing_articles = pmc_full_text.difference(pmc_abstract)
print(len(missing_articles))
pmc_journal_mapped_df = pmc_journal_mapped_df.query(
    f"document not in {list(missing_articles)}"
)

# # Calculate Distances

# ## biorxiv -> published versions

biorxiv_published = (
    biorxiv_journal_mapped_df.query("pmcid.notnull()")
    .query("pmcoa == True")
    .sort_values("pmcid", ascending=True)
    .drop_duplicates("pmcid")
    .set_index("pmcid")
)
biorxiv_published.head()

PMC_published = (
    pmc_journal_mapped_df.query(f"document in {biorxiv_published.index.tolist()}")
    .sort_values("document", ascending=True)
    .set_index("document")
)
PMC_published.head()

# ### Full Text

article_distances = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()].drop(
        ["document", "preprint_doi", "published_doi", "pmcoa"], axis=1
    ),
    PMC_published.drop(["doi", "journal"], axis=1),
    "euclidean",
)
article_distances.shape

articles_distance_original_df = (
    biorxiv_published.loc[PMC_published.index.tolist()]
    .reset_index()[["document", "pmcid"]]
    .assign(
        distance=np.diag(article_distances, k=0), journal=PMC_published.journal.tolist()
    )
)
articles_distance_original_df.head()

# ### Abstracts

biorxiv_published_abstract = (
    biorxiv_journal_mapped_abstract_df.query("pmcid.notnull()")
    .query("pmcoa == True")
    .sort_values("pmcid", ascending=True)
    .drop_duplicates("pmcid")
    .set_index("pmcid")
)
biorxiv_published_abstract.head()

PMC_published_abstract = (
    pmc_journal_mapped_abstract_df.query(
        f"document in {biorxiv_published_abstract.index.tolist()}"
    )
    .sort_values("document", ascending=True)
    .set_index("document")
)
PMC_published_abstract.head()

article_distances = cdist(
    biorxiv_published_abstract.loc[PMC_published_abstract.index.tolist()].drop(
        ["document", "preprint_doi", "published_doi", "pmcoa"], axis=1
    ),
    PMC_published_abstract.drop(["doi", "journal"], axis=1),
    "euclidean",
)
article_distances.shape

articles_distance_abstract_df = (
    biorxiv_published_abstract.loc[PMC_published_abstract.index.tolist()]
    .reset_index()[["document", "pmcid"]]
    .assign(
        distance=np.diag(article_distances, k=0),
        journal=PMC_published_abstract.journal.tolist(),
    )
)
articles_distance_abstract_df.head()

# ## biorxiv -> random paper same journal

PMC_off_published = (
    pmc_journal_mapped_df.drop("doi", axis=1)
    .query(f"document not in {biorxiv_published.index.tolist()}")
    .query(f"journal in {articles_distance_original_df.journal.unique().tolist()}")
    .groupby("journal", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=100))
)
PMC_off_published.head()

journal_mapper = {
    journal: col for col, journal in enumerate(PMC_off_published.journal.tolist())
}
list(journal_mapper.items())[0:10]

# ### Full Text

off_article_dist = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()]
    .drop(["document", "preprint_doi", "published_doi", "pmcoa"], axis=1)
    .values,
    PMC_off_published.drop(["document", "journal"], axis=1).values,
    "euclidean",
)
off_article_dist.shape

data = []
for idx, row in tqdm.tqdm(articles_distance_original_df.iterrows()):
    if row["journal"] in journal_mapper:
        data.append(
            {
                "document": row["document"],
                "pmcid": (
                    PMC_off_published.query(f"journal=='{row['journal']}'")
                    .reset_index()
                    .document.values[0]
                ),
                "journal": row["journal"],
                "distance": off_article_dist[idx, journal_mapper[row["journal"]]],
            }
        )

final_original_df = articles_distance_original_df.assign(
    label="pre_vs_published"
).append(pd.DataFrame.from_records(data).assign(label="pre_vs_random"))
final_original_df.head()

# ### Abstract

PMC_off_published_abstract = pmc_journal_mapped_abstract_df.query(
    f"document in {PMC_off_published.document.tolist()}"
).sort_values("journal")
PMC_off_published_abstract.head()

off_article_dist = cdist(
    biorxiv_published_abstract.loc[PMC_published_abstract.index.tolist()]
    .drop(["document", "preprint_doi", "published_doi", "pmcoa"], axis=1)
    .values,
    PMC_off_published_abstract.drop(["document", "journal", "doi"], axis=1).values,
    "euclidean",
)
off_article_dist.shape

remaining_journal_mapper = list(
    set(PMC_off_published_abstract.journal.tolist()).intersection(
        set(journal_mapper.keys())
    )
)
remaining_journal_mapper = dict(
    zip(sorted(remaining_journal_mapper), range(len(remaining_journal_mapper)))
)

data = []
for idx, row in tqdm.tqdm(articles_distance_abstract_df.iterrows()):
    if row["journal"] in remaining_journal_mapper:
        data.append(
            {
                "document": row["document"],
                "pmcid": (
                    PMC_off_published_abstract.query(f"journal=='{row['journal']}'")
                    .reset_index()
                    .document.values[0]
                ),
                "journal": row["journal"],
                "distance": off_article_dist[
                    idx, remaining_journal_mapper[row["journal"]]
                ],
            }
        )

final_abstract_df = articles_distance_abstract_df.assign(
    label="pre_vs_published"
).append(pd.DataFrame.from_records(data).assign(label="pre_vs_random"))
final_abstract_df.head()

final_abstract_df = biorxiv_journal_df[["document", "preprint_doi"]].merge(
    final_abstract_df
)
final_abstract_df.to_csv(
    "output/annotated_links/article_distances_abstract_only.tsv", sep="\t", index=False
)
final_abstract_df.head()

# # Distribution plot

g = (
    p9.ggplot(
        final_original_df.replace(
            {
                "pre_vs_published": "preprint-published",
                "pre_vs_random": "preprint-random",
            }
        )
    )
    + p9.aes(x="label", y="distance")
    + p9.geom_violin(fill="#a6cee3")
    + p9.labs(x="Document Pair Groups", y="Euclidean Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=2)
    + p9.theme(figure_size=(11, 8.5))
)
print(g)

g = (
    p9.ggplot(
        final_abstract_df.replace(
            {
                "pre_vs_published": "preprint-published",
                "pre_vs_random": "preprint-random",
            }
        )
    )
    + p9.aes(x="label", y="distance")
    + p9.geom_violin(fill="#a6cee3")
    + p9.labs(x="Document Pair Groups", y="Euclidean Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=2)
    + p9.theme(figure_size=(11, 8.5))
)
print(g)

# # Examine the top N predictions using Recall and Precision

data_rows = []
for df, model_label in zip(
    [final_original_df, final_abstract_df], ["Full Text", "Abstract Only"]
):
    for k in tqdm.tqdm(range(1, 34503, 200)):
        recall = recall_score(
            df.sort_values("distance").iloc[0:k].label.tolist(),
            ["pre_vs_published"] * k
            if k <= df.shape[0]
            else ["pre_vs_published"] * df.shape[0],
            pos_label="pre_vs_published",
        )

        precision = precision_score(
            df.sort_values("distance").iloc[0:k].label.tolist(),
            ["pre_vs_published"] * k
            if k <= df.shape[0]
            else ["pre_vs_published"] * df.shape[0],
            pos_label="pre_vs_published",
        )

        data_rows.append(
            {"recall": recall, "precision": precision, "N": k, "model": model_label}
        )

plot_df = pd.DataFrame.from_records(data_rows)
plot_df.head()

g = (
    p9.ggplot(plot_df, p9.aes(x="N", y="recall", color="model"))
    + p9.geom_point()
    + p9.labs(x="Top N predictions", y="Recall")
)
g.save("output/figures/abstract_vs_full_text_top_k_recall.png", dpi=600)
print(g)

g = (
    p9.ggplot(plot_df, p9.aes(x="N", y="precision", color="model"))
    + p9.geom_point()
    + p9.labs(x="Top N predictions", y="Precision")
)
g.save("output/figures/abstract_vs_full_text_top_k_precision.png", dpi=600)
print(g)

# Take Home Points:
#
# 1. Abstract only document embeddings appear to have a small increase in performance compared to using full text alone.
# 2. My hunch is that abstracts haven't drastically changed compared to full text being changed.
