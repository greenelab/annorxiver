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

# # Find published articles missing from bioRxiv

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
biorxiv_journal_mapped_df.head()

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

# ### Euclidean Distance

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

# ### Cosine Proxy L2 Euclidean Distance

biorxiv_published_normalized = biorxiv_published.reset_index().set_index(
    ["pmcid", "document", "preprint_doi", "published_doi", "pmcoa"]
)
vector_norm = np.linalg.norm(biorxiv_published_normalized, axis=1)
biorxiv_published_normalized = biorxiv_published_normalized / np.tile(
    vector_norm[:, np.newaxis], 300
)
biorxiv_published_normalized = biorxiv_published_normalized.reset_index().set_index(
    "pmcid"
)

PMC_published_normalized = PMC_published.reset_index().set_index(
    ["document", "doi", "journal"]
)
vector_norm = np.linalg.norm(PMC_published_normalized, axis=1)
PMC_published_normalized = PMC_published_normalized / np.tile(
    vector_norm[:, np.newaxis], 300
)
PMC_published_normalized = PMC_published_normalized.reset_index().set_index("document")

article_distances = cdist(
    biorxiv_published_normalized.loc[PMC_published.index.tolist()].drop(
        ["document", "preprint_doi", "published_doi", "pmcoa"], axis=1
    ),
    PMC_published_normalized.drop(["doi", "journal"], axis=1),
    "euclidean",
)
article_distances.shape

articles_distance_cosine_proxy_df = (
    biorxiv_published.loc[PMC_published.index.tolist()]
    .reset_index()[["document", "pmcid"]]
    .assign(
        distance=np.diag(article_distances, k=0), journal=PMC_published.journal.tolist()
    )
)
articles_distance_cosine_proxy_df.head()

# ### Cosine Distance

article_distances = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()].drop(
        ["document", "preprint_doi", "published_doi", "pmcoa"], axis=1
    ),
    PMC_published.drop(["doi", "journal"], axis=1),
    "cosine",
)
article_distances.shape

articles_distance_cosine_df = (
    biorxiv_published.loc[PMC_published.index.tolist()]
    .reset_index()[["document", "pmcid"]]
    .assign(
        distance=np.diag(article_distances, k=0), journal=PMC_published.journal.tolist()
    )
)
articles_distance_cosine_df.head()

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

# ### Euclidean Distance

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

# ### Cosine Proxy Distance

PMC_off_published_normalized = PMC_off_published.set_index(["document", "journal"])
vector_norm = np.linalg.norm(PMC_off_published_normalized, axis=1)
PMC_off_published_normalized = PMC_off_published_normalized / np.tile(
    vector_norm[:, np.newaxis], 300
)
PMC_off_published_normalized = PMC_off_published_normalized.reset_index()

off_article_dist = cdist(
    biorxiv_published_normalized.loc[PMC_published.index.tolist()]
    .drop(["document", "preprint_doi", "published_doi", "pmcoa"], axis=1)
    .values,
    PMC_off_published_normalized.drop(["document", "journal"], axis=1).values,
    "euclidean",
)
off_article_dist.shape

data = []
for idx, row in tqdm.tqdm(articles_distance_cosine_proxy_df.iterrows()):
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

final_cosine_proxy_df = articles_distance_cosine_proxy_df.assign(
    label="pre_vs_published"
).append(pd.DataFrame.from_records(data).assign(label="pre_vs_random"))
final_cosine_proxy_df.head()

final_cosine_proxy_df = biorxiv_journal_df[["document", "preprint_doi"]].merge(
    final_cosine_proxy_df
)
final_cosine_proxy_df.to_csv(
    "output/annotated_links/article_distances_cosine_proxy.tsv", sep="\t", index=False
)
final_cosine_proxy_df.head()

# ### Cosine Distance

off_article_dist = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()]
    .drop(["document", "preprint_doi", "published_doi", "pmcoa"], axis=1)
    .values,
    PMC_off_published.drop(["document", "journal"], axis=1).values,
    "cosine",
)
off_article_dist.shape

data = []
for idx, row in tqdm.tqdm(articles_distance_cosine_df.iterrows()):
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

final_cosine_df = articles_distance_cosine_df.assign(label="pre_vs_published").append(
    pd.DataFrame.from_records(data).assign(label="pre_vs_random")
)
final_cosine_df.head()

final_cosine_df = biorxiv_journal_df[["document", "preprint_doi"]].merge(
    final_cosine_df
)
final_cosine_df.to_csv(
    "output/annotated_links/article_distances_cosine.tsv", sep="\t", index=False
)
final_cosine_df.head()

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
        final_cosine_proxy_df.replace(
            {
                "pre_vs_published": "preprint-published",
                "pre_vs_random": "preprint-random",
            }
        )
    )
    + p9.aes(x="label", y="distance")
    + p9.geom_violin(fill="#a6cee3")
    + p9.labs(x="Document Pair Groups", y="Euclidean (L2 Norm) Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=2)
    + p9.theme(figure_size=(11, 8.5))
)
print(g)

g = (
    p9.ggplot(
        final_cosine_df.replace(
            {
                "pre_vs_published": "preprint-published",
                "pre_vs_random": "preprint-random",
            }
        )
    )
    + p9.aes(x="label", y="distance")
    + p9.geom_violin(fill="#a6cee3")
    + p9.labs(x="Document Pair Groups", y="Cosine Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=2)
    + p9.theme(figure_size=(11, 8.5))
)
print(g)

# # Examine the top N predictions using Recall and Precision

data_rows = []
for df, distance_label in zip(
    [final_original_df, final_cosine_proxy_df, final_cosine_df],
    ["euclidean", "euclidean (L2)", "cosine"],
):
    for k in tqdm.tqdm(range(1, 34503, 200)):
        recall = recall_score(
            df.sort_values("distance").iloc[0:k].label.tolist(),
            ["pre_vs_published"] * k,
            pos_label="pre_vs_published",
        )

        precision = precision_score(
            df.sort_values("distance").iloc[0:k].label.tolist(),
            ["pre_vs_published"] * k,
            pos_label="pre_vs_published",
        )

        data_rows.append(
            {
                "recall": recall,
                "precision": precision,
                "N": k,
                "distance": distance_label,
            }
        )

plot_df = pd.DataFrame.from_records(data_rows)
plot_df.head()

g = (
    p9.ggplot(plot_df, p9.aes(x="N", y="recall", color="distance"))
    + p9.geom_point()
    + p9.labs(x="Top N predictions", y="Recall")
)
g.save("output/figures/distance_metrics_top_k_recall.png", dpi=600)
print(g)

g = (
    p9.ggplot(plot_df, p9.aes(x="N", y="precision", color="distance"))
    + p9.geom_point()
    + p9.labs(x="Top N predictions", y="Precision")
)
g.save("output/figures/distance_metrics_top_k_precision.png", dpi=600)
print(g)

# Take Home Points:
# 1. For this particular task the type of distance metric doesn't matter as performance remains the same.
# 2. Recall is the same regardless of the prediction label, while looking at precision we notice a change in performance.
# 3. As we incorporate more predictions precision suffers which makes sense given that the true negatives are going to be incorporated as well.
# 4. Main argument is distance metric doesn't matter in this case but cosine distance/euclidean normalized distance is superior in general tasks.
