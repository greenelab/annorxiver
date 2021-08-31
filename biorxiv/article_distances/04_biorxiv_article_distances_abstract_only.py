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

import numpy as np
import scipy.stats
import pandas as pd
import plotnine as p9
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV
import tqdm
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
    / "biorxiv_all_articles_300_abstract_only.tsv.xz",
    sep="\t",
)
biorxiv_embed_df = biorxiv_embed_df.dropna()
biorxiv_embed_df.head()

biorxiv_journal_mapped_df = biorxiv_journal_df[
    ["document", "published_doi", "pmcid", "pmcoa"]
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
    / Path("pmc_document_vectors_300_abstract_only.tsv.xz"),
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

article_distances = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()].drop(
        ["document", "published_doi", "pmcoa"], axis=1
    ),
    PMC_published.drop(["doi", "journal"], axis=1),
    "euclidean",
)
article_distances.shape

articles_distance_df = (
    biorxiv_published.loc[PMC_published.index.tolist()]
    .reset_index()[["document", "pmcid"]]
    .assign(
        distance=np.diag(article_distances, k=0), journal=PMC_published.journal.tolist()
    )
)
articles_distance_df.head()

# ## biorxiv -> random paper same journal

PMC_off_published = (
    pmc_journal_mapped_df.drop("doi", axis=1)
    .query(f"document not in {biorxiv_published.index.tolist()}")
    .query(f"journal in {articles_distance_df.journal.unique().tolist()}")
    .groupby("journal", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=100))
)
PMC_off_published.head()

journal_mapper = {
    journal: col for col, journal in enumerate(PMC_off_published.journal.tolist())
}
list(journal_mapper.items())[0:10]

off_article_dist = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()]
    .drop(["document", "published_doi", "pmcoa"], axis=1)
    .values,
    PMC_off_published.drop(["document", "journal"], axis=1).values,
    "euclidean",
)
off_article_dist.shape

data = []
for idx, row in tqdm.tqdm(articles_distance_df.iterrows()):
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

final_df = articles_distance_df.assign(label="pre_vs_published").append(
    pd.DataFrame.from_records(data).assign(label="pre_vs_random")
)
final_df.head()

final_df = biorxiv_journal_df[["document", "preprint_doi"]].merge(final_df)
final_df.to_csv(
    "output/annotated_links/article_distances_abstract_only.tsv", sep="\t", index=False
)
final_df.head()

# # Distribution plot

g = (
    p9.ggplot(
        final_df.replace(
            {
                "pre_vs_published": "preprint-published",
                "pre_vs_random": "preprint-random",
            }
        )
    )
    + p9.aes(x="label", y="distance")
    + p9.geom_violin(fill="#a6cee3")
    + p9.labs(x="Document Pair Groups", y="Euclidean Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.35)
)
g.save("output/figures/biorxiv_article_distance_abstract_only.svg", dpi=250)
g.save("output/figures/biorxiv_article_distance_abstract_only.png", dpi=250)
print(g)

# # Plot Abstract Only vs Full Text Only

abstract_only = final_df
full_text = pd.read_csv("output/annotated_links/article_distances.tsv", sep="\t")

plot_df = (
    full_text.query("label=='pre_vs_published'")
    .rename(index=str, columns={"distance": "full_text_distance"})[
        ["document", "full_text_distance"]
    ]
    .merge(
        abstract_only.query("label=='pre_vs_published'").rename(
            index=str, columns={"distance": "abstract_only_distance"}
        )[["document", "abstract_only_distance"]],
        on="document",
    )
    .assign(
        abstract_only_distance_log10=lambda x: -np.log10(x.abstract_only_distance),
        full_text_distance_log10=lambda x: -np.log10(x.full_text_distance),
    )
)
plot_df.head()

g = (
    p9.ggplot(plot_df)
    + p9.aes(x="full_text_distance_log10", y="abstract_only_distance_log10")
    + p9.geom_point(fill="#a6cee3")
    + p9.labs(x="Full Text Distance (-log 10)", y="Abstract Only Distance (-log 10)")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.35)
)
g.save("output/figures/biorxiv_full_text_v_abstract_only.svg", dpi=250)
g.save("output/figures/biorxiv_full_text_v_abstract_only.png", dpi=250)
print(g)

plot_df.sort_values("abstract_only_distance_log10", ascending=False).head(10)

# The negative values for abstract only consist of high confidence scores for a true match. The table for the first few entries have an abstract only distance of practically zero. This means the log version will be highly negative.
