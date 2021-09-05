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
    # .assign(
    #    abstract_only_distance_log10=lambda x: -np.log10(x.abstract_only_distance),
    #    full_text_distance_log10=lambda x: -np.log10(x.full_text_distance),
    # )
)
plot_df.head()

# Pearson's R for correlation
# Shows a weak but positive correlation
scipy.stats.pearsonr(plot_df.full_text_distance, plot_df.abstract_only_distance)

g = (
    p9.ggplot(plot_df)
    + p9.aes(x="full_text_distance", y="abstract_only_distance")
    + p9.geom_point(fill="#a6cee3")
    + p9.scale_y_continuous(trans="log10")
    + p9.labs(x="Full Text Distance", y="Abstract Only Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.35)
)
g.save("output/figures/biorxiv_full_text_v_abstract_only.svg", dpi=250)
g.save("output/figures/biorxiv_full_text_v_abstract_only.png", dpi=250)
print(g)

# Remove outliers for shape of distribution
g = (
    p9.ggplot(plot_df.query("abstract_only_distance>1e-3"))
    + p9.aes(x="full_text_distance", y="abstract_only_distance")
    + p9.geom_point(fill="#a6cee3")
    + p9.scale_y_continuous(trans="log10")
    + p9.labs(x="Full Text Distance", y="Abstract Only Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.35)
)
print(g)

# +
sns.set_theme(
    context="paper", style="white", rc={"figure.figsize": (11, 8.5), "font.size": 22}
)

g = sns.jointplot(
    x=plot_df.full_text_distance,
    y=plot_df.abstract_only_distance,
    kind="hist",
    height=8.5,
)

g.set_axis_labels("Full Text Distance", "Abstract Only Distance", fontsize=22)
g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), size=22)
g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), size=22)
plt.tight_layout()

plt.savefig("output/abstract_full_text_histogram_plot.svg")
plt.savefig("output/abstract_full_text_histogram_plot.png", dpi=500)
# -

plot_df.sort_values("abstract_only_distance").head(10)

plot_df.sort_values("abstract_only_distance", ascending=False).head(10)

# Take Home Points:
# 1. Abstract only distances are greater than full text as I suspect the vectors generated are susceptible to minor changes compared to full text.
# 2. Both the abstract only and full text distributions have majority of their distances centered around 0-5
# 3. Since majority of both distributions are around that way, I'd argue that using abstracts alone could suffice in matching preprints with their published counter parts. By using only abstracts we can detect documents that are published closed access instead of relying full text to be available.
# 4. The pairs with distances close to zero (abstract only) are practically the same abstract. There might be minor word or phrase changes, but those changes haven't affected the vector much.
# 5. The points with the highest distance either have a structural change or significant phrase changes.
#
# Feel free to manually check these via [diffchecker](https://www.diffchecker.com) the preprint abstract and its published version.
