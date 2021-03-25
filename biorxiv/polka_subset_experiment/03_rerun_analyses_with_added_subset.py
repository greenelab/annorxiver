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

# # Re-Run Analyses with Polka et. al. Subset

# +
from datetime import timedelta, date
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
from mizani.breaks import date_breaks
from mizani.formatters import timedelta_format
import numpy as np
import pandas as pd
import plotnine as p9
import requests
from scipy.spatial.distance import cdist
from scipy.stats import linregress
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import spacy
import tqdm

from annorxiver_modules.corpora_comparison_helper import (
    aggregate_word_counts,
    calculate_confidence_intervals,
    get_term_statistics,
)

sys.path.append(str(Path("../../../preprint_similarity_search/server").resolve()))
from SAUCIE import SAUCIE, Loader  # noqa: E402

mpl.rcParams["figure.dpi"] = 250

# +
# Set up porting from python to R
# and R to python :mindblown:

import rpy2.rinterface  # noqa: E402

# %load_ext rpy2.ipython
# -

# # Corpora Comparison Preprint-Published Update

# ## BioRxiv to Published Mapping

mapped_doi_df = (
    pd.read_csv("../journal_tracker/output/mapped_published_doi.tsv", sep="\t")
    .query("published_doi.notnull()")
    .query("pmcid.notnull()")
    .groupby("preprint_doi")
    .agg(
        {
            "author_type": "first",
            "heading": "first",
            "category": "first",
            "document": "first",
            "preprint_doi": "last",
            "published_doi": "last",
            "pmcid": "last",
        }
    )
    .reset_index(drop=True)
)
mapped_doi_df.head()

polka_et_al_mapped_df = pd.read_csv(
    "output/polka_et_al_pmc_mapped_subset.tsv", sep="\t"
)
polka_et_al_mapped_df.head()

spacy_nlp = spacy.load("en_core_web_sm")
stop_word_list = list(spacy_nlp.Defaults.stop_words)

# ## BioRxiv Token Counts

polka_preprints = list(Path("output/biorxiv_word_counts").rglob("*tsv"))

# +
preprint_count = aggregate_word_counts(polka_preprints)

preprint_count_df = (
    pd.DataFrame.from_records(
        [
            {
                "lemma": token[0],
                "pos_tag": token[1],
                "dep_tag": token[2],
                "count": preprint_count[token],
            }
            for token in preprint_count
        ]
    )
    .query(f"lemma not in {stop_word_list}")
    .groupby("lemma")
    .agg({"count": "sum"})
    .reset_index()
    .sort_values("count", ascending=False)
)

preprint_count_df.head()
# -

# ## PMCOA Token Counts

polka_published = list(Path("output/pmcoa_word_counts").rglob("*tsv"))

# +
published_count = aggregate_word_counts(polka_published)

published_count_df = (
    pd.DataFrame.from_records(
        [
            {
                "lemma": token[0],
                "pos_tag": token[1],
                "dep_tag": token[2],
                "count": published_count[token],
            }
            for token in published_count
        ]
    )
    .query(f"lemma not in {stop_word_list}")
    .groupby("lemma")
    .agg({"count": "sum"})
    .reset_index()
    .sort_values("count", ascending=False)
)

published_count_df.head()
# -

# ## Get Token Stats

preprint_vs_published = get_term_statistics(preprint_count_df, published_count_df, 100)
preprint_vs_published.to_csv(
    "output/updated_preprint_to_published_comparison.tsv", sep="\t", index=False
)
preprint_vs_published

full_plot_df = calculate_confidence_intervals(preprint_vs_published)
full_plot_df.head()

plot_df = (
    full_plot_df.sort_values("odds_ratio", ascending=True)
    .iloc[3:]
    .head(20)
    .append(full_plot_df.sort_values("odds_ratio", ascending=False).head(20))
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x)),
    )
)
plot_df.head()

g = (
    p9.ggplot(
        plot_df, p9.aes(y="lemma", x="lower_odds", xend="upper_odds", yend="lemma")
    )
    + p9.geom_segment(color="#253494", size=3.5, alpha=0.7)
    + p9.scale_y_discrete(
        limits=(plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    )
    + p9.scale_x_continuous(limits=(-11, 11))
    + p9.geom_vline(p9.aes(xintercept=0), linetype="--", color="grey")
    + p9.annotate(
        "segment",
        x=2,
        xend=8,
        y=1.5,
        yend=1.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="Published Enriched", x=5, y=2.5, size=12, alpha=0.7)
    + p9.annotate(
        "segment",
        x=-2,
        xend=-8,
        y=39.5,
        yend=39.5,
        colour="black",
        size=0.5,
        alpha=1,
        arrow=p9.arrow(length=0.1),
    )
    + p9.annotate("text", label="Preprint Enriched", x=-5, y=38.5, size=12, alpha=0.7)
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.1, font="Arial")
    + p9.theme(
        figure_size=(10, 6),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(y=None, x="Preprint vs Published log2(Odds Ratio)")
)
g.save("output/figures/preprint_published_frequency_odds.svg")
g.save("output/figures/preprint_published_frequency_odds.png", dpi=250)
print(g)

# # Document Embeddings

# ## Load the Documents

biorxiv_documents_df = pd.read_csv(
    "../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz",
    sep="\t",
)
biorxiv_documents_df.head()

polka_preprints_df = pd.read_csv("output/polka_et_al_biorxiv_embeddings.tsv", sep="\t")
polka_preprints_df.head()

pca_components = pd.read_csv(
    Path("../pca_association_experiment/output/word_pca_similarity/pca_components.tsv"),
    sep="\t",
)
pca_components.head()

# ## PCA Components

document_pca_sim = 1 - cdist(
    polka_preprints_df.drop("document", axis=1).values, pca_components.values, "cosine"
)
print(document_pca_sim.shape)
document_pca_sim

document_to_pca_map = {
    document: document_pca_sim[idx, :]
    for idx, document in enumerate(polka_preprints_df.document.tolist())
}

polka_pca_sim_df = (
    pd.DataFrame.from_dict(document_to_pca_map, orient="index")
    .rename(index=str, columns={col: f"pc{col+1}" for col in range(int(300))})
    .reset_index()
    .rename(index=str, columns={"index": "document"})
)
polka_pca_sim_df.to_csv("output/polka_pca_enrichment.tsv", sep="\t")
polka_pca_sim_df = polka_pca_sim_df.assign(label="polka")
polka_pca_sim_df.head()

document_pca_sim = 1 - cdist(
    biorxiv_documents_df.drop("document", axis=1).values,
    pca_components.values,
    "cosine",
)
print(document_pca_sim.shape)
document_pca_sim

document_to_pca_map = {
    document: document_pca_sim[idx, :]
    for idx, document in enumerate(biorxiv_documents_df.document.tolist())
}

biorxiv_pca_sim_df = (
    pd.DataFrame.from_dict(document_to_pca_map, orient="index")
    .rename(index=str, columns={col: f"pc{col+1}" for col in range(int(300))})
    .reset_index()
    .rename(index=str, columns={"index": "document"})
    .assign(label="biorxiv")
)
# biorxiv_pca_sim_df.to_csv("output/polka_pca_enrichment.tsv", sep="\t")
biorxiv_pca_sim_df.head()

# ## PC Regression

dataset_df = biorxiv_pca_sim_df.sample(60, random_state=100).append(polka_pca_sim_df)
dataset_df.head()

model = LogisticRegressionCV(
    cv=10, Cs=20, max_iter=1000, penalty="l1", solver="liblinear"
)
model.fit(
    StandardScaler().fit_transform(dataset_df[[f"pc{idx+1}" for idx in range(50)]]),
    dataset_df["label"],
)

best_result = list(filter(lambda x: x[1] == model.C_, enumerate(model.Cs_)))[0]
print(best_result)

print("Best CV Fold")
print(model.scores_["polka"][:, best_result[0]])
model.scores_["polka"][:, best_result[0]].mean()

model_weights_df = pd.DataFrame.from_dict(
    {
        "weight": model.coef_[0],
        "pc": list(range(1, 51)),
    }
)
model_weights_df["pc"] = pd.Categorical(model_weights_df["pc"])
model_weights_df.head()

g = (
    p9.ggplot(model_weights_df, p9.aes(x="pc", y="weight"))
    + p9.geom_col(position=p9.position_dodge(width=5), fill="#253494")
    + p9.coord_flip()
    + p9.scale_x_discrete(limits=list(sorted(range(1, 51), reverse=True)))
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.1, font="Arial")
    + p9.theme(figure_size=(10, 8))
    + p9.labs(
        title="Regression Model Weights", x="Princpial Component", y="Model Weight"
    )
)
g.save("output/figures/pca_log_regression_weights.svg")
g.save("output/figures/pca_log_regression_weights.png", dpi=250)
print(g)

# ## Saucie Subset

saucie_model = SAUCIE(
    300,
    restore_folder=str(Path("../../pmc/journal_recommendation/output/model").resolve()),
)

coordinates = saucie_model.get_embedding(
    Loader(polka_preprints_df.drop("document", axis=1).values)
)

subset_df = pd.DataFrame(coordinates, columns=["dim1", "dim2"])
subset_df.head()

pmc_data_df = pd.read_csv(
    Path("../../pmc/journal_recommendation/output")
    / Path("paper_dataset/paper_dataset_full_tsne.tsv"),
    sep="\t",
)
pmc_data_df.head()

# + magic_args="-i pmc_data_df -i subset_df" language="R"
#
# library(ggplot2)
#
# bin_num <- 50
# g <- (
#     ggplot(pmc_data_df, aes(x=dim1, y=dim2))
# + geom_bin2d(bins=bin_num, binwidth=0.85)

# + theme(legend.position="left")

# + geom_point(data=subset_df, aes(x=dim1, y=dim2, colour = "red"))
# )
# print(g)
# -

# ## Publication Time Analysis

# ### Get publication dates

url = "https://api.biorxiv.org/pub/2019-11-01/3000-01-01/"

# +
# Get preprint publication dates for 2019 -> 2020
already_downloaded = Path("output/biorxiv_published_dates_post_2019.tsv").exists()
if not already_downloaded:
    collection = []
    page_size = 100
    total = 23948
    for i in tqdm.tqdm(range(0, total, page_size), total=total / page_size):
        collection += requests.get(url + str(i)).json()["collection"]
    published_dates = pd.DataFrame(collection)
    published_dates.to_csv(
        "output/biorxiv_published_dates_post_2019.tsv", sep="\t", index=False
    )
else:
    published_dates = pd.read_csv(
        "output/biorxiv_published_dates_post_2019.tsv", sep="\t"
    )

published_dates = published_dates.assign(
    preprint_date=lambda x: pd.to_datetime(x.preprint_date.tolist()),
    published_date=lambda x: pd.to_datetime(
        x.published_date.apply(lambda y: y[0 : y.index(":")] if ":" in y else y)
    ),
).assign(time_to_published=lambda x: x.published_date - x.preprint_date)
print(published_dates.shape)
published_dates.head()
# -

polka_preprints_df = polka_preprints_df.assign(
    biorxiv_base=lambda x: x.document.apply(lambda y: y.split("_")[0]),
    version_count=lambda x: x.document.apply(
        lambda y: int(y[y.index("v") + 1 :].split(".")[0])
    ),
)
polka_preprints_df.head()

polka_published_df = pd.read_csv("output/polka_et_al_pmcoa_embeddings.tsv", sep="\t")
polka_published_df.head()

polka_published_preprint_df = (
    polka_et_al_mapped_df.drop(
        ["Version", "MID", "IsCurrent", "IsLive", "ReleaseDate", "Msg"], axis=1
    )
    .assign(biorxiv_base=lambda x: x.biorxiv_doi.apply(lambda y: y.split("/")[1]))
    .merge(
        polka_preprints_df.drop([f"feat_{idx}" for idx in range(300)], axis=1),
        on="biorxiv_base",
    )
    .drop("document", axis=1)
    .merge(
        published_dates.drop(["published_citation_count", "preprint_title"], axis=1),
        on=["biorxiv_doi", "published_doi"],
    )
    .query(f"PMCID in {polka_published_df['document'].tolist()}")
)
polka_published_preprint_df.head()

# +
for col in ["preprint_date", "published_date"]:
    polka_published_preprint_df[col] = pd.to_datetime(polka_published_preprint_df[col])

polka_published_preprint_df["time_to_published"] = pd.to_timedelta(
    polka_published_preprint_df["time_to_published"]
)
polka_published_preprint_df["days_to_published"] = polka_published_preprint_df[
    "time_to_published"
].dt.days
print(polka_published_preprint_df.shape)
polka_published_preprint_df.head()
# -

# ### Document version count plot

biorxiv_published_distances = pd.read_csv(
    "../publication_delay_experiment/output/preprint_published_distances.tsv", sep="\t"
)
biorxiv_published_distances["time_to_published"] = pd.to_timedelta(
    biorxiv_published_distances["time_to_published"]
)
biorxiv_published_distances["days_to_published"] = biorxiv_published_distances[
    "time_to_published"
].dt.days
biorxiv_published_distances.head()

# +
# Get smoothed linear regression line
x = biorxiv_published_distances.version_count.values.tolist()

y = biorxiv_published_distances.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

xseq_2 = np.linspace(np.min(x), np.max(x), 80)

results_2 = linregress(x, y)
print(results_2)
# -

x_line = np.array(
    [
        biorxiv_published_distances["version_count"].min(),
        biorxiv_published_distances["version_count"].max(),
    ]
)
y_line = x_line * results_2.slope + results_2.intercept

# Graph here?
plt.figure(figsize=(8, 5))
g = sns.violinplot(
    x="version_count",
    y="days_to_published",
    data=biorxiv_published_distances,
    cut=0,
    scale="width",
    palette="YlGnBu",
)
_ = g.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
_ = g.set_xlabel("# of Preprint Versions")
_ = g.plot(x_line - 1, y_line, "--k")
_ = g.scatter(
    polka_published_preprint_df["version_count"] - 1,
    polka_published_preprint_df["days_to_published"],
    c="red",
    s=12,
)
_ = g.annotate(f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}", (7, 1470))
_ = g.set_xlim(-0.5, 11.5)
_ = g.set_ylim(0, g.get_ylim()[1])
plt.savefig("output/figures/version_count_vs_publication_time_violin.svg", dpi=500)
plt.savefig("output/figures/version_count_vs_publication_time_violin.png", dpi=500)

# ### Document embedding pair

polka_published_df = polka_published_df.set_index("document")
polka_published_df.head()

polka_preprints_df = polka_preprints_df.set_index("biorxiv_base").drop(
    ["document", "version_count"], axis=1
)
polka_preprints_df.head()

dist = np.diag(
    cdist(
        polka_preprints_df.loc[polka_published_preprint_df["biorxiv_base"]],
        polka_published_df.loc[polka_published_preprint_df["PMCID"]],
    )
)
print(dist)

polka_published_preprint_df = polka_published_preprint_df.assign(doc_distances=dist)
polka_published_preprint_df.head()

# +
# Get smoothed linear regression line
x = biorxiv_published_distances.doc_distances.values.tolist()

y = biorxiv_published_distances.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

xseq_2 = np.linspace(np.min(x), np.max(x), 80)

results_2 = linregress(x, y)
print(results_2)
# -

x_line = np.array(
    [
        biorxiv_published_distances["doc_distances"].min(),
        biorxiv_published_distances["doc_distances"].max(),
    ]
)
y_line = x_line * results_2.slope + results_2.intercept

# graph here?
plt.figure(figsize=(6, 5))
ax = plt.hexbin(
    biorxiv_published_distances["doc_distances"],
    biorxiv_published_distances["days_to_published"],
    gridsize=50,
    cmap="YlGnBu_r",
    norm=mpl.colors.LogNorm(),
    mincnt=1,
    linewidths=(0.15,)
    #     edgecolors=None
)
ax = plt.gca()
ax.plot(x_line, y_line, "--k")
ax.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (8, 1490),
)
_ = ax.set_xlabel("Euclidian Distance of Preprint-Published Versions")
_ = ax.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
cbar = plt.colorbar()
_ = cbar.ax.set_ylabel("count", rotation=270)
_ = ax.scatter(
    polka_published_preprint_df["doc_distances"],
    polka_published_preprint_df["days_to_published"],
    c="red",
    s=6,
)
plt.savefig("output/figures/article_distance_vs_publication_time_hex.svg", dpi=250)
plt.savefig("output/figures/article_distance_vs_publication_time_hex.png", dpi=250)
