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

from cairosvg import svg2png
from IPython.display import Image
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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import spacy
import tqdm

from annorxiver_modules.corpora_comparison_helper import (
    aggregate_word_counts,
    calculate_confidence_intervals,
    create_lemma_count_df,
    get_term_statistics,
    plot_bargraph,
    plot_point_bar_figure,
)

sys.path.append(str(Path("../../../preprint_similarity_search/server").resolve()))
from SAUCIE import SAUCIE, Loader  # noqa: E402

mpl.rcParams["figure.dpi"] = 250
# -

import rpy2.robjects as robjects  # noqa: E402
from rpy2.robjects import pandas2ri  # noqa: E402

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

preprint_vs_published = get_term_statistics(published_count_df, preprint_count_df, 100)
preprint_vs_published.to_csv(
    "output/updated_preprint_to_published_comparison.tsv", sep="\t", index=False
)
preprint_vs_published

full_plot_df = calculate_confidence_intervals(preprint_vs_published)
full_plot_df.head()

plot_df = (
    full_plot_df.sort_values("odds_ratio", ascending=False)
    .iloc[4:]
    .head(20)
    .append(full_plot_df.sort_values("odds_ratio", ascending=True).head(20))
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
    + p9.geom_segment(color="#253494", size=6, alpha=0.7)
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
    + p9.annotate("text", label="Preprint Enriched", x=-5, y=38.5, size=14, alpha=0.7)
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.8, font="Arial")
    + p9.theme(
        figure_size=(11, 8.5),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(y=None, x="Preprint vs Published log2(Odds Ratio)")
)
g.save("output/figures/preprint_published_frequency_odds.svg")
g.save("output/figures/preprint_published_frequency_odds.png", dpi=250)
print(g)

count_plot_df = create_lemma_count_df(plot_df, "published", "preprint").assign(
    repository=lambda x: pd.Categorical(
        x.repository.tolist(), categories=["preprint", "published"]
    )
)
count_plot_df.head()

g = plot_bargraph(count_plot_df, plot_df)
g.save("output/figures/preprint_published_frequency_bar.svg")
print(g)

# +
fig_output_path = "output/figures/polka_preprint_published_frequency.png"

fig = plot_point_bar_figure(
    "output/figures/preprint_published_frequency_odds.svg",
    "output/figures/preprint_published_frequency_bar.svg",
)

# save generated SVG files
svg2png(bytestring=fig.to_str(), write_to=fig_output_path, dpi=75)

Image(fig_output_path)
# -

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

# This section aims to see which principal components have a high association with Polka et al's subset. Furthermore, we also aim to see if we can use linear models to explain which PCs affect preprint prediction.

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
biorxiv_pca_sim_df.head()

# ## PC Regression

# ### Logistic Regression

# Goal here is to determine if we can figure out which PCs separate the bioRxiv subset from Polka et al.'s subset. Given that their dataset is only 60 papers we downsampled our dataset to contain only 60 papers.

dataset_df = biorxiv_pca_sim_df.sample(60, random_state=100).append(polka_pca_sim_df)
dataset_df.head()

model = LogisticRegressionCV(
    cv=10, Cs=100, max_iter=1000, penalty="l1", solver="liblinear"
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

fold_features = model.coefs_paths_["polka"].transpose(1, 0, 2)
model_performance_df = pd.DataFrame.from_dict(
    {
        "feat_num": ((fold_features.astype(bool).sum(axis=1)) > 0).sum(axis=1),
        "C": model.Cs_,
        "score": model.scores_["polka"].mean(axis=0),
    }
)
model_performance_df.head()

# +
fig, ax1 = plt.subplots()
ax1.set_xscale("log")
ax2 = plt.twinx()

ax1.plot(
    model_performance_df.C.tolist(),
    model_performance_df.feat_num.tolist(),
    label="Features",
    marker=".",
)
ax1.set_ylabel("# of Features")
ax1.set_xlabel("Inverse Regularization (C)")
ax1.legend(loc=0)

ax2.plot(
    model_performance_df.C.tolist(),
    model_performance_df.score.tolist(),
    label="Score",
    marker=".",
    color="green",
)
ax2.set_ylabel("Score (Accuracy %)")
ax2.legend(loc=4)
plt.savefig("output/preprint_classifier_results.png")
# -

plot_path = list(
    zip(
        model.Cs_,
        model.scores_["polka"].transpose(),
        model.coefs_paths_["polka"].transpose(1, 0, 2),
    )
)

data_records = []
for cs in plot_path[33:40]:
    model = LogisticRegression(C=cs[0], max_iter=1000, penalty="l1", solver="liblinear")
    model.fit(
        StandardScaler().fit_transform(dataset_df[[f"pc{idx+1}" for idx in range(50)]]),
        dataset_df["label"],
    )
    data_records.append(
        {
            "C": cs[0],
            "PCs": ",".join(map(str, model.coef_.nonzero()[1] + 1)),
            "feat_num": len(model.coef_.nonzero()[1]),
            "accuracy": cs[1].mean(),
        }
    )

model_coefs_df = pd.DataFrame.from_records(data_records)
model_coefs_df

# ### Decision Tree

# Given that Logistic regression doesn't return sparse weights, we may get better insight into this analysis by using a decision tree to determine which PCs are important in prediction.

model = DecisionTreeClassifier(random_state=100)
search_grid = GridSearchCV(
    model, {"criterion": ["gini", "entropy"], "max_features": ["auto", None]}, cv=10
)
search_grid.fit(dataset_df[[f"pc{idx+1}" for idx in range(50)]], dataset_df["label"])

print(search_grid.best_params_)
print(search_grid.best_score_)

export_graphviz(
    search_grid.best_estimator_,
    out_file="output/figures/pca_tree.dot",
    feature_names=[f"pc{idx+1}" for idx in range(50)],
    class_names=["bioRxiv", "polka et al."],
    rotate=True,
)

# ! dot -Tpng output/figures/pca_tree.dot -o output/figures/pca_tree.png

Image(filename="output/figures/pca_tree.png")

# ## Saucie Subset

# Where do the preprints in this subset lie along the SAUCIE map?

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

pandas2ri.activate()
robjects.globalenv["pmc_data_df"] = robjects.conversion.py2rpy(pmc_data_df)
robjects.globalenv["subset_df"] = robjects.conversion.py2rpy(subset_df)
robjects.r.source("saucie_plot.R")
Image(filename="output/figures/saucie_plot.png")
# Publication Time Analysis
# Get publication dates
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
    "../publication_delay_experiment/output/preprint_published_distances.tsv",
    sep="\t",
)
biorxiv_published_distances["time_to_published"] = pd.to_timedelta(
    biorxiv_published_distances["time_to_published"]
)
biorxiv_published_distances["days_to_published"] = biorxiv_published_distances[
    "time_to_published"
].dt.days
biorxiv_published_distances = biorxiv_published_distances.query("days_to_published > 0")
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

# +
# Get smoothed linear regression line
polka_x = polka_published_preprint_df.version_count.values.tolist()

polka_y = polka_published_preprint_df.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

xseq_2 = np.linspace(np.min(x), np.max(x), 80)

results_3 = linregress(polka_x, polka_y)
print(results_3)
# -

polka_x_line = np.array(
    [
        polka_published_preprint_df["version_count"].min(),
        polka_published_preprint_df["version_count"].max(),
    ]
)
polka_y_line = polka_x_line * results_3.slope + results_3.intercept

# +
# Graph here?
plt.figure(figsize=(11, 8.5))
plt.rcParams.update({"font.size": 22})
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
_ = g.plot(polka_x_line - 1, polka_y_line, "--k", color="red")
_ = g.scatter(
    polka_published_preprint_df["version_count"] - 1,
    polka_published_preprint_df["days_to_published"],
    c="red",
    s=12,
)
_ = g.annotate(f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}", (7, 1540))
_ = g.annotate(
    f"Y={results_3.slope:.2f}*X+{results_3.intercept:.2f}", (7, 1460), color="red"
)
_ = g.set_xlim(-0.5, 11.5)
_ = g.set_ylim(0, g.get_ylim()[1])

ax = plt.gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
ax.xaxis.label.set_size(20)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
ax.yaxis.label.set_size(20)

plt.savefig("output/figures/version_count_vs_publication_time_violin.svg", dpi=500)
plt.savefig("output/figures/version_count_vs_publication_time_violin.png", dpi=500)
# -

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

# +
polka_x = (polka_published_preprint_df["doc_distances"].values.tolist(),)
polka_y = polka_published_preprint_df.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

results_3 = linregress(polka_x, polka_y)
print(results_3)
# -

polka_x_line = np.array(
    [
        polka_published_preprint_df["doc_distances"].min(),
        polka_published_preprint_df["doc_distances"].max(),
    ]
)
polka_y_line = polka_x_line * results_3.slope + results_3.intercept

# +
# graph here?
plt.figure(figsize=(11, 8.5))
plt.rcParams.update({"font.size": 22})
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
plt.xlim([0, 12])
plt.ylim([0, 1800])
ax = plt.gca()

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
ax.xaxis.label.set_size(20)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
ax.yaxis.label.set_size(20)

ax.plot(x_line, y_line, "--k")
ax.plot(polka_x_line, polka_y_line, "--k", color="red")
ax.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (5.5, 1530),
)
_ = ax.annotate(
    f"Y={results_3.slope:.2f}*X+{results_3.intercept:.2f}", (5.5, 1450), color="red"
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
