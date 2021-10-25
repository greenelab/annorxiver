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

# # Polka et al Analysis Reviewer Request

# A reviewer requested that the regression analysis for document distances be filtered to only include documents that were published within the same time frame as preprints mentioned in polka et al subset. Rather than edit the old notebook I'm generating a new notebook to only include that updated part of the analysis.

# +
from datetime import timedelta, date
from pathlib import Path
import sys

from cairosvg import svg2png
from gensim.models import Word2Vec
from IPython.display import Image, display, SVG
from lxml import etree
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotnine as p9
from scipy.spatial.distance import cdist
from scipy.stats import linregress
import seaborn as sns
from svgutils.compose import Unit
import svgutils.transform as sg
import tqdm

from annorxiver_modules.document_helper import generate_doc_vector

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.size"] = 12
mpl.rcParams["font.family"] = "Arial"
# -

# # Load the document vectors

biorxiv_documents_df = pd.read_csv(
    "../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz",
    sep="\t",
)
biorxiv_documents_df.head()

polka_preprints_df = pd.read_csv("output/polka_et_al_biorxiv_embeddings.tsv", sep="\t")
polka_preprints_df.head()

polka_preprints_df = polka_preprints_df.assign(
    biorxiv_base=lambda x: x.document.apply(lambda y: y.split("_")[0]),
    version_count=lambda x: x.document.apply(
        lambda y: int(y[y.index("v") + 1 :].split(".")[0])
    ),
)
polka_preprints_df.head()

polka_published_df = pd.read_csv("output/polka_et_al_pmcoa_embeddings.tsv", sep="\t")
polka_published_df.head()

# # Grab the Published Dates

# +
published_dates = pd.read_csv("output/biorxiv_published_dates_post_2019.tsv", sep="\t")

published_dates = published_dates.assign(
    preprint_date=lambda x: pd.to_datetime(x.preprint_date.tolist()),
    published_date=lambda x: pd.to_datetime(
        x.published_date.apply(lambda y: y[0 : y.index(":")] if ":" in y else y)
    ),
).assign(time_to_published=lambda x: x.published_date - x.preprint_date)
print(published_dates.shape)
published_dates.head()
# -

polka_et_al_mapped_df = pd.read_csv(
    "output/polka_et_al_pmc_mapped_subset.tsv", sep="\t"
)
polka_et_al_mapped_df.head()

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

# # Perform Regression Analysis - Version Count

# ## Original

# +
# Get smoothed linear regression line
x = biorxiv_published_distances.version_count.values.tolist()

y = biorxiv_published_distances.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

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
plt.figure(figsize=(11, 8))
plt.rcParams.update({"font.size": 12})
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
test_obj = g.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (9, 1480),
    label="background",
)
test2_obj = g.annotate(
    f"Y={results_3.slope:.2f}*X+{results_3.intercept:.2f}",
    (9, 1400),
    color="red",
    label="test",
)
_ = g.set_xlim(-0.5, 11.5)
_ = g.set_ylim(0, g.get_ylim()[1])

ax = plt.gca()
# for tick in ax.xaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.xaxis.label.set_size(20)

# for tick in ax.yaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.yaxis.label.set_size(20)

ax.legend(
    handles=[
        Patch(color="black", label="bioRxiv"),
        Patch(color="r", label="Polka et al."),
    ],
    fontsize="xx-small",
)
plt.savefig("output/figures/version_count_vs_publication_time_violin.svg")
plt.savefig("output/figures/version_count_vs_publication_time_violin.png", dpi=600)
# -

# ## Background Filtered

# Grab the max amount of days possible
polka_published_preprint_df.time_to_published.max()

# +
# Get smoothed linear regression line
filtered_biorxiv_published_distances = biorxiv_published_distances[
    biorxiv_published_distances["time_to_published"] < timedelta(days=195)
]
x = filtered_biorxiv_published_distances.version_count.values.tolist()

y = filtered_biorxiv_published_distances.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

results_2 = linregress(x, y)
print(results_2)
# -

x_line = np.array(
    [
        filtered_biorxiv_published_distances["version_count"].min(),
        filtered_biorxiv_published_distances["version_count"].max(),
    ]
)
y_line = x_line * results_2.slope + results_2.intercept

# +
# Get smoothed linear regression line
polka_x = polka_published_preprint_df.version_count.values.tolist()

polka_y = polka_published_preprint_df.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

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
plt.figure(figsize=(11, 8))
plt.rcParams.update({"font.size": 12})
g = sns.violinplot(
    x="version_count",
    y="days_to_published",
    data=filtered_biorxiv_published_distances,
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
_ = g.annotate(f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}", (7, 50))
_ = g.annotate(
    f"Y={results_3.slope:.2f}*X+{results_3.intercept:.2f}", (7, 40), color="red"
)
_ = g.set_xlim(-0.5, 9)
_ = g.set_ylim(0, g.get_ylim()[1])

ax = plt.gca()
# for tick in ax.xaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.xaxis.label.set_size(20)

# for tick in ax.yaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.yaxis.label.set_size(20)

ax.legend(
    handles=[
        Patch(color="black", label="bioRxiv"),
        Patch(color="r", label="Polka et al."),
    ],
    fontsize="xx-small",
    loc="lower right",
)

plt.savefig("output/figures/version_count_vs_publication_time_violin_filtered.svg")
plt.savefig(
    "output/figures/version_count_vs_publication_time_violin_filtered.png", dpi=600
)
# -

# # Perform Regression Analysis - Document Distances

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

# ## Original

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
plt.figure(figsize=(11, 8))
plt.rcParams.update({"font.size": 12})
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

# for tick in ax.xaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.xaxis.label.set_size(20)

# for tick in ax.yaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.yaxis.label.set_size(20)

ax.plot(x_line, y_line, "--k")
ax.plot(polka_x_line, polka_y_line, "--k", color="red")
ax.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (9, 1530),
)
_ = ax.annotate(
    f"Y={results_3.slope:.2f}*X+{results_3.intercept:.2f}", (9, 1440), color="red"
)
_ = ax.set_xlabel("Euclidean Distance of Preprint-Published Versions")
_ = ax.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
cbar = plt.colorbar()
_ = cbar.ax.set_ylabel("count", rotation=270)
_ = ax.scatter(
    polka_published_preprint_df["doc_distances"],
    polka_published_preprint_df["days_to_published"],
    c="red",
    s=6,
)
ax.legend(
    handles=[
        Patch(color="black", label="bioRxiv"),
        Patch(color="r", label="Polka et al."),
    ],
    fontsize="xx-small",
)
plt.savefig("output/figures/article_distance_vs_publication_time_hex.svg")
plt.savefig("output/figures/article_distance_vs_publication_time_hex.png", dpi=600)
# -

# ## Background Filtered

# Grab the max amount of days possible
polka_published_preprint_df.time_to_published.max()

# +
# Get smoothed linear regression line
filtered_biorxiv_published_distances = biorxiv_published_distances[
    biorxiv_published_distances["time_to_published"] < timedelta(days=195)
]
x = filtered_biorxiv_published_distances.doc_distances.values.tolist()

y = filtered_biorxiv_published_distances.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

results_2 = linregress(x, y)
print(results_2)
# -

x_line = np.array(
    [
        filtered_biorxiv_published_distances["doc_distances"].min(),
        filtered_biorxiv_published_distances["doc_distances"].max(),
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
plt.figure(figsize=(11, 8))
plt.rcParams.update({"font.size": 12})
ax = plt.hexbin(
    filtered_biorxiv_published_distances["doc_distances"],
    filtered_biorxiv_published_distances["days_to_published"],
    gridsize=40,
    cmap="YlGnBu_r",
    norm=mpl.colors.LogNorm(),
    mincnt=1,
    linewidths=(0.10,)
    #     edgecolors=None
)
plt.xlim([0, 15])
plt.ylim([0, 200])
ax = plt.gca()

# for tick in ax.xaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.xaxis.label.set_size(20)

# for tick in ax.yaxis.get_major_ticks():
#    tick.label.set_fontsize(20)
# ax.yaxis.label.set_size(20)

ax.plot(x_line, y_line, "--k")
ax.plot(polka_x_line, polka_y_line, "--k", color="red")
ax.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (11, 50),
)
_ = ax.annotate(
    f"Y={results_3.slope:.2f}*X+{results_3.intercept:.2f}", (11, 40), color="red"
)
_ = ax.set_xlabel("Euclidean Distance of Preprint-Published Versions")
_ = ax.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
cbar = plt.colorbar()
_ = cbar.ax.set_ylabel("count", rotation=270, labelpad=25)
_ = ax.scatter(
    polka_published_preprint_df["doc_distances"],
    polka_published_preprint_df["days_to_published"],
    c="red",
    s=6,
)

ax.legend(
    handles=[
        Patch(color="black", label="bioRxiv"),
        Patch(color="r", label="Polka et al."),
    ],
    fontsize="xx-small",
    loc="lower right",
)

plt.savefig("output/figures/article_distance_vs_publication_time_hex_filtered.svg")
plt.savefig(
    "output/figures/article_distance_vs_publication_time_hex_filtered.png", dpi=600
)
# -

# # Figure panel generation

# +
fig1 = sg.fromfile(
    "output/figures/version_count_vs_publication_time_violin_filtered.svg"
)
fig2 = sg.fromfile(
    "output/figures/article_distance_vs_publication_time_hex_filtered.svg"
)

fig1_width_size = np.round(float(fig1.root.attrib["width"][:-2]) * 1.33, 0)
fig1_height_size = np.round(float(fig1.root.attrib["height"][:-2]) * 1.33, 0)

fig2_width_size = np.round(float(fig2.root.attrib["width"][:-2]) * 1.33, 0)
fig2_height_size = np.round(float(fig2.root.attrib["height"][:-2]) * 1.33, 0)

fig = sg.SVGFigure(
    Unit((fig1_width_size + fig2_width_size) - 360),
    Unit(min(fig1_height_size, fig2_height_size) - 50),
)

fig.append(
    [etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"})]
)

plot1 = fig1.getroot()
plot1.moveto(10, 30)

plot2 = fig2.getroot()
plot2.moveto(fig1_width_size - 160, 12)

text_A = sg.TextElement(10, 30, "A", size=22, weight="bold")
text_B = sg.TextElement(fig1_width_size - 160, 30, "B", size=22, weight="bold")

fig.append([plot1, plot2, text_A, text_B])
# -

# save generated SVG files
fig.save("output/figures/polka_filtered_background_panels.svg")
svg2png(
    bytestring=fig.to_str(),
    write_to="output/figures/polka_filtered_background_panels.png",
    dpi=600,
)
display(SVG(fig.to_str()))
