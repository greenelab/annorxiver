# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Measure the Difference between Preprint-Published similarity and Published Articles

# This notebook measures the time delay that results from the peer review process. Two plots are generated: one that depict the average publication time delay as changes are demanded from the peer review process and the other that depicts the added time delay as preprints have to undergo multiple versions to be published.

# +
from datetime import timedelta
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import plotnine as p9
import requests
from scipy.spatial.distance import cdist
from scipy.stats import linregress
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
import tqdm

from mizani.breaks import date_breaks
from mizani.formatters import timedelta_format

import warnings

mpl.rcParams["figure.dpi"] = 250
warnings.filterwarnings("ignore")
# -

# # Load the Document Distances

published_date_distances = pd.read_csv(
    "output/preprint_published_distances_rerun.tsv", sep="\t"
)
for col in ["preprint_date", "published_date"]:
    published_date_distances[col] = pd.to_datetime(published_date_distances[col])
published_date_distances["time_to_published"] = pd.to_timedelta(
    published_date_distances["time_to_published"]
)
print(published_date_distances.shape)
published_date_distances.head()

published_date_distances["days_to_published"] = published_date_distances[
    "time_to_published"
].dt.days

remove_negative_time_to_published = True
if remove_negative_time_to_published:
    published_date_distances = published_date_distances[
        published_date_distances["days_to_published"] >= 0
    ]

# # Construct Scatter Plot of Date vs Version Count

# Preprints are delayed on an average of 51 days for each new version posted onto bioRxiv. This section regresses preprint's version counts against the time it takes to have a preprint published. A scatter and square bin plot are generated below.

# +
# Get smoothed linear regression line
x = published_date_distances.version_count.values.tolist()

y = published_date_distances.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

xseq_2 = np.linspace(np.min(x), np.max(x), 80)

results_2 = linregress(x, y)
print(results_2)
# -

x_line = np.array(
    [
        published_date_distances["version_count"].min(),
        published_date_distances["version_count"].max(),
    ]
)
y_line = x_line * results_2.slope + results_2.intercept

g = (
    p9.ggplot(
        published_date_distances,
        p9.aes(x="factor(version_count)", y="time_to_published"),
    )
    + p9.geom_boxplot(fill="#a6cee3")
    + p9.geom_line(
        mapping=p9.aes(x="version_count", y="time_to_published"),
        stat="smooth",
        method="lm",
        linetype="dashed",
        se=False,
        alpha=1,
        size=0.7,
        inherit_aes=False,
    )
    + p9.scale_y_timedelta(labels=timedelta_format("d"))
    + p9.annotate(
        "text",
        x=9,
        y=timedelta(days=1470),
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    )
    + p9.labs(x="# of Preprint Versions", y="Time Elapsed Until Preprint is Published")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.3)
)
# g.save("output/version_count_vs_publication_time.svg", dpi=500)
# g.save("output/version_count_vs_publication_time.png", dpi=500)
print(g)

plt.figure(figsize=(8, 5))
g = sns.boxenplot(
    x="version_count",
    y="days_to_published",
    data=published_date_distances,
    scale="linear",
    palette="YlGnBu",
)
_ = g.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
_ = g.set_xlabel("# of Preprint Versions")
_ = g.plot(x_line - 1, y_line, "--k")
_ = g.annotate(f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}", (7, 1470))

plt.figure(figsize=(8, 5))
g = sns.violinplot(
    x="version_count",
    y="days_to_published",
    data=published_date_distances,
    cut=0,
    scale="width",
    palette="YlGnBu",
)
_ = g.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
_ = g.set_xlabel("# of Preprint Versions")
_ = g.plot(x_line - 1, y_line, "--k")
_ = g.annotate(f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}", (7, 1470))
_ = g.set_xlim(-0.5, 11.5)
_ = g.set_ylim(0, g.get_ylim()[1])
plt.savefig("output/version_count_vs_publication_time_violin_rerun.svg", dpi=500)
plt.savefig("output/version_count_vs_publication_time_violin_rerun.png", dpi=500)

# +
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

bw_adjust = 0.5

g = sns.FacetGrid(
    published_date_distances,
    col="version_count",
    hue="version_count",
    aspect=0.1,
    height=5,
    palette="YlGnBu",
)

# Draw the densities in a few steps
g = g.map(
    sns.kdeplot,
    "days_to_published",
    vertical=True,
    bw_adjust=bw_adjust,
    clip_on=False,
    fill=True,
    alpha=1,
    linewidth=1.5,
)

g = g.map(
    sns.kdeplot,
    "days_to_published",
    clip_on=False,
    color="w",
    lw=2,
    bw_adjust=bw_adjust,
    vertical=True,
)
g = g.map(plt.axvline, x=0, lw=2, clip_on=False)

# reorder so right side is lower than left
for i, ax in enumerate(g.axes[0]):
    ax.set_zorder(ax.zorder - i)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(
        0.03,
        0,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


_ = g.map(label, "days_to_published")

_ = g.set_ylabels("Time Elapsed Until Preprint is Published (Days)")
_ = g.set_xlabels("")

xyA = (0, y_line[0])
xyB = (0, y_line[1])
axA = g.axes[0][0]
axB = g.axes[0][-1]
line = ConnectionPatch(
    xyA=xyA,
    coordsA=axA.transData,
    xyB=xyB,
    coordsB=axB.transData,
    linestyle="--",
    color="k",
)
_ = g.fig.add_artist(line)

# Set the subplots to overlap
_ = g.fig.subplots_adjust(wspace=-0.7)

# Remove axes details that don't play well with overlap
_ = g.set_titles("")
_ = g.set(xticks=[])
_ = g.despine(bottom=True, left=True)

# +
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

g = sns.FacetGrid(
    published_date_distances,
    row="version_count",
    hue="version_count",
    aspect=12,
    height=0.8,
    palette="YlGnBu",
)

# Draw the densities in a few steps
g.map(
    sns.kdeplot,
    "days_to_published",
    bw_adjust=0.5,
    clip_on=False,
    fill=True,
    alpha=1,
    linewidth=1.5,
)
g.map(
    sns.kdeplot,
    "days_to_published",
    clip_on=False,
    color="w",
    lw=2,
    bw_adjust=0.5,
)
g.map(plt.axhline, y=0, lw=2, clip_on=False)


# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(
        0,
        0.2,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


g.map(label, "days_to_published")

# g.set_xlabels("Diversity (1 - gini coefficient)")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-0.7)

# Remove axes details that don't play well with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)
# -

g = sns.lmplot(
    x="version_count",
    y="days_to_published",
    data=published_date_distances,
    x_bins=np.unique(x),
    palette="YlGnBu",
)
_ = g.set_ylabels("Time Elapsed Until Preprint is Published (Days)")
_ = g.set_xlabels("# of Preprint Versions")
_ = g.axes[0][0].annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}", (2, 700)
)
_ = g.axes[0][0].set_xlim(0.5, 12.5)
_ = g.axes[0][0].set_ylim(0, g.axes[0][0].get_ylim()[1])

# # Construct Scatter Plot of Date vs Document Distances

# Preprints are delayed on an average of 17 days as changes are demanded from the peer-review process. This section regresses a preprint's document distance against the time it takes to have a preprint published. A scatter and square bin plot are generated below.


# +
# Get smoothed linear regression line
# Removed negative time here
x = published_date_distances.doc_distances.values.tolist()

y = published_date_distances.time_to_published.apply(
    lambda x: x / timedelta(days=1)
).tolist()

xseq_2 = np.linspace(np.min(x), np.max(x), 80)

results_2 = linregress(x, y)
print(results_2)
# -

g = (
    p9.ggplot(
        published_date_distances, p9.aes(y="time_to_published", x="doc_distances")
    )
    + p9.geom_point()
    + p9.geom_line(
        stat="smooth", method="lm", linetype="dashed", se=False, alpha=0.9, size=0.6
    )
    + p9.scale_y_timedelta(labels=timedelta_format("d"))
    + p9.annotate(
        "text",
        x=10,
        y=timedelta(days=1450),
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    )
    + p9.labs(
        x="Euclidian Distance of Preprints First and Final Versions",
        y="Time Elapsed Until Preprint is Published",
    )
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.3)
)
print(g)

g = (
    p9.ggplot(
        published_date_distances, p9.aes(x="doc_distances", y="time_to_published")
    )
    + p9.geom_bin2d(bins=100)
    + p9.scale_fill_distiller(
        trans="log", direction=-1, type="seq", palette="YlGnBu", name="log(count)"
    )
    + p9.geom_line(
        stat="smooth", method="lm", linetype="dashed", se=False, alpha=1, size=0.7
    )
    + p9.scale_y_timedelta(labels=timedelta_format("d"))
    + p9.annotate(
        "text",
        x=7.5,
        y=timedelta(days=1490),
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    )
    + p9.labs(
        x="Euclidian Distance of Preprint-Published Versions",
        y="Time Elapsed Until Preprint is Published",
        legend="log(count)",
    )
)
print(g)

# # Hex grid options
# A couple hex grid options just to see what they look like

x_line = np.array(
    [
        published_date_distances["doc_distances"].min(),
        published_date_distances["doc_distances"].max(),
    ]
)
y_line = x_line * results_2.slope + results_2.intercept

plt.figure(figsize=(7, 5))
ax = plt.hexbin(
    published_date_distances["doc_distances"],
    published_date_distances["days_to_published"],
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
ax.plot(x_line, y_line, "--k")
ax.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (6, 1490),
)
_ = ax.set_xlabel("Euclidian Distance of Preprint-Published Versions")
_ = ax.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
cbar = plt.colorbar()
_ = cbar.ax.set_ylabel("count", rotation=270)
plt.savefig("output/article_distance_vs_publication_time_hex.svg", dpi=250)
plt.savefig("output/article_distance_vs_publication_time_hex.png", dpi=250)

plt.figure(figsize=(6, 5))
ax = plt.hexbin(
    published_date_distances["doc_distances"],
    published_date_distances["days_to_published"],
    gridsize=50,
    cmap="YlGnBu_r",
    norm=mpl.colors.LogNorm(),
    mincnt=1,
    edgecolors=None,
)
plt.xlim([0, 12])
plt.ylim([0, 1800])
ax = plt.gca()
ax.plot(x_line, y_line, "--k")
ax.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (6, 1490),
)
_ = ax.set_xlabel("Euclidian Distance of Preprint-Published Versions")
_ = ax.set_ylabel("Time Elapsed Until Preprint is Published (Days)")
cbar = plt.colorbar()
_ = cbar.ax.set_ylabel("count", rotation=270)

# +
hexplot = sns.jointplot(
    x="doc_distances",
    y="days_to_published",
    data=published_date_distances,
    kind="hex",
    joint_kws={"cmap": "YlGnBu_r", "mincnt": 1},
    norm=mpl.colors.LogNorm(),
    height=8,
)
hexplot.set_axis_labels(
    xlabel="Euclidian Distance of Preprint-Published Versions",
    ylabel="Time Elapsed Until Preprint is Published (Days)",
)

hexplot.ax_joint.plot(x_line, y_line, "--k")
hexplot.ax_joint.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (6, 1490),
)
# shrink fig so cbar is visible
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([0.85, 0.25, 0.05, 0.4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
_ = cbar_ax.set_ylabel("count", rotation=270)

# +
hexplot = sns.jointplot(
    x="doc_distances",
    y="days_to_published",
    data=published_date_distances,
    kind="hex",
    joint_kws={"cmap": "YlGnBu_r", "mincnt": 1, "edgecolors": None},
    norm=mpl.colors.LogNorm(),
    height=8,
)

hexplot.set_axis_labels(
    xlabel="Euclidian Distance of Preprint-Published Versions",
    ylabel="Time Elapsed Until Preprint is Published (Days)",
)

hexplot.ax_joint.plot(x_line, y_line, "--k")
hexplot.ax_joint.annotate(
    f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    (6, 1490),
)
# shrink fig so cbar is visible
plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([0.85, 0.25, 0.05, 0.4])  # x, y, width, height
plt.colorbar(cax=cbar_ax)
_ = cbar_ax.set_ylabel("count", rotation=270)
