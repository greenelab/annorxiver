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

# # Journal Recommendation - Figure Generator

# +
from collections import Counter
import itertools
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm

import plotnine as p9
# -

sys.path.append(str(Path("..").resolve()))
from SAUCIE import SAUCIE, Loader  # noqa: E402
import tensorflow.compat.v1 as tf  # noqa: E402

# +
# Set up porting from python to R
# and R to python :mindblown:

import rpy2.rinterface  # noqa: E402

# %load_ext rpy2.ipython
# -

# # Plot Accuracy Results

results = [
    {
        "value": 0.00522 / 0.00522,
        "model": "random_baseline",
        "distance": "N/A",
        "dataset": "train (cross validation)",
    },
    {
        "value": 0.39457 / 0.00517,
        "model": "paper_paper",
        "distance": "euclidean",
        "dataset": "train (cross validation)",
    },
    {
        "value": 0.35982 / 0.00517,
        "model": "centroid",
        "distance": "euclidean",
        "dataset": "train (cross validation)",
    },
    {
        "value": 0.39824 / 0.00517,
        "model": "paper_paper",
        "distance": "manhattan",
        "dataset": "train (cross validation)",
    },
    {
        "value": 0.39824 / 0.00517,
        "model": "centroid",
        "distance": "manhattan",
        "dataset": "train (cross validation)",
    },
    {
        "value": 0.01506 / 0.01506,
        "model": "random_baseline",
        "distance": "N/A",
        "dataset": "test",
    },
    {
        "value": 0.15317 / 0.01506,
        "model": "paper_paper",
        "distance": "euclidean",
        "dataset": "test",
    },
    {
        "value": 0.17347 / 0.01506,
        "model": "centroid",
        "distance": "euclidean",
        "dataset": "test",
    },
    {
        "value": 0.20380 / 0.01506,
        "model": "paper_paper",
        "distance": "manhattan",
        "dataset": "test",
    },
    {
        "value": 0.21523 / 0.01506,
        "model": "centroid",
        "distance": "manhattan",
        "dataset": "test",
    },
]

# +
result_df = pd.DataFrame.from_records(results)

result_df["dataset"] = pd.Categorical(
    result_df.dataset.tolist(), categories=["train (cross validation)", "test"]
)
result_df.to_csv("output/knn_results.tsv", sep="\t", index=False)
result_df.head()

# +
g = (
    p9.ggplot(
        result_df.query("distance in ['euclidean']").rename(
            index=str, columns={"value": "fold_change"}
        ),
        p9.aes(x="model", y="fold_change"),
    )
    + p9.geom_col(position="dodge", show_legend=False, fill="#1f78b4")
    + p9.coord_flip()
    + p9.facet_wrap("dataset")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1.3)
    + p9.theme(figure_size=(6.66, 5))
    + p9.labs(y="Fold Change Over Random", fill="Distance Metric")
)

g.save(Path("output") / Path("figures") / Path("knn_result.svg"))

g.save(Path("output") / Path("figures") / Path("knn_result.png"), dpi=250)

print(g)


# -

# # Generate 2D Visualization

# ## Use SAUCIE on PMC

# ### Set Up Grid Evaluation

# Set the seeds to fix the reproducebility issue
def set_seeds():
    np.random.seed(100)
    tf.reset_default_graph()
    tf.set_random_seed(seed=100)
    os.environ["PYTHONHASHSEED"] = str(100)
    random.seed(100)


def run_saucie_param_grid(
    dataset,
    learning_rate_grid=[1e-3],
    lambda_c_grid=[0],
    lambda_d_grid=[0],
    steps_grid=[1000],
):
    plot_df = pd.DataFrame(
        [], columns=["dim1", "dim2", "lambda_c", "lambda_d", "journal"]
    )

    hyper_param_grid = itertools.product(
        learning_rate_grid, lambda_c_grid, lambda_d_grid, steps_grid
    )

    for learning_rate, lambda_c, lambda_d, steps in tqdm(hyper_param_grid):

        set_seeds()

        saucie = SAUCIE(
            dataset.shape[1] - 2,
            lambda_b=0,
            lambda_c=lambda_c,
            lambda_d=lambda_d,
            learning_rate=learning_rate,
            save_folder="output/model",
        )

        loadtrain = Loader(
            dataset.drop(["journal", "document"], axis=1).values,
            pd.Categorical(dataset["journal"].values).codes,
            shuffle=True,
        )

        saucie.train(loadtrain, steps=steps)

        loadeval = Loader(
            dataset.drop(["journal", "document"], axis=1).values,
            pd.Categorical(dataset["journal"].values).codes,
            shuffle=False,
        )

        embedding = saucie.get_embedding(loadeval)

        plot_df = plot_df.append(
            pd.DataFrame(embedding[0], columns=["dim1", "dim2"]).assign(
                steps=steps,
                learning_rate=learning_rate,
                lambda_c=lambda_c,
                lambda_d=lambda_d,
                journal=dataset.journal.tolist(),
            )
        )

    return plot_df


# ### Load the data

full_paper_dataset = pd.read_csv(
    Path("output/paper_dataset") / Path("paper_dataset_full.tsv.xz"), sep="\t"
)
print(full_paper_dataset.shape)
full_paper_dataset.head()

journal_counts = full_paper_dataset.journal.value_counts()
journal_counts[journal_counts > 1000][-4:]

full_paper_dataset_subset = full_paper_dataset.query(
    "journal in " f"{journal_counts[journal_counts > 1000][-4:].index.tolist()}"
)
full_paper_dataset_subset.head()

# ### Evaluate the Grid

# This section involves tuning the hyperparameters of the SAUCIE network. This network sues a shallow autoencoder to project high dimensional data into a low dimensional space. This network takes in three lambda parameters along with a learning rate and number of steps. The plots in this section show the results of different parameters being tunes on a small subset of PMC papers (randomly sampled from four different journals). The best parameters for this model separates the four journals into their own distinct clusters.

lambda_c_grid = np.linspace(1e-6, 1, num=5)
lambda_d_grid = np.linspace(1e-6, 1, num=5)

set_seeds()
plot_df = run_saucie_param_grid(
    full_paper_dataset_subset, lambda_c_grid=lambda_c_grid, lambda_d_grid=lambda_d_grid
)

g = (
    p9.ggplot(plot_df)
    + p9.aes(x="dim1", y="dim2", fill="journal")
    + p9.facet_grid("lambda_d ~ lambda_c", labeller="label_both", scales="free")
    + p9.geom_point()
    + p9.theme(figure_size=(12, 12))
)
print(g)

lambda_c_grid = np.linspace(1e-6, 1e-3, num=5)
lambda_d_grid = np.linspace(1e-6, 1e-3, num=5)

plot_df = run_saucie_param_grid(
    full_paper_dataset_subset, lambda_c_grid=lambda_c_grid, lambda_d_grid=lambda_d_grid
)

g = (
    p9.ggplot(plot_df)
    + p9.aes(x="dim1", y="dim2", fill="journal")
    + p9.facet_grid(
        "lambda_d ~ lambda_c",
        labeller=p9.labeller(
            cols=lambda s: f"lambda_c: {float(s):.3e}",
            rows=lambda s: f"lambda_d: {float(s):.3e}",
        ),
        scales="free",
    )
    + p9.geom_point()
    + p9.theme(figure_size=(12, 12))
)
g.save("output/figures/saucie_hyperparam_lambda_cd.png", dpi=500)
print(g)

learning_rate_grid = np.linspace(1e-6, 1e-3, num=3)
steps_grid = [1000, 3000, 5000, 10000, 10000]

plot_df = run_saucie_param_grid(
    full_paper_dataset_subset,
    lambda_c_grid=[1.000e-3],
    lambda_d_grid=[1.000e-3],
    steps_grid=steps_grid,
    learning_rate_grid=learning_rate_grid,
)

g = (
    p9.ggplot(plot_df)
    + p9.aes(x="dim1", y="dim2", fill="journal")
    + p9.facet_grid(
        "steps ~ learning_rate",
        labeller=p9.labeller(
            cols=lambda s: f"learning_rate: {float(s):.3e}",
            rows=lambda s: f"steps: {s}",
        ),
        scales="free",
    )
    + p9.geom_point()
    # + p9.scale_fill_discrete(guide=False)
    + p9.theme(figure_size=(12, 12))
)
g.save("output/figures/saucie_hyperparam_lr_steps.png", dpi=500)
print(g)

# +
set_seeds()

saucie = SAUCIE(
    full_paper_dataset.shape[1] - 2,
    lambda_b=0,
    lambda_c=1e-3,
    lambda_d=1e-3,
    learning_rate=1e-3,
    save_folder="output/model",
)

loadtrain = Loader(
    full_paper_dataset.drop(["journal", "document"], axis=1).values,
    pd.Categorical(full_paper_dataset["journal"].values).codes,
    shuffle=True,
)

saucie.train(loadtrain, steps=5000)
saucie.save()

loadeval = Loader(
    full_paper_dataset.drop(["journal", "document"], axis=1).values,
    pd.Categorical(full_paper_dataset["journal"].values).codes,
    shuffle=False,
)
embedding = saucie.get_embedding(loadeval)
embedding

# +
full_dataset = pd.DataFrame(embedding[0], columns=["dim1", "dim2"]).assign(
    journal=full_paper_dataset.journal.tolist(),
    document=full_paper_dataset.document.tolist(),
)
print(full_dataset.shape)

full_dataset.to_csv(
    Path("output/paper_dataset") / Path("paper_dataset_full_tsne.tsv"),
    sep="\t",
    index=False,
)

full_dataset.head()
# -

g = (
    p9.ggplot(full_dataset.sample(10000, random_state=100))
    + p9.aes(x="dim1", y="dim2", fill="journal")
    + p9.geom_point()
    + p9.scale_fill_discrete(guide=False)
)
print(g)

# # Generate Bin plots

# ## Square Plot

data_df = pd.read_csv(
    Path("output") / Path("paper_dataset") / Path("paper_dataset_full_tsne.tsv"),
    sep="\t",
)
print(data_df.shape)
data_df.head()

data_df.describe()

# + magic_args="-i data_df -o square_plot_df" language="R"
#
# library(ggplot2)
#
# bin_width = 0.85
# g <- (
#     ggplot(data_df, aes(x=dim1, y=dim2))
# + geom_bin2d(binwidth=bin_width)

# + theme(legend.position="left")
# )
# square_plot_df <- ggplot_build(g)$data[[1]]
# -
print(square_plot_df.shape)  # noqa: F821
square_plot_df.head()  # noqa: F821

full_paper_dataset = pd.read_csv(
    Path("output/paper_dataset") / Path("paper_dataset_full.tsv.xz"), sep="\t"
)
print(full_paper_dataset.shape)
full_paper_dataset.head()

pca_components_df = pd.read_csv(
    Path("../../biorxiv")
    / Path("pca_association_experiment")
    / Path("output")
    / Path("word_pca_similarity")
    / Path("pca_components.tsv"),
    sep="\t",
)
print(pca_components_df.shape)
pca_components_df.head()

# +
mapped_data_df = pd.DataFrame([], columns=data_df.columns.tolist() + ["squarebin_id"])
square_bin_records = []

for idx, (row_idx, square_bin) in tqdm(
    enumerate(square_plot_df.iterrows())  # noqa: F821
):

    top_left = (square_bin["xmin"], square_bin["ymax"])
    bottom_right = (square_bin["xmax"], square_bin["ymin"])

    datapoints_df = data_df.query(
        f"dim1 > {top_left[0]} and dim1 < {bottom_right[0]}"
    ).query(f"dim2 < {top_left[1]} and dim2 > {bottom_right[1]}")

    # sanity check that I'm getting the coordinates correct
    assert datapoints_df.shape[0] == square_bin["count"]

    bin_pca_dist = 1 - cdist(
        pca_components_df,
        (
            full_paper_dataset.query(f"document in {datapoints_df.document.tolist()}")
            .drop(["journal", "document"], axis=1)
            .mean(axis=0)
            .values[:, np.newaxis]
            .T
        ),
        "cosine",
    )

    pca_sim_df = pd.DataFrame(
        {
            "score": bin_pca_dist[:, 0],
            "pc": [f"0{dim+1}" if dim + 1 < 10 else f"{dim+1}" for dim in range(50)],
        }
    )

    pca_sim_df = pca_sim_df.reindex(
        pca_sim_df.score.abs().sort_values(ascending=False).index
    )

    square_bin_records.append(
        {
            "x": square_bin["x"],
            "y": square_bin["y"],
            "xmin": square_bin["xmin"],
            "xmax": square_bin["xmax"],
            "ymin": square_bin["ymin"],
            "ymax": square_bin["ymax"],
            "count": datapoints_df.shape[0],
            "bin_id": idx,
            "pc": pca_sim_df.to_dict(orient="records"),
            "journal": dict(Counter(datapoints_df.journal.tolist()).items()),
        }
    )

    mapped_data_df = mapped_data_df.append(
        datapoints_df.assign(squarebin_id=idx).reset_index(drop=True), ignore_index=True
    )
# -


mapped_data_df.head()

print(mapped_data_df.shape)
mapped_data_df.to_csv(
    Path("output") / Path("paper_dataset") / Path("paper_dataset_tsne_square.tsv"),
    sep="\t",
    index=False,
)

square_map_df = pd.DataFrame.from_records(square_bin_records)
square_map_df.head()

square_map_df.to_json(
    Path("output") / Path("app_plots") / Path("pmc_square_plot.json"),
    orient="records",
    lines=False,
)
