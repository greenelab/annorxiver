#!/usr/bin/env python
# coding: utf-8

# # Journal Recommendation - Figure Generator

# In[1]:


from collections import Counter
import itertools
import os
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm

from saucie_modules import SAUCIE, Loader
import tensorflow.compat.v1 as tf

import plotnine as p9


# In[2]:


# Set up porting from python to R
# and R to python :mindblown:

import rpy2.rinterface
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# # Plot Accuracy Results

# In[3]:


results = [
    {"value":.00517, "model":"random_baseline", "distance":"N/A", "dataset":"train"},
    {"value":.39982, "model":"paper_paper", "distance":"euclidean", "dataset":"train"},
    {"value":.37340, "model":"centroid", "distance":"euclidean", "dataset":"train"},
    {"value":.39824, "model":"paper_paper", "distance":"manhattan", "dataset":"train"},
    {"value":.39824, "model":"centroid", "distance":"manhattan", "dataset":"train"},
    {"value":.00445, "model":"random_baseline", "distance":"N/A", "dataset":"test"},
    {"value":.20638, "model":"paper_paper", "distance":"euclidean", "dataset":"test"},
    {"value":.21636, "model":"centroid", "distance":"euclidean", "dataset":"test"},
    {"value":.20380, "model":"paper_paper", "distance":"manhattan", "dataset":"test"},
    {"value":.21523, "model":"centroid", "distance":"manhattan", "dataset":"test"}
]


# In[4]:


result_df = pd.DataFrame.from_records(results)

result_df['dataset'] = pd.Categorical(
    result_df.dataset.tolist(), 
    categories=['train', 'test']
)

result_df.head()


# In[5]:


g = (
    p9.ggplot(result_df.query("distance in ['euclidean', 'N/A']"), p9.aes(x="model", y="value"))
    + p9.geom_col(p9.aes(fill="factor(distance)"), position="dodge")
    + p9.coord_flip()
    + p9.facet_wrap("dataset")
    + p9.scale_fill_manual(["#808080", "#1f78b4"])
    + p9.theme_seaborn(context='paper')
    + p9.labs(
        y="Accuracy",
        fill="Distance"
    )
)

g.save(
    Path("output")/
    Path("figures")/
    Path("knn_result.svg"), 
    dpi=500
)

g.save(
    Path("output")/
    Path("figures")/
    Path("knn_result.png"),
    dpi=500
)

print(g)


# # Generate 2D Visualization

# ## Use SAUCIE on PMC

# ### Set Up Grid Evaluation

# In[6]:


# Set the seeds to fix the reproducebility issue
def set_seeds():
    np.random.seed(100)
    tf.reset_default_graph()
    tf.set_random_seed(seed=100)
    os.environ['PYTHONHASHSEED'] = str(100)
    random.seed(100)


# In[7]:


def run_saucie_param_grid(
    dataset, learning_rate_grid = [1e-3], 
    lambda_c_grid=[0], lambda_d_grid=[0], 
    steps_grid=[1000]
):
    plot_df = pd.DataFrame(
        [],
        columns=["dim1", "dim2", "lambda_c", "lambda_d", "journal"]
    )
    
    hyper_param_grid = itertools.product(
        learning_rate_grid, lambda_c_grid, 
        lambda_d_grid, steps_grid
    )
    
    for learning_rate, lambda_c, lambda_d, steps in tqdm(hyper_param_grid):

        set_seeds()

        saucie = SAUCIE(
            dataset.shape[1]-2, 
            lambda_b=0, lambda_c=lambda_c, lambda_d=lambda_d,
            learning_rate=learning_rate,
            save_folder="output/model"
        )

        loadtrain = Loader(
            dataset.drop(["journal", "document"], axis=1).values, 
            pd.Categorical(dataset["journal"].values).codes, 
            shuffle=True
        )

        saucie.train(loadtrain, steps=steps)
        
        loadeval = Loader(
            dataset.drop(["journal", "document"], axis=1).values, 
            pd.Categorical(dataset["journal"].values).codes, 
            shuffle=False
        )
        
        embedding = saucie.get_embedding(loadeval)
        
        plot_df = plot_df.append(
            pd.DataFrame(embedding[0], columns=["dim1", "dim2"])
            .assign(
                steps=steps,
                learning_rate=learning_rate,
                lambda_c=lambda_c,
                lambda_d=lambda_d,
                journal=dataset.journal.tolist()
            )
        )
        
    return plot_df


# ### Load the data

# In[8]:


full_paper_dataset = pd.read_csv(
    Path("output/paper_dataset")/
    Path("paper_dataset_full.tsv.xz"), 
    sep="\t"
)
full_paper_dataset.head()


# In[9]:


journal_counts = full_paper_dataset.journal.value_counts()
journal_counts[journal_counts > 1000][-4:]


# In[10]:


full_paper_dataset_subset = (
    full_paper_dataset
    .query(
        "journal in "
        f"{journal_counts[journal_counts > 1000][-4:].index.tolist()}"
    )
)
full_paper_dataset_subset.head()


# ### Evaluate the Grid

# This section involves tuning the hyperparameters of the SAUCIE network. This network sues a shallow autoencoder to project high dimensional data into a low dimensional space. This network takes in three lambda parameters along with a learning rate and number of steps. The plots in this section show the results of different parameters being tunes on a small subset of PMC papers (randomly sampled from four different journals). The best parameters for this model separates the four journals into their own distinct clusters.

# In[11]:


lambda_c_grid = np.linspace(1e-6, 1, num=5)
lambda_d_grid = np.linspace(1e-6, 1, num=5)


# In[12]:


set_seeds()
plot_df = run_saucie_param_grid(
    full_paper_dataset_subset,
    lambda_c_grid=lambda_c_grid,
    lambda_d_grid=lambda_d_grid
)


# In[13]:


g = (
    p9.ggplot(plot_df)
    + p9.aes(x="dim1", y="dim2", fill = "journal")
    + p9.facet_grid("lambda_d ~ lambda_c", labeller="label_both", scales="free")
    + p9.geom_point()
    + p9.theme(
        figure_size=(12, 12)
    )
)
print(g)


# In[14]:


lambda_c_grid = np.linspace(1e-6, 1e-3, num=5)
lambda_d_grid = np.linspace(1e-6, 1e-3, num=5)


# In[15]:


plot_df = run_saucie_param_grid(
    full_paper_dataset_subset,
    lambda_c_grid=lambda_c_grid,
    lambda_d_grid=lambda_d_grid
)


# In[16]:


g = (
    p9.ggplot(plot_df)
    + p9.aes(x="dim1", y="dim2", fill = "journal")
    + p9.facet_grid(
        "lambda_d ~ lambda_c", 
        labeller=p9.labeller(
            cols=lambda s:f"lambda_c: {float(s):.3e}",
            rows=lambda s:f"lambda_d: {float(s):.3e}"
        ),
        scales="free"
    )
    + p9.geom_point()
    + p9.theme(
        figure_size=(12, 12)
    )
)
g.save("output/figures/saucie_hyperparam_lambda_cd.png", dpi=500)
print(g)


# In[17]:


learning_rate_grid = np.linspace(1e-6, 1e-3, num=3)
steps_grid = [1000, 3000, 5000, 10000, 10000]


# In[18]:


plot_df = run_saucie_param_grid(
    full_paper_dataset_subset,
    lambda_c_grid=[1.000e-3],
    lambda_d_grid=[1.000e-3], 
    steps_grid=steps_grid,
    learning_rate_grid=learning_rate_grid
)


# In[19]:


g = (
    p9.ggplot(plot_df)
    + p9.aes(x="dim1", y="dim2", fill = "journal")
    + p9.facet_grid(
        "steps ~ learning_rate", 
        labeller=p9.labeller(
            cols=lambda s:f"learning_rate: {float(s):.3e}",
            rows=lambda s:f"steps: {s}"
        ),
        scales="free"
    )
    + p9.geom_point()
    #+ p9.scale_fill_discrete(guide=False)
    + p9.theme(
        figure_size=(12, 12)
    )
)
g.save("output/figures/saucie_hyperparam_lr_steps.png", dpi=500)
print(g)


# In[20]:


set_seeds()

saucie = SAUCIE(
    full_paper_dataset.shape[1]-2, 
    lambda_b=0, lambda_c=1e-3, lambda_d=1e-3,
    learning_rate=1e-3,
    save_folder="output/model"
)

loadtrain = Loader(
    full_paper_dataset.drop(["journal", "document"], axis=1).values, 
    pd.Categorical(full_paper_dataset["journal"].values).codes, 
    shuffle=True
)

saucie.train(loadtrain, steps=10000)
saucie.save()

loadeval = Loader(
    full_paper_dataset.drop(["journal", "document"], axis=1).values, 
    pd.Categorical(full_paper_dataset["journal"].values).codes, 
    shuffle=False
)
embedding = saucie.get_embedding(loadeval)
embedding


# In[21]:


full_dataset = (
    pd.DataFrame(
        embedding[0], 
        columns=["dim1", "dim2"]
    )
    .assign(
        journal = full_paper_dataset.journal.tolist(),
        document = full_paper_dataset.document.tolist()
    )
)

full_dataset.to_csv(
    Path("output/paper_dataset")/
    Path("paper_dataset_full_tsne.tsv"),
    sep="\t", index=False
)

full_dataset.head()


# In[22]:


g = (
    p9.ggplot(full_dataset.sample(10000, random_state=100))
    + p9.aes(x="dim1", y="dim2", fill="journal")
    + p9.geom_point()
    + p9.scale_fill_discrete(guide=False)
)
print(g)


# # Generate Bin plots

# ## Square Plot

# In[23]:


data_df = pd.read_csv(
    Path("output")/
    Path("paper_dataset")/
    Path("paper_dataset_full_tsne.tsv"), 
    sep="\t"
)
data_df.head()


# In[24]:


data_df.describe()


# In[25]:


get_ipython().run_cell_magic('R', '-i data_df -o square_plot_df', '\nlibrary(ggplot2)\n\nbin_num <- 50\ng <- (\n    ggplot(data_df, aes(x=dim1, y=dim2))\n    + geom_bin2d(bins=bin_num, binwidth=0.85)\n)\nsquare_plot_df <- ggplot_build(g)$data[[1]]\nprint(g)')


# In[26]:


print(square_plot_df.shape)
square_plot_df.head()


# In[27]:


full_paper_dataset = pd.read_csv(
    Path("output/paper_dataset")/
    Path("paper_dataset_full.tsv.xz"),
    sep="\t"
)
print(full_paper_dataset.shape)
full_paper_dataset.head()


# In[28]:


pca_components_df = pd.read_csv(
    Path("../../biorxiv")/
    Path("pca_association_experiment")/
    Path("output")/
    Path("word_pca_similarity")/
    Path("pca_components.tsv"),
    sep="\t"
)
print(pca_components_df.shape)
pca_components_df.head()


# In[29]:


mapped_data_df = pd.DataFrame(
    [], 
    columns=data_df.columns.tolist()+['squarebin_id']
)
square_bin_records = []

for idx, (row_idx, square_bin) in tqdm(enumerate(square_plot_df.iterrows())):
    
    top_left = (square_bin["xmin"], square_bin["ymax"])
    bottom_right = (square_bin["xmax"], square_bin["ymin"])
    
    datapoints_df = (
        data_df
        .query(f"dim1 > {top_left[0]} and dim1 < {bottom_right[0]}")
        .query(f"dim2 < {top_left[1]} and dim2 > {bottom_right[1]}")
    )
    
    # sanity check that I'm getting the coordinates correct
    assert datapoints_df.shape[0] == square_bin["count"]
    
    bin_pca_dist = 1 - cdist(
        pca_components_df, 
        (
            full_paper_dataset
            .query(f"document in {datapoints_df.document.tolist()}")
            .drop(["journal", "document"], axis=1)
            .mean(axis=0)
            .values
            [:, np.newaxis]
            .T
        ),
        "cosine"
    )

    pca_sim_df = (
        pd.DataFrame({
            "score":bin_pca_dist[:,0], 
            "pc":[f"0{dim+1}" if dim+1 < 10 else f"{dim+1}" for dim in range(50)]
        })
    )

    pca_sim_df = (
        pca_sim_df
        .reindex(
            pca_sim_df
            .score
            .abs()
            .sort_values(ascending=False)
            .index
        )
    )

    square_bin_records.append({
        "x": square_bin["x"], "y":square_bin["y"], 
        "xmin":square_bin["xmin"], "xmax":square_bin["xmax"],
        "ymin":square_bin["ymin"], "ymax":square_bin["ymax"],
        "count":datapoints_df.shape[0], "bin_id":idx,
        "pc":pca_sim_df.to_dict(orient="records"),
        "journal":dict(
            Counter(datapoints_df.journal.tolist())
            .items()
        )
    })

    mapped_data_df = (
        mapped_data_df
        .append(
            datapoints_df
            .assign(squarebin_id=idx)
            .reset_index(drop=True),
            ignore_index=True
        )
    )


# In[30]:


mapped_data_df.head()


# In[31]:


mapped_data_df.to_csv(
    Path("output")/
    Path("paper_dataset")/
    Path("paper_dataset_tsne_square.tsv"), 
    sep="\t", index=False
) 


# In[32]:


square_map_df = pd.DataFrame.from_records(square_bin_records)
square_map_df.head()


# In[33]:


square_map_df.to_json(
    Path("output")/
    Path("app_plots")/
    Path("pmc_square_plot.json"),
    orient = 'records',
    lines = False
)

