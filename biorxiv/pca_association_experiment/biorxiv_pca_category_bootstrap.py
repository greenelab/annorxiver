#!/usr/bin/env python
# coding: utf-8

# # Determine Word to PCA Associations

# This notebook is designed to run PCA over the document embeddings and calculate category-pca associations with each principal component.

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook


# In[2]:


journal_map_df = (
    pd.read_csv(
        Path("..")/
        Path("exploratory_data_analysis")/
        Path("output/biorxiv_article_metadata.tsv"), 
        sep="\t"
    )
    .groupby("doi")
    .agg({
        "doi":"last",
        "document":"first",
        "category":"last",
    })
)
journal_map_df.head()


# # PCA the Documents

# Run PCA over the documents. Generates 50 principal components, but can generate more or less.

# In[3]:


n_components = 50
random_state = 100


# In[4]:


biorxiv_articles_df = pd.read_csv(
    Path("..")/
    Path("word_vector_experiment")/
    Path("output/word2vec_output")/
    Path("biorxiv_all_articles_300.tsv.xz"), 
    sep="\t"
)
biorxiv_articles_df.head()


# In[5]:


reducer = PCA(
    n_components = n_components,
    random_state = random_state
)

reducer.fit(
    biorxiv_articles_df[[f"feat_{idx}" for idx in range(300)]]
    .values
)


# In[6]:


document_categories_df = (
    journal_map_df[["document", "category"]]
    .merge(biorxiv_articles_df, on="document")
)
document_categories_df.head()


# # Bootstrap 95% Confidence Intervals Cosine sim

# In[7]:


boostraped_iterations = 10000
pc_1 = []
pc_2 = []

# The for loop performs the following algorithm to estimate 95% confidence intervals:
# 
# 1. Group document embeddings by category
# 2. Randomly sample a document from each category
# 3. Calculate the similarity scores between the documents and the first two principal components
# 4. Repeat the above steps for 10000 iterations.
# 5. Finally take the 25th percentile and the 97.5th percentile to make up the interval

for iteration in tqdm_notebook(range(boostraped_iterations)):
    sampled_df = (
        document_categories_df
        .groupby("category")
        .apply(lambda x: x.sample(1, random_state=iteration))
        .reset_index(drop=True)
        .sort_values("category")
    )
    
    document_distance = (
        1 - cdist(
            (
                sampled_df
                .drop(["document", "category"], axis=1)
                .values
            ), 
            reducer.components_[0:2], 
            'cosine'
        )
    )
    
    pc_1.append(document_distance[:, 0])
    pc_2.append(document_distance[:, 1])

pc_1 = np.stack(pc_1)
pc_2 = np.stack(pc_2)


# In[8]:


mean_similarity_df = (
    pd.DataFrame(
        (
            1 - cdist(
                document_categories_df
                .drop(["document", "category"], axis=1)
                .values, 
                reducer.components_[0:2], 
                'cosine'
            )
        ),
        columns=["pca1_cossim", "pca2_cossim"]
    )
    .assign(
        category = document_categories_df.category.tolist(),
        document = document_categories_df.document.tolist()
    )
)
mean_similarity_df.head()


# In[9]:


category_sim_df = (
    mean_similarity_df
    .groupby("category")
    .agg({
        "pca1_cossim": "mean",
        "pca2_cossim": "mean"
    })
    .assign(
        pca1_cossim_upper = np.percentile(pc_1, 97.5, axis=0),
        pca1_cossim_lower = np.percentile(pc_1, 25, axis=0),
        pca2_cossim_upper = np.percentile(pc_2, 97.5, axis=0),
        pca2_cossim_lower = np.percentile(pc_2, 25, axis=0)
    )
    .reset_index()
    [
        [
            "category", "pca1_cossim_lower",
            "pca1_cossim", "pca1_cossim_upper",
            "pca2_cossim_lower", "pca2_cossim",
            "pca2_cossim_upper"
        ]
    ]
)
category_sim_df.head()


# In[10]:


category_sim_df.to_csv(
    "output/category_cossim_95_ci.tsv", 
    sep="\t", index=False
)


# In[11]:


g = (
    p9.ggplot(category_sim_df)
    + p9.aes(
        x="category", y="pca1_cossim",
        ymin="pca1_cossim_lower", ymax="pca1_cossim_upper"
    )
    + p9.geom_pointrange()
    + p9.coord_flip()
    + p9.theme_bw()
    + p9.scale_x_discrete(limits=category_sim_df.category.tolist()[::-1])
    + p9.theme(
        figure_size=(11, 7),
        text=p9.element_text(size=12),
        panel_grid_major_y=p9.element_blank()
    )
    + p9.labs(
        y = "PC1 Cosine Similarity"
    )
)
g.save("output/pca_plots/figures/category_pca1_95_ci.svg", dpi=500)
g.save("output/pca_plots/figures/category_pca1_95_ci.png", dpi=500)
print(g)


# In[12]:


g = (
    p9.ggplot(category_sim_df)
    + p9.aes(
        x="category", y="pca2_cossim", 
        ymax="pca2_cossim_upper", ymin="pca2_cossim_lower"
    )
    + p9.geom_pointrange()
    + p9.coord_flip()
    + p9.theme_bw()
    + p9.scale_x_discrete(limits=category_sim_df.category.tolist()[::-1])
    + p9.theme(
        figure_size=(11, 7),
        text=p9.element_text(size=12),
        panel_grid_major_y=p9.element_blank()
    )
    + p9.labs(
        y = "PC2 Cosine Similarity"
    )
)
g.save("output/pca_plots/figures/category_pca2_95_ci.svg", dpi=500)
g.save("output/pca_plots/figures/category_pca2_95_ci.png", dpi=500)
print(g)


# # Plot Documents Projected on PCs Grouped by Category

# In[13]:


projected_documents = reducer.transform(
    document_categories_df[[f"feat_{idx}" for idx in range(300)]]
)
projected_documents.shape


# In[14]:


projected_documents_df = (
    pd.DataFrame(
        projected_documents,
        columns=[f"PC_{dim+1}" for dim in range(n_components)]
    )
    .assign(
        category = document_categories_df.category.tolist(),
        document = document_categories_df.document.tolist()
    )
)
projected_documents_df


# In[15]:


g = (
    p9.ggplot(projected_documents_df)
    + p9.aes(x="factor(category)", y="PC_1")
    + p9.geom_boxplot()
    + p9.coord_flip()
    + p9.theme(
        figure_size=(11, 11),
        text=p9.element_text(size=12),
        panel_grid_major_y=p9.element_blank()
    )
    + p9.scale_x_discrete(
        limits=(
            projected_documents_df
            .sort_values("category")
            .category
            .unique()
            .tolist()
            [::-1]
         )
    )
    + p9.labs(
        x = "Article Category",
        y = "PC1"
    )
)
g.save("output/pca_plots/figures/category_box_plot_pc1.png", dpi=500)
g.save("output/pca_plots/figures/category_box_plot_pc1.svg", dpi=500)
print(g)


# In[17]:


g = (
    p9.ggplot(projected_documents_df)
    + p9.aes(x="factor(category)", y="PC_2")
    + p9.geom_boxplot()
    + p9.coord_flip()
    + p9.theme(
        figure_size=(11, 11),
        text=p9.element_text(size=12),
        panel_grid_major_y=p9.element_blank()
    )
    + p9.scale_x_discrete(
        limits=(
            projected_documents_df
            .sort_values("category")
            .category
            .unique()
            .tolist()
            [::-1]
         )
    )
    + p9.labs(
        x = "Article Category",
        y = "PC2"
    )
)
g.save("output/pca_plots/figures/category_box_plot_pc2.png", dpi=500)
g.save("output/pca_plots/figures/category_box_plot_pc2.svg", dpi=500)
print(g)

