#!/usr/bin/env python
# coding: utf-8

# # PCA Plots of bioRxiv

# This notebook is designed to run PCA over the document embeddings and plot various components against each other. The goal here is to understand the concepts best captured by each PC.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pathlib import Path
import os
import re

from gensim.models import Word2Vec
import itertools
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
from PIL import ImageColor
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook
import wordcloud


# In[2]:


def display_clouds(pc_cloud_1, pc_cloud_2):
    return display(
        HTML(
            f"""
            <table>
                <tr>
                    <td>
                    <img src={pc_cloud_1}>
                    </td>
                    <td>
                    <img src={pc_cloud_2}>
                    </td>
                </tr>
            </table>
            """
        )
    )


# In[3]:


journal_map_df = pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
journal_map_df.head()


# # PCA the Documents

# Run PCA over the documents. Generates 50 principal components, but can generate more or less.

# In[4]:


n_components = 50
random_state = 100


# In[5]:


biorxiv_articles_df = pd.read_csv(
    "../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz", 
    sep="\t"
)


# In[6]:


reducer = PCA(
    n_components = n_components,
    random_state = random_state
)

embedding = reducer.fit_transform(
    biorxiv_articles_df[[f"feat_{idx}" for idx in range(300)]].values
)

pca_df = (
    pd.DataFrame(embedding, columns=[f"pca{dim}" for dim in range(1, n_components+1, 1)])
    .assign(document=biorxiv_articles_df.document.values.tolist())
    .merge(journal_map_df[["category", "document", "doi"]], on="document")
)

latest_version = (
    pca_df
    .groupby("doi", as_index=False)
    .agg({"document":"last"})
    .document
    .values
    .tolist()
)

pca_df = (
    pca_df
    .query(f"document in {latest_version}")
    .reset_index(drop=True)
)


# In[7]:


reducer.explained_variance_


# In[8]:


reducer.explained_variance_ratio_


# # Plot the PCA axes

# This section visualizes PCA axes and attempts to provide an explanation for each plotted PC.
# Give there are 50 pcs generated this notebook/section will only focus on PCs that generate a clear enough signal.

# ## Basis PCs

# When analyzing PCs, it is generally a good idea to start off with the PCs that are easily distinguishable and then branch out to PCs that are harder to interpret. This section focuses on PCs: 1,2,3,4 and 5.
# 
# These PCs correspond to the following concepts:
# 
# | PC | Concept (negative vs positive)|
# | --- | --- |
# | PCA1 | Quantitative Biology vs Molecular Biology |
# | PCA2 | Neuroscience vs Genomics |
# | PCA3 | Sequencing vs Disease |
# | PCA4 | Microbiology vs Cell Biology |
# | PCA5 | RNA-seq vs Evolutional Biology | 

# ### PCA1 vs PCA2

# In[9]:


display_clouds(
    'output/word_pca_similarity/figures/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_02_cossim_word_cloud.png',
)


# These word clouds depict the following concepts: quantitative biology vs molecular biology (left) and genomics vs neuroscience (right). The cells below provide evidence for the previous claim

# In[10]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca01_v_pca02.png", dpi=500)
print(g)


# Take note that pca2 clusters neruoscience papers on the negative axis while genomics papers are on the positive axis (up and down). PCA 1 places papers that are more focused on quantitative biology on the right and molecular biology to the left. Hence why bioinforamtics papers are shifted more to the right and cell biology papers are shifted more to the left. This plot visually confirms the above finding.

# ### PCA1 vs PCA 3

# In[11]:


display_clouds(
    'output/word_pca_similarity/figures/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_03_cossim_word_cloud.png'
)


# These word clouds depict the following concepts: quantitative biology vs molecular biology (left) and disease vs sequencing (right)

# In[12]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca1", y="pca3", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca01_v_pca03.png", dpi=500)
print(g)


# Take note that pca3 clusters sequencing based papers on the negative axis (down) while disease papers are on the positive axis (up). When plotted against PCA1 it looks like epidemiology papers dominate the top right quadrant, while bottom right quadrant contains bioinformatics papers. This makes sense as many bioinformatic papers consist of some sort of sequencing technologies, while epidemiology is focused on measuring disease and its progression. Both take quantitative views, which is why they are on the positive size of PCA1.

# ### PCA2 vs PCA3

# In[13]:


display_clouds(
    'output/word_pca_similarity/figures/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_03_cossim_word_cloud.png'
)


# These word clouds depict the following concepts: neuroscience to genomics (left) and disease vs sequencing (right)

# In[14]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca2", y="pca3", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca02_v_pca03.png", dpi=500)
print(g)


# Note that bottom right quadrant comprises of mainly bioinformatics papers, which makes sense given that quadrant represents sequencing and genomics related papers (hence bioinformatics). The bottom left quadrant contains papers that have sequencing terms mentioned, but are more related to neuroscience than genomics (thats what forms the biophysics clusters). The top left are papers that relate to neuroscience and focused on disease while top right are genomics related papers that focus on disease.

# ### PCA3 vs PCA5 

# In[15]:


display_clouds(
    'output/word_pca_similarity/figures/pca_03_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_05_cossim_word_cloud.png'
)


# These word clouds depict the following concepts: sequencing vs disease (left) and RNA-seq vs evolutionary biology (right)

# In[16]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca3", y="pca5", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca03_v_pca05.png", dpi=500)
print(g)


# In[17]:


(
    pca_df[["pca3", "pca5", "category", "doi"]]
    .query("pca3 > 0 & pca5 > 0")
    .category
    .value_counts()
    .head(10)
)


# In[18]:


(
    pca_df[["pca3", "pca5", "category", "doi"]]
    .query("pca3 < 0 & pca5 < 0")
    .category
    .value_counts()
    .head(10)
)


# Looking at the top-right quadrant and bottom-left quadrant, the top 10 categories provide evidence for the concepts mentioned above. Since PCA5 contains RNA-seq concepts on the negative sdes and PCA3 has sequencing as well on its negative axis, one would expect the top category for the bottom-left quadrant be bioinformatics related. Likewise the top right should be focused on evolutionary biology and possibly disease.

# ### PCA1 vs PCA4

# In[19]:


display_clouds(
    'output/word_pca_similarity/figures/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_04_cossim_word_cloud.png'
)


# These word cloud produces the following concepts: qunatitative biology vs molecular biology (left) and marine biology vs cell biology (right).

# In[20]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca1", y="pca4", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca01_v_pca04.png", dpi=500)
print(g)


# In[21]:


(
    pca_df
    [["pca1", "pca4", "category", "doi"]]
    .query("pca1 < -2 & pca4 > 0")
    .category
    .value_counts()
    .head(10)
)


# In[22]:


(
    pca_df
    [["pca1", "pca4", "category", "doi"]]
    .query("pca1 < 0 & pca4 < 0")
    .category
    .value_counts()
    .head(10)
)


# Looking on the left size of PCA1 (negative), it looks like the top quadrant contains mainly cell biology papers and variants of cell biology. The bottom quadrant contains papers that are related to microbiology; Fun anecdote is that I super convinced that PCA4 was marine biology, but upon closer inspection turns out I was wrong it is microbio.

# # Non-straightforward PCs

# This section of the notebook aims to take a look at PCs that are not as straightforward as the ones above.

# ## PCA1 vs PCA6

# In[23]:


display_clouds(
    'output/word_pca_similarity/figures/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_06_cossim_word_cloud.png'
)


# The right word cloud appears to represent mathematics vs scientific communication or at least popular buzz words scientist used to promote their research. The next few cells will look more into it.

# In[24]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca1", y="pca6", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca01_v_pca06.png", dpi=500)
print(g)


# In[25]:


(
    pca_df[["pca1", "pca6", "category", "doi"]]
    .drop_duplicates("doi")
    .query("pca1 > 0 & pca6 > 3")
    .category
    .value_counts()
    .head(10)
)


# In[26]:


(
    pca_df[["pca1", "pca6", "category", "doi"]]
    .drop_duplicates("doi")
    .query("pca1 > 0 & pca6 < -1.5")
    .category
    .value_counts()
    .head(10)
)


# Looking at the top categories for the top and bottom right quadrants it seems that the papers follow the patterns captures by the word clouds above; however the positive axis still remains difficult to judge without taking a look at the individual papers.

# In[27]:


(
    pca_df[["pca1", "pca6", "category", "doi"]]
    .drop_duplicates("doi")
    .query("pca1 > 0 & pca6 > 3")
    .sort_values("pca6", ascending=False)
    .head(10)
)


# ## PCA2 vs PCA15

# In[28]:


display_clouds(
    'output/word_pca_similarity/figures/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_15_cossim_word_cloud.png'
)


# The word cloud on the right seems to contain the following concepts: facial recognition and behavior vs neuron biochemistry.

# In[29]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca2", y="pca15", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca02_v_pca15.png", dpi=500)
print(g)


# This graph depicts diversity within the neuroscience field as some papers are about facial recognition (negative) and other papers are about biochemistry (positive).

# In[30]:


(
    pca_df
    [["pca2", "pca15", "category", "document","doi"]]
    .sort_values(["pca15", "pca2"], ascending=[False, False])
    .head(10)
)


# In[31]:


(
    pca_df
    [["pca2", "pca15", "category", "document","doi"]]
    .sort_values(["pca15", "pca2"], ascending=[False, False])
    .tail(10)
)


# These papers confirm that the negative axis of PCA15 is facial recognition.

# ## PCA1 vs PCA8

# In[32]:


display_clouds(
    'output/word_pca_similarity/figures/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_08_cossim_word_cloud.png'
)


# The wordcloud on the right seems to represent the following concept:  biochemistry vs developmental biology. Main evidence for this appears in the plot below.

# In[33]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca2", y="pca8", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca02_v_pca08.png", dpi=500)
print(g)


# In[34]:


(
    pca_df
    [["pca2", "pca8", "category", "doi"]]
    .query("pca2 > -2 & pca2 < 2 & pca8 < -1")
    .category
    .value_counts()
    .head(10)
)


# In[35]:


(
    pca_df
    [["pca2", "pca8", "category", "doi"]]
    .query("pca2 > -2 & pca2 < 2 & pca8 > 1")
    .category
    .value_counts()
    .head(10)
)


# Looking at the top left and bottom left quadrants the top categories are: biochemistry and developmental biology. Based on this confirmation I'd argue that pca8 covers both of these concepts.

# ## PCA3 VS PCA13

# In[36]:


display_clouds(
    'output/word_pca_similarity/figures/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_13_cossim_word_cloud.png'
)


# Based on a quick google search the wordcloud on the right represents: viruses (immunology) vs model organisms.

# In[37]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca2", y="pca13", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca02_v_pca13.png", dpi=500)
print(g)


# In[38]:


(
    pca_df
    [["pca2", "pca13", "category", "doi"]]
    .sort_values("pca13", ascending=False)
    .head(10)
)


# In[39]:


(
    pca_df
    [["pca2", "pca13", "category", "doi"]]
    .sort_values("pca13", ascending=True)
    .head(10)
)


# Looking at the extremes values along PCA13, the categories seem to confirm my suspicions.

# ## PCA17 vs PCA20

# In[40]:


display_clouds(
    'output/word_pca_similarity/figures/pca_04_cossim_word_cloud.png',
    'output/word_pca_similarity/figures/pca_20_cossim_word_cloud.png'
)


# PCA20 represents the following concepts: immunology and cancer biology.

# In[41]:


g = (
    p9.ggplot(pca_df)
    + p9.aes(x="pca4", y="pca20", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word Dim: 300)",
        color="Article Category"
    )
)
g.save("output/pca_plots/pca04_v_pca20.png", dpi=500)
print(g)


# In[42]:


(
    pca_df
    [["pca4", "pca20", "category", "doi"]]
    .query("pca4 < 0 & pca20 < 0")
    .category
    .value_counts()
    .head(10)
)


# In[43]:


(
    pca_df
    [["pca4", "pca20", "category", "doi"]]
    .query("pca4 > 0 & pca20 > 0")
    .category
    .value_counts()
    .head(10)
)

