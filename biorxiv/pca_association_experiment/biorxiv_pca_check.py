#!/usr/bin/env python
# coding: utf-8

# # Determine Word to PCA Associations

# This notebook is designed to check if there is a bug for the PCA plots. Discovered in [greenelab/annorxiver#2](https://github.com/greenelab/annorxiver/pull/2) the PCA plots there look practically the same. This highlights that there might be a bug when trying to train a PCA model (i.e. feeding the same embeddings into the PCA model (300 dim instead of the intended 150 dim). 
# 
# Conclusion: turns out there isn't a bug and the signal captured in the first two components is quite strong.

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook


# In[2]:


journal_map_df = pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
journal_map_df.head()


# # PCA the Documents

# Run PCA over the documents. Generates 2 principal components as in [greenelab/annorxiver#2](https://github.com/greenelab/annorxiver/pull/2). Needed to make sure PCA is trained with the same hyperparamter.

# In[3]:


n_components = 2
random_state = 100


# In[4]:


biorxiv_articles_dict = {
    150: pd.read_csv("../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_150.tsv.xz", sep="\t"),
    250: pd.read_csv("../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_250.tsv.xz", sep="\t"),
    300: pd.read_csv("../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz", sep="\t")
}


# In[5]:


biorxiv_articles_dict[150].shape


# In[6]:


biorxiv_articles_dict[250].shape


# In[7]:


biorxiv_articles_dict[300].shape


# In[10]:


pca_dict = {}
for embedding_dim in biorxiv_articles_dict:
    
    reducer = PCA(
        n_components = n_components,
        random_state = random_state
    )

    embedding = reducer.fit_transform(
        biorxiv_articles_dict[embedding_dim][[f"feat_{idx}" for idx in range(embedding_dim)]].values
    )

    pca_dict[embedding_dim] = (
        pd.DataFrame(embedding, columns=[f"pca{dim}" for dim in range(1, n_components+1, 1)])
        .assign(document=biorxiv_articles_dict[embedding_dim].document.values.tolist())
        .merge(journal_map_df[["category", "document", "doi"]], on="document")
    )


# # PCA Embedding Check

# The sanity check here is to make sure the values of pca1 and pca2 are not the exact same. If they were then there is a bug somewhere above.

# In[11]:


pca_dict[150].head(3)


# In[12]:


pca_dict[250].head(3)


# In[13]:


pca_dict[300].head(3)


# # PCA Plots

# This is the biggest test. If any of these plots look drastically different than in [greenelab/annorxiver#2](https://github.com/greenelab/annorxiver/pull/2) that would mean there is a bug I would have to go find. Luckily, there isn't a bug as the PCs captured here are the same regardless of embedding dimensions.

# In[14]:


g = (
    p9.ggplot(pca_dict[150])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word dim:150)",
        color="Article Category"
    )
)
print(g)


# In[15]:


g = (
    p9.ggplot(pca_dict[250])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word dim:250)",
        color="Article Category"
    )
)
print(g)


# In[16]:


g = (
    p9.ggplot(pca_dict[300])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word dim:300)",
        color="Article Category"
    )
)
print(g)

