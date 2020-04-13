#!/usr/bin/env python
# coding: utf-8

# # Visualize BioRxiv Document Embeddings

# This notebook is designed to visualize umap and tsne representations of bioRxiv document embeddings.
# Each document embedding is generated via an average of every word in a given article.

# In[1]:


import itertools
from pathlib import Path
import re

import pandas as pd
import plotnine as p9


# In[2]:


journal_map_df = pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
journal_map_df.head()


# In[3]:


biorxiv_umap_models = {
    int(re.search(r"(\d+)", str(file)).group(1)):pd.read_csv(str(file), sep="\t")
    for file in Path().rglob("output/embedding_output/umap/biorxiv_umap*tsv")
}

# Keep only current versions of articles
biorxiv_umap_models_lastest = {
    key: (
        biorxiv_umap_models[key]
        .groupby("doi")
        .agg({
            "doi": "last",
            "document": "last",
            "umap1": "last",
            "umap2": "last",
            "category":"last",
        })
        .reset_index(drop=True)
    )
    for key in biorxiv_umap_models
}


# In[4]:


biorxiv_tsne_models = {
    int(re.search(r"(\d+)", str(file)).group(1)):pd.read_csv(str(file), sep="\t")
    for file in Path().rglob("output/embedding_output/tsne/biorxiv_tsne*tsv")
}

# Keep only current versions of articles
biorxiv_tsne_models_lastest = {
    key: (
        biorxiv_tsne_models[key]
        .groupby("doi")
        .agg({
            "doi": "last",
            "document": "last",
            "tsne1": "last",
            "tsne2": "last",
            "category":"last",
        })
        .reset_index(drop=True)
    )
    for key in biorxiv_tsne_models
}


# In[5]:


biorxiv_pca_models = {
    int(re.search(r"(\d+)", str(file)).group(1)):pd.read_csv(str(file), sep="\t")
    for file in Path().rglob("output/embedding_output/pca/biorxiv_pca*tsv")
}

# Keep only current versions of articles
biorxiv_pca_models_lastest = {
    key: (
        biorxiv_pca_models[key]
        .groupby("doi")
        .agg({
            "doi": "last",
            "document": "last",
            "pca1": "last",
            "pca2": "last",
            "category":"last",
        })
        .reset_index(drop=True)
    )
    for key in biorxiv_pca_models
}


# # UMAP of the Documents

# In[6]:


g = (
    p9.ggplot(biorxiv_umap_models_lastest[150])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="UMAP of BioRxiv (Word dim: 150)",
        color="Article Category"
    )
)
g.save("output/embedding_output/umap/figures/biorxiv_umap_150.png", dpi=500)
print(g)


# In[7]:


g = (
    p9.ggplot(biorxiv_umap_models_lastest[250])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="UMAP of BioRxiv (Word dim: 250)",
        color="Article Category"
    )
)
g.save("output/embedding_output/umap/figures/biorxiv_umap_250.png", dpi=500)
print(g)


# In[8]:


g = (
    p9.ggplot(biorxiv_umap_models_lastest[300])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="UMAP of BioRxiv (Word dim: 300)",
        color="Article Category"
    )
)
g.save("output/embedding_output/umap/figures/biorxiv_umap_300.png", dpi=500)
print(g)


# Looking at the plots it seems that documents in the same category mostly cluster together, which is expected. The plots appear to be rotated versions of each other, which suggests that dimensionality doesn't hugely affect umap's results. Interesting cases are the outliers that appear within the plot. Question that arises: what are those outliers?

# ## UMAP Outliers

# In[9]:


g = (
    p9.ggplot(biorxiv_umap_models_lastest[300])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.annotate("segment", x=10, y=7, xend=9.2, yend=5.2, arrow=p9.arrow(), color="red")
    + p9.annotate("segment", x=10, y=-2.5, xend=12, yend=-0.6, arrow=p9.arrow(), color="red")
    + p9.annotate("segment", x=7.5, y=-2.2, xend=6.2, yend=-2.2, arrow=p9.arrow(), color="red")
    + p9.annotate("segment", x=7.5, y=1.5, xend=5.7, yend=1.5, arrow=p9.arrow(), color="red")
    + p9.labs(
        title="Outlier Detection of BioRxiv UMAP (Word dim: 300)",
        color="Article Category"
    )
)
print(g)


# In[10]:


(
    biorxiv_umap_models_lastest[300]
    .query("umap1 > 5")
)


# Using the doi link (doi.org/doi url) so far the bioinformatics papers should be reclassified as Epigenetics (10.1101/290825) and Cancer biology (10.1101/599225). The animal behavior and cognition category (10.1101/066423 and 10.1101/045062) has shockingly similar titles, which means I found a duplicate of preprints. The articles have two different ids, but have the same authors and unsurprisingly close titles. Lastly, the genetics (10.1101/045666) preprint sounds like it should be a neuroscience preprint. So far it seems like the outliers are category errors/mysterious duplication.

# In[11]:


g = (
    p9.ggplot(biorxiv_umap_models_lastest[300])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.annotate("segment", x=-5.5, y=5, xend=-5.6, yend=2.5, arrow=p9.arrow(), color="red")
    + p9.annotate("segment", x=-4.8, y=5, xend=-4.8, yend=3.6, arrow=p9.arrow(), color="red")
    + p9.labs(
        title="Outlier Detection of BioRxiv UMAP (Word dim: 300)",
        color="Article Category"
    )
)
print(g)


# In[12]:


(
    biorxiv_umap_models_lastest[300]
    .query("umap1 < -4.5")
)


# This article is mainly about evolutionary biology; however, it discusses intellegency quotient (IQ). This means the outlier is a combination of both fields; hence the point is in close proximity of the evolutionary biology section. The cancer biology preprint (10.1101/034132) seems like a biophysics paper or something to that nature.

# In[13]:


g = (
    p9.ggplot(biorxiv_umap_models_lastest[300])
    + p9.aes(x="umap1", y="umap2", color="factor(category)")
    + p9.geom_point()
    + p9.annotate("rect", xmin=0, xmax=3, ymin=7.5, ymax=9.5, alpha = 0.4)
    + p9.labs(
        title="Outlier Detection of BioRxiv UMAP (Word dim: 300)",
        color="Article Category"
    )
)
print(g)


# In[14]:


(
    biorxiv_umap_models_lastest[300]
    .query("umap1 > 0 & umap1 <= 3")
    .query("umap2 > 7.5")
    .query("category=='bioinformatics'")
)


# Looking at (10.1101/2020.01.11.899831) it sounds like the authors used word2vec on fMRI documents. This is bioinformatics; however, I'd argue that this could fall under neuroscience. Plus (10.1101/038919) should be a neuroscience paper since it involves statistical modeling of multitasking behavior. From what I'm gathering majority of these outliers are papers that fall out of the categorical bounds.

# # TSNE of the Documents

# In[15]:


g = (
    p9.ggplot(biorxiv_tsne_models_lastest[150])
    + p9.aes(x="tsne1", y="tsne2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="TSNE of BioRxiv (Word dim: 150)",
        color="Article Category"
    )
)
g.save("output/embedding_output/tsne/figures/biorxiv_tsne_150.png", dpi=500)
print(g)


# In[16]:


g = (
    p9.ggplot(biorxiv_tsne_models_lastest[250])
    + p9.aes(x="tsne1", y="tsne2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="TSNE of BioRxiv (Word dim: 250)",
        color="Article Category"
    )
)
g.save("output/embedding_output/tsne/figures/biorxiv_tsne_250.png", dpi=500)
print(g)


# In[17]:


g = (
    p9.ggplot(biorxiv_tsne_models_lastest[300])
    + p9.aes(x="tsne1", y="tsne2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="TSNE of BioRxiv (Word dim: 300)",
        color="Article Category"
    )
)
g.save("output/embedding_output/tsne/figures/biorxiv_tsne_300.png", dpi=500)
print(g)


# These plots are similar to each other which results in documents grouping together based on their assigned category. Interestingly, there seems to be a big divide between the neuroscience categories. Plus the bioinformatics papers have a separated blob. Question to follow up on is: why the discrepancies? 

# ## TSNE Outliers

# In[18]:


g = (
    p9.ggplot(biorxiv_tsne_models_lastest[300])
    + p9.aes(x="tsne1", y="tsne2", color="factor(category)")
    + p9.geom_point()
    + p9.annotate("segment", x=60, y=60, xend=45, yend=45, arrow=p9.arrow(), color="red")
    + p9.labs(
        title="Outlier Detection of BioRxiv TSNE (Word dim: 300)",
        color="Article Category"
    )
)
print(g)


# In[19]:


(
    biorxiv_tsne_models_lastest[300]
    .query("tsne1 > 30 & tsne1 < 45")
    .query("tsne2 > 31 & tsne2 < 45")
    .query("category != 'bioinformatics'")
    .category
    .value_counts()
)


# Looking into the bioinformatics blob approximately 370 articles are off category.

# In[20]:


(
    biorxiv_tsne_models_lastest[300]
    .query("tsne1 > 30 & tsne1 < 45")
    .query("tsne2 > 31 & tsne2 < 45")
    .query("category in ['pharmacology', 'none', 'physiology', 'biochemistry', 'animal behavior and cognition']")
)


# (10.1101/008896) is a bioinformatics and biophysics paper, but the authors didn't provide a category for the preprint. (10.1101/248898) is a preprint that monitors data availability, which suggests that it should have the bioinformatics category as well. (10.1101/377960) should have the bioinformatics category as well.

# In[21]:


g = (
    p9.ggplot(biorxiv_tsne_models_lastest[300])
    + p9.aes(x="tsne1", y="tsne2", color="factor(category)")
    + p9.geom_point()
    + p9.annotate("rect", xmin=-30, xmax=-20, ymin=40, ymax=55, alpha=0.4)
    + p9.labs(
        title="Outlier Detection of BioRxiv TSNE (Word dim: 300)",
        color="Article Category"
    )
)
print(g)


# In[22]:


(
    biorxiv_tsne_models_lastest[300]
    .query("tsne1 > -30 & tsne1 < -20")
    .query("tsne2 > 40 & tsne2 < 55")
    .query("category != 'neuroscience'")
    .category
    .value_counts()
)


# Looking into the bioinformatics blob approximately 370 articles are off category.

# In[23]:


(
    biorxiv_tsne_models_lastest[300]
    .query("tsne1 > -30 & tsne1 < -20")
    .query("tsne2 > 40 & tsne2 < 55")
    .query("category in ['biophysics', 'genomics', 'bioinformatics', 'clinical trials', 'ecology', 'cancer biology']")
)


# (10.1101/374801) is a preprint on glucose monitoring in infants. Not sure how this preprint fell into the neurscience category or even in the bioinformatics category.. (10.1101/395293) is about using ultrasound to improve corticosteroid injection. Once again not sure how this paper got grouped with other neuroscience papers. (10.1101/574673) is a preprint about transcriptional profiling of cerebrovascular traits in mice (paraphrased). This paper could also receive a neuroscience label given that the cell location type is neuroscience related. 

# # PCA of the Documents

# In[25]:


g = (
    p9.ggplot(biorxiv_pca_models_lastest[150])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word dim: 150)",
        color="Article Category"
    )
)
g.save("output/embedding_output/pca/figures/biorxiv_pca_150.png", dpi=500)
print(g)


# In[26]:


g = (
    p9.ggplot(biorxiv_pca_models_lastest[250])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word dim: 250)",
        color="Article Category"
    )
)
g.save("output/embedding_output/pca/figures/biorxiv_pca_250.png", dpi=500)
print(g)


# In[27]:


g = (
    p9.ggplot(biorxiv_pca_models_lastest[300])
    + p9.aes(x="pca1", y="pca2", color="factor(category)")
    + p9.geom_point()
    + p9.labs(
        title="PCA of BioRxiv (Word dim: 300)",
        color="Article Category"
    )
)
g.save("output/embedding_output/pca/figures/biorxiv_pca_300.png", dpi=500)
print(g)


# Overall, it seems that average word vectors can detect category errors within the biorxiv repository. Really cool given the fact that this technique can provide a elegant solution to a complex problem.
