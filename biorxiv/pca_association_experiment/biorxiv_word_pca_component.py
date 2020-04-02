#!/usr/bin/env python
# coding: utf-8

# # Determine Word to PCA Associations

# This notebook is designed to run PCA over the document embeddings and calculate words-pca associations and document centroid-pca associations for each principal component. 

# In[1]:


from pathlib import Path
import os
import re

from gensim.models import Word2Vec
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
from PIL import ImageColor
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook
import wordcloud


# In[2]:


journal_map_df = pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
journal_map_df.head()


# # Get the Word Vectors

# Save the word vectors to disk, so later sections have easy access.

# In[3]:


model = Word2Vec.load("../word_vector_experiment/output/word2vec_models/300/biorxiv_300.model")


# In[4]:


word_vector_map = {
    word:model.wv[word]
    for word in tqdm_notebook(model.wv.vocab.keys())
}


# In[5]:


word_vector_df = pd.DataFrame.from_dict(word_vector_map, orient='index')

word_vector_df.to_csv(
    "output/word_pca_similarity/word_vectors_300.tsv.xz", 
    sep="\t", compression="xz"
)

word_vector_df.head()


# # PCA the Documents

# Run PCA over the documents. Generates 50 principal components, but can generate more or less.

# In[3]:


n_components = 50
random_state = 100


# In[4]:


biorxiv_articles_df = pd.read_csv("output/word2vec_output/biorxiv_all_articles_300.tsv.xz", sep="\t")


# In[5]:


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


# # Calculate Word-PCA Cosine Similarity

# Once PCA has finished, there are now 50 different principal components. The association between every word and principal component is calculated via [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) (cosine of the angle between two vectors).

# In[6]:


word_vector_df = pd.read_csv(
    "output/word_pca_similarity/word_vectors_300.tsv.xz",
    sep="\t", index_col=0
)


# In[7]:


# 1 - cosine distance = cosine similarity
word_pca_similarity = (
    1 - cdist(
        word_vector_df.values, 
        reducer.components_, 
        'cosine'
    )
)

word_pca_similarity.shape


# In[8]:


word_pca_sim_df = (
    pd.DataFrame(word_pca_similarity, columns=[f"pca{dim}_cossim" for dim in range(1, n_components+1, 1)])
    .assign(word=word_vector_df.index.tolist())
)

# for files greater than a 1GB
if n_components > 40:
    word_pca_sim_df.to_csv(
        f"output/word_pca_similarity/word_pca_cos_sim_{n_components}_pcs.tsv.xz", 
        sep="\t", index=False, compression="xz"
    )

else:
    word_pca_sim_df.to_csv(
        f"output/word_pca_similarity/word_pca_cos_sim_{n_components}_pcs.tsv", 
        sep="\t", index=False
    )

word_pca_sim_df.head()


# # Generate Word Clouds for the PC dimensions

# Given word to principal component association, next step is to generate word clouds for each principal component. The word clouds have orange representing words that are most similar to the principal component and blue as words most dissimilar to the principal component.

# In[9]:


class PolarityColorFunc():
    def __init__(
        self, word_class_map, 
        positive_key="positive", positive="#ef8a62",
        negative_key="negative", negative="#67a9cf", 
        default="#f7f7f7"
    ):
        self.positive_class = positive
        self.negative_class = negative
        self.positive_key = positive_key
        self.negative_key = negative_key
        self.default_class = default
        
        self.words_to_color_map = word_class_map
    
    def get_color_mapper(self, word):
        if word in self.words_to_color_map[self.positive_key]:
            return ImageColor.getrgb(self.positive_class)
        #wordcloud.get_single_color_func(self.positive_class)
        elif word in self.words_to_color_map[self.negative_key]:
            return ImageColor.getrgb(self.negative_class)
        else:
            return ImageColor.getrgb(self.default_class)
    
    def __call__(self, word, **kwargs):
        return self.get_color_mapper(word)


# In[10]:


pca_dimensions = [f"pca{dim}_cossim" for dim in range(1, n_components+1, 1)]
for component in tqdm_notebook(pca_dimensions):
    word_class_map = {}
    
    word_class_map['negative'] = (
        word_pca_sim_df
        .sort_values(component, ascending=True)
        .head(100)
        .assign(**{component:lambda x: x[component].abs().values.tolist()})
        .assign(**{component:lambda x: x[component]/x[component].max()})
        [["word", component]]
        .to_dict(orient='records')
    )

    word_class_map['positive'] = (
        word_pca_sim_df
        .sort_values(component, ascending=False)
        .assign(**{component:lambda x: x[component]/x[component].max()})
        .head(100)
        [["word", component]]
        .to_dict(orient='records')
    )
    
    polarity_color_map = PolarityColorFunc({
        word_class:set(map(lambda x: x['word'], word_class_map[word_class]))
        for word_class in word_class_map
    })
    
    polarity_cloud = (
        wordcloud
        .WordCloud(background_color="white", width=1024, height=768, collocations=False)
        .generate_from_frequencies({
            record['word']:record[component]
            for word_class in word_class_map
            for record in word_class_map[word_class]
        })
        .recolor(color_func=polarity_color_map)
        .to_file(f"output/word_pca_similarity/figures/{component}_word_cloud.png")
    )


# # Document Centroid Cosine Similarity

# Finally this section calculates document centroid to principal component associations. This means the higher score the higher the association a document is with a given principal component.

# In[11]:


document_centroid_df = (
    journal_map_df[["document", "category"]]
    .merge(biorxiv_articles_df, on="document")
    .groupby("category")
    .agg({
        f"feat_{dim}":"mean"
        for dim in range(300)
    })
    .reset_index()
)
document_centroid_df.head()


# In[12]:


# 1 - cosine distance = cosine similarity
centroid_pca_similarity = (
    1 - cdist(
        document_centroid_df[[f"feat_{dim}" for dim in range(300)]].values, 
        reducer.components_, 
        'cosine'
    )
)

centroid_pca_similarity.shape


# In[13]:


centroid_pca_df = (
    pd.DataFrame(centroid_pca_similarity, columns=[f"pca{dim}_cossim" for dim in range(1, n_components+1, 1)])
    .assign(category=document_centroid_df.category.tolist())
    [["category"] + [f"pca{dim}_cossim" for dim in range(1, n_components+1, 1)]]
)
centroid_pca_df.to_csv("output/word_pca_similarity/centroid_pca_cos_sim.tsv", sep="\t", index=False)
centroid_pca_df.head()

