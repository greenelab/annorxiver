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

# # Determine Word to PCA Associations

# This notebook is designed to run PCA over the document embeddings and calculate words-pca associations and document centroid-pca associations for each principal component.

# +
from pathlib import Path
import os
import re

from gensim.models import Doc2Vec
import itertools
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as p9
from PIL import ImageColor
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook
import wordcloud

matplotlib.use("SVG")  # set the backend to SVG
# -

journal_map_df = pd.read_csv(
    "../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t"
)
journal_map_df.head()

# # Get the Word Vectors

# Save the word vectors to disk, so later sections have easy access.

model = Doc2Vec.load(
    "../word_vector_experiment/output/doc2vec_output/biorxiv_300.model"
)

# # PCA the Documents

# Run PCA over the documents. Generates 50 principal components, but can generate more or less.

n_components = 50
random_state = 100

# +
reducer = PCA(n_components=n_components, random_state=random_state)

embedding = reducer.fit_transform(model.docvecs.vectors_docs)

pca_df = (
    pd.DataFrame(
        embedding, columns=[f"pca{dim}" for dim in range(1, n_components + 1, 1)]
    )
    .assign(document=[f"{str(tag)}.xml" for tag in model.docvecs.doctags])
    .merge(journal_map_df[["category", "document", "doi"]], on="document")
)
# -

(
    pd.DataFrame(
        reducer.components_,
        columns=[f"{dim+1}" for dim in range(reducer.components_.shape[1])],
    ).to_csv(
        "output/word_pca_similarity/pca_components_doc2vec.tsv", sep="\t", index=False
    )
)

# # Calculate Word-PCA Cosine Similarity

# Once PCA has finished, there are now 50 different principal components. The association between every word and principal component is calculated via [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) (cosine of the angle between two vectors).

# +
# 1 - cosine distance = cosine similarity
word_pca_similarity = 1 - cdist(model.wv.vectors, reducer.components_, "cosine")

word_pca_similarity.shape

# +
word_pca_sim_df = pd.DataFrame(
    word_pca_similarity,
    columns=[f"pca{dim}_cossim" for dim in range(1, n_components + 1, 1)],
).assign(word=model.wv.index2word)

word_pca_sim_df.to_csv(
    f"output/word_pca_similarity/word_pca_cos_sim_{n_components}_pcs_doc2vec.tsv",
    sep="\t",
    index=False,
)

# Remove those pesky citations from the word pca similarity
word_pca_sim_df = word_pca_sim_df.query(
    "~word.str.match(r'^(\(|\[)', na=False)"  # noqa: W605
)
word_pca_sim_df.head()


# -

# # Generate Word Clouds for the PC dimensions

# Given word to principal component association, next step is to generate word clouds for each principal component. The word clouds have orange representing words that are most similar to the principal component and blue as words most dissimilar to the principal component.

class PolarityColorFunc:
    def __init__(
        self,
        word_class_map,
        positive_key="positive",
        positive="#ef8a62",
        negative_key="negative",
        negative="#67a9cf",
        default="#f7f7f7",
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
        # wordcloud.get_single_color_func(self.positive_class)
        elif word in self.words_to_color_map[self.negative_key]:
            return ImageColor.getrgb(self.negative_class)
        else:
            return ImageColor.getrgb(self.default_class)

    def __call__(self, word, **kwargs):
        return self.get_color_mapper(word)


pca_dimensions = [f"pca{dim}_cossim" for dim in range(1, n_components + 1, 1)]
for pc, component in tqdm_notebook(enumerate(pca_dimensions, start=1)):
    word_class_map = {}

    word_class_map["negative"] = (
        word_pca_sim_df.sort_values(component, ascending=True)
        .head(100)
        .assign(**{component: lambda x: x[component].abs().values.tolist()})
        .assign(**{component: lambda x: x[component] / x[component].max()})[
            ["word", component]
        ]
        .to_dict(orient="records")
    )

    word_class_map["positive"] = (
        word_pca_sim_df.sort_values(component, ascending=False)
        .assign(**{component: lambda x: x[component] / x[component].max()})
        .head(100)[["word", component]]
        .to_dict(orient="records")
    )

    polarity_color_map = PolarityColorFunc(
        {
            word_class: set(map(lambda x: x["word"], word_class_map[word_class]))
            for word_class in word_class_map
        }
    )

    pc = f"{pc:02d}"

    polarity_cloud = (
        wordcloud.WordCloud(
            background_color="white", width=1024, height=768, collocations=False
        )
        .generate_from_frequencies(
            {
                record["word"]: record[component]
                for word_class in word_class_map
                for record in word_class_map[word_class]
            }
        )
        .recolor(color_func=polarity_color_map)
        .to_file(
            f"output/word_pca_similarity/figure_pieces/pca_{pc}_cossim_word_cloud_doc2vec.png"
        )
    )
