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

# # Generate BioRxiv Document Embeddings

# This notebook is designed to generate document embeddings for every article in bioRxiv. After submitting my manuscript to PLOS Biology, I got a comment on why I chose not to use Doc2Vec to generate document vectors. With that being said this notebook will explore using Doc2Vec on bioRxiv to see if any of my results change.

# +
from pathlib import Path
import re
import sys

import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook
import umap

from annorxiver_modules.document_helper import DocIterator, generate_doc_vector
# -

journal_map_df = pd.read_csv(
    "../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t"
)
journal_map_df.head()

biorxiv_xpath_str = "//abstract/p|//abstract/title|//abstract/sec/*|//body/sec//p|//body/sec//title|//body/p"

model = api.load("word2vec-google-news-300")
model.save_word2vec_format("output/pretrained_output/temp_model.bin", binary=True)

# # Train Doc2Vec

# This section trains the word2vec model (continuous bag of words [CBOW]). Since the number of dimensions can vary I decided to train multiple models: 150, 250, 300. Each model is saved into is own respective directory.

# +
word_embedding_sizes = [300]
doc_iterator = DocIterator("output/word2vec_input/biorxiv_text.txt")
for size in word_embedding_sizes:

    # Create save path
    word_path = Path("output/pretrained_output")
    word_path.mkdir(parents=True, exist_ok=True)

    # If model exists don't run again
    if Path(f"{str(word_path.resolve())}/biorxiv_{size}.model").exists():
        continue

    # Create model with biorxiv
    model = Word2Vec(size=300, min_count=1)
    model.build_vocab(doc_iterator)

    # inject model with pretrained vectors
    model.intersect_word2vec_format(
        "output/pretrained_output/temp_model.bin", binary=True, lockf=1.0
    )

    # Run Word2Vec
    model.train(doc_iterator, epochs=20, total_examples=model.corpus_count)

# Save the model for future use
model.save(f"{str(word_path.resolve())}/biorxiv_{size}_pretrained.model")
# -

model = Word2Vec.load(f"{str(word_path.resolve())}/biorxiv_{size}_pretrained.model")

if not Path(
    "output/word2vec_output/biorxiv_all_articles_300_pretrained.tsv.xz"
).exists():
    biorxiv_document_map = {
        document: generate_doc_vector(
            model,
            document_path=f"../biorxiv_articles/{document}",
            xpath=biorxiv_xpath_str,
        )
        for document in tqdm_notebook(journal_map_df.document.tolist())
    }

    biorxiv_vec_df = (
        pd.DataFrame.from_dict(biorxiv_document_map, orient="index")
        .rename(columns={col: f"feat_{col}" for col in range(int(300))})
        .rename_axis("document")
        .reset_index()
    )

    biorxiv_vec_df.to_csv(
        "output/word2vec_output/biorxiv_all_articles_300_pretrained.tsv.xz",
        sep="\t",
        index=False,
        compression="xz",
    )

# # PCA the Documents

n_components = 2
random_state = 100

# +
reducer = PCA(n_components=n_components, random_state=random_state)

embedding = reducer.fit_transform(
    biorxiv_vec_df.dropna()[[f"feat_{idx}" for idx in range(300)]].values
)

pca_df = (
    pd.DataFrame(embedding, columns=["pca1", "pca2"])
    .assign(document=[str(tag) for tag in biorxiv_vec_df.dropna().document.values])
    .merge(journal_map_df[["category", "document", "doi"]], on="document")
)

pca_df.to_csv(
    "output/embedding_output/pca/biorxiv_pca_300_pretrained.tsv",
    sep="\t",
    index=False,
)
# -

# # UMAP the Documents

# After generating document embeddings, the next step is to visualize all the documents. In order to visualize the embeddings a low dimensional representation is needed. UMAP is an algorithm that can generate this representation, while grouping similar embeddings together.

random_state = 100
n_neighbors = journal_map_df.category.unique().shape[0]
n_components = 2

# +
reducer = umap.UMAP(
    n_components=n_components, n_neighbors=n_neighbors, random_state=random_state
)

# Doc2vec already has document vectors
embedding = reducer.fit_transform(
    biorxiv_vec_df.dropna()[[f"feat_{idx}" for idx in range(300)]].values
)

umapped_df = (
    pd.DataFrame(embedding, columns=["umap1", "umap2"])
    .assign(document=[str(tag) for tag in biorxiv_vec_df.dropna().document.values])
    .merge(journal_map_df[["category", "document", "doi"]], on="document")
)

umapped_df.to_csv(
    "output/embedding_output/umap/biorxiv_umap_300_pretrained.tsv",
    sep="\t",
    index=False,
)
