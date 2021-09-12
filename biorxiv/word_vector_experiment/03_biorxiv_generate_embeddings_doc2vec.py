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

from gensim.models.doc2vec import Doc2Vec
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook
import umap

from annorxiver_modules.document_helper import TaggedDocIterator
# -

journal_map_df = pd.read_csv(
    "../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t"
)
journal_map_df.head()

biorxiv_xpath_str = (
    "//abstract/p|//abstract/title|//body/sec//p|//body/sec//title|//body/p"
)

# # Train Doc2Vec

# This section trains the word2vec model (continuous bag of words [CBOW]). Since the number of dimensions can vary I decided to train multiple models: 150, 250, 300. Each model is saved into is own respective directory.

word_embedding_sizes = [300]
for size in word_embedding_sizes:

    # Create save path
    word_path = Path("output/doc2vec_output")
    word_path.mkdir(parents=True, exist_ok=True)

    # If model exists don't run again
    if Path(f"{str(word_path.resolve())}/biorxiv_{size}.model").exists():
        continue

    # Run Word2Vec
    doc_model = Doc2Vec(
        TaggedDocIterator(
            list(Path("../biorxiv_articles").rglob("*xml")), biorxiv_xpath_str
        ),
        size=size,
        epochs=20,
        seed=100,
        workers=4,
    )

    # Save the model for future use
    doc_model.save(f"{str(word_path.resolve())}/biorxiv_{size}.model")

doc_model = Doc2Vec.load(f"{str(word_path.resolve())}/biorxiv_{size}.model")

# # PCA the Documents

n_components = 2
random_state = 100

# +
reducer = PCA(n_components=n_components, random_state=random_state)

embedding = reducer.fit_transform(doc_model.docvecs.vectors_docs)

pca_df = (
    pd.DataFrame(embedding, columns=["pca1", "pca2"])
    .assign(document=[f"{str(tag)}.xml" for tag in doc_model.docvecs.doctags])
    .merge(journal_map_df[["category", "document", "doi"]], on="document")
)

pca_df.to_csv(
    "output/embedding_output/pca/biorxiv_pca_300_doc2vec.tsv",
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
embedding = reducer.fit_transform(doc_model.docvecs.vectors_docs)

umapped_df = (
    pd.DataFrame(embedding, columns=["umap1", "umap2"])
    .assign(document=[f"{str(tag)}.xml" for tag in doc_model.docvecs.doctags])
    .merge(journal_map_df[["category", "document", "doi"]], on="document")
)

umapped_df.to_csv(
    "output/embedding_output/umap/biorxiv_umap_300_doc2vec.tsv",
    sep="\t",
    index=False,
)
