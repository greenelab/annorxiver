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

# This notebook is designed to generate document embeddings for every article in bioRxiv.

# +
from pathlib import Path
import re
import sys

from gensim.models import Word2Vec
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm_notebook
import umap

from annorxiver_modules.document_helper import (
    generate_doc_vector,
    DocIterator,
    dump_article_text,
)
# -

journal_map_df = pd.read_csv(
    "../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t"
)
journal_map_df.head()

biorxiv_xpath_str = (
    "//abstract/p|//abstract/title|//body/sec//p|//body/sec//title|//body/p"
)

# # Output Documents to File

# This section dumps all of biorxiv text into a single document in order to train the word2vec model. This is for ease of training the model.

# +
# Only use the most current version of the documents
latest_journal_version = journal_map_df.groupby("doi").agg(
    {"document": "first", "doi": "last"}
)

if not Path("output/word2vec_input/biorxiv_text.txt").exists():
    with open("output/word2vec_input/biorxiv_text.txt", "w") as f:
        for article in tqdm_notebook(latest_journal_version.document.tolist()):
            document_text = dump_article_text(
                file_path=f"../biorxiv_articles/{article}",
                xpath_str=biorxiv_xpath_str,
                remove_stop_words=True,
            )

            f.write("\n".join(document_text))
            f.write("\n\n")
# -

# # Train Word2Vec

# This section trains the word2vec model (continuous bag of words [CBOW]). Since the number of dimensions can vary I decided to train multiple models: 150, 250, 300. Each model is saved into is own respective directory.

word_embedding_sizes = [150, 250, 300]
for size in word_embedding_sizes:

    # Create save path
    word_path = Path(f"output/word2vec_models/{size}")
    word_path.mkdir(parents=True, exist_ok=True)

    # If model exists don't run again
    if Path(f"{str(word_path.resolve())}/biorxiv_{size}.model").exists():
        continue

    # Run Word2Vec
    words = Word2Vec(
        DocIterator("output/word2vec_input/biorxiv_text.txt"),
        size=size,
        iter=20,
        seed=100,
    )

    # Save the model for future use
    words.save(f"{str(word_path.resolve())}/biorxiv_{size}.model")

# # Generate Document Embeddings

# After training the word2vec models, the next step is to generate a document embeddings. For this experiment each document embedding is generated via an average of all word vectors contained in the document. There are better approaches towards doing this, but this can serve as a simple baseline.

for word_model_path in Path().rglob("output/word2vec_models/*/*.model"):
    model_dim = word_model_path.parents[0].stem
    word_model = Word2Vec.load(str(word_model_path.resolve()))

    biorxiv_document_map = {
        document: generate_doc_vector(
            word_model,
            document_path=f"../biorxiv_articles/{document}",
            xpath=biorxiv_xpath_str,
        )
        for document in tqdm_notebook(journal_map_df.document.tolist())
    }

    biorxiv_vec_df = (
        pd.DataFrame.from_dict(biorxiv_document_map, orient="index")
        .rename(columns={col: f"feat_{col}" for col in range(int(model_dim))})
        .rename_axis("document")
        .reset_index()
    )

    biorxiv_vec_df.to_csv(
        f"output/word2vec_output/biorxiv_all_articles_{model_dim}.tsv.xz",
        sep="\t",
        index=False,
        compression="xz",
    )

# # UMAP the Documents

# After generating document embeddings, the next step is to visualize all the documents. In order to visualize the embeddings a low dimensional representation is needed. UMAP is an algorithm that can generate this representation, while grouping similar embeddings together.

random_state = 100
n_neighbors = journal_map_df.category.unique().shape[0]
n_components = 2

for biorxiv_doc_vectors in Path().rglob(
    "output/word2vec_output/biorxiv_all_articles*.tsv.xz"
):
    model_dim = int(re.search(r"(\d+)", biorxiv_doc_vectors.stem).group(1))
    biorxiv_articles_df = pd.read_csv(str(biorxiv_doc_vectors.resolve()), sep="\t")

    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, random_state=random_state
    )

    embedding = reducer.fit_transform(
        biorxiv_articles_df[[f"feat_{idx}" for idx in range(model_dim)]].values
    )

    umapped_df = (
        pd.DataFrame(embedding, columns=["umap1", "umap2"])
        .assign(document=biorxiv_articles_df.document.values.tolist())
        .merge(journal_map_df[["category", "document", "doi"]], on="document")
    )

    umapped_df.to_csv(
        f"output/embedding_output/umap/biorxiv_umap_{model_dim}.tsv",
        sep="\t",
        index=False,
    )

# # TSNE the Documents

# After generating document embeddings, the next step is to visualize all the documents. In order to visualize the embeddings a low dimensional representation is needed. TSNE is an another algorithm (besides UMAP) that can generate this representation, while grouping similar embeddings together.

n_components = 2
random_state = 100

for biorxiv_doc_vectors in Path().rglob(
    "output/word2vec_output/biorxiv_all_articles*.tsv.xz"
):
    model_dim = int(re.search(r"(\d+)", biorxiv_doc_vectors.stem).group(1))
    biorxiv_articles_df = pd.read_csv(str(biorxiv_doc_vectors.resolve()), sep="\t")

    reducer = TSNE(n_components=n_components, random_state=random_state)

    embedding = reducer.fit_transform(
        biorxiv_articles_df[[f"feat_{idx}" for idx in range(model_dim)]].values
    )

    tsne_df = (
        pd.DataFrame(embedding, columns=["tsne1", "tsne2"])
        .assign(document=biorxiv_articles_df.document.values.tolist())
        .merge(journal_map_df[["category", "document", "doi"]], on="document")
    )

    tsne_df.to_csv(
        f"output/embedding_output/tsne/biorxiv_tsne_{model_dim}.tsv",
        sep="\t",
        index=False,
    )

# # PCA the Documents

n_components = 2
random_state = 100

for biorxiv_doc_vectors in Path().rglob(
    "output/word2vec_output/biorxiv_all_articles*.tsv.xz"
):
    model_dim = int(re.search(r"(\d+)", biorxiv_doc_vectors.stem).group(1))
    biorxiv_articles_df = pd.read_csv(str(biorxiv_doc_vectors.resolve()), sep="\t")

    reducer = PCA(n_components=n_components, random_state=random_state)

    embedding = reducer.fit_transform(
        biorxiv_articles_df[[f"feat_{idx}" for idx in range(model_dim)]].values
    )

    pca_df = (
        pd.DataFrame(embedding, columns=["pca1", "pca2"])
        .assign(document=biorxiv_articles_df.document.values.tolist())
        .merge(journal_map_df[["category", "document", "doi"]], on="document")
    )

    pca_df.to_csv(
        f"output/embedding_output/pca/biorxiv_pca_{model_dim}.tsv",
        sep="\t",
        index=False,
    )
