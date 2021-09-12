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

biorxiv_xpath_str = "//abstract/p|//abstract/title|//abstract/sec/*"

# +
word_model_path = list(Path().rglob("output/word2vec_models/300/*.model"))[0]
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
    f"output/word2vec_output/biorxiv_all_articles_{model_dim}_abstract_only.tsv.xz",
    sep="\t",
    index=False,
    compression="xz",
)
