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

# # Get Token Counts and Word Vectors

# This notebook is designed to calculate token frequencies for each token within processed preprints. Plus, this notebook generates document vector representations for preprints to be analyzed downstream of the pipeline.

# +
import csv
from pathlib import Path

from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm_notebook

from annorxiver_modules.corpora_comparison_helper import get_word_stats
from annorxiver_modules.document_helper import generate_doc_vector
# -

# # BioRxiv

mapped_documents_df = pd.read_csv("output/polka_et_al_pmc_mapped_subset.tsv", sep="\t")
mapped_documents_df.head()

biorxiv_documents = [
    Path(x.name) for x in list(Path("output/biorxiv_xml_files").rglob("*xml"))
]

# ## BioRxiv -> Term counts

Path("output/biorxiv_word_counts").mkdir(exist_ok=True)
sentence_length = get_word_stats(
    document_list=biorxiv_documents,
    document_folder="output/biorxiv_xml_files",
    tag_path="//abstract/p|//abstract/title|//body/sec//p|//body/sec//title",
    output_folder="output/biorxiv_word_counts",
)

# ## BioRxiv -> Doc Embeddings

biorxiv_xpath_str = "//abstract/p|//abstract/title|//body/sec//p|//body/sec//title"
word_model = Word2Vec.load(
    str(Path("../word_vector_experiment/output/word2vec_models/300/biorxiv_300.model"))
)

biorxiv_document_map = {
    document: generate_doc_vector(
        word_model,
        document_path=str(Path("output/biorxiv_xml_files") / document),
        xpath=biorxiv_xpath_str,
    )
    for document in tqdm_notebook(biorxiv_documents)
}

# +
biorxiv_vec_df = (
    pd.DataFrame.from_dict(biorxiv_document_map, orient="index")
    .rename(columns={col: f"feat_{col}" for col in range(int(300))})
    .rename_axis("document")
    .reset_index()
)

biorxiv_vec_df.to_csv(
    "output/polka_et_al_biorxiv_embeddings.tsv", sep="\t", index=False
)

biorxiv_vec_df.head().T
# -

# # PMCOA

pmcoa_documents = [
    Path(f"{x.parent.stem}/{x.name}")
    for x in list(Path("output/pmcoa_xml_files").rglob("*nxml"))
]

# ## PMCOA -> Term counts

Path("output/pmcoa_word_counts").mkdir(exist_ok=True)
sentence_length = get_word_stats(
    document_list=pmcoa_documents,
    document_folder="output/pmcoa_xml_files",
    tag_path="//abstract/sec/*|//abstract/p|//body/sec/*|//body/p",
    output_folder="output/pmcoa_word_counts",
)

# ## PMCOA -> Doc Vectors

pmcoa_vec_map = {
    document.stem: generate_doc_vector(
        word_model,
        str(Path("output/pmcoa_xml_files") / Path(document)),
        "//abstract/sec/*|//abstract/p|//body/sec/*|//body/p",
    )
    for document in pmcoa_documents
}

# +
pmcoa_vec_df = (
    pd.DataFrame.from_dict(pmcoa_vec_map, orient="index")
    .rename(columns={col: f"feat_{col}" for col in range(int(300))})
    .rename_axis("document")
    .reset_index()
)

pmcoa_vec_df.to_csv("output/polka_et_al_pmcoa_embeddings.tsv", sep="\t", index=False)

pmcoa_vec_df.head().T
