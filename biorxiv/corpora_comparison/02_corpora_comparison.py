# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:annorxiver]
#     language: python
#     name: conda-env-annorxiver-py
# ---

# %% [markdown]
# # Comparative Linguistic Analysis of bioRxiv and PMC

# %%
# %load_ext autoreload
# %autoreload 2

from collections import defaultdict, Counter
import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import spacy
from scipy.stats import chi2_contingency
from tqdm import tqdm_notebook

from annorxiver_modules.corpora_comparison_helper import (
    aggregate_word_counts,
    dump_to_dataframe,
    get_term_statistics,
    KL_divergence,
)

# %% [markdown]
# # Full Text Comparison (Global)

# %% [markdown]
# ## Gather Word Frequencies

# %%
biorxiv_count_path = Path("output/total_word_counts/biorxiv_total_count.tsv")
pmc_count_path = Path("output/total_word_counts/pmc_total_count.tsv")
nytac_count_path = Path("output/total_word_counts/nytac_total_count.tsv")

# %%
if not biorxiv_count_path.exists():
    biorxiv_corpus_count = aggregate_word_counts(
        list(Path("output/biorxiv_word_counts").rglob("*tsv"))
    )
    dump_to_dataframe(biorxiv_corpus_count, "output/biorxiv_total_count.tsv")
    biorxiv_corpus_count.most_common(10)

# %%
if not pmc_count_path.exists():
    pmc_corpus_count = aggregate_word_counts(
        list(Path("../../pmc/pmc_corpus/pmc_word_counts").rglob("*tsv"))
    )
    dump_to_dataframe(pmc_corpus_count, "output/pmc_total_count.tsv")
    pmc_corpus_count.most_common(10)

# %%
if not nytac_count_path.exists():
    nytac_corpus_count = aggregate_word_counts(
        list(Path("../../nytac/corpora_stats/output").rglob("*tsv"))
    )
    dump_to_dataframe(nytac_corpus_count, "output/nytac_total_count.tsv")
    nytac_corpus_count.most_common(10)

# %%
biorxiv_total_count_df = pd.read_csv(biorxiv_count_path.resolve(), sep="\t")

pmc_total_count_df = pd.read_csv(pmc_count_path.resolve(), sep="\t")

nytac_total_count_df = pd.read_csv(nytac_count_path.resolve(), sep="\t")

# %%
biorxiv_sentence_length = pickle.load(open("output/biorxiv_sentence_length.pkl", "rb"))
pmc_sentence_length = pickle.load(
    open("../../pmc/pmc_corpus/pmc_sentence_length.pkl", "rb")
)
nytac_sentence_length = pickle.load(
    open("../../nytac/corpora_stats/nytac_sentence_length.pkl", "rb")
)

# %%
spacy_nlp = spacy.load("en_core_web_sm")
stop_word_list = list(spacy_nlp.Defaults.stop_words)

# %% [markdown]
# ## Get Corpora Comparison Stats

# %%
biorxiv_sentence_len_list = list(biorxiv_sentence_length.items())
biorxiv_data = {
    "document_count": len(biorxiv_sentence_length),
    "sentence_count": sum(map(lambda x: len(x[1]), biorxiv_sentence_len_list)),
    "token_count": biorxiv_total_count_df["count"].sum(),
    "stop_word_count": (
        biorxiv_total_count_df.query(f"lemma in {stop_word_list}")["count"].sum()
    ),
    "avg_document_length": np.mean(
        list(map(lambda x: len(x[1]), biorxiv_sentence_len_list))
    ),
    "avg_sentence_length": np.mean(
        list(itertools.chain(*list(map(lambda x: x[1], biorxiv_sentence_len_list))))
    ),
    "negatives": (biorxiv_total_count_df.query(f"dep_tag =='neg'")["count"].sum()),
    "coordinating_conjunctions": (
        biorxiv_total_count_df.query(f"dep_tag =='cc'")["count"].sum()
    ),
    "coordinating_conjunctions%": (
        biorxiv_total_count_df.query(f"dep_tag =='cc'")["count"].sum()
    )
    / biorxiv_total_count_df["count"].sum(),
    "pronouns": (biorxiv_total_count_df.query(f"pos_tag =='PRON'")["count"].sum()),
    "pronouns%": (biorxiv_total_count_df.query(f"pos_tag =='PRON'")["count"].sum())
    / biorxiv_total_count_df["count"].sum(),
    "passives": (
        biorxiv_total_count_df.query(
            f"dep_tag in ['auxpass', 'nsubjpass', 'csubjpass']"
        )["count"].sum()
    ),
    "passive%": (
        biorxiv_total_count_df.query(
            f"dep_tag in ['auxpass', 'nsubjpass', 'csubjpass']"
        )["count"].sum()
    )
    / biorxiv_total_count_df["count"].sum(),
}

# %%
pmc_sentence_len_list = list(pmc_sentence_length.items())
pmc_data = {
    "document_count": len(pmc_sentence_length),
    "sentence_count": sum(map(lambda x: len(x[1]), pmc_sentence_len_list)),
    "token_count": pmc_total_count_df["count"].sum(),
    "stop_word_count": (
        pmc_total_count_df.query(f"lemma in {stop_word_list}")["count"].sum()
    ),
    "avg_document_length": np.mean(
        list(map(lambda x: len(x[1]), pmc_sentence_len_list))
    ),
    "avg_sentence_length": np.mean(
        list(itertools.chain(*list(map(lambda x: x[1], pmc_sentence_len_list))))
    ),
    "negatives": (pmc_total_count_df.query(f"dep_tag =='neg'")["count"].sum()),
    "coordinating_conjunctions": (
        pmc_total_count_df.query(f"dep_tag =='cc'")["count"].sum()
    ),
    "coordinating_conjunctions%": (
        pmc_total_count_df.query(f"dep_tag =='cc'")["count"].sum()
    )
    / pmc_total_count_df["count"].sum(),
    "pronouns": (pmc_total_count_df.query(f"pos_tag =='PRON'")["count"].sum()),
    "pronouns%": (pmc_total_count_df.query(f"pos_tag =='PRON'")["count"].sum())
    / pmc_total_count_df["count"].sum(),
    "passives": (
        pmc_total_count_df.query(f"dep_tag in ['auxpass', 'nsubjpass', 'csubjpass']")[
            "count"
        ].sum()
    ),
    "passive%": (
        pmc_total_count_df.query(f"dep_tag in ['auxpass', 'nsubjpass', 'csubjpass']")[
            "count"
        ].sum()
    )
    / pmc_total_count_df["count"].sum(),
}

# %%
nytac_sentence_len_list = list(nytac_sentence_length.items())
nytac_data = {
    "document_count": len(nytac_sentence_length),
    "sentence_count": sum(map(lambda x: len(x[1]), nytac_sentence_len_list)),
    "token_count": nytac_total_count_df["count"].sum(),
    "stop_word_count": (
        nytac_total_count_df.query(f"lemma in {stop_word_list}")["count"].sum()
    ),
    "avg_document_length": np.mean(
        list(map(lambda x: len(x[1]), nytac_sentence_len_list))
    ),
    "avg_sentence_length": np.mean(
        list(itertools.chain(*list(map(lambda x: x[1], nytac_sentence_len_list))))
    ),
    "negatives": (nytac_total_count_df.query(f"dep_tag =='neg'")["count"].sum()),
    "coordinating_conjunctions": (
        nytac_total_count_df.query(f"dep_tag =='cc'")["count"].sum()
    ),
    "coordinating_conjunctions%": (
        nytac_total_count_df.query(f"dep_tag =='cc'")["count"].sum()
    )
    / nytac_total_count_df["count"].sum(),
    "pronouns": (nytac_total_count_df.query(f"pos_tag =='PRON'")["count"].sum()),
    "pronouns%": (nytac_total_count_df.query(f"pos_tag =='PRON'")["count"].sum())
    / nytac_total_count_df["count"].sum(),
    "passives": (
        nytac_total_count_df.query(f"dep_tag in ['auxpass', 'nsubjpass', 'csubjpass']")[
            "count"
        ].sum()
    ),
    "passive%": (
        nytac_total_count_df.query(f"dep_tag in ['auxpass', 'nsubjpass', 'csubjpass']")[
            "count"
        ].sum()
    )
    / nytac_total_count_df["count"].sum(),
}

# %%
# This dataframe contains document statistics for each Corpus
# document count - the number of documents within the corpus
# Sentence count - the number of sentences within the corpus
# Token count - the number of tokens within the corpus
# Stop word counts - the number of stop words within the corpus
# Average document length - the average number of sentences within a document for a given corpus
# Average sentence length - the average number of words within a sentence for a given corpus
# Negatives - the number of negations (e.g. placing not in within a sentence) within a given corpus
# Coordinating Conjunctions - the number of coordinating conjunctions (and, but, for etc.) within a given corpus
# Pronouns - the number of pronouns within a given corpus
# Passive - the number of passive words within a given corpus

token_stats_df = pd.DataFrame.from_records(
    [biorxiv_data, pmc_data, nytac_data], index=["bioRxiv", "PMC", "NYTAC"]
).T
token_stats_df.to_csv("output/figures/corpora_token_stats.tsv", sep="\t")
token_stats_df

# %% [markdown]
# ## LogLikelihood + Odds Ratio + KL Divergence Calculations

# %% [markdown]
# The goal here is to compare word frequencies between bioRxiv and pubmed central. The problem when comparing word frequencies is that non-meaningful words (aka stopwords) such as the, of, and, be, etc., appear the most often. To account for this problem the first step here is to remove those words from analyses.

# %% [markdown]
# ### Remove Stop words

# %%
biorxiv_total_count_df = (
    biorxiv_total_count_df.query(f"lemma not in {stop_word_list}")
    .groupby("lemma")
    .agg({"count": "sum"})
    .reset_index()
    .sort_values("count", ascending=False)
)
biorxiv_total_count_df

# %%
pmc_total_count_df = (
    pmc_total_count_df.query(f"lemma not in {stop_word_list}")
    .groupby("lemma")
    .agg({"count": "sum"})
    .reset_index()
    .sort_values("count", ascending=False)
    .iloc[2:]
)
pmc_total_count_df

# %%
nytac_total_count_df = (
    nytac_total_count_df.query(f"lemma not in {stop_word_list}")
    .groupby("lemma")
    .agg({"count": "sum"})
    .reset_index()
    .sort_values("count", ascending=False)
)
nytac_total_count_df

# %% [markdown]
# ### Calculate LogLikelihoods and Odds ratios

# %%
biorxiv_vs_pmc = get_term_statistics(biorxiv_total_count_df, pmc_total_count_df, 100)

biorxiv_vs_pmc.to_csv(
    "output/comparison_stats/biorxiv_vs_pmc_comparison.tsv", sep="\t", index=False
)

biorxiv_vs_pmc

# %%
biorxiv_vs_nytac = get_term_statistics(
    biorxiv_total_count_df, nytac_total_count_df, 100
)
biorxiv_vs_nytac.to_csv(
    "output/comparison_stats/biorxiv_nytac_comparison.tsv", sep="\t", index=False
)
biorxiv_vs_nytac

# %%
pmc_vs_nytac = get_term_statistics(pmc_total_count_df, nytac_total_count_df, 100)

pmc_vs_nytac.to_csv(
    "output/comparison_stats/pmc_nytac_comparison.tsv", sep="\t", index=False
)

pmc_vs_nytac

# %% [markdown]
# ## Calculate KL Divergence

# %%
term_grid = [100, 200, 300, 400, 500, 1000, 1500, 2000, 3000, 5000]
kl_data = []
for num_terms in tqdm_notebook(term_grid):
    kl_data.append(
        {
            "num_terms": num_terms,
            "KL_divergence": KL_divergence(
                biorxiv_total_count_df, pmc_total_count_df, num_terms=num_terms
            ),
            "comparison": "biorxiv_vs_pmc",
        }
    )

    kl_data.append(
        {
            "num_terms": num_terms,
            "KL_divergence": KL_divergence(
                biorxiv_total_count_df, nytac_total_count_df, num_terms=num_terms
            ),
            "comparison": "biorxiv_vs_nytac",
        }
    )

    kl_data.append(
        {
            "num_terms": num_terms,
            "KL_divergence": KL_divergence(
                pmc_total_count_df, nytac_total_count_df, num_terms=num_terms
            ),
            "comparison": "pmc_vs_nytac",
        }
    )

# %%
kl_metrics = pd.DataFrame.from_records(kl_data)
kl_metrics.to_csv(
    "output/comparison_stats/corpora_kl_divergence.tsv", sep="\t", index=False
)
kl_metrics

# %% [markdown]
# # Preprint to Published View

# %%
mapped_doi_df = (
    pd.read_csv("../journal_tracker/output/mapped_published_doi.tsv", sep="\t")
    .query("published_doi.notnull()")
    .query("pmcid.notnull()")
    .groupby("preprint_doi")
    .agg(
        {
            "author_type": "first",
            "heading": "first",
            "category": "first",
            "document": "first",
            "preprint_doi": "last",
            "published_doi": "last",
            "pmcid": "last",
        }
    )
    .reset_index(drop=True)
)
mapped_doi_df.tail()

# %%
print(f"Total # of Preprints Mapped: {mapped_doi_df.shape[0]}")
print(f"Total % of Mapped: {mapped_doi_df.shape[0]/71118}")

# %%
preprint_count = aggregate_word_counts(
    [
        Path(f"output/biorxiv_word_counts/{Path(file)}.tsv")
        for file in mapped_doi_df.document.values.tolist()
        if Path(f"output/biorxiv_word_counts/{Path(file)}.tsv").exists()
    ]
)

preprint_count_df = (
    pd.DataFrame.from_records(
        [
            {
                "lemma": token[0],
                "pos_tag": token[1],
                "dep_tag": token[2],
                "count": preprint_count[token],
            }
            for token in preprint_count
        ]
    )
    .query(f"lemma not in {stop_word_list}")
    .groupby("lemma")
    .agg({"count": "sum"})
    .reset_index()
    .sort_values("count", ascending=False)
)

preprint_count_df.head()

# %%
published_count = aggregate_word_counts(
    [
        Path(f"../../pmc/pmc_corpus/pmc_word_counts/{file}.tsv")
        for file in mapped_doi_df.pmcid.values.tolist()
        if Path(f"../../pmc/pmc_corpus/pmc_word_counts/{file}.tsv").exists()
    ]
)

published_count_df = (
    pd.DataFrame.from_records(
        [
            {
                "lemma": token[0],
                "pos_tag": token[1],
                "dep_tag": token[2],
                "count": published_count[token],
            }
            for token in published_count
        ]
    )
    .query(f"lemma not in {stop_word_list}")
    .groupby("lemma")
    .agg({"count": "sum"})
    .reset_index()
    .sort_values("count", ascending=False)
)

published_count_df.head()

# %%
preprint_vs_published = get_term_statistics(preprint_count_df, published_count_df, 100)

preprint_vs_published.to_csv(
    "output/comparison_stats/preprint_to_published_comparison.tsv",
    sep="\t",
    index=False,
)

preprint_vs_published

# %% [markdown]
# Main takeaways from this analysis:
# 1. On a global scale bioRxiv contains more field specific articles as top words consist of: neuron, gene, genome, network
# 2. "Patients" appear more correlated with PMC as most preprints involving patients are shipped over to medRxiv.
# 3. Many words associated with PMC are health related which ties back to the medRxiv note.
# 4. Citation styles change as preprints transition to published versions. Et Al. has a greater association within bioRxiv compared to PMC.
# 5. On a local scale published articles contain more statistical concepts (e.g., t-test) as well as quantitative measures (e.g. degree signs). (High associated lemmas are t, -, degree sign etc.)
# 6. Publish articles have a focus shift on mentioning figures, adding supplementary data etc compared to preprints.
# 7. Preprints have a universal way of citing published works by using the et al. citation. Hard to pinpoint if leading factor is because of peer review or journal style, but it will be an interesting point to discuss in the paper.
