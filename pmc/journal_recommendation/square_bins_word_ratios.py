#!/usr/bin/env python
# coding: utf-8

# # Calculate Odds Ratios for each Square Bin

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from collections import Counter
import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm_notebook

from annorxiver_modules.corpora_comparison_helper import (
    aggregate_word_counts,
    get_term_statistics
)


# # Gather Paper Bins Dataframe

# In[2]:


pmc_df = pd.read_csv(
    "output/paper_dataset/paper_dataset_tsne_square.tsv",
    sep="\t"
)
pmc_df.head()


# In[3]:


word_count_folder = Path("../pmc_corpus/pmc_word_counts/")


# In[4]:


bin_group = pmc_df.groupby("squarebin_id")


# In[5]:


spacy_nlp = spacy.load('en_core_web_sm')
stop_word_list = list(spacy_nlp.Defaults.stop_words)


# In[6]:


global_word_counter = Counter()
for name, group in tqdm_notebook(bin_group):
    files = [
        f"{word_count_folder.resolve()}/{doc}.tsv"
        for doc in group.document.tolist()
    ]
    
    agg_word_count = aggregate_word_counts(files, disable_progressbar=True)
    
    filtered_agg_word_count = {
       term[0]:agg_word_count[term] 
        for term in agg_word_count 
        if term[1] != 'SPACE' and term[0] not in stop_word_list
    }
    
    global_word_counter.update(Counter(filtered_agg_word_count))


# In[8]:


for bin_id, group in tqdm_notebook(bin_group):
    files = [
        f"{word_count_folder.resolve()}/{doc}.tsv"
        for doc in group.document.tolist()
    ]
    
    agg_word_count = aggregate_word_counts(files, disable_progressbar=True)
    
    filtered_agg_word_count = {
       term[0]:agg_word_count[term] 
        for term in agg_word_count 
        if term[1] != 'SPACE' and "\\" not in repr(term[0]) and term[0] not in stop_word_list
    }
    
    bin_counter = Counter(filtered_agg_word_count)
    remaining_words = (
        Counter({
            term:global_word_counter[term] 
            for term in filtered_agg_word_count
        })  - bin_counter
    )

    bin_df = (
        pd.DataFrame.from_dict(
            dict(bin_counter),
            orient="index",
            columns=["count"]
        )
        .rename_axis("lemma")
        .reset_index()
    )
    
    background_df = (
        pd.DataFrame.from_dict(
            {
                key:remaining_words[key]
                for key in bin_counter
                if key in remaining_words
            },
            orient="index",
            columns=["count"]
        )
        .rename_axis("lemma")
        .reset_index()
    )
    
    # Calculate the odds ratio
    word_odds_df = get_term_statistics(
        bin_df,
        background_df,
        100, 
        psudeocount=1,
        disable_progressbar=True
    )
    
    (
        word_odds_df
        .to_csv(
            f"output/word_odds/word_odds_bin_{bin_id}.tsv", 
            sep="\t", index=False
        )
    )


# # Insert Bin Word Associations in JSON File

# In[2]:


square_bin_plot_df = pd.read_json(
    open(
        Path("output")/
        Path("app_plots")/
        Path("pmc_square_plot.json")
    )
)
square_bin_plot_df.head()


# In[3]:


lemma_bin_records = []
for bin_id in tqdm_notebook(square_bin_plot_df.bin_id.tolist()):
    bin_assoc_df = pd.read_csv(
        f"output/word_odds/word_odds_bin_{bin_id}.tsv",
        sep="\t"
    )
    
    high_odds_words = (
        bin_assoc_df
        .sort_values("odds_ratio", ascending=False)
        .head(15)
        [["lemma", "odds_ratio"]]
    )
      
    lemma_bin_records.append([
        {
            "lemma": pair[0],
            "odds_ratio": pair[1]
        }
        for pair in zip(high_odds_words.lemma, high_odds_words.odds_ratio)
    ])


# In[8]:


(
    square_bin_plot_df
    .assign(bin_odds=lemma_bin_records)
    .to_json(
        Path("output")/
        Path("app_plots")/
        Path("pmc_square_plot.json"),
        orient = 'records',
        lines = False
    )
)

