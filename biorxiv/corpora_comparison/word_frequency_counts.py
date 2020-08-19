#!/usr/bin/env python
# coding: utf-8

# # Compare Pubmed Central Corpus with bioRxiv Corpus

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from collections import Counter
import csv
from pathlib import Path
import pickle
import string

import pandas as pd
import spacy
from tqdm import tqdm

from annorxiver_modules.document_helper import dump_article_text
from annorxiver_modules.corpora_comparison_helper import get_word_stats


# In[2]:


lemma_model = spacy.load("en_core_web_sm")
lemma_model.max_length = 9000000 


# # Calculate Word Frequency of bioRxiv

# In[3]:


biorxiv_map_df = (
    pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
    .groupby("doi")
    .agg({"document":"first", "doi":"last"})
)
print(biorxiv_map_df.shape)
biorxiv_map_df.head()


# In[4]:


Path("output/biorxiv_word_counts/").mkdir(parents=True, exist_ok=True)


# In[5]:


sentence_length = get_word_stats(
    document_list=biorxiv_map_df.document.tolist(),
    document_folder="output/biorxiv_word_counts/",
    tag_path="//abstract/p|//abstract/title|//body/sec//p|//body/sec//title",
    output_folder="output/biorxiv_word_counts/"
)


# In[6]:


pickle.dump(
    sentence_length, 
    open("output/biorxiv_sentence_length.pkl", "wb")
)


# # Calculate Word Frequency of Pubmed Central

# In[3]:


pmc_map_df = (
    pd.read_csv(
        "../../pmc/exploratory_data_analysis/output/pubmed_central_journal_paper_map.tsv.xz", 
        sep="\t"
    )
    .query("article_type=='research-article'")
)
print(pmc_map_df.shape)
pmc_map_df.head()


# In[4]:


Path("../../pmc/pmc_corpus/pmc_word_counts/").mkdir(parents=True, exist_ok=True)


# In[ ]:


pmc_path_list = [
    Path(f"{doc_path[0]}/{doc_path[1]}.nxml")
    for doc_path in pmc_map_df[["journal", "pmcid"]].values.tolist()
]

sentence_length = get_word_stats(
    document_list=pmc_path_list,
    document_folder="../../pmc/journals/",
    tag_path="//abstract/sec/*|//body/sec/*",
    output_folder="../../pmc/pmc_corpus/pmc_word_counts/",
    skip_condition=lambda folder, document: (
        Path(f"{folder}/{str(document)}").exists() or 
        Path(f"../../pmc/pmc_corpus/pmc_word_counts/{document.stem}.tsv").exists()
    )
)


# In[ ]:


pickle.dump(
    sentence_length, 
    open("../../pmc/pmc_corpus/pmc_sentence_length.pkl", "wb")
)

