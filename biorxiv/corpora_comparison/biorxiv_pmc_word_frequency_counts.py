#!/usr/bin/env python
# coding: utf-8

# # Compare Pubmed Central Corpus with bioRxiv Corpus

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from collections import Counter, defaultdict
import csv
from pathlib import Path
import pickle
import string
import sys

sys.path.append("../../modules/")

import gensim
import pandas as pd
import spacy
from tqdm import tqdm_notebook

from document_helper import dump_article_text


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
biorxiv_map_df.head()


# In[4]:


Path('output/biorxiv_word_counts').mkdir(exist_ok=True)


# In[5]:


for document in tqdm_notebook(biorxiv_map_df.document.tolist()):
    
    document_text = dump_article_text(
        file_path=f"../biorxiv_articles/{document}",
        xpath_str="//abstract/p|//abstract/title|//body/sec//p|//body/sec//title",
        remove_stop_words=False
    )

    doc = lemma_model(" ".join(document_text),  disable = ['ner', 'parser'])
    tokens = [tok for tok in doc if tok.text.lower() not in string.punctuation]
    
    with open(f"output/biorxiv_word_counts/{Path(document).stem}.tsv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=["lemma", "count"], delimiter="\t")
        writer.writeheader()
        
        lemma_freq = Counter(
            list(
                map(
                    lambda x: (
                        x.lemma_.lower() 
                        if x.lemma_.lower() != '-pron-' 
                        else x.text.lower()
                    ), 
                    tokens
                )
            )
        )
        
        writer.writerows([
            {"lemma":val[0], "count":val[1]}
            for val in lemma_freq.items()
        ])


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


Path("output/pmc_word_counts").mkdir(exist_ok=True)


# In[5]:


for document in tqdm_notebook(pmc_map_df[["journal", "pmcid"]].values.tolist()):
    
    #Skip files that dont exist or files already parsed
    if (
        not Path(f"../../pmc/journals/{document[0]}/{document[1]}.nxml").exists()
        or Path(f"output/pmc_word_counts/{document[1]}.tsv").exists()
    ):
        continue
    
    document_text = dump_article_text(
        file_path=f"../../pmc/journals/{document[0]}/{document[1]}.nxml",
        xpath_str="//abstract/sec/*|//body/sec/*",
        remove_stop_words=False
    )
    
    doc = lemma_model(" ".join(document_text),  disable = ['ner', 'parser'])
    tokens = [tok for tok in doc if tok.text.lower() not in string.punctuation]
    with open(f"output/pmc_word_counts/{document[1]}.tsv", "w") as file:
        writer = csv.DictWriter(file, fieldnames=["lemma", "count"],delimiter="\t")
        writer.writeheader()

        lemma_freq = Counter(
            list(
                map(
                    lambda x: (
                        x.lemma_.lower() 
                        if x.lemma_.lower() != '-pron-' 
                        else x.text.lower()
                    ), 
                    tokens
                )
            )
        )
              
        writer.writerows([
            {"lemma":val[0], "count":val[1]}
            for val in lemma_freq.items()
        ])
        

