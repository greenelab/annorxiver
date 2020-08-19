#!/usr/bin/env python
# coding: utf-8

# # Get Word Frequency and Statistics on the New York Times Annotated Corpus

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

import lxml.etree as ET
from annorxiver_modules.document_helper import dump_article_text


# In[2]:


lemma_model = spacy.load("en_core_web_sm")
lemma_model.max_length = 9000000 


# # Get the Listing of NYTAC documents

# In[3]:


document_gen = list(
    Path("../nyt_corpus/extracted_data")
    .rglob("*.xml")
)
print(len(document_gen))


# 
# # Parse the Corpus

# In[4]:


document_list = [
    f"{doc.stem}.xml"
    for doc in document_gen
]

sentence_length = get_word_stats(
    document_list=document_list,
    document_folder="../nyt_corpus/extracted_data",
    tag_path="//body/body.head/headline/hl1|//body/body.content/block/p",
    output_folder="output/",
)


# In[5]:


pickle.dump(
    sentence_length, 
    open("nytac_sentence_length.pkl", "wb")
)

