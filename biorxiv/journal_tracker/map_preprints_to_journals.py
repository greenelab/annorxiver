#!/usr/bin/env python
# coding: utf-8

# # Following the Preprint to Published Path

# The goal of this notebook is to map preprint dois to published dois and published dois to Pubmed Central articles.

# In[1]:


import json

from habanero import Crossref
import pandas as pd
from ratelimit import limits, sleep_and_retry
import tqdm
from urllib.error import HTTPError


# In[2]:


preprints_df = pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
preprints_df.head()


# In[3]:


dois = (
    preprints_df
    .doi
    .unique()
)
print(len(dois))


# In[4]:


credentials = json.load("credentials.json")

if "@" not in credentials:
    raise Exception("Please input a valid email address.")
    
cf = Crossref(mailto=credentials['email'])


# In[5]:


TEN_MINUTES = 600

@sleep_and_retry
@limits(calls=10, period=TEN_MINUTES)
def call_crossref(doi_ids):
    responses = []
    for doi in doi_ids:
        try:
            response = cf.works(ids=doi)
            responses.append(response)
        except:
            responses.append({
                "message":{
                    "relation":{"none":"none"}, 
                    "DOI":doi
                }
            })
        
    return responses


# # Map preprint DOIs to Published DOIs

# In[8]:


batch_limit = 100
doi_mapper_records = []

for batch in tqdm.tqdm(range(0, len(dois), batch_limit)):
    response = call_crossref(dois[batch:batch+batch_limit])
    doi_mapper_records += [
        {
         "preprint_doi":result['message']['DOI'],
         "published_doi":(
                 result['message']['relation']['is-preprint-of'][0]['id'] 
                 if "is-preprint-of" in result['message']['relation'] 
                 else ""
             )
        }
        for result in response
    ]


# In[9]:


published_doi_df = (
    pd.DataFrame.from_records(doi_mapper_records)
    .append(mapped_preprints_df)
)
published_doi_df.to_csv("output/mapped_published_doi.tsv", sep="\t", index=False)


# # Map Journal Titles to DOI

# In[8]:


published_doi_df = pd.read_csv("output/mapped_published_doi.tsv", sep="\t")


# In[11]:


journal_dois = (
    published_doi_df
    .published_doi
    .unique()
    .tolist()
)


# In[14]:


batch_limit = 100
journal_mapper_records = []

for batch in tqdm.tqdm(range(0, len(journal_dois), batch_limit)):
    response = call_crossref(journal_dois[batch:batch+batch_limit])
    journal_mapper_records += [
        {
            "published_doi": result['message']['DOI'],
            "journal": (
                result['message']['container-title'][0] 
                if 'container-title' in result['message'] and len(result['message']['container-title']) > 0
                else "fill_me_in"
            ),
        }
        for result in response
    ]


# In[15]:


journal_mapper_df = pd.DataFrame.from_records(journal_mapper_records)
journal_mapper_df.head()


# In[12]:


final_df = (
    preprints_df
    .merge(published_doi_df, left_on="doi", right_on="preprint_doi")
    .merge(journal_mapper_df, on="published_doi", how="left")
    .drop("preprint_doi", axis=1)
)
final_df.head()


# In[13]:


final_df.to_csv("output/mapped_published_doi.tsv", sep="\t", index=False)


# # Map Published Articles to PMC

# In[6]:


preprint_df = pd.read_csv("output/mapped_published_doi.tsv", sep="\t")
preprint_df.head()


# In[7]:


pmc_df = pd.read_csv(
    "../../pmc/exploratory_data_analysis/output/pubmed_central_journal_paper_map.tsv.xz", 
    sep="\t"
)
pmc_df.head()


# In[8]:


final_df = (
    preprint_df
    .merge(
        pmc_df[["doi", "pmcid"]].dropna(), 
        how="left", left_on="published_doi", 
        right_on="doi"
    )
    .drop("doi_y", axis=1)
    .rename(index=str, columns={"doi_x":"doi"})
)
final_df.head()


# In[9]:


final_df.to_csv("output/mapped_published_doi.tsv", sep="\t", index=False)

