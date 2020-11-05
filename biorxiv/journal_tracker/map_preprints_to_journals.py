#!/usr/bin/env python
# coding: utf-8

# # Following the Preprint to Published Path

# The goal of this notebook is to map preprint dois to published dois and published dois to Pubmed Central articles.

# In[1]:


import json
import re

import numpy as np
import pandas as pd
from ratelimit import limits, sleep_and_retry
import requests
import tqdm
from urllib.error import HTTPError


# In[2]:


preprints_df = pd.read_csv(
    "../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", 
    sep="\t"
)
preprints_df.head()


# In[3]:


dois = (
    preprints_df
    .doi
    .unique()
)
print(len(dois))


# In[4]:


FIVE_MINUTES = 300

@sleep_and_retry
@limits(calls=100, period=FIVE_MINUTES)
def call_biorxiv(doi_ids):
    url = "https://api.biorxiv.org/details/biorxiv/"
    responses = []
    for doi in doi_ids:
        try:
            response = requests.get(url+doi).json()
            responses.append(response)
        except:
            responses.append({
                "message":{
                    "relation":{"none":"none"}, 
                    "DOI":doi
                }
            })
        
    return responses


# In[5]:


FIVE_MINUTES = 300

@sleep_and_retry
@limits(calls=300, period=FIVE_MINUTES)
def call_pmc(doi_ids, tool_name, email):
    query = (
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?"
        f"ids={','.join(doi_ids)}"
        f"&tool={tool_name}"
        f"&email={email}"
        "&format=json"
    )
    
    return requests.get(query)


# # Map preprint DOIs to Published DOIs

# In[6]:


batch_limit = 100
doi_mapper_records = []

for batch in tqdm.tqdm(range(0, len(dois), batch_limit)):
    response = call_biorxiv(dois[batch:batch+batch_limit])
    doi_mapper_records += [
        {
            "preprint_doi": collection['doi'],
            "posted_date": collection['date'],
            "published_doi": collection['published'],
            "version": collection['version']
        }
        for result in response
        for collection in result['collection']
    ]


# In[7]:


(
    pd.DataFrame
    .from_records(doi_mapper_records)
    .to_csv("output/mapped_published_doi_part1.tsv", sep="\t", index=False)
)


# # Map Journal Titles to DOI

# In[6]:


published_doi_df = pd.read_csv(
    "output/mapped_published_doi_part1.tsv", 
    sep="\t"
)
print(published_doi_df.shape)
published_doi_df.head()


# In[9]:


mapped_preprints_df = (
    preprints_df
    .assign(
        version=lambda x: x.document.apply(lambda doc: int(doc.split(".")[0][-1])),
    )
    .rename(index=str, columns={"doi":"preprint_doi"})
    .merge(
        published_doi_df.assign(
            published_doi=lambda x: x.published_doi.apply(
                lambda url: re.sub(r"http(s)?://doi.org/", '', url) 
                if type(url) == str else url
            )
        ), 
        on=["preprint_doi", "version"]
    )
)
print(mapped_preprints_df.shape)
mapped_preprints_df.head()


# In[11]:


mapped_preprints_df.to_csv(
    "output/mapped_published_doi_part2.tsv", 
    sep="\t", index=False
)


# # Map Published Articles to PMC

# In[6]:


preprint_df = pd.read_csv("output/mapped_published_doi_part2.tsv", sep="\t")
print(preprint_df.shape)
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
    .assign(published_doi=preprint_df.published_doi.str.lower())
    .merge(
        pmc_df[["doi", "pmcid"]]
        .assign(doi=pmc_df.doi.str.lower())
        .dropna()
        .rename(index=str, columns={"doi":"published_doi"}), 
        how="left", on="published_doi"
    )
)
print(final_df.shape)
final_df.head()


# In[9]:


# Fill in missing links
missing_ids = (
    final_df
    .query("published_doi.notnull()&pmcid.isnull()")
    .published_doi
    .unique()
)
print(len(missing_ids))


# In[10]:


chunksize=100
data = []
for chunk in tqdm.tqdm(range(0, len(missing_ids), chunksize)):
    query_ids = missing_ids[chunk:chunk+chunksize]
    response = call_pmc(query_ids, 'model_name', 'email@server.com').json()
    
    for potential_match in response['records']:
        if "pmcid" not in potential_match:
            continue
        
        data.append({
            "pmcid": potential_match["pmcid"], 
            "published_doi": potential_match['doi']
        })


# In[11]:


missing_pmcids = pd.DataFrame.from_records(data)
missing_pmcids.head()


# In[28]:


(
    final_df
    .merge(
        missing_pmcids.assign(published_doi=lambda x:x.published_doi.str.lower()),
        on="published_doi", how="left"
    )
    .assign(
        final_pmcid=lambda x: x.pmcid_x.fillna('') + x.pmcid_y.fillna(''),
        pmcoa=final_df.pmcid.isin(pmc_df.pmcid.values.tolist())
    )
    .drop(["pmcid_x", "pmcid_y"], axis=1)
    .rename(index=str, columns={"final_pmcid":"pmcid"})
    .replace('', np.nan)
    .to_csv(
        "output/mapped_published_doi.tsv",
        sep="\t", index=False
    )
)

