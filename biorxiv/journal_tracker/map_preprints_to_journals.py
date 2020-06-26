#!/usr/bin/env python
# coding: utf-8

# # Following the Preprint to Published Path

# The goal of this notebook is to map preprint dois to published dois and published dois to Pubmed Central articles.

# In[1]:


import json

from habanero import Crossref
import pandas as pd
from ratelimit import limits, sleep_and_retry
import requests
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


credentials = json.load(open("credentials.json", "r"))

if "@" not in credentials['email']:
    raise Exception("Please input a valid email address.")
    
if credentials['tool_name'] == 'insert tool name here':
    raise Exception("Please input a name for the tool you are using.")
    
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


# In[6]:


TEN_MINUTES = 600

@sleep_and_retry
@limits(calls=50, period=TEN_MINUTES)
def call_pmc(doi_ids):
    query = (
        "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?"
        f"ids={','.join(doi_ids)}"
        f"&tool={credentials['tool_name']}"
        f"&email={credentials['email']}"
        "&format=json"
    )
    
    return requests.get(query)


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

# In[7]:


preprint_df = pd.read_csv("output/mapped_published_doi.tsv", sep="\t").drop("pmcid", axis=1)
preprint_df.head()


# In[8]:


pmc_df = pd.read_csv(
    "../../pmc/exploratory_data_analysis/output/pubmed_central_journal_paper_map.tsv.xz", 
    sep="\t"
)
pmc_df.head()


# In[9]:


final_df = (
    preprint_df
    .assign(published_doi=preprint_df.published_doi.str.lower())
    .merge(
        pmc_df[["doi", "pmcid"]].assign(doi=pmc_df.doi.str.lower()).dropna(), 
        how="left", left_on="published_doi", 
        right_on="doi"
    )
    .drop("doi_y", axis=1)
    .rename(index=str, columns={"doi_x":"doi"})
)
final_df.head()


# In[10]:


# Fill in missing links
missing_ids = (
    final_df
    .query("published_doi.notnull()&pmcid.isnull()")
    .published_doi
    .unique()
)
print(len(missing_ids))


# In[12]:


chunksize=100
data = []
for chunk in tqdm.tqdm(range(0, len(missing_ids), chunksize)):
    query_ids = missing_ids[chunk:chunk+chunksize]
    response = call_pmc(query_ids).json()
    
    for potential_match in response['records']:
        if "pmcid" not in potential_match:
            continue
            
        final_df.loc[
            final_df["published_doi"] == potential_match['doi'], 
            "pmcid"
        ] = potential_match["pmcid"]


# In[13]:


final_df.head()


# In[17]:


(
    final_df
    .assign(
        pmcoa=final_df.pmcid.isin(pmc_df.pmcid.values.tolist())
    )
    .to_csv(
        "output/mapped_published_doi.tsv",
        sep="\t", index=False
    )
)

