#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from tqdm.notebook import tqdm
import pandas as pd


# In[2]:


url = "https://api.biorxiv.org/pub/2000-01-01/3000-01-01/"


# In[3]:


response = requests.get(url + "0").json()


# In[4]:


total = response["messages"][0]["total"]


# In[5]:


collection = []
page_size = 100
for i in tqdm(range(0, total, page_size), total=total/page_size):
    collection += requests.get(url + str(i)).json()["collection"]


# In[6]:


pd.DataFrame(collection).to_csv("biorxiv_published_api_data.tsv", sep="\t", index=False)


# In[ ]:




