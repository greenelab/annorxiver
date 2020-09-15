#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from lifelines import KaplanMeierFitter
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


preprints_df = pd.read_csv(
    "/home/thielk/gitlab/annorxiver/biorxiv/exploratory_data_analysis/output/biorxiv_article_metadata.tsv",
    sep="\t",
)


# In[3]:


preprints_df["date_received"] = pd.to_datetime(preprints_df["date_received"])


# In[4]:


xml_df = (
    preprints_df.sort_values(by="date_received")
    .dropna(subset=["date_received"])
    .groupby("doi")
    .first()
)


# In[5]:


api_df = pd.read_csv(
    "/home/thielk/gitlab/ctha-biorxiv-analysis/notebooks/biorxiv_published_api_data.tsv", sep="\t"
)


# In[6]:


api_df[api_df["published_date"].str.contains(":")]


# In[7]:


index = api_df[api_df["published_date"].str.contains(":")].index
api_df.loc[index, "published_date"] = (
    api_df.loc[index, "published_date"].str.split(":").str[0]
)


# In[8]:


for col in ["preprint_date", "published_date"]:
    api_df[col] = pd.to_datetime(api_df[col])


# In[9]:


api_df.set_index("biorxiv_doi")


# In[10]:


merged_df = pd.merge(
    xml_df,
    api_df.set_index("biorxiv_doi"),
    left_index=True,
    right_index=True,
    how="outer",
)


# In[11]:


merged_df


# In[12]:


merged_df["document"].isna().sum()


# In[13]:


merged_df["published_doi"].isna().sum()


# In[14]:


len(merged_df)


# In[15]:


# lets ignore papers we don't have xmls for
merged_df = pd.merge(
    xml_df,
    api_df.set_index("biorxiv_doi"),
    left_index=True,
    right_index=True,
    how="left",
)


# In[16]:


merged_df["published"] = ~merged_df["published_doi"].isna()


# In[17]:


# I should change this to when the data was pulled, but I didn't record that for now :(
merged_df.loc[merged_df["published"], "observation_date"] = merged_df.loc[
    merged_df["published"], "published_date"
]
merged_df.loc[~merged_df["published"], "observation_date"] = pd.datetime.today()


# In[18]:


merged_df["observation_duration"] = (
    merged_df["observation_date"] - merged_df["date_received"]
)


# In[19]:


(merged_df["observation_duration"] < pd.Timedelta(0)).sum()


# In[20]:


merged_df = merged_df[merged_df["observation_duration"] > pd.Timedelta(0)]


# In[21]:


ax = sns.distplot(merged_df["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365)


# In[22]:


kmf = KaplanMeierFitter()


# In[23]:


kmf.fit(
    merged_df["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365,
    event_observed=merged_df["published"],
)
ax = kmf.plot(label="all papers", logx=True)
_ = ax.set_ylabel("proportion of unpublished biorxiv papers")
_ = ax.set_xlabel("timeline (years)")
_ = ax.set_ylim(0, 1)


# In[24]:


f = plt.figure(figsize=(10, 8))

ax = None
for category, cat_group in merged_df.groupby("category"):
    kmf.fit(
        cat_group["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365,
        event_observed=cat_group["published"],
    )
    ax = kmf.plot(label=category, ax=ax, ci_show=False, logx=True)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
_ = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Biorxiv category")

_ = ax.set_ylabel("proportion of unpublished biorxiv papers")
_ = ax.set_xlabel("timeline (years)")
_ = ax.set_ylim(0, 1)


# In[25]:


merged_df["doi_prefix"] = merged_df["published_doi"].str.split("/").str[0]


# In[26]:


get_ipython().run_cell_magic('time', '', 'f = plt.figure(figsize=(10, 8))\n\nax = None\nfor category, cat_group in merged_df.groupby("doi_prefix"):\n    if len(cat_group) > 100:\n        kmf.fit(\n            cat_group["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365,\n            event_observed=cat_group["published"],\n        )\n        ax = kmf.plot(label=category, ax=ax, ci_show=False, logx=True)\n\n# Shrink current axis by 20%\nbox = ax.get_position()\nax.set_position([box.x0, box.y0, box.width * 0.8, box.height])\n\n# Put a legend to the right of the current axis\n_ = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="DOI prefix")\n\n_ = ax.set_ylabel("proportion of unpublished biorxiv papers")\n_ = ax.set_xlabel("timeline (years)")\n_ = ax.set_ylim(0, 1)')


# In[27]:


get_ipython().run_cell_magic('time', '', 'doi_prefix_df = merged_df.groupby("doi_prefix").apply(\n    lambda cat_group: pd.Series(\n        {\n            "count": len(cat_group),\n            "80th_percentile": kmf.fit(\n                cat_group["observation_duration"].dt.total_seconds()\n                / 60\n                / 60\n                / 24,\n                event_observed=cat_group["published"],\n            ).percentile(0.8),\n        }\n    )\n)')


# In[28]:


doi_prefix_df[doi_prefix_df["count"] > 50].sort_values("80th_percentile").head()


# F1000 Research Ltd <== 10.12688
# 
# MDPI AG <== 10.3390 - wikipedia notes questionable quality of peer-review
