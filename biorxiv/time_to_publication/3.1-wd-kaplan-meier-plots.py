#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
from lifelines import KaplanMeierFitter
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


def show_sample(df):
    print(f"len = {len(df)}")
    display(df[:2])


# In[3]:


preprints_df = pd.read_csv("output/biorxiv_article_metadata.tsv", sep="\t",)


# In[4]:


preprints_df["date_received"] = pd.to_datetime(preprints_df["date_received"])


# In[5]:


xml_df = (
    preprints_df.sort_values(by="date_received")
    .dropna(subset=["date_received"])
    .groupby("doi")
    .first()
)


# In[6]:


print(f"preprints_df len: {len(preprints_df)}")
show_sample(xml_df)


# In[7]:


header_list = [
    "document",  # file_path
    "title",
    "version",
    "category",
    "author_type",
    "heading",
    "doi",
    "publisher",  # actually, the publisher of biorxiv Cold Spring Harbor Laboratory
    "rcvd_day",
    "rcvd_month",
    "rcvd_year",
]
xmlmeta_df_0 = pd.read_csv(
    "output/biorxiv-basic-meta.tsv", names=header_list, sep="\t",
)


# ### Get crossref member data
# Downloaded from https://www.crossref.org/reporting/members-with-closed-references/ (and open)

# In[8]:


xref_member_closed_df = pd.read_csv("output/crossref-members-closed.csv")
xref_member_open_df = pd.read_csv("output/crossref-members-open.csv")
xref_member_df = pd.concat([xref_member_closed_df, xref_member_open_df])
xref_member_df["doi_prefix"] = xref_member_df["Sponsored member & prefix"].str.extract(
    r"(10.[0-9]+)$"
)
xref_member_df[["mem_name", "mem_id"]] = xref_member_df["Member Name & ID"].str.extract(
    r"(.*) \(ID ([0-9]+)\)$"
)
show_sample(xref_member_df)


# In[9]:


xref_pref_table = xref_member_df[["doi_prefix", "mem_name"]]
xref_pref_table[:5]


# In[10]:


xref_pref_table.to_csv("output/xref_pref_table.tsv", sep="\t", index=False)


# In[11]:


header_list = [
    "document",  # file_path
    "title",
    "version",
    "category",
    "author_type",
    "heading",
    "doi",
    "publisher",  # actually, the publisher of biorxiv Cold Spring Harbor Laboratory
    "rcvd_day",
    "rcvd_month",
    "rcvd_year",
]
xmlmeta_df_0 = pd.read_csv(
    "output/biorxiv-basic-meta.tsv", names=header_list, sep="\t",
)


# In[12]:


xmlmeta_df_0.groupby("publisher").count()


# In[13]:


xmlmeta_df_0[:2]


# In[14]:


xmlmeta_df_0["date_received"] = (
    xmlmeta_df_0["rcvd_year"]
    + "-"
    + xmlmeta_df_0["rcvd_month"]
    + "-"
    + xmlmeta_df_0["rcvd_day"]
)
xmlmeta_df_0["date_received"] = pd.to_datetime(
    xmlmeta_df_0["date_received"], errors="coerce"
)
xmlmeta_df = xmlmeta_df_0.dropna(subset=["date_received"]).drop(
    ["rcvd_day", "rcvd_month", "rcvd_year"], axis=1
)
show_sample(xmlmeta_df_0)


# In[15]:


xmlmeta_df = xmlmeta_df.groupby("doi").first()
show_sample(xmlmeta_df)


# ### Read data retrived from the API

# In[16]:


api_df = pd.read_csv("output/biorxiv_published_api_data.tsv", sep="\t")


# In[17]:


api_df[api_df["published_date"].str.contains(":")]


# In[18]:


api_df["doi_prefix"] = api_df["published_doi"].str.split("/").str[0]


# In[19]:


api_df[:5]


# In[20]:


doi_prefixes_df = api_df["doi_prefix"].drop_duplicates()


# In[21]:


len(doi_prefixes_df)


# In[22]:


index = api_df[api_df["published_date"].str.contains(":")].index
api_df.loc[index, "published_date"] = (
    api_df.loc[index, "published_date"].str.split(":").str[0]
)


# In[23]:


for col in ["preprint_date", "published_date"]:
    api_df[col] = pd.to_datetime(api_df[col])


# In[24]:


api_df.set_index("biorxiv_doi")


# In[25]:


merged_df = pd.merge(
    xml_df,
    # xmlmeta_df,
    api_df.set_index("biorxiv_doi"),
    left_index=True,
    right_index=True,
    how="outer",
)


# In[26]:


show_sample(merged_df)


# In[27]:


merged_df["document"].isna().sum()


# In[28]:


merged_df["published_doi"].isna().sum()


# In[29]:


len(merged_df)


# In[30]:


# lets ignore papers we don't have xmls for
merged_df = pd.merge(
    xml_df,
    # xmlmeta_df,
    api_df.set_index("biorxiv_doi"),
    left_index=True,
    right_index=True,
    how="left",
)


# In[31]:


len(merged_df)


# In[32]:


merged_df = pd.merge(merged_df, xref_pref_table, on="doi_prefix")


# In[33]:


print(len(merged_df))
merged_df[:3]


# In[34]:


merged_df["published"] = ~merged_df["published_doi"].isna()


# In[35]:


# I should change this to when the data was pulled, but I didn't record that for now :(
merged_df.loc[merged_df["published"], "observation_date"] = merged_df.loc[
    merged_df["published"], "published_date"
]
merged_df.loc[~merged_df["published"], "observation_date"] = datetime.datetime.today()


# In[36]:


merged_df["observation_duration"] = (
    merged_df["observation_date"] - merged_df["date_received"]
)


# In[37]:


(merged_df["observation_duration"] < pd.Timedelta(0)).sum()


# In[38]:


merged_df = merged_df[merged_df["observation_duration"] > pd.Timedelta(0)]


# In[39]:


merged_df["observation_duration"].astype("timedelta64[s]")


# In[40]:


ax = sns.distplot(
    merged_df["observation_duration"].astype("timedelta64[D]")
)  # timedelta in days


# In[41]:


kmf = KaplanMeierFitter()


# In[42]:


kmf.fit(
    merged_df["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365,
    event_observed=merged_df["published"],
)
ax = kmf.plot(label="all papers", logx=True)
_ = ax.set_ylabel("proportion of unpublished biorxiv papers")
_ = ax.set_xlabel("timeline (years)")
_ = ax.set_ylim(0, 1)


# In[43]:


ax = kmf.plot(label="all papers", logx=False)
_ = ax.set_ylabel("proportion of unpublished biorxiv papers")
_ = ax.set_xlabel("timeline (years)")
_ = ax.set_ylim(0, 1)


# In[44]:


kmf.median_survival_time_


# In[45]:


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
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
_ = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Biorxiv category")

_ = ax.set_ylabel("proportion of unpublished biorxiv papers")
_ = ax.set_xlabel("timeline (years)")
_ = ax.set_ylim(0, 1)


# In[46]:


def run_plot_by_group(df, group_name="category", min_group_size=0, selected=None):
    f = plt.figure(figsize=(10, 8))
    ax = None
    for category, cat_group in df.groupby(group_name):
        if selected is not None and category not in selected:
            continue
        if len(cat_group) < min_group_size:
            continue
        kmf.fit(
            cat_group["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365,
            event_observed=cat_group["published"],
        )
        ax = kmf.plot(label=category, ax=ax, ci_show=False, logx=True)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    _ = ax.legend(
        loc="center left", bbox_to_anchor=(1, 0.5), title=f"Biorxiv {group_name}"
    )

    _ = ax.set_ylabel("proportion of unpublished biorxiv papers")
    _ = ax.set_xlabel("timeline (years)")
    _ = ax.set_ylim(0, 1)


# In[47]:


run_plot_by_group(
    merged_df,
    "category",
    min_group_size=0,
    selected=["biochemistry", "cell biology", "clinical trials", "zoology"],
)


# In[48]:


run_plot_by_group(merged_df, "category", min_group_size=5000)


# In[49]:


run_plot_by_group(merged_df, "mem_name", min_group_size=500)


# In[50]:


get_ipython().run_cell_magic('time', '', 'f = plt.figure(figsize=(10, 8))\n\nax = None\nfor category, cat_group in merged_df.groupby("mem_name"):\n    if len(cat_group) > 100:\n        kmf.fit(\n            cat_group["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365,\n            event_observed=cat_group["published"],\n        )\n        ax = kmf.plot(label=category, ax=ax, ci_show=False, logx=True)\n\n# Shrink current axis by 20%\n# box = ax.get_position()\n# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])\n\n# Put a legend to the right of the current axis\n_ = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Publisher")\n\n_ = ax.set_ylabel("proportion of unpublished biorxiv papers")\n_ = ax.set_xlabel("timeline (years)")\n_ = ax.set_ylim(0, 1)')


# In[51]:


# merged_df["doi_prefix"] = merged_df["published_doi"].str.split("/").str[0]


# In[52]:


get_ipython().run_cell_magic('time', '', 'f = plt.figure(figsize=(10, 8))\n\nax = None\nfor category, cat_group in merged_df.groupby("doi_prefix"):\n    if len(cat_group) > 100:\n        kmf.fit(\n            cat_group["observation_duration"].dt.total_seconds() / 60 / 60 / 24 / 365,\n            event_observed=cat_group["published"],\n        )\n        ax = kmf.plot(label=category, ax=ax, ci_show=False, logx=True)\n\n# Shrink current axis by 20%\nbox = ax.get_position()\nax.set_position([box.x0, box.y0, box.width * 0.8, box.height])\n\n# Put a legend to the right of the current axis\n_ = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="DOI prefix")\n\n_ = ax.set_ylabel("proportion of unpublished biorxiv papers")\n_ = ax.set_xlabel("timeline (years)")\n_ = ax.set_ylim(0, 1)')


# In[53]:


import warnings
warnings.filterwarnings("ignore")


# In[54]:


get_ipython().run_cell_magic('time', '', 'doi_prefix_df = merged_df.groupby("doi_prefix").apply(\n    lambda cat_group: pd.Series(\n        {\n            "count": len(cat_group),\n            "80th_percentile": kmf.fit(\n                cat_group["observation_duration"].dt.total_seconds() / 60 / 60 / 24,\n                event_observed=cat_group["published"],\n            ).percentile(0.8),\n        }\n    )\n)')


# In[55]:


doi_prefix_df[doi_prefix_df["count"] > 50].sort_values("80th_percentile").head()


# F1000 Research Ltd <== 10.12688
# 
# MDPI AG <== 10.3390 - wikipedia notes questionable quality of peer-review

# In[56]:


get_ipython().run_cell_magic('time', '', 'doi_prefix_df = merged_df.groupby("mem_name").apply(\n    lambda cat_group: pd.Series(\n        {\n            "count": len(cat_group),\n            "80th_percentile": kmf.fit(\n                cat_group["observation_duration"].dt.total_seconds() / 60 / 60 / 24,\n                event_observed=cat_group["published"],\n            ).percentile(0.8),\n        }\n    )\n)')


# In[57]:


doi_prefix_df[doi_prefix_df["count"] > 50].sort_values("80th_percentile").head(10)


# In[ ]:




