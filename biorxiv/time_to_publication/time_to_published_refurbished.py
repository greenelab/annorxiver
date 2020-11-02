#!/usr/bin/env python
# coding: utf-8

# # Restructure Hazard Function Calculations

# In[1]:


from datetime import timedelta, date
from pathlib import Path

from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd
import plotnine as p9
import tqdm

from mizani.breaks import date_breaks
from mizani.formatters import timedelta_format


# # Load the Data

# In[2]:


published_dates = (
    pd.read_csv(
        "../publication_delay_experiment/output/biorxiv_published_dates.tsv", 
        sep="\t"
    )
    .assign(
        preprint_date = lambda x: pd.to_datetime(x.preprint_date.tolist()),
        published_date = lambda x: pd.to_datetime(
            x.published_date.apply(lambda y: y[0:y.index(":")] if ":" in y else y)
        )
    )
)
print(published_dates.shape)
published_dates.head()


# In[3]:


biorxiv_journal_df = (
    pd.read_csv(
        "../journal_tracker/output/mapped_published_doi.tsv", 
        sep="\t"
    )
    .groupby("preprint_doi")
    .agg({
        "document":"first",
        "category":"first",
        "preprint_doi":"count",
        "published_doi":"first",  
        "pmcid":"first", 
        "pmcoa":"first",
        "posted_date":"first"
    })
    .rename(index=str, columns={"preprint_doi":"version_count"})
    .reset_index()
)
print(biorxiv_journal_df.shape)
biorxiv_journal_df.head()


# In[4]:


preprints_w_published_dates = (
    biorxiv_journal_df
    .sort_values("document")
    .merge(
        published_dates
        [["biorxiv_doi", "published_date"]]
        .rename(index=str, columns={"biorxiv_doi":"preprint_doi"}),
        on="preprint_doi",
        how="left"
    )
    .assign(
        published_date=lambda x: x.published_date.fillna(date.today())
    )
    .assign(
        time_to_published = lambda x: pd.to_datetime(x.published_date) - pd.to_datetime(x.posted_date)
    )
)
preprints_w_published_dates = (
    preprints_w_published_dates[
        preprints_w_published_dates.time_to_published > pd.Timedelta(0)
    ]
)
print(preprints_w_published_dates.shape)
preprints_w_published_dates.head()


# # Calculate Overall Survival Function

# In[5]:


kmf = KaplanMeierFitter()


# In[6]:


kmf.fit(
    preprints_w_published_dates["time_to_published"].dt.total_seconds() / 60 / 60 / 24,
    event_observed= ~preprints_w_published_dates["published_doi"].isna(),
)


# In[7]:


kmf.median_survival_time_


# In[8]:


overall_preprint_survival = (
    kmf.survival_function_
    .reset_index()
    .assign(label="all_papers")
)
overall_preprint_survival.head()


# In[9]:


g = (
    p9.ggplot(overall_preprint_survival, p9.aes(x="timeline", y="KM_estimate", color="label"))
    + p9.geom_point()
    + p9.ylim(0,1)
    + p9.xlim(0, 2550)
)
print(g)


# # Calculate Category Survival Function

# In[10]:


entire_preprint_df = pd.DataFrame([], columns=["timeline", "KM_estimate", "category"])
half_life = []
for cat, grouped_df in preprints_w_published_dates.groupby("category"):
    temp_df = preprints_w_published_dates.query(f"category=='{cat}'")
    kmf.fit(
        temp_df["time_to_published"].dt.total_seconds() / 60 / 60 / 24,
        event_observed= ~temp_df["published_doi"].isna(),
    )
    
    half_life.append({
        "category":cat, 
        "half_life_time":kmf.median_survival_time_
    })
    
    entire_preprint_df = (
        entire_preprint_df
        .append(
            kmf.survival_function_
            .reset_index()
            .assign(category=cat)
        )
    )


# In[11]:


g = (
    p9.ggplot(
        entire_preprint_df, 
        p9.aes(x="timeline", y="KM_estimate", color='category')
    )
    + p9.geom_line()
    + p9.ylim(0,1)
    + p9.xlim(0,2550)
    + p9.labs(
        x="timeline (days)",
        y="proportion of unpublished biorxiv paper"
    )
)
print(g)


# In[12]:


category_half_life = (
    pd.DataFrame
    .from_records(half_life)
    .replace(np.inf, (temp_df["time_to_published"].dt.total_seconds() / 60 / 60 / 24).max())
)
category_half_life


# In[13]:


g = (
    p9.ggplot(
        category_half_life.query("category!='none'"),
        p9.aes(x="category", y="half_life_time")
    )
    + p9.geom_col()
    + p9.scale_x_discrete(
        limits=(
            category_half_life
            .query("category!='none'")
            .sort_values("half_life_time", ascending=False)
            .category
            .tolist()
        )
    )
    + p9.coord_flip()
    + p9.labs(
        x="preprint category",
        y="Days Until 50% of Preprints are Published",
        title="Preprint Category Half-Life"
    )
)
g.save("output/preprint_category_halflife.svg", dpi=500)
g.save("output/preprint_category_halflife.png", dpi=500)
print(g)

