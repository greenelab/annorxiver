#!/usr/bin/env python
# coding: utf-8

# # Tree Plots for bioRxiv and Pubmed Central

# In[1]:


import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# # bioRxiv vs Pubmed Central

# In[2]:


full_text_comparison = pd.read_csv("output/full_corpus_comparison_stats.tsv", sep="\t")
full_text_comparison.head()


# In[3]:


full_text_comparison = (
    full_text_comparison
    .assign(repository=lambda x: x.odds_ratio.apply(lambda ratio: "bioRxiv" if ratio > 1 else "PMC"))
    .sort_values("odds_ratio", ascending=False)
)
full_text_comparison


# In[4]:


subset = 20
plot_df = (
    full_text_comparison.head(subset).append(full_text_comparison.tail(subset))
    .assign(odds_ratio=lambda x: x.odds_ratio.apply(lambda ratio: ratio if ratio > 1 else 1/ratio))
    .assign(label=lambda x: x.apply(
        lambda row: (
            f"<b>{row.lemma}</b><br />bioRxiv:{row.biorxiv_count:,}"
            f"<br />PMC:{row.pmc_count:,}<br />"
            f"Odds Ratio:{row.odds_ratio:.2f}"
        ),
        axis=1
    ))
    
)
plot_df.head()


# In[5]:


fig=go.Figure(
    go.Treemap(
        branchvalues="remainder",
        labels = ["bioRxiv","PMC"] + plot_df.label.tolist(),
        parents = ["", ""] + plot_df.repository.tolist(),
        values = [0,0] + plot_df.odds_ratio.tolist(),
        textinfo = "label",
        textfont = {'size':17}
    )
)
fig.update_layout(
    width=1280, 
    height=720,
    treemapcolorway=["#a6cee3", "#b2df8a"],
    title={
        'text':f"Comparison of Top {plot_df.shape[0]} Words between bioRxiv and Pubmed Central (PMC)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.update_layout(uniformtext=dict(minsize=10))
fig.show()


# In[6]:


fig.write_image("output/figures/biorxiv_vs_pubmed_central.svg", scale=2.5)
fig.write_image("output/figures/biorxiv_vs_pubmed_central.png", scale=2.5)


# # Preprint vs Published

# In[7]:


preprint_published_comparison = pd.read_csv("output/preprint_to_published_comparison.tsv", sep="\t")
preprint_published_comparison.head()


# In[8]:


preprint_published_comparison = (
    preprint_published_comparison
    .assign(repository=lambda x: x.odds_ratio.apply(lambda ratio: "preprint" if ratio > 1 else "published"))
    .sort_values("odds_ratio", ascending=False)
)
preprint_published_comparison


# In[9]:


subset = 20
plot_df = (
    preprint_published_comparison.head(subset).append(preprint_published_comparison.tail(subset))
    .assign(odds_ratio=lambda x: x.odds_ratio.apply(lambda ratio: ratio if ratio > 1 else 1/ratio))
    .assign(label=lambda x: x.apply(
        lambda row: (
            f"<b>{row.lemma}</b><br />Preprint:{row.preprint_count:,}"
            f"<br />Published:{row.published_count:,}<br />"
            f"Odds Ratio:{row.odds_ratio:.2f}"
        ),
        axis=1
    ))
    
)
plot_df.head()


# In[10]:


fig=go.Figure(
    go.Treemap(
        branchvalues="remainder",
        labels = ["preprint","published"] + plot_df.label.tolist(),
        parents = ["", ""] + plot_df.repository.tolist(),
        values = [0,0] + plot_df.odds_ratio.tolist(),
        textinfo = "label",
        textfont = {
            'size':15
        }
    )
)
fig.update_layout(
    width=1280, 
    height=720,
    treemapcolorway=["#a6cee3", "#b2df8a"],
    title={
        'text':f"Comparison of Top {plot_df.shape[0]} Words between Preprints and Published Papers",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.update_layout(uniformtext=dict(minsize=10))
fig.show()


# In[11]:


fig.write_image("output/figures/preprint_published_comparison.svg", scale=2.5)
fig.write_image("output/figures/preprint_published_comparison.png", scale=2.5)

