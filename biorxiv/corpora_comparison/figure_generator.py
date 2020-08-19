#!/usr/bin/env python
# coding: utf-8

# # Figures for Corpora Comparison between bioRxiv,  Pubmed Central, New York Times

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd

from cairosvg import svg2png
from IPython.display import Image
import plotnine as p9

from annorxiver_modules.corpora_comparison_helper import(
    calculate_confidence_intervals,
    create_lemma_count_df,
    plot_pointplot,
    plot_bargraph,
    plot_point_bar_figure
)


# In[2]:


subset = 20


# # KL Divergence Graph

# In[3]:


kl_divergence_df = pd.read_csv(
    "output/comparison_stats/corpora_kl_divergence.tsv", 
    sep="\t"
)
kl_divergence_df.head()


# In[4]:


g = (
    p9.ggplot(
        kl_divergence_df.replace({
        "biorxiv_vs_nytac":"biorxiv_vs_reference",
        "pmc_vs_nytac": "pmc_vs_reference"
        })
    )
    + p9.aes(x="factor(num_terms)", y="KL_divergence", fill="comparison")
    + p9.geom_col(stat='identity', position='dodge')
    + p9.scale_fill_manual(["#b2df8a", "#1f78b4", "#a6cee3"])
    + p9.labs(
        x = "Number of terms evaluated",
        y = "KL Divergence"
    )
    + p9.theme_seaborn(context='paper')
)
g.save("output/figures/corpora_kl_divergence.png", dpi=500)
print(g)


# # bioRxiv vs Pubmed Central

# In[5]:


full_text_comparison = pd.read_csv(
    "output/comparison_stats/biorxiv_vs_pmc_comparison.tsv",
    sep="\t"
)
full_text_comparison.head()


# ## Line Plots

# In[6]:


full_plot_df = calculate_confidence_intervals(
    full_text_comparison
)
full_plot_df.head()


# In[7]:


plot_df = (
    full_plot_df
    .sort_values("odds_ratio", ascending=False)
    .head(subset)
    .append(
        full_plot_df
        .sort_values("odds_ratio", ascending=False)
        .iloc[:-2]
        .tail(subset)
    )
    .replace('rna', 'RNA')
)
plot_df.head()


# In[8]:


g = plot_pointplot(plot_df, "bioRxiv vs PMC Odds Ratio")
g.save("output/svg_files/biorxiv_pmc_frequency_odds.svg", dpi=75)
print(g)


# In[9]:


count_plot_df = create_lemma_count_df(plot_df, 'bioRxiv', 'pmc')
count_plot_df.head()


# In[10]:


g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/biorxiv_pmc_frequency_bar.svg", dpi=75)
print(g)


# In[11]:


fig_output_path = "output/figures/biorxiv_vs_pubmed_central.png"

fig = plot_point_bar_figure(
    "output/svg_files/biorxiv_pmc_frequency_odds.svg",
    "output/svg_files/biorxiv_pmc_frequency_bar.svg"
)

# save generated SVG files
svg2png(
    bytestring=fig.to_str(), 
    write_to=fig_output_path,
    dpi=500
)

Image(fig_output_path)


# # bioRxiv vs Reference

# In[12]:


full_text_comparison = pd.read_csv(
    "output/comparison_stats/biorxiv_nytac_comparison.tsv",
    sep="\t"
)
full_text_comparison.head()


# ## Line Plots

# In[13]:


full_plot_df = calculate_confidence_intervals(
    full_text_comparison
)
full_plot_df.head()


# In[14]:


plot_df = (
    full_plot_df
    .sort_values("odds_ratio", ascending=False)
    .head(subset)
    .append(
        full_plot_df
        .sort_values("odds_ratio", ascending=False)
        .iloc[:-2]
        .tail(subset)
    )
    .replace('rna', 'RNA')
)
plot_df.head()


# In[15]:


g = plot_pointplot(plot_df, "bioRxiv vs Reference Odds Ratio", True)
g.save("output/svg_files/biorxiv_nytac_frequency_odds.svg", dpi=75)
print(g)


# In[16]:


count_plot_df = create_lemma_count_df(
    plot_df, 'bioRxiv', 'reference'
)
count_plot_df.head()


# In[17]:


g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/biorxiv_nytac_frequency_bar.svg", dpi=75)
print(g)


# In[18]:


fig_output_path = "output/figures/biorxiv_vs_reference.png"

fig = plot_point_bar_figure(
    "output/svg_files/biorxiv_nytac_frequency_odds.svg",
    "output/svg_files/biorxiv_nytac_frequency_bar.svg"
)

# save generated SVG files
svg2png(
    bytestring=fig.to_str(), 
    write_to=fig_output_path,
    dpi=500
)

Image(fig_output_path)


# # PMC vs Reference

# In[19]:


full_text_comparison = pd.read_csv(
    "output/comparison_stats/pmc_nytac_comparison.tsv",
    sep="\t"
)
full_text_comparison.head()


# ## Line Plots

# In[20]:


full_plot_df = calculate_confidence_intervals(
    full_text_comparison
)
full_plot_df.head()


# In[21]:


plot_df = (
    full_plot_df
    .sort_values("odds_ratio", ascending=False)
    .iloc[1:]
    .head(subset)
    .append(
        full_plot_df
        .sort_values("odds_ratio", ascending=False)
        .iloc[:-2]
        .tail(subset)
    )
    .replace('rna', 'RNA')
)
plot_df.head()


# In[22]:


g = plot_pointplot(plot_df, "PMC vs Reference Odds Ratio", True)
g.save("output/svg_files/pmc_nytac_frequency_odds.svg", dpi=75)
print(g)


# In[23]:


count_plot_df = create_lemma_count_df(
    plot_df, 'pmc', 'reference'
)
count_plot_df.head()


# In[24]:


g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/pmc_nytac_frequency_bar.svg", dpi=75)
print(g)


# In[25]:


fig_output_path = "output/figures/pmc_vs_reference.png"

fig = plot_point_bar_figure(
    "output/svg_files/pmc_nytac_frequency_odds.svg",
    "output/svg_files/pmc_nytac_frequency_bar.svg"
)

# save generated SVG files
svg2png(
    bytestring=fig.to_str(), 
    write_to=fig_output_path,
    dpi=500
)

Image(fig_output_path)


# # Preprint vs Published

# In[26]:


preprint_published_comparison = pd.read_csv(
    "output/comparison_stats/preprint_to_published_comparison.tsv", 
    sep="\t"
)
preprint_published_comparison.head()


# ## Line Plot

# In[27]:


full_plot_df = calculate_confidence_intervals(
    preprint_published_comparison
)
full_plot_df.head()


# In[28]:


plot_df = (
    full_plot_df
    .sort_values("odds_ratio", ascending=False)
    .head(subset)
    .append(
        full_plot_df
        .sort_values("odds_ratio", ascending=False)
        .iloc[:-2]
        .tail(subset)
    )
)
plot_df.head()


# In[29]:


g = plot_pointplot(plot_df, "Preprint vs Published Odds Ratio")
g.save("output/svg_files/preprint_published_frequency_odds.svg", dpi=75)
print(g)


# In[30]:


count_plot_df = create_lemma_count_df(
    plot_df, 'preprint', 'published'
)
count_plot_df.head()


# In[31]:


g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/preprint_published_frequency_bar.svg", dpi=75)
print(g)


# In[32]:


fig_output_path = "output/figures/preprint_published_comparison.png"

fig = plot_point_bar_figure(
    "output/svg_files/preprint_published_frequency_odds.svg",
    "output/svg_files/preprint_published_frequency_bar.svg"
)

# save generated SVG files
svg2png(
    bytestring=fig.to_str(), 
    write_to=fig_output_path,
    dpi=500
)

Image(fig_output_path)
