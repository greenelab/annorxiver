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
        kl_divergence_df
        .replace({
            "biorxiv_vs_pmc": "bioRxiv-PMC",
            "biorxiv_vs_nytac":"bioRxiv-NYTAC",
            "pmc_vs_nytac": "PMC-NYTAC",
        })
        .rename(
            index=str, 
            columns={"comparison": "Comparison"}
        )
    )
    + p9.aes(
        x="factor(num_terms)", y="KL_divergence", 
        fill="Comparison", color="Comparison",
        group = "Comparison"
    )
    + p9.geom_point(size=2)
    + p9.geom_line(linetype='dashed')
    + p9.scale_fill_brewer(
        type='qual', 
        palette='Paired',
        direction=-1
    )
    + p9.scale_color_brewer(
        type='qual', 
        palette='Paired',
        direction=-1,
    )
    + p9.labs(
        x = "Number of terms evaluated",
        y = "Kullback–Leibler Divergence",
    )
    + p9.theme_seaborn(
        context='paper',
        style="ticks",
        font_scale=1.2,
    )
    + p9.theme(
        figure_size=(10, 6),
        text=p9.element_text(family="Arial")
    )
)
g.save("output/svg_files/corpora_kl_divergence.svg")
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
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x))
    )
)
plot_df.head()


# In[8]:


g = (
    p9.ggplot(
        plot_df, 
        p9.aes(
            y="lemma", x="odds_ratio", 
            xmin="lower_odds", xmax="upper_odds"
        )
    )
    + p9.geom_point(
        #position=p9.position_dodge(width=1),
        size=1,
        color="#253494"
    )
    + p9.scale_y_discrete(
        limits=(
            plot_df
            .sort_values("odds_ratio", ascending=True)
            .lemma
            .tolist()
        )
    )
    + p9.scale_x_continuous(
        limits=(-3, 3)
    )
    + p9.geom_vline(
        p9.aes(xintercept=0), 
        linetype = '--', color='grey'
    )
    + p9.annotate(
        "segment", 
        x = 0.5, xend = 2.5, 
        y = 1.5, yend = 1.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="bioRxiv Enriched",
        x=1.5, y=2.5,
        size=12,
        alpha=0.7
    )
    + p9.annotate(
        "segment", 
        x = -0.5, xend = -2.5, 
        y = 39.5, yend = 39.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="PMC Enriched",
        x=-1.5, y=38.5,
        size=12,
        alpha=0.7
    )
    + p9.theme_seaborn(
        context='paper', 
        style="ticks", 
        font_scale=1.1, 
        font='Arial'
    )
    + p9.theme(
        figure_size=(10, 6),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(
        y=None,
        x="bioRxiv vs PMC log2(Odds Ratio)"
    )
)
g.save("output/svg_files/biorxiv_pmc_frequency_odds.svg")
g.save("output/svg_files/biorxiv_pmc_frequency_odds.png", dpi=75)
print(g)


# In[9]:


count_plot_df = (
    create_lemma_count_df(plot_df, 'bioRxiv', 'pmc')
    .replace({"pmc": "PMC"})
    .assign(
        repository = lambda x: pd.Categorical(
            x.repository.tolist(), 
            categories=["bioRxiv", "PMC"]
        )
    )
)
count_plot_df.head()


# In[10]:


g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/biorxiv_pmc_frequency_bar.svg")
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
    dpi=75
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
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x))
    )
)
plot_df.head()


# In[15]:


g = (
    p9.ggplot(
        plot_df, 
        p9.aes(
            y="lemma", x="odds_ratio", 
            xmin="lower_odds", xmax="upper_odds"
        )
    )
    + p9.geom_point(
        size=1,
        color="#253494"
    )
    + p9.scale_y_discrete(
        limits=(
            plot_df
            .sort_values("odds_ratio", ascending=True)
            .lemma
            .tolist()
        )
    )
    + p9.scale_x_continuous(
        limits=(-20, 20)
    )
    + p9.geom_vline(
        p9.aes(xintercept=0), 
        linetype = '--', color='grey'
    )
    + p9.annotate(
        "segment", 
        x = 5, xend = 17, 
        y = 1.5, yend = 1.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="bioRxiv Enriched",
        x=9, y=2.5,
        size=12,
        alpha=0.7
    )
    + p9.annotate(
        "segment", 
        x = -5, xend = -17, 
        y = 39.5, yend = 39.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="NYTAC Enriched",
        x=-9, y=38.5,
        size=12,
        alpha=0.7
    )
    + p9.theme_seaborn(
        context='paper', 
        style="ticks", 
        font_scale=1.1, 
        font='Arial'
    )
    + p9.theme(
        figure_size=(10, 6),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(
        y=None,
        x="bioRxiv vs NYTAC log2(Odds Ratio)"
    )
)

g.save("output/svg_files/biorxiv_nytac_frequency_odds.svg")
g.save("output/svg_files/biorxiv_nytac_frequency_odds.png", dpi=250)
print(g)


# In[16]:


count_plot_df = (
    create_lemma_count_df(
        plot_df, 'bioRxiv', 'NYTAC'
    )
    .assign(
        repository = lambda x: pd.Categorical(
            x.repository.tolist(), 
            categories=["bioRxiv", "NYTAC"]
        )
    )
)
count_plot_df.head()


# In[17]:


g = plot_bargraph(count_plot_df, plot_df)
g.save("output/svg_files/biorxiv_nytac_frequency_bar.svg")
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
    dpi=75
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
    .drop([152, 160])
    .head(subset)
    .append(
        full_plot_df
        .sort_values("odds_ratio", ascending=False)
        .iloc[:-2]
        .tail(subset)
    )
    .replace('rna', 'RNA')
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x))
    )
)
plot_df.head()


# In[22]:


g = (
    p9.ggplot(
        plot_df, 
        p9.aes(
            y="lemma", x="odds_ratio", 
            xmin="lower_odds", xmax="upper_odds"
        )
    )
    + p9.geom_point(
        size=1,
        color="#253494"
    )
    + p9.scale_y_discrete(
        limits=(
            plot_df
            .sort_values("odds_ratio", ascending=True)
            .lemma
            .tolist()
        )
    )
    + p9.scale_x_continuous(
        limits=(-20, 20)
    )
    + p9.geom_vline(
        p9.aes(xintercept=0), 
        linetype = '--', color='grey'
    )
    + p9.annotate(
        "segment", 
        x = 5, xend = 17, 
        y = 1.5, yend = 1.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="bioRxiv Enriched",
        x=9, y=2.5,
        size=12,
        alpha=0.7
    )
    + p9.annotate(
        "segment", 
        x = -5, xend = -17, 
        y = 39.5, yend = 39.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="NYTAC Enriched",
        x=-9, y=38.5,
        size=12,
        alpha=0.7
    )
    + p9.theme_seaborn(
        context='paper', 
        style="ticks", 
        font_scale=1.1, 
        font='Arial'
    )
    + p9.theme(
        figure_size=(10, 6),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(
        y=None,
        x="PMC vs NYTAC log2(Odds Ratio)"
    )
)
g.save("output/svg_files/pmc_nytac_frequency_odds.svg")
g.save("output/svg_files/pmc_nytac_frequency_odds.png", dpi=250)
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
    dpi=75
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
        .iloc[:-3]
        .tail(subset)
    )
    .assign(
        odds_ratio=lambda x: x.odds_ratio.apply(lambda x: np.log2(x)),
        lower_odds=lambda x: x.lower_odds.apply(lambda x: np.log2(x)),
        upper_odds=lambda x: x.upper_odds.apply(lambda x: np.log2(x))
    )
)
plot_df.head()


# In[29]:


g = (
    p9.ggplot(
        plot_df, 
        p9.aes(
            y="lemma", x="odds_ratio", 
            xmin="lower_odds", xmax="upper_odds"
        )
    )
    + p9.geom_point(
        #position=p9.position_dodge(width=1),
        size=1,
        color="#253494"
    )
    + p9.scale_y_discrete(
        limits=(
            plot_df
            .sort_values("odds_ratio", ascending=True)
            .lemma
            .tolist()
        )
    )
    + p9.scale_x_continuous(
        limits=(-3, 3)
    )
    + p9.geom_vline(
        p9.aes(xintercept=0), 
        linetype = '--', color='grey'
    )
    + p9.annotate(
        "segment", 
        x = 0.5, xend = 2.5, 
        y = 1.5, yend = 1.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="Preprint Enriched",
        x=1.5, y=2.5,
        size=12,
        alpha=0.7
    )
    + p9.annotate(
        "segment", 
        x = -0.5, xend = -2.5, 
        y = 39.5, yend = 39.5, 
        colour = "black", 
        size=0.5, alpha=1,
        arrow=p9.arrow(length=0.1)
    )
    + p9.annotate(
        "text",
        label="Published Enriched",
        x=-1.5, y=38.5,
        size=12,
        alpha=0.7
    )
    + p9.theme_seaborn(
        context='paper', 
        style="ticks", 
        font_scale=1.1, 
        font='Arial'
    )
    + p9.theme(
        figure_size=(10, 6),
        panel_grid_minor=p9.element_blank(),
    )
    + p9.labs(
        y=None,
        x="Preprint vs Published log2(Odds Ratio)"
    )
)
g.save("output/svg_files/preprint_published_frequency_odds.svg")
g.save("output/svg_files/preprint_published_frequency_odds.png", dpi=250)
print(g)


# In[30]:


count_plot_df = (
    create_lemma_count_df(
        plot_df, 'preprint', 'published'
    )
    .replace({"preprint":"Preprint", "published":"Published"})
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
    dpi=75
)

Image(fig_output_path)

