#!/usr/bin/env python
# coding: utf-8

# # Tree Plots for bioRxiv and Pubmed Central

# In[1]:


import numpy as np
import pandas as pd

from cairosvg import svg2png
from IPython.display import Image
from lxml import etree
from mizani.formatters import custom_format
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotnine as p9
import svgutils.transform as sg


# In[2]:


subset = 20


# # bioRxiv vs Pubmed Central

# In[3]:


full_text_comparison = pd.read_csv("output/full_corpus_comparison_stats.tsv", sep="\t")
full_text_comparison.head()


# ## Tree Map

# In[4]:


full_text_comparison_old = (
    full_text_comparison
    .assign(repository=lambda x: x.odds_ratio.apply(lambda ratio: "bioRxiv" if ratio > 1 else "PMC"))
    .sort_values("odds_ratio", ascending=False)
)
full_text_comparison_old


# In[5]:


plot_df = (
    full_text_comparison_old.head(subset).append(full_text_comparison_old.tail(subset))
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


# In[6]:


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
        'text':f"Comparison of Top {plot_df.shape[0]} Most Frequent Words between bioRxiv and Pubmed Central (PMC)",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.update_layout(uniformtext=dict(minsize=10))
fig.show()


# In[7]:


fig.write_image("output/svg_files/biorxiv_vs_pubmed_central_treemap.svg", scale=2.5)


# ## Line Plots

# In[8]:


full_plot_df = (
    full_text_comparison
    .assign(
        lower_odds=lambda x: np.exp(
            np.log(x.odds_ratio) - 1.96*(
                # log(odds) - z_alpha/2*sqrt(1/a+1/b+1/c+1/d)
                np.sqrt(
                    1/x.biorxiv_count + 1/x.pmc_count + 1/x.biorxiv_total + 1/x.pmc_total
                )
            )
        ),
        upper_odds=lambda x: np.exp(
            np.log(x.odds_ratio) + 1.96*(
                # log(odds)+ z_alpha/2*sqrt(1/a+1/b+1/c+1/d)
                np.sqrt(
                    1/x.biorxiv_count + 1/x.pmc_count + 1/x.biorxiv_total + 1/x.pmc_total
                )
            )
        )
    )
)
full_plot_df.head()


# In[9]:


plot_df = (
    full_plot_df
    .sort_values("odds_ratio", ascending=False)
    .head(subset)
    .append(
        full_plot_df
        .sort_values("odds_ratio", ascending=False)
        .tail(subset)
    )
    .replace('rna', 'RNA')
)
plot_df.head()


# In[10]:


g = (
    p9.ggplot(plot_df, p9.aes(x="lemma", y="odds_ratio"))
    + p9.geom_pointrange(p9.aes(ymin="lower_odds", ymax="upper_odds"), position=p9.position_dodge(width=5))
    + p9.scale_x_discrete(limits=plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    + p9.scale_y_continuous(limits=[0,3.2])
    + p9.geom_hline(p9.aes(yintercept=1), linetype = '--', color='grey')
    + p9.coord_flip()
    + p9.theme_seaborn(context='paper')
    + p9.theme(
        # 1024, 768
        figure_size=(13.653333333333334, 10.24),
        axis_text_y=p9.element_text(family='DejaVu Sans', size=12),
        panel_grid_minor=p9.element_blank(),
        axis_title=p9.element_text(size=15),
        axis_text_x=p9.element_text(size=11, weight="bold")
    )
    + p9.labs(
        x=None,
        y="bioRxiv vs PMC Odds Ratio"
    )
)
g.save("output/svg_files/biorxiv_pmc_frequency_odds.svg", dpi=75)
print(g)


# In[11]:


count_plot_df = pd.DataFrame(
    list(
        zip(
            plot_df.lemma.tolist(), 
            plot_df.biorxiv_count.tolist(), 
            plot_df.assign(label='bioRxiv').label.tolist()

        )
    )
    +
    list(
        zip(
            plot_df.lemma.tolist(), 
            plot_df.pmc_count.tolist(), 
            plot_df.assign(label='pmc').label.tolist()

        )
    ),
    columns=["lemma", "count", "repository"]
)
count_plot_df.head()


# In[12]:


g = (
    p9.ggplot(count_plot_df.astype({"count":int}), p9.aes(x="lemma", y="count"))
    + p9.geom_col(position=p9.position_dodge(width=0.5))
    + p9.coord_flip()
    + p9.facet_wrap("repository", scales='free_x')
    + p9.scale_x_discrete(limits=plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    + p9.scale_y_continuous(labels=custom_format('{:,.0f}'))
    + p9.labs(x=None)
    + p9.theme_seaborn(context='paper')
    +  p9.theme(
        # 1024, 768
        figure_size=(13.653333333333334, 10.24),
        axis_text_y=p9.element_text(family='DejaVu Sans', size=12),
        panel_grid_minor=p9.element_blank(),
        axis_title=p9.element_text(size=15),
        axis_text_x=p9.element_text(size=11, weight="bold"),
        strip_text=p9.element_text(size=13)
    )
)
g.save("output/svg_files/biorxiv_pmc_frequency_bar.svg", dpi=75)
print(g)


# In[13]:


fig_output_path = "output/figures/biorxiv_vs_pubmed_central.png"
fig = sg.SVGFigure("2080", "768")
fig.append([etree.Element("rect", {"width":"100%", "height":"100%", "fill":"white"})])

fig1 = sg.fromfile("output/svg_files/biorxiv_pmc_frequency_odds.svg")
plot1 = fig1.getroot()
plot1.moveto(0, 25, scale=1.2)

fig2 = sg.fromfile("output/svg_files/biorxiv_pmc_frequency_bar.svg")
plot2 = fig2.getroot()
plot2.moveto(1024, 0, scale=1.2)

fig.append([plot1,plot2])

text_A = sg.TextElement(10, 30, "A", size=22, weight="bold")
text_B = sg.TextElement(1044, 30, "B", size=22, weight="bold")

fig.append([text_A, text_B])

# save generated SVG files
svg2png(
    bytestring=fig.to_str(), 
    write_to=fig_output_path,
    dpi=500
)

Image(fig_output_path)


# # Preprint vs Published

# In[14]:


preprint_published_comparison = pd.read_csv("output/preprint_to_published_comparison.tsv", sep="\t")
preprint_published_comparison.head()


# ## Tree Map

# In[15]:


preprint_published_comparison_old = (
    preprint_published_comparison
    .assign(repository=lambda x: x.odds_ratio.apply(lambda ratio: "preprint" if ratio > 1 else "published"))
    .sort_values("odds_ratio", ascending=False)
)
preprint_published_comparison_old


# In[16]:


plot_df = (
    preprint_published_comparison_old.head(subset).append(preprint_published_comparison_old.tail(subset))
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


# In[17]:


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
        'text':f"Comparison of Top {plot_df.shape[0]} Most Frequent Words between Preprints and Published Papers",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}
)
fig.update_layout(uniformtext=dict(minsize=10))
fig.show()


# In[18]:


fig.write_image("output/svg_files/preprint_published_comparison_treemap.svg", scale=2.5)


# ## Line Plot

# In[19]:


full_plot_df = (
    preprint_published_comparison
    .assign(
        lower_odds=lambda x: np.exp(
            np.log(x.odds_ratio) - 1.96*(
                # log(odds) - z_alpha/2*sqrt(1/a+1/b+1/c+1/d)
                np.sqrt(
                    1/x.preprint_count + 1/x.published_count + 1/x.preprint_total + 1/x.published_total
                )
            )
        ),
        upper_odds=lambda x: np.exp(
            np.log(x.odds_ratio) + 1.96*(
                # log(odds)+ z_alpha/2*sqrt(1/a+1/b+1/c+1/d)
                np.sqrt(
                    1/x.preprint_count + 1/x.published_count + 1/x.preprint_total + 1/x.published_total
                )
            )
        )
    )
)
full_plot_df.head()


# In[20]:


plot_df = (
    full_plot_df
    .sort_values("odds_ratio", ascending=False)
    .head(subset)
    .append(
        full_plot_df
        .sort_values("odds_ratio", ascending=False)
        .tail(subset)
    )
)
plot_df.head()


# In[21]:


g = (
    p9.ggplot(plot_df, p9.aes(x="lemma", y="odds_ratio"))
    + p9.geom_pointrange(p9.aes(ymin="lower_odds", ymax="upper_odds"), position=p9.position_dodge(width=5))
    + p9.scale_x_discrete(limits=plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    + p9.geom_hline(p9.aes(yintercept=1), linetype = '--', color='grey')
    + p9.coord_flip()
    + p9.theme_seaborn(context='paper')
    + p9.theme(
        # 1024, 768
        figure_size=(13.653333333333334, 10.24),
        axis_text_y=p9.element_text(family='DejaVu Sans', size=12),
        panel_grid_minor=p9.element_blank(),
        axis_title=p9.element_text(size=15),
        axis_text_x=p9.element_text(size=11, weight="bold")
    )
    + p9.labs(
        x=None,
        y="Preprint vs Published Odds Ratio"
    )
)
g.save("output/svg_files/preprint_published_frequency_odds.svg", dpi=75)
print(g)


# In[22]:


count_plot_df = pd.DataFrame(
    list(
        zip(
            plot_df.lemma.tolist(), 
            plot_df.preprint_count.tolist(), 
            plot_df.assign(label='preprint').label.tolist()

        )
    )
    +
    list(
        zip(
            plot_df.lemma.tolist(), 
            plot_df.published_count.tolist(), 
            plot_df.assign(label='published').label.tolist()

        )
    ),
    columns=["lemma", "count", "category"]
)
count_plot_df.head()


# In[23]:


g = (
    p9.ggplot(count_plot_df, p9.aes(x="lemma", y="count"))
    + p9.geom_col(position=p9.position_dodge(width=0.5))
    + p9.coord_flip()
    + p9.facet_wrap("category")
    + p9.scale_x_discrete(limits=plot_df.sort_values("odds_ratio", ascending=True).lemma.tolist())
    + p9.scale_y_continuous(labels=custom_format('{:,g}'))
    + p9.labs(x=None)
    + p9.theme_seaborn(context='paper')
    +  p9.theme(
        # 1024, 768
        figure_size=(13.653333333333334, 10.24),
        axis_text_y=p9.element_text(family='DejaVu Sans', size=12),
        panel_grid_minor=p9.element_blank(),
        axis_title=p9.element_text(size=15),
        axis_text_x=p9.element_text(size=11, weight="bold"),
        strip_text=p9.element_text(size=13)
    )
)
g.save("output/svg_files/preprint_published_frequency_bar.svg", dpi=75)
print(g)


# In[25]:


fig_output_path = "output/figures/preprint_published_comparison.png"
fig = sg.SVGFigure("2080", "768")
fig.append([etree.Element("rect", {"width":"100%", "height":"100%", "fill":"white"})])

fig1 = sg.fromfile("output/svg_files/preprint_published_frequency_odds.svg")
plot1 = fig1.getroot()
plot1.moveto(0, 25, scale=1.2)

fig2 = sg.fromfile("output/svg_files/preprint_published_frequency_bar.svg")
plot2 = fig2.getroot()
plot2.moveto(1004, 0, scale=1.2)

fig.append([plot1,plot2])

text_A = sg.TextElement(10, 30, "A", size=22, weight="bold")
text_B = sg.TextElement(1024, 30, "B", size=22, weight="bold")

fig.append([text_A, text_B])

# save generated SVG files
svg2png(
    bytestring=fig.to_str(), 
    write_to=fig_output_path,
    dpi=500
)

Image(fig_output_path)

