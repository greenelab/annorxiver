#!/usr/bin/env python
# coding: utf-8

# # PCA Plots of bioRxiv

# This notebook is designed to run PCA over the document embeddings and plot various components against each other. The goal here is to understand the concepts best captured by each PC.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys

import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from annorxiver_modules.pca_plot_helper import *


# In[2]:


# Set up porting from python to R
# and R to python :mindblown:

import rpy2.rinterface
get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[3]:


journal_map_df = pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_metadata.tsv", sep="\t")
journal_map_df.head()


# # PCA the Documents

# Run PCA over the documents. Generates 50 principal components, but can generate more or less.

# In[4]:


n_components = 50
random_state = 100


# In[ ]:


biorxiv_articles_df = pd.read_csv(
    "../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz", 
    sep="\t"
)


# In[ ]:


reducer = PCA(
    n_components = n_components,
    random_state = random_state
)

embedding = reducer.fit_transform(
    biorxiv_articles_df[[f"feat_{idx}" for idx in range(300)]].values
)

pca_df = (
    pd.DataFrame(embedding, columns=[f"pca{dim}" for dim in range(1, n_components+1, 1)])
    .assign(document=biorxiv_articles_df.document.values.tolist())
    .merge(journal_map_df[["category", "document", "doi"]], on="document")
)

latest_version = (
    pca_df
    .groupby("doi", as_index=False)
    .agg({"document":"last"})
    .document
    .values
    .tolist()
)

pca_df = (
    pca_df
    .query(f"document in {latest_version}")
    .reset_index(drop=True)
)

pca_df.head()


# In[ ]:


reducer.explained_variance_


# In[ ]:


reducer.explained_variance_ratio_


# In[ ]:


(
    pca_df
    .category
    .sort_values()
    .unique()
)


# # Plot the PCA axes

# This section visualizes PCA axes and attempts to provide an explanation for each plotted PC.
# Give there are 50 pcs generated this notebook/section will only focus on PCs that generate a clear enough signal.

# ## Basis PCs

# When analyzing PCs, it is generally a good idea to start off with the PCs that are easily distinguishable and then branch out to PCs that are harder to interpret. This section focuses on PCs: 1,2,3,4 and 5.
# 
# These PCs correspond to the following concepts:
# 
# | PC | Concept (negative vs positive)|
# | --- | --- |
# | PCA1 | Quantitative Biology vs Molecular Biology |
# | PCA2 | Neuroscience vs Genomics |
# | PCA3 | Sequencing vs Disease |
# | PCA4 | Microbiology vs Cell Biology |
# | PCA5 | RNA-seq vs Evolutional Biology | 

# In[ ]:


global_color_palette = [
    '#a6cee3','#1f78b4',
    '#b2df8a','#33a02c',
    '#fb9a99'
]


# ### PCA1 vs PCA2

# In[ ]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png'
)


# These word clouds depict the following concepts: quantitative biology vs molecular biology (left) and genomics vs neuroscience (right). The cells below provide evidence for the previous claim

# In[ ]:


selected_categories = [
    'biochemistry', 'bioinformatics',
    'cell biology', 'neuroscience',
    'scientific communication'
]


# In[ ]:


pca_sample_df = (
    pca_df
    .query(f"category in {selected_categories}")
    .groupby("category")
    .apply(lambda x: x.sample(200, random_state=100) if len(x) > 200 else x)
    .reset_index(drop=True)
)
pca_sample_df.head()


# In[ ]:


get_ipython().run_cell_magic('R', '-i pca_sample_df', '# have to switch to R as it has a better "layout manager"\n# https://github.com/has2k1/plotnine/issues/46\nlibrary(ggplot2)\n\ncolor_mapper <- c(\n    \'biochemistry\' = \'#a6cee3\', \n    \'bioinformatics\'= \'#1f78b4\',\n    \'cell biology\'=\'#b2df8a\',\n    \'neuroscience\'=\'#33a02c\',\n    \'scientific communication\'=\'#fb9a99\'\n)\n\ng <- (\n        ggplot(pca_sample_df)\n        + aes(x=pca1, y=pca2, color=factor(category))\n        + theme_bw()\n        + theme(\n            legend.position="left",\n            text=element_text(family = "Arial", size=16),\n            rect=element_rect(color="black"),\n            panel.grid.major = element_blank(),\n            panel.grid.minor = element_blank()\n        )\n        + geom_point()\n        + scale_y_continuous(position="right")\n        + scale_color_manual(values=color_mapper)\n        + labs(\n            x="PC1",\n            y="PC2",\n            color="Article Category",\n            title="PCA of BioRxiv (Word Dim: 300)"\n        )\n)\n\nCairo::CairoSVG(\n    file="output/pca_plots/svg_files/scatterplot_files/pca01_v_pca02_reversed.svg",\n    height=5,\n    width=10,\n)\n\nprint(g)')


# In[ ]:


generate_scatter_plots(
    pca_df,
    x="pca1", y="pca2",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca01_v_pca02.svg"
)


# In[ ]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca01_v_pca02.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca01_v_pca02_figure.png"
)


# Take note that pca2 clusters neruoscience papers on the negative axis while genomics papers are on the positive axis (up and down). PCA 1 places papers that are more focused on quantitative biology on the right and molecular biology to the left. Hence why bioinforamtics papers are shifted more to the right and cell biology papers are shifted more to the left. This plot visually confirms the above finding.

# ### PCA1 vs PCA 3

# In[17]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_03_cossim_word_cloud.png'
)


# These word clouds depict the following concepts: quantitative biology vs molecular biology (left) and disease vs sequencing (right)

# In[18]:


selected_categories = [
    'bioinformatics','epidemiology', 
    'genetics', 'paleontology',
    'pathology'
]


# In[19]:


generate_scatter_plots(
    pca_df,
    x="pca1", y="pca3",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca01_v_pca03.svg"
)


# In[20]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca01_v_pca03.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_03_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca01_v_pca03_figure.png"
)


# Take note that pca3 clusters sequencing based papers on the negative axis (down) while disease papers are on the positive axis (up). When plotted against PCA1 it looks like epidemiology papers dominate the top right quadrant, while bottom right quadrant contains bioinformatics papers. This makes sense as many bioinformatic papers consist of some sort of sequencing technologies, while epidemiology is focused on measuring disease and its progression. Both take quantitative views, which is why they are on the positive size of PCA1.

# ### PCA2 vs PCA3

# In[21]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_03_cossim_word_cloud.png'
)


# These word clouds depict the following concepts: neuroscience to genomics (left) and disease vs sequencing (right)

# In[22]:


selected_categories = [
    'epidemiology', 'genetics', 
    'genomics', 'neuroscience',
    'pathology'
]


# In[23]:


generate_scatter_plots(
    pca_df,
    x="pca2", y="pca3",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca02_v_pca03.svg"
)


# In[24]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca02_v_pca03.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_03_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca02_v_pca03_figure.png"
)


# Note that bottom right quadrant comprises of mainly bioinformatics papers, which makes sense given that quadrant represents sequencing and genomics related papers (hence bioinformatics). The bottom left quadrant contains papers that have sequencing terms mentioned, but are more related to neuroscience than genomics (thats what forms the biophysics clusters). The top left are papers that relate to neuroscience and focused on disease while top right are genomics related papers that focus on disease.

# ### PCA3 vs PCA5 

# In[25]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_03_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_05_cossim_word_cloud.png'
)


# These word clouds depict the following concepts: sequencing vs disease (left) and RNA-seq vs evolutionary biology (right)

# In[26]:


selected_categories = [
    'bioinformatics','ecology',
    'evolutionary biology','epidemiology',
    'paleontology'
]


# In[27]:


generate_scatter_plots(
    pca_df,
    x="pca3", y="pca5",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca03_v_pca05.svg"
)


# In[28]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca03_v_pca05.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_03_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_05_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca03_v_pca05_figure.png"
)


# In[29]:


(
    pca_df[["pca3", "pca5", "category", "doi"]]
    .query("pca3 > 0 & pca5 > 0")
    .category
    .value_counts()
    .head(10)
)


# In[30]:


(
    pca_df[["pca3", "pca5", "category", "doi"]]
    .query("pca3 < 0 & pca5 < 0")
    .category
    .value_counts()
    .head(10)
)


# Looking at the top-right quadrant and bottom-left quadrant, the top 10 categories provide evidence for the concepts mentioned above. Since PCA5 contains RNA-seq concepts on the negative sdes and PCA3 has sequencing as well on its negative axis, one would expect the top category for the bottom-left quadrant be bioinformatics related. Likewise the top right should be focused on evolutionary biology and possibly disease.

# ### PCA1 vs PCA4

# In[31]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_04_cossim_word_cloud.png'
)


# These word cloud produces the following concepts: qunatitative biology vs molecular biology (left) and microbiology vs cell biology (right).

# In[32]:


selected_categories = [
    'cell biology', 'epidemiology',
    'immunology', 'microbiology',
    'systems biology'
]


# In[33]:


generate_scatter_plots(
    pca_df,
    x="pca1", y="pca4",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca01_v_pca04.svg"
)


# In[34]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca01_v_pca04.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_04_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca01_v_pca04_figure.png"
)


# In[35]:


(
    pca_df
    [["pca1", "pca4", "category", "doi"]]
    .query("pca1 < -2 & pca4 > 0")
    .category
    .value_counts()
    .head(10)
)


# In[36]:


(
    pca_df
    [["pca1", "pca4", "category", "doi"]]
    .query("pca1 < 0 & pca4 < 0")
    .category
    .value_counts()
    .head(10)
)


# Looking on the left size of PCA1 (negative), it looks like the top quadrant contains mainly cell biology papers and variants of cell biology. The bottom quadrant contains papers that are related to microbiology; Fun anecdote is that I super convinced that PCA4 was marine biology, but upon closer inspection turns out I was wrong it is microbio.

# # Non-straightforward PCs

# This section of the notebook aims to take a look at PCs that are not as straightforward as the ones above.

# ## PCA1 vs PCA6

# In[37]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_06_cossim_word_cloud.png'
)


# The right word cloud appears to represent mathematics vs scientific communication or at least popular buzz words scientist used to promote their research. The next few cells will look more into it.

# In[38]:


selected_categories = [
    'biophysics', 'bioengineering',
    'clinical trials', 'scientific communication', 
    'synthetic biology'
]


# In[39]:


generate_scatter_plots(
    pca_df,
    x="pca1", y="pca6",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca01_v_pca06.svg"
)


# In[40]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca01_v_pca06.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_01_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_06_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca01_v_pca06_figure.png"
)


# In[41]:


(
    pca_df[["pca1", "pca6", "category", "doi"]]
    .drop_duplicates("doi")
    .query("pca1 > 0 & pca6 > 3")
    .category
    .value_counts()
    .head(10)
)


# In[42]:


(
    pca_df[["pca1", "pca6", "category", "doi"]]
    .drop_duplicates("doi")
    .query("pca1 > 0 & pca6 < -1.5")
    .category
    .value_counts()
    .head(10)
)


# Looking at the top categories for the top and bottom right quadrants it seems that the papers follow the patterns captures by the word clouds above; however the positive axis still remains difficult to judge without taking a look at the individual papers.

# In[43]:


(
    pca_df[["pca1", "pca6", "category", "doi"]]
    .drop_duplicates("doi")
    .query("pca1 > 0 & pca6 > 3")
    .sort_values("pca6", ascending=False)
    .head(10)
)


# ## PCA2 vs PCA15

# In[44]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_15_cossim_word_cloud.png'
)


# The word cloud on the right seems to contain the following concepts: facial recognition and behavior vs neuron biochemistry.

# In[45]:


selected_categories = [
    'animal behavior and cognition',
    'biochemistry','cell biology',
    'molecular biology',
    'neuroscience'
]


# In[46]:


generate_scatter_plots(
    pca_df,
    x="pca2", y="pca15",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca02_v_pca15.svg"
)


# In[47]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca02_v_pca15.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_15_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca02_v_pca15_figure.png"
)


# This graph depicts diversity within the neuroscience field as some papers are about facial recognition (negative) and other papers are about biochemistry (positive).

# In[48]:


(
    pca_df
    [["pca2", "pca15", "category", "document","doi"]]
    .sort_values(["pca15", "pca2"], ascending=[False, False])
    .head(10)
)


# In[49]:


(
    pca_df
    [["pca2", "pca15", "category", "document","doi"]]
    .sort_values(["pca15", "pca2"], ascending=[False, False])
    .tail(10)
)


# These papers confirm that the negative axis of PCA15 is facial recognition.

# ## PCA2 vs PCA8

# In[50]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_08_cossim_word_cloud.png'
)


# The wordcloud on the right seems to represent the following concept:  biochemistry vs developmental biology. Main evidence for this appears in the plot below.

# In[51]:


selected_categories = [
    'biochemistry', 'biophysics',
    'cell biology', 'developmental biology', 
    'plant biology'
]


# In[52]:


generate_scatter_plots(
    pca_df,
    x="pca2", y="pca8",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca02_v_pca08.svg"
)


# In[53]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca02_v_pca08.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_08_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca02_v_pca08_figure.png"
)


# In[54]:


(
    pca_df
    [["pca2", "pca8", "category", "doi"]]
    .query("pca2 > -2 & pca2 < 2 & pca8 < -1")
    .category
    .value_counts()
    .head(10)
)


# In[55]:


(
    pca_df
    [["pca2", "pca8", "category", "doi"]]
    .query("pca2 > -2 & pca2 < 2 & pca8 > 1")
    .category
    .value_counts()
    .head(10)
)


# Looking at the top left and bottom left quadrants the top categories are: biochemistry and developmental biology. Based on this confirmation I'd argue that pca8 covers both of these concepts.

# ## PCA2 VS PCA13

# In[56]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_13_cossim_word_cloud.png'
)


# Based on a quick google search the wordcloud on the right represents: viruses (immunology) vs model organisms.

# In[57]:


selected_categories = [
    'animal behavior and cognition','developmental biology'
    'genetics', 'immunology',
    'microbiology'
]


# In[58]:


generate_scatter_plots(
    pca_df,
    x="pca2", y="pca13",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca02_v_pca13.svg"
)


# In[59]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca02_v_pca13.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_02_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_13_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca02_v_pca13_figure.png"
)


# In[60]:


(
    pca_df
    [["pca2", "pca13", "category", "doi"]]
    .sort_values("pca13", ascending=False)
    .head(10)
)


# In[61]:


(
    pca_df
    [["pca2", "pca13", "category", "doi"]]
    .sort_values("pca13", ascending=True)
    .head(10)
)


# Looking at the extremes values along PCA13, the categories seem to confirm my suspicions.

# ## PCA04 vs PCA20

# In[62]:


display_clouds(
    'output/word_pca_similarity/figure_pieces/pca_04_cossim_word_cloud.png',
    'output/word_pca_similarity/figure_pieces/pca_20_cossim_word_cloud.png'
)


# PCA20 represents the following concepts: immunology and cancer biology.

# In[63]:


selected_categories = [
    'cancer biology', 'immunology',
    'molecular biology','microbiology',
    'neuroscience'
]


# In[64]:


generate_scatter_plots(
    pca_df,
    x="pca4", y="pca20",
    nsample=200, random_state=100,
    selected_categories=selected_categories,
    color_palette=global_color_palette,
    save_file_path="output/pca_plots/svg_files/scatterplot_files/pca04_v_pca20.svg"
)


# In[65]:


plot_scatter_clouds(
    scatter_plot_path = "output/pca_plots/svg_files/scatterplot_files/pca04_v_pca20.svg", 
    word_cloud_x_path = "output/word_pca_similarity/figure_pieces/pca_04_cossim_word_cloud.png",
    word_cloud_y_path = "output/word_pca_similarity/figure_pieces/pca_20_cossim_word_cloud.png",
    final_figure_path = "output/pca_plots/figures/pca04_v_pca20_figure.png"
)


# In[66]:


(
    pca_df
    [["pca4", "pca20", "category", "doi"]]
    .query("pca4 < 0 & pca20 < 0")
    .category
    .value_counts()
    .head(10)
)


# In[67]:


(
    pca_df
    [["pca4", "pca20", "category", "doi"]]
    .query("pca4 > 0 & pca20 > 0")
    .category
    .value_counts()
    .head(10)
)

