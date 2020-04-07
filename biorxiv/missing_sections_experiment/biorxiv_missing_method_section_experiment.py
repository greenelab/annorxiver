#!/usr/bin/env python
# coding: utf-8

# # BioRxiv Missing Method Section Detection

# This notebook is designed to plot preprints where the method section cannot be detected using my document parser.

# In[1]:


import pandas as pd
import plotnine as p9


# In[2]:


biorxiv_sections_df = (
    pd.read_csv("../exploratory_data_analysis/output/biorxiv_article_sections.tsv", sep="\t")
    .assign(document=lambda x: x.document.apply(lambda x: f"{x}.xml"))
)
biorxiv_sections_df.head()


# # PCA Embeddings for bioRxiv

# In[3]:


biorxiv_embeddings_df = pd.read_csv(
    "../word_vector_experiment/output/embedding_output/pca/biorxiv_pca_300.tsv", 
    sep="\t"
)
biorxiv_embeddings_df.head()


# In[4]:


biorxiv_pca_method_section_df = (
    pd.merge(
        biorxiv_embeddings_df, 
        biorxiv_sections_df.query("section=='material and methods'").drop_duplicates(),
        on="document",
        how="left"
    )
    .assign(section=lambda x: x.section.apply(lambda x: "has_methods" if x=="material and methods" else "no_methods"))
    .groupby("doi")
    .agg({
        "doi": "last",
        "document": "last",
        "pca1": "last",
        "pca2": "last",
        "category":"last",
        "section": "last",
    })
    .reset_index(drop=True)
)
biorxiv_pca_method_section_df.head()


# ## Global View of PCA plot

# In[5]:


g = (
    p9.ggplot(
        biorxiv_pca_method_section_df
    )
    + p9.aes(x="pca1", y="pca2", color="category")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.labs(
        title="TSNE Methods Section (300 dim)"
    )
)
print(g)


# ## Neuroscience Methods Section

# In[6]:


g = (
    p9.ggplot(
        biorxiv_pca_method_section_df
        .query("category=='neuroscience'")
    )
    + p9.aes(x="pca1", y="pca2", color="section")
    + p9.geom_point(position=p9.position_dodge(width=0.2))
    + p9.facet_wrap("section")
    + p9.theme_bw()
    + p9.theme(
        subplots_adjust={'wspace':0.10}
    )
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Neuroscience Methods Section"
    )
)
g.save("output/pca/neuroscience_missing_methods.png", dpi=500)
print(g)


# In[7]:


g = (
    p9.ggplot(
        biorxiv_pca_method_section_df
        .query("category=='neuroscience'")
    )
    + p9.aes(x="pca1", y="pca2", color="section")
    + p9.geom_point(position=p9.position_dodge(width=0.2))
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Neuroscience Methods Section"
    )
)
g.save("output/pca/neuroscience_missing_methods_overlapped.png", dpi=500)
print(g)


# # TSNE Embeddings for bioRxiv

# In[8]:


biorxiv_embeddings_df = pd.read_csv(
    "../word_vector_experiment/output/embedding_output/tsne/biorxiv_tsne_300.tsv", 
    sep="\t"
)
biorxiv_embeddings_df.head()


# In[9]:


biorxiv_tsne_method_section_df = (
    pd.merge(
        biorxiv_embeddings_df, 
        biorxiv_sections_df.query("section=='material and methods'").drop_duplicates(),
        on="document",
        how="left"
    )
    .assign(section=lambda x: x.section.apply(lambda x: "has_methods" if x=="material and methods" else "no_methods"))
    .groupby("doi")
    .agg({
        "doi": "last",
        "document": "last",
        "tsne1": "last",
        "tsne2": "last",
        "category":"last",
        "section": "last",
    })
    .reset_index(drop=True)
)
biorxiv_tsne_method_section_df.head()


# ## Global View of tSNE plot

# In[10]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
    )
    + p9.aes(x="tsne1", y="tsne2", color="category")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.labs(
        title="TSNE Methods Section (300 dim)"
    )
)
print(g)


# ## Neuroscience Methods Section

# In[11]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
        .query("category=='neuroscience'")
    )
    + p9.aes(x="tsne1", y="tsne2", color="section")
    + p9.geom_point(position=p9.position_dodge(width=0.2))
    + p9.facet_wrap("section")
    + p9.theme_bw()
    + p9.theme(
        subplots_adjust={'wspace':0.10}
    )
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Neuroscience Methods Section"
    )
)
g.save("output/tsne/neuroscience_missing_methods.png", dpi=500)
print(g)


# In[12]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
        .query("category=='neuroscience'")
    )
    + p9.aes(x="tsne1", y="tsne2", color="section")
    + p9.geom_point(position=p9.position_dodge(width=0.2))
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Neuroscience Methods Section"
    )
)
g.save("output/tsne/neuroscience_missing_methods_overlapped.png", dpi=500)
print(g)


# In[13]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
        .query("category=='neuroscience'")
    )
    + p9.aes(x="tsne1", y="tsne2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Neuroscience Methods Section"
    )
    + p9.annotate("rect", xmin=-30, xmax=0, ymin=-50, ymax=-30, alpha=0.4)
)
print(g)


# In[14]:


(
    biorxiv_tsne_method_section_df
    .query("category=='neuroscience'")
    .query("tsne1 > -30 & tsne1 <=0")
    .query("tsne2 < -30")
)


# Personal Inspection:

# | Document | Comment |
# | --- | --- |
# | [10.1101/053827](https://doi.org/10.1101/053827) | Dense evolutionary biology paper/neuroscience? |
# | [10.1101/164780](https://doi.org/10.1101/164780) | Dense evolutionary biology paper/neuroscience? |
# | [10.1101/164780](https://doi.org/10.1101/164780) |  has a brief methods section at the end of the paper, but it isn't really a methods section |
# | [10.1101/053827](https://doi.org/10.1101/053827) | doesn't have a methods section at all |
# | [10.1101/585760](https://doi.org/10.1101/585760) | has the methods section in the supplement|
# | [10.1101/004176](https://doi.org/10.1101/004176) | A prime candidate for missing sections. This is a brief article, which is missing a lot of sections except for results. |

# ## Bioinformatics Methods Section

# In[15]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
        .query("category=='bioinformatics'")
    )
    + p9.aes(x="tsne1", y="tsne2", color="section")
    + p9.geom_point()
    + p9.theme_bw()

    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Bioinformatics Methods Section"
    )
)
g.save("output/tsne/bioinformatics_missing_methods_overlapped.png", dpi=500)
print(g)


# In[16]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
        .query("category=='bioinformatics'")
    )
    + p9.aes(x="tsne1", y="tsne2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Bioinformatics Methods Section"
    )
    + p9.annotate("rect", xmin=0, xmax=-15, ymin=-60, ymax=-40, alpha=0.4)
)
print(g)


# In[17]:


(
    biorxiv_tsne_method_section_df
    .query("category=='bioinformatics'")
    .query("tsne1 > 0 & tsne1 < 15")
    .query("tsne2 < -40")
)


# Personal Interpretation:

# | Document | Comment |
# | --- | --- |
# | [10.1101/015503](https://doi.org/10.1101/015503) | I don't think should be in biorxiv given that it is just a technical note and not a new discovery. (Plus doesn't need a methods section |
# | [10.1101/031047](https://doi.org/10.1101/031047) | describes the algorithm used, which can be considered a methods section, but I argue that this isn't a regular research article nor is it really bio related. More like statistics. |
# | [10.1101/234948](https://doi.org/10.1101/234948) | a statistics paper that talks about population dynamics testing. At first I thought it was a population genetics paper, but it really is a statistics paper |
# | [10.1101/392944](https://doi.org/10.1101/392944) | Statistics for microbe growth modeling. It talks about a software package, so no direct methods section |
# | [10.1101/835181](https://doi.org/10.1101/835181) | has a method section, so this is a false positive. Could be an xml parsing issue. | 

# In[18]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
        .query("category=='bioinformatics'")
    )
    + p9.aes(x="tsne1", y="tsne2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Bioinformatics Methods Section"
    )
    + p9.annotate("rect", xmin=30, xmax=45, ymin=30, ymax=50, alpha=0.4)
)
print(g)


# In[19]:


(
    biorxiv_tsne_method_section_df
    .query("category=='bioinformatics'")
    .query("tsne1 > 30 & tsne1 < 45")
    .query("tsne2 > 30")
    .query("section=='no_methods'")
)


# Personal Interpretation:

# | Document | Comment |
# | --- | --- |
# | [10.1101/002824](https://doi.org/10.1101/002824) | software paper so no real need for a methods section |
# | [10.1101/002881](https://doi.org/10.1101/002881) | software paper so no real need for a methods section |
# | [10.1101/003723](https://doi.org/10.1101/003723) | software paper so no real need for a methods section |
# | [10.1101/859900](https://doi.org/10.1101/859900) | software paper so no real need for a methods section |
# | [10.1101/866087](https://doi.org/10.1101/866087) | software paper so no real need for a methods section |
# | [10.1101/870170](https://doi.org/10.1101/870170) | software paper so no real need for a methods section | 
# | [10.1101/845529](https://doi.org/10.1101/845529) | software paper so no real need for a methods section |

# ## Microbiology Methods Section

# In[20]:


g = (
    p9.ggplot(
        biorxiv_tsne_method_section_df
        .query("category=='microbiology'")
    )
    + p9.aes(x="tsne1", y="tsne2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Microbiology Methods Section"
    )
    + p9.annotate("rect", xmin=30, xmax=45, ymin=30, ymax=60, alpha=0.4)
)
print(g)


# In[21]:


(
    biorxiv_tsne_method_section_df
    .query("category=='microbiology'")
    .query("tsne1 > 30 & tsne1 < 45")
    .query("tsne2 > 30")
)


# Personal Interpretation:

# | Document | Comment |
# | --- | --- |
# | [10.1101/2020.01.20.913202](https://doi.org/10.1101/2020.01.20.913202) | article withdrawals, so obviously a method section is not needed |
# | [10.1101/612093](https://doi.org/10.1101/612093) | article withdrawals, so obviously a method section is not needed |
# | [10.1101/176784](https://doi.org/10.1101/176784) | software paper |
# | [10.1101/215525](https://doi.org/10.1101/215525) | software paper |
# | [10.1101/482380](https://doi.org/10.1101/482380) | software paper |
# | [10.1101/532267](https://doi.org/10.1101/532267) | data repository paper so I guess it falls under the software group |
# | [10.1101/347625](https://doi.org/10.1101/347625) | no headings at all, which could be a prime example for missing methods section |

# # UMAP Embeddings for BioRxiv

# In[22]:


biorxiv_embeddings_df = pd.read_csv(
    "../word_vector_experiment/output/embedding_output/umap/biorxiv_umap_300.tsv", 
    sep="\t"
)
biorxiv_embeddings_df.head()


# In[23]:


biorxiv_umap_method_section_df = (
    pd.merge(
        biorxiv_embeddings_df, 
        biorxiv_sections_df.query("section=='material and methods'").drop_duplicates(),
        on="document",
        how="left"
    )
    .assign(section=lambda x: x.section.apply(lambda x: "has_methods" if x=="material and methods" else "no_methods"))
    .groupby("doi")
    .agg({
        "doi": "last",
        "document": "last",
        "umap1": "last",
        "umap2": "last",
        "category":"last",
        "section": "last",
    })
    .reset_index(drop=True)
)
biorxiv_umap_method_section_df.head()


# ## Global View of Umap Plot

# In[24]:


g = (
    p9.ggplot(
        biorxiv_umap_method_section_df
    )
    + p9.aes(x="umap1", y="umap2", color="category")
    + p9.geom_point(size=2)
    + p9.theme_bw()
    + p9.labs(
        title="UMAP Methods Section (300 dim)"
    )
)
print(g)


# ## Neuroscience Methods Section

# In[25]:


g = (
    p9.ggplot(
        biorxiv_umap_method_section_df
        .query("category=='neuroscience'")
    )
    + p9.aes(x="umap1", y="umap2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Neuroscience Methods Section"
    )
    + p9.annotate("rect", xmin=-3.75, xmax=-2.5, ymin=0, ymax=3, alpha=0.4)
)
print(g)


# In[26]:


(
    biorxiv_umap_method_section_df
    .query("category=='neuroscience'")
    .query("umap1 < -2.5 & umap1 > -3.75")
    .query("umap2 > 0")
)


# Personal Interpretation:

# | Document | Comment |
# | --- | --- |
# | [[10.1101/004176](https://doi.org/10.1101/004176)](https://doi.org/[10.1101/004176](https://doi.org/10.1101/004176)) | is a short article that doesn't have any section headers |
# | [10.1101/053827](https://doi.org/10.1101/053827) | already identified look at section 2.2 |
# | [10.1101/164780](https://doi.org/10.1101/164780) | already identified look at section 2.2 |
# | [10.1101/585760](https://doi.org/10.1101/585760) | already identified look at section 2.2 |
# | [10.1101/347070](https://doi.org/10.1101/347070) | has methods section labeled as "the proposed approach". It also seems like this paper should belong in PsyRxiv rather than biorxiv |
# | [10.1101/541243](https://doi.org/10.1101/541243) | has a methods section, but labeled as something else, which is why I couldn't detect it with my parser |

# In[27]:


g = (
    p9.ggplot(
        biorxiv_umap_method_section_df
        .query("category=='neuroscience'")
    )
    + p9.aes(x="umap1", y="umap2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Neuroscience Methods Section"
    )
    + p9.annotate("rect", xmin=-1.75, xmax=0, ymin=3, ymax=5, alpha=0.4)
)
print(g)


# In[28]:


(
    biorxiv_umap_method_section_df
    .query("category=='neuroscience'")
    .query("umap1 >-1.75 & umap1 < 0")
    .query("umap2 > 3 & umap2 < 5")
    .query("section=='no_methods'")
)


# Personal Interpretation:

# | Document | Comment |
# | --- | --- |
# | [10.1101/001586](https://doi.org/10.1101/001586) | a theorem proof, which shouldn't have a methods section | 
# | [10.1101/019398](https://doi.org/10.1101/019398) | math heavy paper on decision making |
# | [10.1101/034199](https://doi.org/10.1101/034199) | math focused paper |
# | [10.1101/043976](https://doi.org/10.1101/043976) | Math paper about modeling a neural network (neuroscience context) |
# | [10.1101/066480](https://doi.org/10.1101/066480) | Math paper |

# ## Bioinformatics Method Section

# In[29]:


g = (
    p9.ggplot(
        biorxiv_umap_method_section_df
        .query("category=='bioinformatics'")
    )
    + p9.aes(x="umap1", y="umap2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Bioinformatics Methods Section"
    )
    + p9.annotate("rect", xmin=-5, xmax=0, ymin=-7, ymax=-5, alpha=0.4)
)
print(g)


# In[30]:


(
    biorxiv_umap_method_section_df
    .query("category=='bioinformatics'")
    .query("umap1 < 0 & umap1 > -5")
    .query("umap2 < -5")
    .query("section=='no_methods'")
)


# Personal Interpretation:

# | Document | Comment |
# | --- | --- |
# | [10.1101/050047](https://doi.org/10.1101/050047) | doesn't have any sections, but quick glance only shows results. Could be a candidate for "needs a methods section" hunt. |
# | [10.1101/103283](https://doi.org/10.1101/103283) | is a false positive. It has a methods and material section. Could be a parser error. |
# | [10.1101/134288](https://doi.org/10.1101/134288) | is a weirdly formatted paper, but I didn't see a methods section in sight. |
# | [10.1101/140376](https://doi.org/10.1101/140376) | is a borderline article where it is written like a literature survey, but has analysis and results. No method section though. |

# In[31]:


g = (
    p9.ggplot(
        biorxiv_umap_method_section_df
        .query("category=='bioinformatics'")
    )
    + p9.aes(x="umap1", y="umap2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Bioinformatics Methods Section"
    )
    + p9.annotate("rect", xmin=-1, xmax=2, ymin=5, ymax=8, alpha=0.4)
)
print(g)


# In[32]:


(
    biorxiv_umap_method_section_df
    .query("umap1 > -1 & umap1 < 2")
    .query("umap2 > 5")
    .query("section=='no_methods'")
    .query("category=='bioinformatics'")
)


# Personal Interpretation:

# | Document | Comment |
# | --- | --- |
# | [10.1101/026898](https://doi.org/10.1101/026898) | diffusion fmri dataset paper |
# | [10.1101/038919](https://doi.org/10.1101/038919) | Multitask modeling...? Something about human multitasking and how they model that |
# | [10.1101/066910](https://doi.org/10.1101/066910) | fmri deep learning paper. Has methods section labeled as "background and algorithms" |
# | [10.1101/067702](https://doi.org/10.1101/067702) | Methods section is relabeled as approach |

# ## Microbiology Methods Section

# In[33]:


g = (
    p9.ggplot(
        biorxiv_umap_method_section_df
        .query("category=='microbiology'")
    )
    + p9.aes(x="umap1", y="umap2", color="section")
    + p9.geom_point()
    + p9.theme_bw()
    + p9.scale_color_manual({
        "has_methods": "#d8b365",
        "no_methods": "#5ab4ac"
    })
    + p9.labs(
        title="Microbiology Methods Section"
    )
    + p9.annotate("rect", xmin=-4, xmax=-3, ymin=0, ymax=2, alpha=0.4)
)
print(g)


# In[34]:


(
    biorxiv_umap_method_section_df
    .query("category=='microbiology'")
    .query("umap1 > -4 & umap1 < -3")
    .query("umap2 > 0 & umap2 < 2")
)


# Personal Interpretation:

# | Document | Comment | 
# | --- | --- | 
# | [10.1101/028209](https://doi.org/10.1101/028209) | No defined method section header however they do describe their model |
# | [10.1101/065128](https://doi.org/10.1101/065128) | No defined method section but definitely has it integrated in the paper |
# | [10.1101/095745](https://doi.org/10.1101/095745) | is a literature review about hand drying and cleanliness... |
# | [10.1101/101436](https://doi.org/10.1101/101436) | No section headers except for abstract |
