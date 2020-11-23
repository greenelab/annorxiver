#!/usr/bin/env python
# coding: utf-8

# # Evaluation Statistics on bioRxiv - PMC links

# In[1]:


from pathlib import Path

import pandas as pd
import pickle
import plotnine as p9

from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegressionCV


# # Equal Dimension Distance Calculation

# In[2]:


potential_links_df = pd.read_csv(
    "output/annotated_links/potential_biorxiv_pmc_links.tsv", 
    sep="\t"
)
potential_links_df.head()


# In[3]:


annotated_df = pd.read_csv(
    "output/annotated_links/final_biorxiv_pmc_links_curation.tsv", 
    sep="\t"
)
annotated_df.head()


# In[4]:


kappa_score = cohen_kappa_score(
    annotated_df.is_same_paper_1, 
    annotated_df.is_same_paper_2
)
print(f"The interrator reliability (kappa score) is {kappa_score:%}.")


# In[5]:


final_annotated_df = (
    annotated_df
    .merge(
        potential_links_df
        [["biorxiv_doi_url", "pmcid_url", "distance_bin"]]
    )
    .assign(
        final_same_paper=lambda x:(
            x[["is_same_paper_1", "is_same_paper_2", "is_same_paper_3"]]
            .mode(axis=1)
        )
    )
)
final_annotated_df.head()


# In[6]:


binned_stats_df = (
    final_annotated_df
    .groupby("distance_bin")
    .final_same_paper
    .mean()
    .to_frame()
    .rename(index=str, columns={"final_same_paper":"frac_correct"})
    .reset_index()
    .replace({
        "distance_bin":{
            "[0, 25%ile)": '0-25th',
            '[25%ile, 50%ile)':'25th-50th',
            '[50%ile, 75%ile)':'50th-75th',
            '[75%, min(same-journal-no-known-link))': '75th-min(Randomized Journal Pairs)'
        }
    })
)
binned_stats_df


# In[7]:


g = (
    p9.ggplot(binned_stats_df, p9.aes(x="distance_bin", y="frac_correct"))
    + p9.geom_col(fill="#a6cee3")
    + p9.coord_flip()
    + p9.labs(
        y="Fraction Correct",
        x="Euclidean Distance Percentile Bins"
    )
    + p9.theme_seaborn(
        context="paper",
        style="ticks",
        font="Arial",
        font_scale=1.5
    )
    
)
g.save("output/figures/distance_bin_accuracy.svg")
g.save("output/figures/distance_bin_accuracy.png", dpi=250)
print(g)


# # Logsitic Regression Performance

# In[8]:


biorxiv_embed_df = (
    pd.read_csv(
        Path("../word_vector_experiment/output/") /
        "word2vec_output/" /
        "biorxiv_all_articles_300.tsv.xz",
        sep="\t"
    )
    .set_index("document")
)
biorxiv_embed_df.head()


# In[9]:


pmc_embed_df = (
    pd.read_csv(
        list(
            Path("../../pmc/word_vector_experiment/output/")
            .glob("*300.tsv.xz")
        )[0],
        sep="\t"
    )
    .set_index("document")
)
pmc_embed_df.head()


# In[10]:


id_mapper_df = (
    potential_links_df
    [["document", "biorxiv_doi_url", "pmcid", "pmcid_url"]]
    .merge(final_annotated_df, on=["biorxiv_doi_url", "pmcid_url"])
)
id_mapper_df.head()


# In[11]:


data_records = []
for idx, row in id_mapper_df[["document", "pmcid"]].iterrows():
    data_records.append(
        dict(
            biorxiv_document=row['document'],
            pmcid=row['pmcid'],
            **(
                biorxiv_embed_df.loc[row['document']]-
                pmc_embed_df.loc[row['pmcid']]
            ).to_dict()
        )
    ) 


# In[12]:


final_df = pd.DataFrame.from_records(data_records)
final_df.head()


# In[13]:


model = pickle.load(open("output/optimized_model.pkl", "rb"))


# In[14]:


model_predictions = (
    model.predict_proba(
        final_df.drop(["biorxiv_document", "pmcid"], axis=1)
    )
    [:,1]
)
model_predictions


# In[15]:


id_mapper_df = (
    id_mapper_df
    .assign(lr_model=list(map(lambda x: True if x > 0.5 else False,model_predictions)))
)
id_mapper_df.head()


# In[16]:


binned_model_stats_df = (
    id_mapper_df
    .groupby("distance_bin")
    .agg({
        "final_same_paper": "mean", 
        "lr_model": "mean"
    })
    .rename(index=str, columns={"final_same_paper":"frac_correct"})
    .reset_index()
)
binned_model_stats_df

