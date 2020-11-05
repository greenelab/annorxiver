#!/usr/bin/env python
# coding: utf-8

# # Measure the Difference between Preprint-Published similarity and Published Articles

# This notebook measures the time delay that results from the peer review process. Two plots are generated: one that depict the average publication time delay as changes are demanded from the peer review process and the other that depicts the added time delay as preprints have to undergo multiple versions to be published.

# In[1]:


from datetime import timedelta
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
import requests
from scipy.spatial.distance import cdist
from scipy.stats import linregress
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
import tqdm

from mizani.breaks import date_breaks
from mizani.formatters import timedelta_format


# # Gather Preprint-Published Pairs

# ## bioRxiv Document Embeddings

# In[2]:


url = "https://api.biorxiv.org/pub/2000-01-01/3000-01-01/"


# In[3]:


already_downloaded = Path("output/biorxiv_published_dates.tsv").exists()
if not already_downloaded:
    collection = []
    page_size = 100
    total = 44397
    for i in tqdm.tqdm(range(0, total, page_size), total=total/page_size):
        collection += requests.get(url + str(i)).json()["collection"]
    published_dates = pd.DataFrame(collection)
    published_dates.to_csv("output/biorxiv_published_dates.tsv", sep="\t", index=False)
else:
    published_dates = pd.read_csv("output/biorxiv_published_dates.tsv", sep="\t")

print(published_dates.shape)
published_dates.head()


# In[4]:


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
    })
    .rename(index=str, columns={"preprint_doi":"version_count"})
    .reset_index()
)
print(biorxiv_journal_df.shape)
biorxiv_journal_df.head()


# In[5]:


biorxiv_embed_df = (
    pd.read_csv(
        Path("../word_vector_experiment/output/") /
        "word2vec_output/" /
        "biorxiv_all_articles_300.tsv.xz",
        sep="\t"
    )   
)
print(biorxiv_embed_df.shape)
biorxiv_embed_df.head()


# ## PMC Article Embeddings

# In[6]:


pmc_articles_df = (
    pd.read_csv(
        Path("../../pmc/exploratory_data_analysis/") /
        "output/pubmed_central_journal_paper_map.tsv.xz", 
        sep="\t"
    )
    .query("article_type=='research-article'")
)
print(pmc_articles_df.shape)
pmc_articles_df.head()


# In[7]:


pmc_embed_df = (
    pd.read_csv(
        list(
            Path("../../pmc/word_vector_experiment/output/")
            .glob("*300.tsv.xz")
        )[0],
        sep="\t"
    )
)
pmc_embed_df.head()


# In[8]:


biorxiv_journal_df = (
    biorxiv_journal_df
    .merge(
        pmc_articles_df[["journal", "pmcid"]],
        on="pmcid",
        how="left"
    )
)
print(biorxiv_journal_df.shape)
biorxiv_journal_df.head()


# # Gather Golden bioRxiv Set

# In[9]:


matched_preprint_published_pairs = (
    biorxiv_journal_df
    .query("pmcoa==True")
    .sort_values("document")
    .merge(
        published_dates
        [["biorxiv_doi", "preprint_date", "published_date"]]
        .rename(index=str, columns={"biorxiv_doi":"preprint_doi"}),
        on="preprint_doi"
    )
    .assign(
        preprint_date = lambda x: pd.to_datetime(
            x.preprint_date
            .tolist()
        ),
        published_date = lambda x: pd.to_datetime(
            x
            .published_date
            .apply(lambda y: y[0:y.index(":")] if ":" in y else y)
        )
    )
    .assign(
        time_to_published = lambda x: x.published_date - x.preprint_date
    )
)
print(matched_preprint_published_pairs.shape)
matched_preprint_published_pairs.head()


# # Calculate the Document Distances

# This block calculates the euclidean distance between preprint's first version and their final published version.

# In[10]:


biorxiv_documents = (
    biorxiv_embed_df
    .query(f"document in {matched_preprint_published_pairs.document.tolist()}")
    .set_index("document")
    .reindex(matched_preprint_published_pairs.document.tolist())
    .fillna(0)
)
biorxiv_documents.head()


# In[11]:


pmc_documents = (
    pmc_embed_df
    .query(f"document in {matched_preprint_published_pairs.pmcid.tolist()}")
    .set_index("document")
    .reindex(matched_preprint_published_pairs.pmcid.tolist())
    .fillna(0)
)
pmc_documents.head()


# In[12]:


published_date_distances = (
    matched_preprint_published_pairs
    .assign(
        doc_distances = np.diag(cdist(
            biorxiv_documents.values, 
            pmc_documents.values, 
            'euclidean'
        ))
    )
    .replace(0, np.nan)
    .dropna()
    .query("doc_distances.notnull()")
)
print(published_date_distances.shape)
published_date_distances.head()


# # Construct Scatter Plot of Date vs Version Count

# Preprints are delayed on an average of 51 days for each new version posted onto bioRxiv. This section regresses preprint's version counts against the time it takes to have a preprint published. A scatter and square bin plot are generated below.

# In[13]:


# Get smoothed linear regression line
x = (
     published_date_distances
    .version_count
    .values
    .tolist()
)

y = (
    published_date_distances
    .time_to_published
    .apply(lambda x: x/timedelta(days=1))
    .tolist()
)

xseq_2 = (
    np.linspace(np.min(x), np.max(x), 80)
)

results_2 = linregress(x, y)
print(results_2)


# In[14]:


g = (
    p9.ggplot(
        published_date_distances, 
        p9.aes(x="factor(version_count)", y="time_to_published")
    )
    + p9.geom_boxplot(fill="#ffffcc")
    + p9.geom_line(
        mapping=p9.aes(x="version_count", y="time_to_published"),
        stat="smooth", method='lm', linetype='dashed', 
        se=False, alpha=1, size=0.7, inherit_aes=False,
    )
    + p9.scale_y_timedelta(labels=timedelta_format('d'))
    + p9.annotate(
        'text', x=9, y=timedelta(days=1470), 
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}",
    )
    + p9.labs(
        x="# of Preprint Versions",
        y="Time Elapsed Until Preprint is Published"
    )
)
g.save("output/version_count_vs_publication_time.svg", dpi=500)
g.save("output/version_count_vs_publication_time.png", dpi=500)
print(g)


# # Construct Scatter Plot of Date vs Document Distances

# Preprints are delayed on an average of 17 days as changes are demanded from the peer-review process. This section regresses a preprint's document distance against the time it takes to have a preprint published. A scatter and square bin plot are generated below.

# In[15]:


# Get smoothed linear regression line
x = (
     published_date_distances
    .doc_distances
    .values
    .tolist()
)

y = (
    published_date_distances
    .time_to_published
    .apply(lambda x: x/timedelta(days=1))
    .tolist()
)

xseq_2 = (
    np.linspace(np.min(x), np.max(x), 80)
)

results_2 = linregress(x, y)
print(results_2)


# In[16]:


g = (
    p9.ggplot(
        published_date_distances, 
        p9.aes(y="time_to_published", x="doc_distances")
    )
    + p9.geom_point()
    + p9.geom_line(
        stat="smooth",method='lm', linetype='dashed', 
        se=False, alpha=0.9, size=0.6
    )
    + p9.scale_y_timedelta(labels=timedelta_format('d'))
    + p9.annotate(
        'text', x=10, y=timedelta(days=1450), 
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}"
    )
    + p9.labs(
        x="Eucledian Distance of Preprints First and Final Versions",
        y="Time Elapsed Until Preprint is Published"
    )
)
print(g)


# In[23]:


g = (
    p9.ggplot(
        published_date_distances, 
        p9.aes(x="doc_distances", y="time_to_published")
    )
    + p9.geom_bin2d(bins=100)
    + p9.scale_fill_distiller(
        trans="log", direction=-1, type='seq', 
        palette='YlGnBu', name = "log(count)" 
    )
    + p9.geom_line(
        stat="smooth",method='lm', linetype='dashed', 
        se=False, alpha=0.6, size=0.7
    )
    + p9.scale_y_timedelta(labels=timedelta_format('d'))
    + p9.annotate(
        'text', x=15, y=timedelta(days=1470), 
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}"
    )
    + p9.labs(
        x="Eucledian Distance of Preprint-Published Versions",
        y="Time Elapsed Until Published",
        legend="log(count)"
    )
)
g.save("output/article_distance_vs_publication_time.svg", dpi=500)
g.save("output/article_distance_vs_publication_time.png", dpi=500)
print(g)


# # Contextualize Document Distances

# The goal here is to understand what a unit of distance represents for two document embeddings. It is already established that low distances can indicate similar documents, but question remains how does a unit of distances relate to time taken to get published? To answer this question I randomly sampled two preprints from the following groups: same journal, same preprint category and conglomeration of all bioRxiv preprints. Sampled preprints have their distance measured and I report the average distance of each group.

# In[18]:


def random_combination(iterable, r, size=100, seed=100):
    "Random selection from itertools.combinations(iterable, r)"
    random.seed(seed)
    
    pool = tuple(iterable)
    n = len(pool)
    
    indices = [
        sorted(random.sample(range(n), r)) 
        for elem in range(size)
    ]
    
    for i in indices:
        yield (pool[i[0]], pool[i[1]])

def article_distances(iterable, embed_df, combination_size=2, sample_size=100, seed=100):
    article_pair_generator = (
        random_combination(
            iterable,
            combination_size,
            sample_size,
            seed
        )
    )

    paper_one, paper_two = zip(*article_pair_generator)
    temp_index = embed_df.set_index("document")

    return (
        np.diag(
            cdist(
                temp_index.loc[list(paper_one)].values, 
                temp_index.loc[list(paper_two)].values, 
                'euclidean'
            )
        )
    )


# In[19]:


# Randomly sample two papers from the same journal 1000 times - plos one
plos_one_distances = (
    article_distances(
        biorxiv_journal_df
        .query("journal=='PLoS_Genet'")
        .document
        .tolist(),
        biorxiv_embed_df,
        2,
        1000
    )
)

print(
    f"Genetics Stats Mean: {np.mean(plos_one_distances):.3f}, "
    f"Std: {np.std(plos_one_distances):.3f}"
)


# In[20]:


# Randomly sample two papers from the same field category 1000 times - bioinformatics
bioinformatic_distances = (
    article_distances(
        biorxiv_journal_df
        .query("category == 'bioinformatics'")
        .query(f"document in {biorxiv_embed_df.document.tolist()}")
        .document
        .tolist(),
        biorxiv_embed_df,
        2,
        1000
    )
)

print(
    f"Bioinformatics Stats Mean: {np.mean(bioinformatic_distances):.3f}, "
    f"Std: {np.std(bioinformatic_distances):.3f}"
)


# In[21]:


# Randomly sample two papers from the entire bioRxiv corpus 1000 times
biorxiv_distances = (
    article_distances(
        biorxiv_journal_df
        .query(f"document in {biorxiv_embed_df.document.tolist()}")
        .document
        .tolist(),
        biorxiv_embed_df,
        2,
        1000
    )
)

print(
    f"Biorxiv Stats Mean: {np.mean(biorxiv_distances):.3f}, "
    f"Std: {np.std(biorxiv_distances):.3f}"
)


# Take home results:
#     1. It takes approximately 51 days for a new preprint version to be posted onto bioRxiv.
#     2. Making peer review changes takes on average 17 days to make for given preprints.
#     3. A distance unit reflects about an 18% ((6.210-5.068)/6.210) change of a preprints textual content
