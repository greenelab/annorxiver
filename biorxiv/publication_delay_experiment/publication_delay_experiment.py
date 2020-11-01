#!/usr/bin/env python
# coding: utf-8

# # Measure the Difference between Preprint-Published similarity and Published Articles

# In[1]:


from datetime import timedelta
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
import requests
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import linregress
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

published_dates.head()


# In[4]:


biorxiv_journal_df = (
    pd.read_csv(
        "../journal_tracker/output/mapped_published_doi.tsv", 
        sep="\t"
    )
    .groupby("doi")
    .agg({
        "document":"first",
        "category":"first",
        "journal":"first",
        "doi":"count",
        "published_doi":"first",  
        "pmcid":"first", 
        "pmcoa":"first",
    })
    .rename(index=str, columns={"doi":"version_count"})
    .reset_index()
)
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


# # Gather Golden bioRxiv Set

# In[8]:


matched_preprint_published_pairs = (
    biorxiv_journal_df
    .query("pmcoa==True")
    .sort_values("document")
    .merge(
        published_dates
        [["biorxiv_doi", "preprint_date", "published_date"]]
        .rename(index=str, columns={"biorxiv_doi":"doi"}),
        on="doi"
    )
    .assign(
        preprint_date = lambda x: pd.to_datetime(x.preprint_date.tolist()),
        published_date = lambda x: pd.to_datetime(
            x.published_date.apply(lambda y: y[0:y.index(":")] if ":" in y else y)
        )
    )
    .assign(
        time_to_published = lambda x: x.published_date - x.preprint_date
    )
)
matched_preprint_published_pairs.head()


# # Calculate the Document Distances

# In[9]:


biorxiv_documents = (
    biorxiv_embed_df
    .query(f"document in {matched_preprint_published_pairs.document.tolist()}")
    .set_index("document")
    .loc[matched_preprint_published_pairs.document.tolist()]
)
biorxiv_documents.head()


# In[10]:


pmc_documents = (
    pmc_embed_df
    .query(f"document in {matched_preprint_published_pairs.pmcid.tolist()}")
    .set_index("document")
    .loc[matched_preprint_published_pairs.pmcid.tolist()]
)
pmc_documents.head()


# In[11]:


published_date_distances = (
    matched_preprint_published_pairs
    .assign(
        doc_distances = np.diag(cdist(biorxiv_documents.values, pmc_documents.values, 'euclidean'))
    )
    .query("doc_distances.notnull()")
)
print(published_date_distances.shape)
published_date_distances.head()


# # Construct Scatter Plot of Date vs Version Count

# In[12]:


# Get smoothed linear regression line
x = (
    published_date_distances
    .time_to_published
    .apply(lambda x: x/timedelta(days=1))
    .tolist()
)

y = (
    published_date_distances
    .version_count
    .values
    .tolist()
)

xseq = (
    np.linspace(np.min(x), np.max(x), 80)
)
results = linregress(x, y)
print(results)


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
        p9.aes(y="time_to_published", x="version_count")
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
)
g.save("output/article_distance_vs_publication_time_dots.png", dpi=500)
print(g)


# In[15]:


g = (
    p9.ggplot(
        published_date_distances, 
        p9.aes(x="version_count", y="time_to_published")
    )
    + p9.geom_bin2d(bins=100)
    + p9.geom_line(
        stat="smooth",method='lm', linetype='dashed', 
        se=False, alpha=0.9, size=0.6
    )
    + p9.scale_y_timedelta(labels=timedelta_format('d'))
    + p9.annotate(
        'text', x=10, y=timedelta(days=1450), 
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}"
    )
)
g.save("output/article_distance_vs_publication_time.svg", dpi=500)
g.save("output/article_distance_vs_publication_time.png", dpi=500)
print(g)


# # Construct Scatter Plot of Date vs Document Distances

# In[16]:


# Get smoothed linear regression line
x = (
    published_date_distances
    .time_to_published
    .apply(lambda x: x/timedelta(days=1))
    .tolist()
)

y = (
    published_date_distances
    .doc_distances
    .values
    .tolist()
)

xseq = (
    np.linspace(np.min(x), np.max(x), 80)
)
results = linregress(x, y)
print(results)


# In[17]:


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


# In[18]:


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
)
g.save("output/article_distance_vs_publication_time_dots.png", dpi=500)
print(g)


# In[19]:


g = (
    p9.ggplot(
        published_date_distances, 
        p9.aes(x="doc_distances", y="time_to_published")
    )
    + p9.geom_bin2d(bins=100)
    + p9.geom_line(
        stat="smooth",method='lm', linetype='dashed', 
        se=False, alpha=0.9, size=0.6
    )
    + p9.scale_y_timedelta(labels=timedelta_format('d'))
    + p9.annotate(
        'text', x=10, y=timedelta(days=1450), 
        label=f"Y={results_2.slope:.2f}*X+{results_2.intercept:.2f}"
    )
)
g.save("output/article_distance_vs_publication_time.svg", dpi=500)
g.save("output/article_distance_vs_publication_time.png", dpi=500)
print(g)


# # Contextualize Document Distances

# The goal here is to understand what a unit of distance represents for two document embeddings. It is already established that low distances can indicate similar documents, but question remains how does a unit of distances relate to time taken to get published?

# In[20]:


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


# In[33]:


# Randomly sample two papers from the same journal 1000 times - plos one
plos_one_distances = (
    article_distances(
        biorxiv_journal_df
        .query("journal=='Genetics'")
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


# In[29]:


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


# In[30]:


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

