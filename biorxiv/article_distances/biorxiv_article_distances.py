#!/usr/bin/env python
# coding: utf-8

# # Find published articles missing from bioRxiv

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9
from scipy.spatial.distance import cdist
import tqdm


# # Load Embeddings

# ## bioRxiv

# In[2]:


biorxiv_journal_df = (
    pd.read_csv(
        "../journal_tracker/output/mapped_published_doi.tsv", 
        sep="\t"
    )
    .groupby("doi")
    .agg({
        "document":"last",
        "category":"first",
        "journal":"first",
        "doi":"last",
        "published_doi":"first",  
        "pmcid":"first", 
        "pmcoa":"first"
    })
    .reset_index(drop=True)
)
biorxiv_journal_df.head()


# In[3]:


biorxiv_embed_df = (
    pd.read_csv(
        Path("../word_vector_experiment/output/") /
        "word2vec_output/" /
        "biorxiv_all_articles_300.tsv.xz",
        sep="\t"
    )   
)
biorxiv_embed_df.head()


# In[4]:


biorxiv_journal_mapped_df = (
    biorxiv_journal_df[["document", "published_doi", "pmcid", "pmcoa"]]
    .merge(biorxiv_embed_df, on="document")
)
biorxiv_journal_mapped_df.head()


# ## Pubmed Central

# In[5]:


pmc_articles_df = (
    pd.read_csv(
        Path("../../pmc/exploratory_data_analysis/") /
        "output/pubmed_central_journal_paper_map.tsv.xz", 
        sep="\t"
    )
    .query("article_type=='research-article'")
)
pmc_articles_df.head()


# In[6]:


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


# In[7]:


pmc_journal_mapped_df = (
    pmc_articles_df[["journal", "doi", "pmcid"]]
    .merge(pmc_embed_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
)
pmc_journal_mapped_df.head()


# # Calculate Distances

# ## biorxiv -> published versions

# In[8]:


biorxiv_published = (
    biorxiv_journal_mapped_df
    .query("pmcid.notnull()")
    .query("pmcoa == True")
    .sort_values("pmcid", ascending=True)
    .drop_duplicates("pmcid")
    .set_index("pmcid")
)
biorxiv_published.head()


# In[9]:


PMC_pubished = (
    pmc_journal_mapped_df
    .query(f"document in {biorxiv_published.reset_index().pmcid.tolist()}")
    .sort_values("document", ascending=True)
    .set_index("document")
)
PMC_pubished.head()


# In[10]:


article_distances = cdist(
    biorxiv_published
    .loc[PMC_pubished.index.tolist()]
    .drop(["document", "published_doi", "pmcoa"], axis=1), 
    PMC_pubished.drop(["journal", "doi"], axis=1),
    'euclidean'
)
article_distances.shape


# In[12]:


articles_distance_df = (
    biorxiv_published
    .loc[PMC_pubished.index.tolist()]
    .reset_index()
    [["document", "pmcid"]]
    .assign(
            distance=np.diag(article_distances, k=0),
            journal=PMC_pubished.journal.tolist()
    )
)
articles_distance_df.head()


# ## biorxiv -> random paper same journal

# In[13]:


PMC_off_published = (
    pmc_journal_mapped_df
    .drop("doi", axis=1)
    .query(f"document not in {biorxiv_published.reset_index().pmcid.tolist()}")
    .query(f"journal in {articles_distance_df.journal.unique().tolist()}")
    .groupby("journal", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=100))
    .set_index("document")
)
PMC_off_published.head()


# In[14]:


journal_mapper = {
    journal:col
    for col, journal in enumerate(PMC_off_published.journal.tolist())
}
list(journal_mapper.items())[0:10]


# In[17]:


off_article_dist = cdist(
    biorxiv_published
    .loc[PMC_pubished.index.tolist()]
    .drop(["document", "published_doi", "pmcoa"], axis=1), 
    PMC_off_published.drop("journal", axis=1),
    'euclidean'
)
off_article_dist.shape


# In[18]:


data = []
for idx, row in tqdm.tqdm(articles_distance_df.iterrows()):
    if row['journal'] in journal_mapper:
        data.append(
            {
                "document": row['document'],
                "pmcid":  (
                    PMC_off_published
                    .query(f"journal=='{row['journal']}'")
                    .reset_index()
                    .document
                    .values[0]
                ),
                "journal": row['journal'],
                "distance": off_article_dist[idx, journal_mapper[row['journal']]]
            }
        )


# In[19]:


final_df = (
    articles_distance_df
    .assign(label="pre_vs_published")
    .append(
        pd.DataFrame
        .from_records(data)
        .assign(label="pre_vs_random")
    )
)
final_df.head()


# In[20]:


final_df = (
    biorxiv_journal_df[["document", "doi"]]
    .merge(final_df)
)
final_df.to_csv("output/article_distances.tsv", sep="\t", index=False)
final_df.head()


# # Distribution plot

# In[21]:


g = (
    p9.ggplot(final_df)
    + p9.aes(x="label",y="distance")
    + p9.geom_boxplot()
    + p9.coord_flip()
    + p9.theme_seaborn()
)
g.save("output/biorxiv_article_distance.svg", dpi=500)
g.save("output/biorxiv_article_distance.png", dpi=500)
print(g)


# # Find bioRxiv unpublished ->  published PMC articles

# In[31]:


biorxiv_unpublished = (
    biorxiv_journal_mapped_df
    .query("published_doi.isnull()")
    .drop(["published_doi", "pmcid", "pmcoa"], axis=1)
)
print(biorxiv_unpublished.shape)
biorxiv_unpublished.head()


# In[23]:


PMC_unlinked = (
    pmc_journal_mapped_df
    .query(
        f"""
        document not in {
            biorxiv_published
            .reset_index()
            .pmcid
            .tolist()
        }
        """
    )
)
print(PMC_unlinked.shape)
PMC_unlinked.head()


# In[24]:


cutoff_score = (
    final_df
    .query("label=='pre_vs_random'")
    .distance
    .min()
)
cutoff_score


# In[27]:


chunksize=100
chunk_iterator = range(
    0, biorxiv_unpublished.shape[0], 
    chunksize
)


# In[32]:


for idx, chunk in tqdm.tqdm(enumerate(chunk_iterator)):
    
    # Chunk the documents so memory doesn't break
    biorxiv_subset = (
        biorxiv_unpublished
        .iloc[chunk:chunk+chunksize]
    )
    
    # Calculate distances
    paper_distances = cdist(
        biorxiv_subset.drop(["document"], axis=1), 
        PMC_unlinked.drop(["journal", "document", "doi"], axis=1),
        'euclidean'
    )
    
    # Get elements less than threshold
    sig_indicies = np.where(paper_distances < cutoff_score)
    results = zip(
        sig_indicies[0],
        sig_indicies[1],
        paper_distances[paper_distances < cutoff_score]
    )
    
    # Map the results into records for pandas to parse
    results = list(
        map(
            lambda x: dict(
                document=biorxiv_subset.iloc[x[0]].document,
                pmcid=PMC_unlinked.iloc[x[1]].document,
                distance=x[2]
            ),
            results
        )
    )
    
    # There may be cases where there are no matches
    if len(results) > 0:
        # Generate pandas dataframe
        potential_matches_df = (
            biorxiv_journal_df[["document", "doi"]]
            .merge(
                pd.DataFrame
                .from_records(results)
            )
            .sort_values("distance")
        )
    
        # Write to file
        if idx == 0:
            potential_matches_df.to_csv(
                "output/potential_biorxiv_pmc_links.tsv", 
                sep="\t", index=False
            )

        else:
            potential_matches_df.to_csv(
                "output/potential_biorxiv_pmc_links.tsv", 
                sep="\t", index=False,
                mode="a", header=False
            )


# # Bin Potential Matches

# In[33]:


potential_matches_df = pd.read_csv(
    "output/potential_biorxiv_pmc_links.tsv",
    sep="\t"
)
potential_matches_df.head()


# In[34]:


potential_matches_df = (
    potential_matches_df
    .rename(index=str, columns={"doi":"biorxiv_doi"})
    .drop_duplicates(["document", "biorxiv_doi","pmcid"])
    .assign(
        pmcid_url=lambda x:(
            x
            .pmcid
            .apply(
                lambda paper: f"https://www.ncbi.nlm.nih.gov/pmc/{paper}"
            )
        ),
        biorxiv_doi_url=lambda x:(
            x
            .biorxiv_doi
            .apply(
                lambda paper: f"https://doi.org/{paper}"
            )
        )
    )
    
)
potential_matches_df.head()


# In[35]:


distance_bins = np.squeeze(
    final_df
    .query("label=='pre_vs_published'")
    .describe()
    .loc[["25%", "50%", "75%"]]
    .values
)

distance_bins = np.append(
        [0.0],
        distance_bins,
)

distance_bins = np.append(
    distance_bins,
    (
        final_df
        .query("label=='pre_vs_random'")
        .distance
        .min()
    )   
)

distance_bins


# In[36]:


potential_matches_df = (
    potential_matches_df
    .assign(
        distance_bin=(
            pd.cut(
                potential_matches_df.distance,
                distance_bins,
                right=False,
                labels=[
                    "[0, 25%ile)",
                    "[25%ile, 50%ile)",
                    "[50%ile, 75%ile)",
                    "[75%, min(same-journal-no-known-link))"
                ]
            )
        )
    )
    [[
        "document", "biorxiv_doi", 
        "biorxiv_doi_url", "pmcid",
        "pmcid_url", "distance",
        "distance_bin"
    ]]
)

potential_matches_df.to_csv(
    "output/potential_biorxiv_pmc_links.tsv", 
    sep="\t", index=False
)

potential_matches_df.head()

