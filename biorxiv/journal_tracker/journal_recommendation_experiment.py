#!/usr/bin/env python
# coding: utf-8

# # Journal Recommendation for Preprints

# The goal of this notebook is to help users know which journal would be most appropriate for their preprint. The central idea is to use euclidean distance between documents to gauge which journal similar works have been sent.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm_notebook


# # Load bioRxiv Document Vectors

# In[2]:


biorxiv_journal_df = (
    pd.read_csv("output/mapped_published_doi.tsv", sep="\t")
    .groupby("doi")
    .agg({
        "document":"last",
        "category":"first",
        "journal":"first",
        "doi":"last",
        "published_doi":"first",  
        "pmcid":"first", 
    })
    .reset_index(drop=True)
)
biorxiv_journal_df.head()


# In[3]:


# Count number of Non-NaN elements
print(f"Number of Non-NaN entries: {biorxiv_journal_df.pmcid.count()}")
print(f"Total number of entries: {biorxiv_journal_df.shape[0]}")
print(f"Percent Covered: {biorxiv_journal_df.pmcid.count()/biorxiv_journal_df.shape[0]:.2f}")


# In[4]:


golden_set_df = biorxiv_journal_df.query("pmcid.notnull()")
golden_set_df.head()


# # Load Pubmed Central Document Vectors

# In[6]:


pmc_articles_df = (
    pd.read_csv(
        "../../pmc/exploratory_data_analysis/output/pubmed_central_journal_paper_map.tsv.xz", 
        sep="\t"
    )
    .query("article_type=='research-article'")
)
pmc_articles_df.head()


# In[7]:


print(pmc_articles_df.journal.value_counts().shape)
journals = pmc_articles_df.journal.value_counts()
journals


# In[8]:


# Filter out low count journals
pmc_articles_df = pmc_articles_df.query(f"journal in {journals[journals > 100].index.tolist()}")
print(pmc_articles_df.shape)
pmc_articles_df.head()


# In[9]:


pmc_embedding_df = pd.read_csv(
    "../../pmc/word_vector_experiment/output/pmc_document_vectors.tsv.xz", 
    sep="\t"
)
pmc_embedding_df.head()


# # Train Recommendation System

# In[10]:


def cross_validation(dataset, cv=10, n_neighbors=10, random_state=100, centroid=False):
    
    folds = KFold(n_splits=cv, random_state = random_state, shuffle=True)
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    cv_fold_accs = []
    
    for train, val in folds.split(dataset):
        
        X = (
            dataset
            .drop("journal", axis=1)
            .values
            .astype('float32')
        )
        
        Y = (
            dataset
            .journal
            .values
        )
        
        if centroid:
            journal_centroid_df = (
                dataset
                .iloc[train]
                .groupby("journal")
                .agg("mean")
                .reset_index()
            )
            
            centroid_X = (
                journal_centroid_df
                .drop("journal", axis=1)
                .values
                .astype('float32')
            )
            
            centroid_Y = (
                journal_centroid_df
                .journal
                .values
            )
            knn_model.fit(centroid_X, centroid_Y)
            
            distance, neighbors = knn_model.kneighbors(X[val])
            
            accs = [
                 (
                     1 if Y[val[data_idx]] in centroid_Y[neighbor_group]
                     else 0 
                 )
                 for data_idx, neighbor_group in enumerate(neighbors)
            ]
            
        else:
            knn_model.fit(X[train],Y[train])
        
            distance, neighbors = knn_model.kneighbors(X[val])

            accs = [
                 (
                     1 if Y[val[data_idx]] in Y[train][neighbor_group]
                     else 0 
                 )
                 for data_idx, neighbor_group in enumerate(neighbors)
            ]
        
        cv_fold_accs.append(np.sum(accs)/len(accs))
        print(f"{np.sum(accs)} out of {len(accs)}")
    print(np.mean(cv_fold_accs))
        


# ## KNearestNeighbors Paper by Paper Comparison

# The first idea for a classifier is to compare which papers are similar to other papers. Due to the overflow of PLOS One papers I sub-sampled each journal to have only 100 papers for representation. Then trained a KNearestNeighbors to determine how often does the correct journal appear in the top ten neighbors as well as top twenty neighbors.

# In[11]:


training_dataset = (
    pmc_articles_df
    .query(f"pmcid not in {golden_set_df.pmcid.tolist()}")
    [["pmcid", "journal"]]
    .merge(pmc_embedding_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
    .groupby("journal", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 100), random_state=100))
    .set_index("document")
)
print(training_dataset.shape)
training_dataset.head()


# In[12]:


cross_validation(training_dataset, cv=10, n_neighbors=10, random_state=100)


# In[13]:


cross_validation(training_dataset, cv=10, n_neighbors=20, random_state=100)


# ## KNearestNeighbors Centroid analysis

# Following up on the original idea, I thought a helpful experiment would be to perform a centroid analysis (i.e. take the average of all papers within each journal). Similar to above I trained a KNearestNeighbors classifier to see if the correct journal will appear in the top 10/20 neighbors.

# In[11]:


training_dataset = (
    pmc_articles_df
    .query(f"pmcid not in {golden_set_df.pmcid.tolist()}")
    [["pmcid", "journal"]]
    .merge(pmc_embedding_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
    .groupby("journal", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), 100), random_state=100))
    .set_index("document")
)
print(training_dataset.shape)
training_dataset.head()


# In[12]:


cross_validation(training_dataset, cv=10, n_neighbors=10, random_state=100, centroid=True)


# In[13]:


cross_validation(training_dataset, cv=10, n_neighbors=20, random_state=100, centroid=True)


# # Golden Set Analysis

# In[ ]:


#biorxiv_journal_embedding_df = pd.read_csv(
#    "../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz", 
#    sep="\t"
#)
#biorxiv_journal_embedding_df.head()


# In[ ]:


#golden_set_df


# Conclusions for this notebook:
# 1. Prediction accuracy is low when it comes to journal predictions on pubmed central data.
# 2. Centroid analysis performs a bit worse compared to paper by paper basis.
# 3. 300 Dimensions might not be the correct number of dimensions when prediction journals. A parameter sweep on embeddings might be needed.
