#!/usr/bin/env python
# coding: utf-8

# # Journal Recommendation for Preprints

# The goal of this notebook is to help users know which journal would be most appropriate for their preprint. The central idea is to use euclidean distance between documents to gauge which journal similar works have been sent.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier

from tqdm import tqdm_notebook


# In[2]:


cluster = not True


# # Load bioRxiv Document Vectors

# In[3]:


if cluster:
    biorxiv_journal_df = (
        pd.read_csv("mapped_published_doi.tsv", sep="\t")
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
else:
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


# In[4]:


# Count number of Non-NaN elements
print(f"Number of Non-NaN entries: {biorxiv_journal_df.pmcid.count()}")
print(f"Total number of entries: {biorxiv_journal_df.shape[0]}")
print(f"Percent Covered: {biorxiv_journal_df.pmcid.count()/biorxiv_journal_df.shape[0]:.2f}")


# In[5]:


golden_set_df = biorxiv_journal_df.query("pmcid.notnull()")
golden_set_df.head()


# # Load Pubmed Central Document Vectors

# In[6]:


if cluster:
    pmc_articles_df = (
        pd.read_csv(
            "pubmed_central_journal_paper_map.tsv.xz", 
            sep="\t"
        )
        .query("article_type=='research-article'")
    )
else:
    pmc_articles_df = (
        pd.read_csv(
            "../../pmc/exploratory_data_analysis/output/pubmed_central_journal_paper_map.tsv.xz", 
            sep="\t"
        )
        .query("article_type=='research-article'")
    )
print(pmc_articles_df.head())


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


if cluster:
    pmc_embedding_dict = {
        int(path.stem[-7:-4]):pd.read_csv(
            str(path), 
            sep="\t"
        )
        for path in Path("pmc_vectors/").rglob("*tsv.xz")
    }
else:
    pmc_embedding_dict = {
        int(path.stem[-7:-4]):pd.read_csv(
            str(path), 
            sep="\t"
        )
        for path in Path("../../pmc/word_vector_experiment/output/").rglob("*tsv.xz")
    }
pmc_embedding_dict[300].head()


# In[10]:


full_training_dataset = {
    dim : (
        pmc_articles_df
        .query(f"pmcid not in {golden_set_df.pmcid.tolist()}")
        [["pmcid", "journal"]]
        .merge(pmc_embedding_dict[dim], left_on="pmcid", right_on="document")
        .drop("pmcid", axis=1)
        .set_index("document")
    )
    for dim in pmc_embedding_dict
}


# In[11]:


subsampled_training_dataset = {
    dim : (
        pmc_articles_df
        .query(f"pmcid not in {golden_set_df.pmcid.tolist()}")
        [["pmcid", "journal"]]
        .merge(pmc_embedding_dict[dim], left_on="pmcid", right_on="document")
        .drop("pmcid", axis=1)
        .groupby("journal", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 100), random_state=100))
        .set_index("document")
    )
    for dim in pmc_embedding_dict
}


# # Train Recommendation System

# In[12]:


def cross_validation(model, dataset, evaluate, cv=10, random_state=100, **kwargs):
    
    folds = KFold(n_splits=cv, random_state = random_state, shuffle=True)
    cv_fold_accs = []
    
    fold_predictions = []
    for train, val in folds.split(dataset):
        
        prediction, true_labels = evaluate(
            model, dataset.iloc[train], 
            dataset.iloc[val], **kwargs
        )

        accs = [
                 (
                     1 if true_labels[data_idx] in prediction_row 
                     else 0 
                 )
                 for data_idx, prediction_row in enumerate(prediction)
        ]
        
        cv_fold_accs.append(np.sum(accs)/len(accs))
        print(f"{np.sum(accs)} out of {len(accs)}")
        
        fold_predictions.append(prediction)
        
    print(f"Total Accuracy: {np.mean(cv_fold_accs)*100:.3f}%")
    return fold_predictions


# ## Random Journal Prediction

# The central idea here is to answer the question what is the accuracy when journals are recommended at random?

# In[13]:


def dummy_evaluate(model, training_data, validation_data, **kwargs):
    top_X = kwargs.get("top_predictions", 10)
    random_states = kwargs.get("dummy_seed", [100,200,300,400,500,600,700,800,900,1000])
    
    X_train = (
        training_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_train = (
        training_data
        .journal
        .values
    )
    
    X_val = (
        validation_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    Y_val = (
        validation_data
        .journal
        .values
    )
    
    predictions = []
    for i, seed in zip(range(top_X), random_states):
        model.random_state = seed
        model.fit(X_train, Y_train)
        predictions.append(model.predict(X_val))

    return np.stack(predictions).transpose(), Y_val


# In[16]:


model = DummyClassifier(strategy='uniform')


# In[17]:


_ = cross_validation(
    model, subsampled_training_dataset[300], 
    dummy_evaluate, cv=10, 
    random_state=100, top_predictions=10
)


# ## KNearestNeighbors Paper by Paper Comparison - Full Dataset

# Assuming I didn't take mega-journal influence into account, what would the initial recommendation accuracy be?

# In[14]:


def knn_evaluate(model, training_data, validation_data, **kwargs):
    
    X_train = (
        training_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_train = (
        training_data
        .journal
        .values
    )
    
    X_val = (
        validation_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_val = (
        validation_data
        .journal
        .values
    )
    
    model.fit(X_train, Y_train)
    distance, neighbors = model.kneighbors(X_val)
    
    predictions = [
        Y_train[neighbor_predict]
        for neighbor_predict in neighbors 
    ]

    return np.stack(predictions), Y_val


# In[18]:


knn_model = KNeighborsClassifier(n_neighbors=10)


# In[ ]:


_ = cross_validation(
    knn_model, full_training_dataset[300], 
    knn_evaluate, cv=10,
    random_state=100
)


# ## KNearestNeighbors Paper by Paper Comparison Subsampled

# The first idea for a classifier is to compare which papers are similar to other papers. Due to the overflow of PLOS One papers I sub-sampled each journal to have only 100 papers for representation. Then trained a KNearestNeighbors to determine how often does the correct journal appear in the top ten neighbors as well as top twenty neighbors.

# In[16]:


knn_model = KNeighborsClassifier(n_neighbors=10)


# In[17]:


result_dict = {}
for dim in subsampled_training_dataset:
    print(dim)
    
    fold_predictions = cross_validation(
        knn_model, subsampled_training_dataset[dim], 
        knn_evaluate, cv=10, 
        random_state=100
    )
    
    print()
    
    result_dict[dim] = fold_predictions


# ## KNearestNeighbors Centroid analysis

# Following up on the original idea, I thought a helpful experiment would be to perform a centroid analysis (i.e. take the average of all papers within each journal). Similar to above I trained a KNearestNeighbors classifier to see if the correct journal will appear in the top 10 neighbors.

# In[15]:


def knn_centroid_evaluate(model, training_data, validation_data, **kwargs):
    
    train_centroid_df = (
        training_data
        .groupby("journal")
        .agg("mean")
        .reset_index()
    )
            
    X_train_centroid = (
        train_centroid_df
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )

    Y_train_centroid = (
        train_centroid_df
        .journal
        .values
    )
    
    
    X_val = (
        validation_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_val = (
        validation_data
        .journal
        .values
    )
    
    knn_model.fit(X_train_centroid, Y_train_centroid)
    distance, neighbors = knn_model.kneighbors(X_val)
    
    predictions = [
        Y_train_centroid[neighbor_predict]
        for neighbor_predict in neighbors 
    ]

    return np.stack(predictions), Y_val


# In[18]:


knn_model = KNeighborsClassifier(n_neighbors=10)


# In[19]:


_ = cross_validation(
    knn_model, subsampled_training_dataset[300], 
    knn_centroid_evaluate, cv=10, 
    random_state=100
)


# ## KNearestNeighbors Centroid Analysis - Full dataset

# This section I'm using the entire dataset to calculate journal centroids and then evaluate performance on the sub-sampled dataset.

# In[20]:


def knn_centroid_full_evaluate(model, training_data, validation_data, **kwargs):
    
    train_centroid_df = (
        kwargs.get("full_dataset")
        .groupby("journal")
        .agg("mean")
        .reset_index()
    )
            
    X_train_centroid = (
        train_centroid_df
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )

    Y_train_centroid = (
        train_centroid_df
        .journal
        .values
    )
    
    
    X_val = (
        validation_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_val = (
        validation_data
        .journal
        .values
    )
    
    knn_model.fit(X_train_centroid, Y_train_centroid)
    distance, neighbors = knn_model.kneighbors(X_val)
    
    predictions = [
        Y_train_centroid[neighbor_predict]
        for neighbor_predict in neighbors 
    ]

    return np.stack(predictions), Y_val


# In[21]:


knn_model = KNeighborsClassifier(n_neighbors=10)


# In[23]:


_ = cross_validation(
    knn_model, subsampled_training_dataset[300], 
    knn_centroid_full_evaluate, cv=10, 
    random_state=100, full_dataset=full_training_dataset[300]
)


# # Golden Set Analysis

# In[25]:


biorxiv_embeddings_df = pd.read_csv(
    Path("../word_vector_experiment/output/word2vec_output/biorxiv_all_articles_300.tsv.xz")
    .resolve(),
    sep="\t"
)

biorxiv_embeddings_df.head()


# In[35]:


golden_dataset = (
    golden_set_df[["document", "pmcid"]]
    .merge(pmc_articles_df[["journal", "pmcid"]], on="pmcid")
    .merge(biorxiv_embeddings_df, on="document")
)
golden_dataset.head()


# ## Centroid Analysis

# In[37]:


train_centroid_df = (
    full_training_dataset[300]
    .groupby("journal")
    .agg("mean")
    .reset_index()
)

X_train_centroid = (
    train_centroid_df
    .drop("journal", axis=1)
    .values
    .astype('float32')
)

Y_train_centroid = (
    train_centroid_df
    .journal
    .values
)
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train_centroid, Y_train_centroid)


# In[38]:


distance, neighbors = knn_model.kneighbors(
    golden_dataset
    .drop(["document", "pmcid", "journal"], axis=1)
    .values
)


# In[40]:


accs = [
    (
         1 if golden_dataset.journal[data_idx] in prediction_row 
         else 0 
     )
     for data_idx, prediction_row in enumerate(
         [
             Y_train_centroid[neighbor_predict]
             for neighbor_predict in neighbors 
         ]
     )
]


# In[43]:


print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)}% correct")


# ## Subsampled Paper Analysis

# In[44]:


X_train = (
    subsampled_training_dataset[300]
    .drop("journal", axis=1)
    .values
    .astype('float32')
)
    
Y_train = (
    subsampled_training_dataset[300]
    .journal
    .values
)
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, Y_train)


# In[45]:


distance, neighbors = knn_model.kneighbors(
    golden_dataset
    .drop(["document", "pmcid", "journal"], axis=1)
    .values
)


# In[46]:


accs = [
    (
         1 if golden_dataset.journal[data_idx] in prediction_row 
         else 0 
     )
     for data_idx, prediction_row in enumerate(
         [
             Y_train[neighbor_predict]
             for neighbor_predict in neighbors 
         ]
     )
]


# In[47]:


print(f"{np.sum(accs)} out of {len(accs)}")
print(f"{np.mean(accs)}% correct")


# Conclusions for this notebook:
# 1. Mega-journals cover a wide range of research topics.
# 2. The correct journal only appears in the top ten about 37-39 percent of the time.
# 3. 300 dimensions gives the best performance compared to the other dimensions.
# 4. Reporting a combination of centroid analysis and individual paper predictions will be needed to go forward.
