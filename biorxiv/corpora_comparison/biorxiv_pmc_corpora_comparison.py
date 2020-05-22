#!/usr/bin/env python
# coding: utf-8

# # Comparative Linguistic Analysis of bioRxiv and PMC

# In[1]:


from collections import defaultdict, Counter
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import spacy
from scipy.stats import chi2_contingency
from tqdm import tqdm_notebook


# In[2]:


def get_term_statistics(corpus_one, corpus_two, term, psudeocount=1, eps=1e-20):
    """
    This function is designed to perform the folllowing calculations:
        - chi square contingency test 
          - log pvalue + an epsilon (1e-20)
        - log likelihood of contingency table
        - log odds ratio
        
    keywords:
        corpus_one - a Counter object with terms as keys and count as values
        corpus_two - a Counter object with terms as keys and count as values
        term - the word of interest
    """
    observed_contingency_table = np.array([
        [corpus_one[term], corpus_two[term]],
        [sum(corpus_one.values()), sum(corpus_two.values())]
    ])
    
    # Chi Squared Test
    (chi_test_stat, p_val, dof, exp) = chi2_contingency(
        observed_contingency_table, 
        correction=False
    )
    
    # Log Likelihood
    
    ## add psudeocount to prevent log(0)
    observed_contingency_table += psudeocount
    
    a, b, c, d = (
        observed_contingency_table[0][0],
        observed_contingency_table[0][1],
        observed_contingency_table[1][0],
        observed_contingency_table[1][1]
    )
    
    # Obtained from (Kilgarriff, 2001) - Comparing Corpora
    LL = lambda a,b,c,d: 2*(
        a*np.log(a) + b*np.log(b) + c*np.log(c) + d*np.log(d)
        - (a+b)*np.log(a+b) - (a+c)*np.log(a+c) - (b+d)*np.log(b+d)
        - (c+d)*np.log(c+d) + (a+b+c+d)*np.log(a+b+c+d)
    )
    log_likelihood = LL(a,b,c,d)
    
    
    # Log Odds
    log_ratio = float((a*d)/(b*c))
    
    return {
        "chi_sq": (
            chi_test_stat, np.log(p_val+eps), dof,
            (observed_contingency_table-psudeocount), exp
        ),
        "log_likelihood":log_likelihood,
        "odds_ratio":log_ratio
    }


# In[3]:


def aggregate_word_counts(doc_iterator):
    global_word_counter = Counter()
    
    for doc in tqdm_notebook(doc_iterator):
        with open(doc, "r") as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter="\t")
            global_word_counter.update({
                row['lemma']:int(row['count'])
                for row in reader
            })

    return global_word_counter


# In[4]:


def remove_stop_words(corpus_one, corpus_two):
    spacy_nlp = spacy.load('en_core_web_sm')
    stop_word_list = list(spacy_nlp.Defaults.stop_words)
    stop_word_list += ['  ', '\t\t\t\t', '\u2009', ' ']
    
    for stopword in tqdm_notebook(stop_word_list):
        if stopword in corpus_one:
            del corpus_one[stopword]

        if stopword in corpus_two:
            del corpus_two[stopword]
            
    return corpus_one, corpus_two


# # Full Text Comparison (Global)

# ## Gather Word Frequencies

# In[4]:


biorxiv_corpus_count = (
    aggregate_word_counts(
        list(Path("output/biorxiv_word_counts").rglob("*tsv"))
    )
)


# In[5]:


pmc_corpus_count = (
    aggregate_word_counts(
        list(Path("output/pmc_word_counts").rglob("*tsv"))
    )
)


# In[8]:


biorxiv_corpus_count.most_common(10)


# In[9]:


pmc_corpus_count.most_common(10)


# In[6]:


pickle.dump(biorxiv_corpus_count, open("output/biorxiv_total_count.pkl", "wb"))
pickle.dump(pmc_corpus_count, open("output/pmc_total_count.pkl", "wb"))


# ## Analysis without Stop Words

# In[5]:


biorxiv_corpus_count = pickle.load(open("output/biorxiv_total_count.pkl", "rb"))
pmc_corpus_count = pickle.load(open("output/pmc_total_count.pkl", "rb"))


# In[6]:


biorxiv_corpus_count, pmc_corpus_count = remove_stop_words(
    biorxiv_corpus_count,
    pmc_corpus_count
)


# In[7]:


top_ten_biorxiv = biorxiv_corpus_count.most_common(100)
top_ten_biorxiv[0:10]


# In[8]:


top_ten_pmc = pmc_corpus_count.most_common(100)
top_ten_pmc[0:10]


# In[9]:


print("Number of words in biorxiv but not in Pubmed Central:")
biorxiv_difference = set(list(biorxiv_corpus_count.keys())) - set(list(pmc_corpus_count.keys()))
print(len(biorxiv_difference))


# In[10]:


[
    biorxiv_difference.pop()
    for i in range(10)
]


# In[11]:


print("Number of words in Pubmed Central but not in biorxiv:")
pmc_difference = set(list(pmc_corpus_count.keys())) - set(list(biorxiv_corpus_count.keys()))
print(len(pmc_difference))


# In[12]:


[
    pmc_difference.pop()
    for i in range(10)
]


# In[13]:


total_words = set(list(dict(top_ten_biorxiv).keys()) + list(dict(top_ten_pmc).keys()))
data = []
for word in tqdm_notebook(total_words):
    
    word_stat = get_term_statistics(
        biorxiv_corpus_count,
        pmc_corpus_count,
        word
    )
    
    data.append({
        "lemma": word,
        "biorxiv_count":biorxiv_corpus_count[word] if word in biorxiv_corpus_count else 0,
        "pmc_count":pmc_corpus_count[word] if word in pmc_corpus_count else 0,
        "biorxiv_total":word_stat['chi_sq'][3][1,0],
        "pmc_total":word_stat['chi_sq'][3][1,1],
        "log_p": word_stat['chi_sq'][1],
        "log_likelihood": word_stat['log_likelihood'],
        "odds_ratio": word_stat['odds_ratio']
    })


# In[14]:


total_word_stats_df = pd.DataFrame.from_records(data)
total_word_stats_df.to_csv(
    "output/full_corpus_comparison_stats.tsv", 
    sep="\t", index=False
)
total_word_stats_df.head()


# In[15]:


(
    total_word_stats_df
    .sort_values("log_likelihood", ascending=False)
    .head(20)
)


# In[16]:


(
    total_word_stats_df
    .sort_values("odds_ratio", ascending=False)
    .head(20)
)


# # Preprint to Published View

# In[17]:


mapped_doi_df = (
    pd.read_csv("../journal_tracker/output/mapped_published_doi.tsv", sep="\t")
    .query("published_doi.notnull()")
    .query("pmcid.notnull()")
    .groupby("doi")
    .agg({
        "author_type":"first",
        "heading":"first",
        "category":"first",
        "document":"first",
        "doi":"last",
        "published_doi":"last",
        "journal":"last",
        "pmcid":"last"
    })
    .reset_index(drop=True)
)
mapped_doi_df.tail()


# In[18]:


print(f"Total # of Preprints Mapped: {mapped_doi_df.shape[0]}")
print(f"Total % of Mapped: {mapped_doi_df.shape[0]/71118}")


# In[19]:


preprint_count = aggregate_word_counts([
    Path(f"output/biorxiv_word_counts/{Path(file).stem}.tsv")
    for file in mapped_doi_df.document.values.tolist()
    if Path(f"output/biorxiv_word_counts/{Path(file).stem}.tsv").exists()
])


# In[20]:


published_count = aggregate_word_counts([
    Path(f"output/pmc_word_counts/{file}.tsv")
    for file in mapped_doi_df.pmcid.values.tolist()
    if Path(f"output/pmc_word_counts/{file}.tsv").exists()
])


# In[21]:


preprint_count, published_count = remove_stop_words(
    preprint_count,
    published_count
)


# In[22]:


top_hundred_preprint = preprint_count.most_common(100)
top_hundred_preprint[0:10]


# In[23]:


top_hundred_published = published_count.most_common(100)
top_hundred_published[0:10]


# In[24]:


print("Number of words in preprint but not in published version:")
preprint_difference = set(list(preprint_count.keys())) - set(list(published_count.keys()))
print(len(preprint_difference))


# In[25]:


[
    preprint_difference.pop()
    for i in range(10)
]


# In[26]:


print("Number of words in published version but not in preprint:")
published_difference = set(list(published_count.keys())) - set(list(preprint_count.keys()))
print(len(published_difference))


# In[27]:


[
    published_difference.pop()
    for i in range(10)
]


# In[28]:


total_words = set(list(dict(top_hundred_preprint).keys()) + list(dict(top_hundred_published).keys()))
data = []
for word in tqdm_notebook(total_words):
    
    word_stat = get_term_statistics(
        preprint_count,
        published_count,
        word
    )
    
    data.append({
        "lemma": word,
        "preprint_count":preprint_count[word] if word in preprint_count else 0,
        "published_count":published_count[word] if word in published_count else 0,
        "preprint_total":word_stat['chi_sq'][3][1,0],
        "published_total":word_stat['chi_sq'][3][1,1],
        "log_p": word_stat['chi_sq'][1],
        "log_likelihood": word_stat['log_likelihood'],
        "odds_ratio": word_stat['odds_ratio']
    })


# In[29]:


published_comparison_stats_df = pd.DataFrame.from_records(data)
published_comparison_stats_df.to_csv(
    "output/preprint_to_published_comparison.tsv", 
    sep="\t", index=False
)
published_comparison_stats_df.head()


# In[30]:


(
    published_comparison_stats_df
    .sort_values("log_likelihood", ascending=False)
    .head(20)
)


# In[31]:


(
    published_comparison_stats_df
    .sort_values("log_likelihood", ascending=True)
    .head(20)
)


# In[32]:


(
    published_comparison_stats_df
    .sort_values("odds_ratio", ascending=False)
    .head(20)
)


# In[33]:


(
    published_comparison_stats_df
    .sort_values("odds_ratio", ascending=True)
    .head(20)
)

