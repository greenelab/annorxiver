from collections import defaultdict, Counter
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import spacy
from scipy.stats import chi2_contingency
from tqdm import tqdm_notebook

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

biorxiv_corpus_count = (
    aggregate_word_counts(
        list(Path("output/biorxiv_word_counts").rglob("*tsv"))
    )
)

pmc_corpus_count = (
    aggregate_word_counts(
        list(Path("output/pmc_word_counts").rglob("*tsv"))
    )
)

biorxiv_corpus_count.most_common(10)

pmc_corpus_count.most_common(10)

pickle.dump(biorxiv_corpus_count, open("output/biorxiv_total_count.pkl", "wb"))
pickle.dump(pmc_corpus_count, open("output/pmc_total_count.pkl", "wb"))

biorxiv_corpus_count = pickle.load(open("output/biorxiv_total_count.pkl", "rb"))
pmc_corpus_count = pickle.load(open("output/pmc_total_count.pkl", "rb"))

biorxiv_corpus_count, pmc_corpus_count = remove_stop_words(
    biorxiv_corpus_count,
    pmc_corpus_count
)

top_ten_biorxiv = biorxiv_corpus_count.most_common(100)
top_ten_biorxiv[0:10]

top_ten_pmc = pmc_corpus_count.most_common(100)
top_ten_pmc[0:10]

print("Number of words in biorxiv but not in Pubmed Central:")
biorxiv_difference = set(list(biorxiv_corpus_count.keys())) - set(list(pmc_corpus_count.keys()))
print(len(biorxiv_difference))

[
    biorxiv_difference.pop()
    for i in range(10)
]

print("Number of words in Pubmed Central but not in biorxiv:")
pmc_difference = set(list(pmc_corpus_count.keys())) - set(list(biorxiv_corpus_count.keys()))
print(len(pmc_difference))

[
    pmc_difference.pop()
    for i in range(10)
]

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

total_word_stats_df = pd.DataFrame.from_records(data)
total_word_stats_df.to_csv(
    "output/full_corpus_comparison_stats.tsv", 
    sep="\t", index=False
)
total_word_stats_df.head()

(
    total_word_stats_df
    .sort_values("log_likelihood", ascending=False)
    .head(20)
)

(
    total_word_stats_df
    .sort_values("odds_ratio", ascending=False)
    .head(20)
)

mapped_doi_df = (
    pd.read_csv("../journal_tracker/output/mapped_published_doi.tsv", sep="\t")
    .query("published_doi.notnull()")
    .query("pmcid.notnull()")
    .groupby("doi")
    .agg({
        "author_type":"first",
        "heading":"first",
        "category":"first",
        "document":"last",
        "doi":"last",
        "published_doi":"last",
        "journal":"last",
        "pmcid":"last"
    })
    .reset_index(drop=True)
)
mapped_doi_df.tail()

preprint_count = aggregate_word_counts(
    [Path("output/biorxiv_word_counts/862847_v1.tsv")]
)

published_count = aggregate_word_counts(
    [Path("output/pmc_word_counts/PMC6933653.tsv")]
)

preprint_count, published_count = remove_stop_words(
    preprint_count,
    published_count
)

top_ten_preprint = preprint_count.most_common(100)
top_ten_preprint[0:10]

top_ten_published = published_count.most_common(100)
top_ten_published[0:10]

print("Number of words in preprint but not in published version:")
preprint_difference = set(list(preprint_count.keys())) - set(list(published_count.keys()))
print(len(preprint_difference))

[
    preprint_difference.pop()
    for i in range(10)
]

print("Number of words in published version but not in preprint:")
published_difference = set(list(published_count.keys())) - set(list(preprint_count.keys()))
print(len(published_difference))

[
    published_difference.pop()
    for i in range(10)
]

total_words = set(list(dict(top_ten_preprint).keys()) + list(dict(top_ten_published).keys()))
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

published_comparison_stats_df = pd.DataFrame.from_records(data)
published_comparison_stats_df.to_csv(
    "output/544536_v2_PMC6687187_comparison.tsv", 
    sep="\t", index=False
)
published_comparison_stats_df.head()

(
    published_comparison_stats_df
    .sort_values("log_likelihood", ascending=False)
    .head(20)
)

(
    published_comparison_stats_df
    .sort_values("odds_ratio", ascending=False)
    .head(20)
)
