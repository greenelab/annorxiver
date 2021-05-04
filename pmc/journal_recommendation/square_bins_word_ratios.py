# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:annorxiver]
#     language: python
#     name: conda-env-annorxiver-py
# ---

# # Calculate Odds Ratios for each Square Bin

# +
# %load_ext autoreload
# %autoreload 2

import csv
from collections import Counter, defaultdict
import json
import lzma
from multiprocessing import Process, Manager
from pathlib import Path
import pickle
import re
import sys
from threading import Thread

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook

from annorxiver_modules.corpora_comparison_helper import (
    aggregate_word_counts,
    get_term_statistics,
)

from annorxiver_modules.word_bin_helper import lemmatize_tokens, write_lemma_counts
# -

# # Gather Paper Bins Dataframe

pmc_df = pd.read_csv("output/paper_dataset/paper_dataset_tsne_square.tsv", sep="\t")
print(pmc_df.shape)
pmc_df.head()

word_count_folder = Path("../pmc_corpus/pmc_word_counts/")

word_counter_file = "output/app_plots/global_doc_word_counter.tsv.xz"
field_names = ["document", "lemma", "count"]
n_jobs = 3
QUEUE_SIZE = 75000  # Queue Size if too big then will need to make smaller
doc_xpath = "//abstract/sec/*|//abstract/p|//body/sec/*|//body/p"

with Manager() as m:

    # Set up the Queue
    doc_path_queue = m.JoinableQueue(QUEUE_SIZE)
    lemma_queue = m.JoinableQueue(QUEUE_SIZE)

    # Start the document object feeder
    t = Thread(
        target=write_lemma_counts,
        args=(word_counter_file, field_names, lemma_queue, n_jobs),
    )
    t.start()

    running_jobs = []
    # Start the jobs
    for job in range(n_jobs):
        p = Process(
            target=lemmatize_tokens, args=(doc_xpath, doc_path_queue, lemma_queue)
        )
        running_jobs.append(p)
        p.start()

    for idx, row in tqdm_notebook(pmc_df.iterrows()):
        doc_path = f"../journals/{row['journal']}/{row['document']}.nxml"
        doc_path_queue.put(doc_path)

    # Poison pill to end running jobs
    for job in running_jobs:
        doc_path_queue.put(None)

    # Wait for jobs to finish
    for job in running_jobs:
        job.join()

    # Wait until thread is done running
    t.join()

with lzma.open(word_counter_file, "rt") as infile:
    reader = csv.DictReader(infile, delimiter="\t")

    background_bin_dictionaries = defaultdict(Counter)
    word_bin_dictionaries = {
        squarebin_id: defaultdict(Counter)
        for squarebin_id in pmc_df.squarebin_id.unique()
    }

    document_mapper = dict(zip(pmc_df.document.tolist(), pmc_df.squarebin_id.tolist()))

    for line in tqdm_notebook(reader):
        squarebin_id = document_mapper[line["document"]]
        background_bin_dictionaries.update({line["lemma"]: int(line["count"])})
        word_bin_dictionaries[squarebin_id].update({line["lemma"]: int(line["count"])})

# +
cutoff_score = 20
background_sum = sum(background_bin_dictionaries.values())
bin_ratios = {}

for squarebin in tqdm_notebook(word_bin_dictionaries):

    bin_dict = word_bin_dictionaries[squarebin]
    bin_sum = sum(word_bin_dictionaries[squarebin].values())

    # Try and filter out low count tokens to speed function up
    filtered_bin_dict = {
        lemma: bin_dict[lemma] for lemma in bin_dict if bin_dict[lemma] > cutoff_score
    }

    if len(filtered_bin_dict) > 0:
        bin_dict = filtered_bin_dict

    # Calculate odds ratio
    bin_words = set(bin_dict.keys())
    background_words = set(background_bin_dictionaries.keys())
    words_to_compute = bin_words & background_words

    word_odd_ratio_records = []
    for idx, word in enumerate(words_to_compute):
        top = float(bin_dict[word] * background_sum)
        bottom = float(background_bin_dictionaries[word] * bin_sum)
        word_odd_ratio_records.append(
            {"lemma": word, "odds_ratio": np.log(top / bottom)}
        )

    sorted(word_odd_ratio_records, key=lambda x: x["odds_ratio"], reverse=True)
    bin_ratios[squarebin] = word_odd_ratio_records[0:20]
# -

# # Insert Bin Word Associations in JSON File

square_bin_plot_df = pd.read_json(
    open(Path("output") / Path("app_plots") / Path("pmc_square_plot.json"))
)
square_bin_plot_df.head()

bin_odds_df = pd.DataFrame.from_records(
    [{"bin_id": key, "bin_odds": bin_ratios[key]} for key in bin_ratios]
)
bin_odds_df.head()

(
    square_bin_plot_df.merge(bin_odds_df, on=["bin_id"]).to_json(
        Path("output") / Path("app_plots") / Path("pmc_square_plot.json"),
        orient="records",
        lines=False,
    )
)
