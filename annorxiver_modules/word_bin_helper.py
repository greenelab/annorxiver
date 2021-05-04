from collections import Counter
import csv
import lzma
from pathlib import Path

import spacy
from annorxiver_modules.document_helper import dump_article_text


def lemmatize_tokens(doc_xpath, document_path_queue, lemma_queue):
    """
    Function designed to take dicionary entries of word counts and append them to a queue.
    Spacy is used to lemmatize each word if possible

    Parameters:
        xpath - extract text from using lxml xpaths
        aggregated_counts_queue - queue that holds aggregated word counts
        lemma_queue - queue to place parsed counts onto for future processing
    """

    while True:
        doc_path = document_path_queue.get()

        if doc_path is None:
            break

        document_text = dump_article_text(file_path=doc_path, xpath_str=doc_xpath)

        word_dict = Counter(document_text)
        lemma_queue.put((Path(doc_path).stem, word_dict))

    # Poison Pill for writer thread
    lemma_queue.put(None)


def write_lemma_counts(file_name, field_names, lemma_queue, n_jobs=3):
    """
    Function designed to take an entry from the lemma queue and write it to a
    tab delimited file

    Parameters:
        file_name - name of the file to write to
        field_names - the header for the tabbed file
        lemma_queue - the queue that contains the lemmaa counts
        n_jobs - the number of processes running for this section
    """

    if "xz" in file_name:
        outfile = lzma.open(file_name, "wt")
    else:
        outfile = open(file_name, "w")

    writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=field_names)
    writer.writeheader()
    job_count = 0

    while True:
        lemma_entry = lemma_queue.get()

        if lemma_entry is None:
            job_count += 1

            if job_count == n_jobs:
                break

            continue

        for lemma_term in lemma_entry[1]:
            writer.writerow(
                {
                    "document": lemma_entry[0],
                    "lemma": lemma_term,
                    "count": lemma_entry[1][lemma_term],
                }
            )

    outfile.close()
