import csv
import lzma
import re

import lxml.etree as ET
import numpy as np
import pandas as pd
import spacy

disabled_pipelines = ["parser", "ner"]
nlp = spacy.load("en_core_web_sm", disable=disabled_pipelines)
nlp.max_length = 9999999

filter_tag_list = [
    "sc",
    "italic",
    "xref",
    "label",
    "sub",
    "sup",
    "inline-formula",
    "fig",
    "disp-formula",
    "bold",
    "table-wrap",
    "table",
    "thead",
    "tbody",
    "caption",
    "tr",
    "td",
]
parser = ET.XMLParser(encoding="UTF-8", recover=True)


def dump_article_text(
    file_path, xpath_str, filter_tags=filter_tag_list, remove_stop_words=True
):
    """
    This method is designed to extract all text from xml documents.
    Every document has specific tags that are striped in order to produce
    clean text output for downstream processing

    Keyword arguments:
        file_path - the file path for xml document
        xpath_str - the xpath string to extract tags from the xml document
        filter_tag_list - the list of tags to strip from the xml document
        remove_stop_words - a flag to indicate if stop words should be removed
    """

    tree = ET.parse(open(file_path, "rb"), parser=parser)

    # Process xml without specified tags
    ET.strip_tags(tree, *filter_tags)

    root = tree.getroot()
    all_tags = root.xpath(xpath_str)
    text = list(map(lambda x: "".join(list(x.itertext())), all_tags))
    text = " ".join(text)

    # Remove stop words
    if remove_stop_words:
        text = list(
            map(
                lambda x: x.lemma_.lower(),
                filter(
                    lambda tok: tok.lemma_ not in nlp.Defaults.stop_words,
                    nlp(text),
                ),
            )
        )
    else:
        text = list(map(lambda x: x.lemma_.lower(), nlp(text)))

    return text


def generate_sectional_vector(model, doc_xpath_obj, filter_Tags=filter_tag_list):
    word_vectors = []

    all_text = list(map(lambda x: "".join(list(x.itertext())), doc_xpath_obj))
    all_text = " ".join(all_text)

    word_vectors += [
        model.wv[text]
        for text in filter(lambda tok: tok in model.wv, all_text.split(" "))
    ]

    # skips weird documents that don't contain text
    if len(word_vectors) > 0:
        return np.stack(word_vectors).mean(axis=0)

    return []


def generate_doc_vector(model, document_path, xpath, filter_tags=filter_tag_list):
    """
    This method is designed to construct document vectors for a given xml document.
    Every document has specific tags that are striped in order to have accurate embeddings

    Keyword arguments:
        model - the model to extract word vectors from
        xpath_str - the xpath string to extract tags from the xml document
        filter_tag_list - the list of tags to strip from the xml document
    """

    word_vectors = []

    tree = ET.parse(open(document_path, "rb"), parser=parser)

    # Process xml without specified tags
    ET.strip_tags(tree, *filter_tags)

    root = tree.getroot()
    all_text = root.xpath(xpath)
    all_text = list(map(lambda x: "".join(list(x.itertext())), all_text))
    all_text = " ".join(all_text)

    all_tokens = list(
        map(
            lambda x: x.lemma_,
            filter(
                lambda tok: tok.lemma_ in model.wv
                and tok.lemma_ not in nlp.Defaults.stop_words,
                nlp(all_text),
            ),
        )
    )

    word_vectors += [model.wv[text] for text in all_tokens]

    # skips weird documents that don't contain text
    if len(word_vectors) > 0:
        return np.stack(word_vectors).mean(axis=0)

    return []


def generate_doc_vector_parallel(
    model, xpath, document_queue, vector_queue, filter_tags=filter_tag_list
):
    """
    This method is designed to construct document vectors for a given xml document.
    Every document has specific tags that are striped in order to have accurate embeddings
        e.g. we want to remove tags such as xref -> (<xref rid="fig4" ref-type="fig">Fig. 4</xref>)

    Keyword arguments:
        model - the model to extract word vectors from
        xpath_str - the xpath string to extract tags from the xml document
        filter_tag_list - the list of tags to strip from the xml document
    """
    while True:

        document_path = document_queue.get()

        if document_path is None:
            break

        word_vectors = []

        tree = ET.parse(open(document_path[2], "rb"), parser=parser)

        # Process xml without specified tags
        ET.strip_tags(tree, *filter_tags)

        root = tree.getroot()
        all_text = root.xpath(xpath)
        all_text = list(map(lambda x: "".join(list(x.itertext())), all_text))
        all_text = " ".join(all_text)

        all_tokens = list(
            map(
                lambda x: x.lemma_,
                filter(
                    lambda tok: tok.lemma_ in model.wv
                    and tok.lemma_ not in nlp.Defaults.stop_words,
                    nlp(all_text),
                ),
            )
        )

        word_vectors += [model.wv[text] for text in all_tokens]

        # skips weird documents that don't contain text
        if len(word_vectors) > 0:
            vector_queue.put(
                (
                    document_path[0],
                    document_path[1],
                    np.stack(word_vectors).mean(axis=0),
                )
            )

    # Poison Pill to end writer
    vector_queue.put(None)


def write_document_vector_parallel(vector_queue, file_path_name, dim, n_jobs):
    """
    This method is designed to write document vectors from a given xml document to a tsv file.
    This is the output part of the parallel processing workframe to parse PubMed articles at a faster rate.

    Keyword arguments:
        vector_queue - the queue that contains an entry for the tsv file
        file_path_name - the name of the file that will contain all document vectors
        dim - the number of dimension for each document vector
        n_jobs - the number of working jobs that convert xml docs to vectors
    """

    if "xz" in file_path_name:
        outfile = lzma.open(file_path_name, "wt")
    else:
        outfile = open(file_path_name, "w")

    fields = ["journal", "document"] + [f"feat_{col}" for col in range(int(dim))]
    writer = csv.DictWriter(outfile, delimiter="\t", fieldnames=fields)
    writer.writeheader()

    job_count = 0
    while True:

        # Get document vector
        doc_vector = vector_queue.get()

        # Count number of feeder jobs
        # If all has finished then terminate
        # this thread
        if doc_vector is None:
            job_count += 1

            if job_count == n_jobs:
                break

            continue

        if len(doc_vector) > 0:

            output_values = zip(
                fields, [doc_vector[0], doc_vector[1]] + list(doc_vector[2])
            )

            writer.writerow(dict(output_values))

    outfile.close()


class DocIterator:
    """
    A class designed to feed lines into the word2vec model
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in open(self.filepath, "r"):
            yield [
                tok.lemma_
                for tok in nlp(line)
                if tok.lemma_ not in nlp.Defaults.stop_words
            ]
