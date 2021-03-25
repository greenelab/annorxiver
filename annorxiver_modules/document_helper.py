import re

from gensim.parsing.preprocessing import remove_stopwords
import lxml.etree as ET
import numpy as np
import pandas as pd


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
    text = list(map(lambda x: list(x.itertext()), list(all_tags)))

    # Remove stop words
    if remove_stop_words:
        text = list(map(lambda x: remove_stopwords(re.sub("\n", "", "".join(x))), text))

    else:
        text = list(map(lambda x: re.sub("\n", "", "".join(x)), text))

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

    word_vectors += [
        model.wv[text]
        for text in filter(lambda tok: tok in model.wv, all_text.split(" "))
    ]

    # skips weird documents that don't contain text
    if len(word_vectors) > 0:
        return np.stack(word_vectors).mean(axis=0)

    return []


class DocIterator:
    """
    A class designed to feed lines into the word2vec model
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        for line in open(self.filepath, "r"):
            yield line.split()
