#!/usr/bin/env python
# coding: utf-8

# This NB is copied from https://github.com/greenelab/annorxiver/ and I parallelized it
# This should probably be cleaned up quite a bit but... this was for a hackathon and I haven't gotten around to it

# # Exploratory Data Analysis-BioRxiv

# This notebook is designed to generate descriptive statistics for a snapshot of the BioRxiv repository. The following information is obtained: 
# 1. if the article is a research article
# 2. if the article is a new, contradictory, or confirmatory analysis
# 3. the category assigned to each research article (pi self assigns)
# 4. the type of section headers contain in each research article

# ## Load the environment to parse BioRxiv

# In[1]:


from pathlib import Path
import re
from itertools import product

import lxml.etree as ET
import pandas as pd

import plotnine as p9

# from tqdm import tqdm_notebook
from tqdm.notebook import tqdm


# In[2]:


biorxiv_files = Path("/home/thielk/gitlab/ctha-biorxiv-analysis/notebooks").rglob("content/*.xml")


# In[3]:


# total_files = len(list(biorxiv_files))
total_files = 110717


# ## Parse BioRxiv

# In[4]:


def header_group_mapper(header):
    if re.search("method", header, flags=re.I):
        return "material and methods"
    if re.search("abstract", header, flags=re.I):
        return "abstract"
    if re.search("conclusion", header, flags=re.I):
        return "conclusion"
    if re.search(r"(supplementary|supplemental) material", header, flags=re.I):
        return "supplemental material"
    if re.search(
        r"(declaration[s]?( of interest[s]?)?)|(competing (financial )?interest[s]?)",
        header,
        flags=re.I,
    ):
        return "conflict of interest"
    if re.search("additional information", header, flags=re.I):
        return "supplemental information"
    if re.search(r"author[s]?[']? contribution[s]?", header, flags=re.I):
        return "author contribution"
    if re.search(r"(supplementary|supporting) information", header, flags=re.I):
        return "supplemental information"
    if re.search("data accessibility", header, flags=re.I):
        return "data availability"
    if re.search(r"experimental procedures", header, flags=re.I):
        return "material and methods"
    return header


# In[5]:


def parse_article_xml(file):
    type_mapper = {
        "author-type": "author_type",
        "heading": "heading",
        "hwp-journal-coll": "category",
    }
    xml_parser = ET.XMLParser(encoding="UTF-8", recover=True)
    article = file.with_suffix("").name
    with open(file, "rb") as f:
        root = ET.parse(f, parser=xml_parser).getroot()

    # Grab the subject category
    metadata = {
        type_mapper[x.attrib["subj-group-type"]]: x.getchildren()[0].text.lower()
        for x in root.xpath("//subj-group")
    }

    metadata.update(
        {
            "document": f"{article}.xml",
            "doi": root.xpath("//article-id")[0].text,
            "date_received": ""
            if not root.xpath("//history")
            else "-".join(
                [out.text for out in reversed(root.xpath("//history")[0][0])]
            ),
        }
    )
    #     article_metadata.append(metadata)

    # Grab the section titles
    section_objs = list(
        filter(
            lambda x: "id" in x.attrib
            and re.search(r"s[\d]+$", x.attrib["id"]) is not None,
            root.xpath("//sec"),
        )
    )

    title_objs = list(map(lambda x: x.xpath("title//text()"), section_objs))
    title_objs = list(filter(lambda x: len(x) > 0, title_objs))

    # edge case in the xml where
    # a tag contains the following: <title>A<sc>bstract</sc></title>
    # why is there a <sc> tag?
    if any(list(map(lambda x: len(x) > 1, title_objs))):

        # filter out weird characters ⓘ
        # cant think of a better way to handle these types of edge cases
        title_objs = list(
            map(
                lambda headers: list(filter(lambda token: token != "ⓘ", headers)),
                title_objs,
            )
        )
        title_objs = list(map(lambda x: x[0] + x[1] if len(x) > 1 else x, title_objs))

    abstract_section = root.xpath("//abstract/title//text()")
    if len(abstract_section) > 0:

        # in case of a parse error that splits A from bstract
        if len(abstract_section) > 1:
            abstract_section = ["".join(abstract_section)]

        title_objs = title_objs + [abstract_section]

    title_objs = list(map(lambda x: x[0].rstrip().lower(), title_objs))

    #     article_sections += list(
    #         map(
    #             lambda x: {'section':header_group_mapper(x[0]), 'document':x[1]},
    #             product(title_objs, [article])
    #         )
    #     )
    return (
        metadata,
        list(
            map(
                lambda x: {"section": header_group_mapper(x[0]), "document": x[1]},
                product(title_objs, [article]),
            )
        ),
    )


# In[6]:


from joblib import Parallel, delayed


# In[7]:


run_in_parallel = True
if run_in_parallel:
    out = Parallel(n_jobs=40)(
        delayed(parse_article_xml)(file)
        for file in tqdm(biorxiv_files, total=total_files)
    )
else:
    out = [parse_article_xml(file) for file in tqdm(bioarxiv_files)]

article_metadata, article_sections = zip(*out)
article_sections = [section for article in article_sections for section in article]


# In[8]:


metadata_df = (
    pd.DataFrame.from_records(article_metadata)
    .fillna({"category": "none", "author_type": "none", "heading": "none"})
    .assign(
        category=lambda x: x.category.apply(
            lambda x: " ".join(x.split("_")) if "_" in x else x
        )
    )
    .replace(
        {
            "heading": {
                "bioinformatics": "none",
                "genomics": "none",
                "zoology": "none",
                "evolutionary biology": "none",
                "animal behavior and cognition": "none",
                "ecology": "none",
                "genetics": "none",
            }
        }
    )
)

metadata_df.to_csv("output/biorxiv_article_metadata.tsv", sep="\t", index=False)
metadata_df.head()


# In[9]:


sections_df = pd.DataFrame.from_records(article_sections)

sections_df.to_csv("output/biorxiv_article_sections.tsv", sep="\t", index=False)
sections_df.head()

