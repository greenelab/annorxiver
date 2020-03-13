#!/usr/bin/env python
# coding: utf-8

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
from tqdm import tqdm_notebook


# In[2]:


biorxiv_files = Path("../biorxiv_articles").rglob("*.xml")


# ## Parse BioRxiv

# In[3]:


def header_group_mapper(header):
    if "method" in header:
        return "material and methods"
    if "abstract" in header:
        return "abstract"
    if "conclusion" in header:
        return "conclusion"
    if re.search(r"(supplementary|supplemental) material", header):
        return "supplemental material"
    if re.search(r"(declaration[s]?( of interest[s]?)?)|(competing (financial )?interest[s]?)", header):
        return "conflict of interest"
    if "additional information" in header:
        return "supplemental information"
    if re.search(r"author[s]?[']? contribution[s]?", header):
        return "author contribution"
    if re.search(r"(supplementary|supporting) information", header):
        return "supplemental information"
    if "data accessibility" in header:
        return "data availability"
    if re.search(r"experimental procedures", header):
        return "material and methods"
    return header


# In[4]:


article_metadata = []
article_sections = []

type_mapper = {
    'author-type':'author_type',
    'heading':'heading',
    'hwp-journal-coll':'category'
}
xml_parser = ET.XMLParser(encoding='UTF-8', recover=True)
for file in tqdm_notebook(biorxiv_files):
    article = file.with_suffix('').name
    root = ET.parse(open(file, "rb"), parser=xml_parser).getroot()
    
    # Grab the subject category
    metadata = {
        type_mapper[x.attrib['subj-group-type']]:x.getchildren()[0].text.lower()
        for x in root.xpath('//subj-group')
    }
    
    metadata.update({'document':f"{article}.xml", 'doi':root.xpath('//article-id')[0].text})
    article_metadata.append(metadata)
    
    # Grab the section titles 
    section_objs = list(
        filter(
            lambda x: "id" in x.attrib and re.search(r"s[\d]+$", x.attrib['id']) is not None,
            root.xpath('//sec')
        )
    )
    
    title_objs = list(map(lambda x: x.xpath('title//text()'), section_objs))
    title_objs = list(filter(lambda x: len(x) > 0, title_objs))
    
    # edge case in the xml where
    # a tag contains the following: <title>A<sc>bstract</sc></title>
    # why is there a <sc> tag?
    if any(list(map(lambda x: len(x) > 1, title_objs))):
    
        # filter out weird characters ⓘ
        # cant think of a better way to handle these types of edge cases
        title_objs = list(map(lambda headers: list(filter(lambda token: token != 'ⓘ', headers)), title_objs))
        title_objs = list(map(lambda x: x[0] + x[1] if len(x) > 1 else x, title_objs))
    
    abstract_section = root.xpath('//abstract/title//text()')
    if len(abstract_section) > 0:
        
        # in case of a parse error that splits A from bstract
        if len(abstract_section) > 1:
            abstract_section = ["".join(abstract_section)]
        
        title_objs = title_objs + [abstract_section]

    title_objs = list(map(lambda x: x[0].rstrip().lower(), title_objs))
    
    article_sections += list(
        map(
            lambda x: {'section':header_group_mapper(x[0]), 'document':x[1]},
            product(title_objs, [article])
        )
    )


# In[5]:


metadata_df = (
    pd.DataFrame
    .from_records(article_metadata)
    .fillna({"category":'none', 'author_type':'none', 'heading':'none'})
    .assign(category=lambda x:x.category.apply(lambda x: " ".join(x.split("_")) if "_" in x else x))
)

metadata_df.to_csv("output/biorxiv_article_metadata.tsv", sep="\t", index=False)
metadata_df.head()


# In[6]:


sections_df = (
    pd.DataFrame
    .from_records(article_sections)
)

sections_df.to_csv("output/biorxiv_article_sections.tsv", sep="\t", index=False)
sections_df.head()


# # Regular Research Articles?

# BioRxiv claims that each article should be a research article. The plot below mainly confirms that statement.

# In[7]:


g = (
    p9.ggplot(metadata_df, p9.aes(x="author_type"))
    + p9.geom_bar(size=10)
    + p9.theme_bw()
)
print(g)


# In[8]:


metadata_df["author_type"].value_counts()


# # BioRxiv Research Article Categories

# Categories assigned to each research article. Neuroscience dominates majority of the articles as expected.

# In[9]:


category_list = metadata_df.category.value_counts().index.tolist()[::-1]

# plot nine doesn't implement reverse keyword for scale x discrete
# ugh...
g = (
    p9.ggplot(metadata_df, p9.aes(x="category"))
    + p9.geom_bar(size=10)
    + p9.scale_x_discrete(limits=category_list)
    + p9.coord_flip()
    + p9.theme_bw()
)
print(g)


# In[10]:


metadata_df["category"].value_counts()


# # New, Confirmatory, Contradictory Results?

# In[11]:


heading_list = metadata_df.heading.value_counts().index.tolist()[::-1]

g = (
    p9.ggplot(metadata_df, p9.aes(x="heading"))
    + p9.geom_bar(size=10)
    + p9.scale_x_discrete(limits=heading_list)
    + p9.coord_flip()
    + p9.theme_bw()
)
print(g)


# In[12]:


metadata_df["heading"].value_counts()


# # BioRxiv Section Articles

# In[14]:


section_list = sections_df.section.value_counts()
section_list = section_list[section_list > 800].index.to_list()[::-1]

g = (
    p9.ggplot(sections_df[sections_df.section.isin(section_list)])
    + p9.aes(x="section")
    + p9.geom_bar(position="dodge")
    + p9.scale_x_discrete(limits=section_list)
    + p9.coord_flip()
    + p9.theme_bw()
)
print(g)


# In[17]:


section_list = sections_df.section.value_counts()
section_list[section_list > 800]

