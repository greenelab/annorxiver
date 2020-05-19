#!/usr/bin/env python
# coding: utf-8

# # Pubmed Central (PMC) Exploratory Data Analysis

# This notebook is designed to generate descriptive statistics for the PMC repository.

# In[1]:


from pathlib import Path 
from collections import Counter

import pandas as pd
import plotnine as p9
import lxml.etree as ET
import tqdm


# # Journal Statistics

# Gather a listing of journals contained in PMC.

# In[2]:


journals = list(Path("../journals").rglob("*.nxml"))


# In[3]:


journal_paper_count = Counter(map(lambda x: x.parent.stem, journals))
journal_records = [{
    'journal':item[0],
    'article_count':item[1]
    }
    for item in journal_paper_count.items()
]


# In[4]:


journal_df = (
    pd.DataFrame
    .from_records(journal_records)
    .sort_values("journal")
    .reset_index(drop=True)
)
journal_df.head()


# ## Map Journals to PMC articles

# In[5]:


journal_type_records = []
for file in tqdm.tqdm_notebook(journals):
    journal = file.parent.stem
    tree = ET.parse(str(file.resolve()))
    root = tree.getroot()
    journal_type_records.append({
        'journal': journal,
        'article_type': root.attrib['article-type'].strip(),
        'doi':(
            root.xpath("//article-meta/article-id[@pub-id-type='doi']")[0].text
            if len(root.xpath("//article-meta/article-id[@pub-id-type='doi']")) > 0
            else ""
        ),
        'pmcid':file.stem
    })


# In[6]:


journal_paper_df = pd.DataFrame.from_records(journal_type_records)
journal_paper_df.to_csv("output/pubmed_central_journal_paper_map.tsv.xz", sep="\t", index=False, compression="xz")
journal_paper_df.head()


# In[7]:


journal_paper_df.journal.unique().shape


# # Types of Articles Contained in PMC

# In[8]:


journal_article_type_list = journal_paper_df['article_type'].value_counts().index.tolist()[::-1]
journal_article_type_list = journal_article_type_list[-15:]

g = (
    p9.ggplot(journal_paper_df.query(f"article_type in {journal_article_type_list}"))
    + p9.aes(x="article_type")
    + p9.geom_bar(position="dodge")
    + p9.scale_x_discrete(limits=journal_article_type_list)
    + p9.coord_flip()
    + p9.theme_bw()
)
g.save("output/figures/article_type.png", dpi=500)
print(g)

