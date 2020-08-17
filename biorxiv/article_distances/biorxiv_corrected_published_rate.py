#!/usr/bin/env python
# coding: utf-8

# # Estimate the Correct Number of Published Preprints

# In[1]:


from pathlib import Path

import lxml.etree as ET
import numpy as np
import pandas as pd
import plotnine as p9
from tqdm import tqdm


# # Load Abdil et al. Score

# In[2]:


file_path="https://elifesciences.org/download/aHR0cHM6Ly9jZG4uZWxpZmVzY2llbmNlcy5vcmcvYXJ0aWNsZXMvNDUxMzMvZWxpZmUtNDUxMzMtZmlnMy1kYXRhMS12Mi5jc3Y-/elife-45133-fig3-data1-v2.csv?_hash=6d5MoXPaQXFZiifDTRaKXoZ29OnVUJq%2FuoHlyuh%2Bg04%3D"


# In[3]:


abdill_df = pd.read_csv(file_path)
abdill_df.head(10)


# # Current Estimated Published Fraction

# ## Gather all published preprints

# In[4]:


biorxiv_journal_df = (
    pd.read_csv(
        Path("../journal_tracker")/
        Path("output/mapped_published_doi.tsv"), 
        sep="\t"
    )
    .groupby("doi")
    .agg({
        "document":"first",
        "category":"first",
        "journal":"first",
        "doi":"last",
        "published_doi":"first",  
        "pmcid":"first", 
        "pmcoa":"first"
    })
    .reset_index(drop=True)
    .assign(published=lambda x: x.published_doi.apply(lambda y: "/" in str(y)))
)
print(biorxiv_journal_df.shape)
biorxiv_journal_df.head()


# In[5]:


print(f"Total Published: {biorxiv_journal_df.published.sum()}")
print(f"Total Published in PMCOA: {biorxiv_journal_df.pmcoa.sum()}")


# In[6]:


annotated_links_df =(
    pd.read_csv(
        Path("output/annotated_links")/
        Path("potential_biorxiv_pmc_links.tsv"), 
        sep="\t"
    )
    .assign(published=True)
    .rename(index=str, columns={"biorxiv_doi":"doi"})
    [["document", "doi", "published", "distance_bin"]]
)
print(annotated_links_df.shape)
annotated_links_df.head()


# In[7]:


missing_link_mapper = {
    row['doi']:[
        row['published'],
        True
    ]
    for row in (
        annotated_links_df
        .query("distance_bin in ['[0, 25%ile)', '[25%ile, 50%ile)']")
        [["doi", "published"]]
        .to_dict(orient="records")
    )
}
new_link_count = len(missing_link_mapper)
print(f"Novel links filled: {new_link_count}")


# In[8]:


remaining_links = {
    row['doi']:[
        row['published'],
        row['pmcoa']
    ]
    for row in (
        biorxiv_journal_df
        [["doi", "published", "pmcoa"]]
        .to_dict(orient="records")
    ) 
    if row['doi'] not in missing_link_mapper
}
print(len(remaining_links))


# In[9]:


missing_link_mapper.update(remaining_links)
print(len(missing_link_mapper))


# In[10]:


updated_biorxiv_journal_df = (
    pd.DataFrame
    .from_records([
        {
            "doi":row[0], 
            "published":row[1][0], 
            "pmcoa":row[1][1]
        } 
        for row in list(remaining_links.items())
    ])
    .merge(
        biorxiv_journal_df
         .drop(["published", "pmcoa"], axis=1), 
        on="doi"
    )
)
print(updated_biorxiv_journal_df.shape)
updated_biorxiv_journal_df.head()


# ## Gather date published for each document

# In[11]:


file_path = Path("output/biorxiv_articles_published_date.tsv")


# In[12]:


if not file_path.exists():
    data_records = []
    xml_parser = ET.XMLParser(encoding='UTF-8', recover=True)
    for document in tqdm(updated_biorxiv_journal_df.document.tolist()):
        root = (
            ET
            .parse(
                f"../biorxiv_articles/{document}", 
                parser=xml_parser
            )
            .getroot()
        )

        date_node = root.xpath(
            "//front/article-meta/history/date"
        )

        if len(date_node) == 0:
            print("Error in xml:")
            print(document)
            continue

        elif len(date_node) < 2:
            date_node = date_node[0]

        else:
            date_node = date_node[1]

        data_row = dict(
            map(
                lambda x: (x.tag, x.text), 
                date_node.getchildren()
            )
        )

        data_records.append({
            "month":f"{data_row['year']}-{int(data_row['month']):02d}",
            "document":document
        })


# In[13]:


if not file_path.exists():
    published_preprint_df = pd.DataFrame.from_records(
        data_records+[
            {"month":"2018-09", "document":"423517_v1.xml"},
            {"month":"2019-01", "document":"528299_v1.xml"}
        ]
    )
else:
    published_preprint_df = pd.read_csv(
        file_path, 
        sep="\t"
    )
print(published_preprint_df.shape)
published_preprint_df.head()


# In[14]:


if not file_path.exists():
    final_mapped_df = (
        published_preprint_df
        .merge(
            updated_biorxiv_journal_df, 
            on="document"
        )
        .rename(index=str, columns={"month":"pub_month"})
    )
    print(final_mapped_df.shape)
else:
    final_mapped_df = published_preprint_df
    print(final_mapped_df.shape)

final_mapped_df


# In[15]:


if not file_path.exists():
    (
        final_mapped_df
        .to_csv(
            "output/biorxiv_articles_published_date.tsv",
            sep="\t", index=False
        )
    )


# In[16]:


snapshot_wo_links_df = (
    biorxiv_journal_df
    .assign(published_closed=lambda x: (x.pmcoa != x.published))
    .merge(
        final_mapped_df[["document", "pub_month"]],
        on="document"
    )
    .replace({"2013-10":"2013-11"})
    .groupby("pub_month")
    .agg({
        "published":lambda x: (x==True).sum(),
        "pub_month":"count",
        "published_closed":lambda x: (x==True).sum(),
    })
    .rename(index=str, columns={"pub_month":"posted"})
    .reset_index()
    .assign(
        label="2020 Snapshot",
        rate=lambda x: x.published/x.posted
    )
)
snapshot_wo_links_df.head()


# In[17]:


snapshot_w_links_df = (
    final_mapped_df
    .replace({"2013-10":"2013-11"})
    .groupby("pub_month")
    .agg({
        "published":lambda x: (x==True).sum(),
        "pub_month":"count"
    })
    .rename(index=str, columns={"pub_month":"posted"})
    .reset_index()
    .assign(
        rate=lambda x: x.published/x.posted,
        label="2020 Snapshot+Missing Links"
    )
)
snapshot_w_links_df.head()


# ## Calculate the publication rate

# In[18]:


publish_rate_df = (
    snapshot_w_links_df
    .append(
        abdill_df
         .assign(label="Abdill et al. (2018)")
         .rename(index=str, columns={"month":"pub_month"}),
        sort=False
    )
    .append(snapshot_wo_links_df.drop("published_closed", axis=1))
    .reset_index(drop=True)
)
publish_rate_df.sample(10, random_state=100)


# In[19]:


publish_rate_df['pub_month'] = (
    pd.Categorical(
        publish_rate_df.pub_month.values.tolist(), 
        ordered=True
    )
)

posted = (
    publish_rate_df
    .query("label=='2020 Snapshot+Missing Links'")
    .posted
    .sum()
)

published = (
    publish_rate_df
    .query("label=='2020 Snapshot+Missing Links'")
    .published
    .sum()
)
print(f"Published: {published}")
print(f"Posted: {posted}")
print(f"Overall proportion published: {published/posted:.4f}")


# In[20]:


if False:
    # Calculate per month missing OA links
    missing_oa_rate = (
        (snapshot_w_links_df.published - snapshot_wo_links_df.published)/
        (snapshot_wo_links_df.posted - snapshot_wo_links_df.published_closed)
    )

    # Calculate published closed rate
    published_closed_rate = (
        snapshot_wo_links_df.published_closed/snapshot_wo_links_df.posted
    )


# In[21]:


if False:
    publish_rate_df = (
        publish_rate_df
        .append(
            publish_rate_df
            .query("label=='2020 Snapshot+Missing Links'")
            .assign(
                published=lambda x: x.published + (missing_oa_rate.values * (
                    snapshot_w_links_df
                    .posted.values - 
                    snapshot_w_links_df
                    .published
                    .values
                    )
                ),
                label="2020 Estimate"
            )
            .assign(
                rate=lambda x: x.published/x.posted
            )
        )
    )
    publish_rate_df.sample(10, random_state=100)


# In[22]:


if False:
    est_published = (
        publish_rate_df
        .query("label=='2020 Estimate'")
        .published
        .sum()
    )

    est_posted = (
        publish_rate_df
        .query("label=='2020 Estimate'")
        .posted
        .sum()
    )


# # Plot Publication Rate

# In[23]:


color_mapper = {
    "2018":"#a6cee3",
    "2020ML":"#33a02c",
    "2020":"#1f78b4",
}


# In[24]:


g = (
    p9.ggplot(
        publish_rate_df
        .rename(index=str, columns={"label":"Label"})
    )
    + p9.aes(
        x="pub_month", y="rate", 
        fill="Label", group="Label", 
        color="Label", linetype="Label",
        shape="Label"
    )
    + p9.geom_point(
        size=2
    )
    + p9.geom_line()
    + p9.scale_linetype_manual(
        ['solid', 'solid', 'solid']
    )
    + p9.theme_seaborn(
        style='ticks', 
        context='paper',
        font_scale=1.3
    )
    + p9.theme(
        figure_size=(10,4.5),
        axis_text_x=p9.element_blank(),
        axis_title_x=p9.element_text(
            weight='bold',
            margin={"t":15}
        ),
        axis_title_y=p9.element_text(weight='bold'),
    )
    + p9.labs(
        y="Proportion Published",
        x="Month"
    )
    + p9.scale_color_manual(
        [
            color_mapper['2020'],
            color_mapper['2020ML'],
            color_mapper['2018']
        ]
    )
    + p9.scale_fill_manual(
        [
            color_mapper['2020'],
            color_mapper['2020ML'],
            color_mapper['2018']
        ]
    )
    + p9.scale_shape_manual(
        ['o', 'o', 'o']
    )
    
    # plot the x axis titles
    + p9.geom_vline(xintercept=[2.5, 14.5, 26.5, 38.5, 50.5, 62.5, 74.5])
    + p9.geom_text(label="2014", x=8.5, y=0, color="black")
    + p9.geom_text(label="2015", x=20.5, y=0, color="black")
    + p9.geom_text(label="2016", x=32.5, y=0, color="black")
    + p9.geom_text(label="2017", x=44.5, y=0, color="black")
    + p9.geom_text(label="2018", x=56.5, y=0, color="black")
    + p9.geom_text(label="2019", x=68.5, y=0, color="black")
    
    # Plot the overall proportion published
    + p9.geom_hline(
        yintercept=0.4196, 
        linetype='solid', 
        color=color_mapper['2018']
    )
    
    + p9.geom_hline(
        yintercept=published/posted, 
        linetype="solid", 
        color=color_mapper['2020ML']
    )
    
    + p9.annotate(
        "text", x=8.5, y=0.395, 
        label="overall: 0.4196", size = 8
    )
    
    + p9.annotate(
        "text", x=8.5, y=0.48, 
        label=f"overall: {published/posted:.4f}", size = 8
    )
)
g.save("output/figures/publication_rate.svg", dpi=500)
g.save("output/figures/publication_rate.png", dpi=500)
print(g)

