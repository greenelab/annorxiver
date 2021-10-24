# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Find published articles missing from bioRxiv

# +
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotnine as p9
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegressionCV
import tqdm
# -

# # Load Embeddings

# ## bioRxiv

biorxiv_journal_df = (
    pd.read_csv(
        "../journal_tracker/output/mapped_published_doi_before_update.tsv", sep="\t"
    )
    .rename(index=str, columns={"doi": "preprint_doi"})
    .groupby("preprint_doi")
    .agg(
        {
            "document": "last",
            "category": "first",
            "preprint_doi": "last",
            "published_doi": "first",
            "pmcid": "first",
            "pmcoa": "first",
        }
    )
    .reset_index(drop=True)
)
biorxiv_journal_df.head()

biorxiv_embed_df = pd.read_csv(
    Path("../word_vector_experiment/output/")
    / "word2vec_output/"
    / "biorxiv_all_articles_300_fixed.tsv.xz",
    sep="\t",
)
biorxiv_embed_df = biorxiv_embed_df.dropna()
biorxiv_embed_df.head()

biorxiv_journal_mapped_df = biorxiv_journal_df[
    ["document", "published_doi", "pmcid", "pmcoa"]
].merge(biorxiv_embed_df, on="document")
biorxiv_journal_mapped_df.head()

# ## Pubmed Central

pmc_articles_df = pd.read_csv(
    Path("../../pmc/exploratory_data_analysis/")
    / "output/pubmed_central_journal_paper_map.tsv.xz",
    sep="\t",
).query("article_type=='research-article'")
pmc_articles_df.head()

pmc_embed_df = pd.read_csv(
    Path("../../pmc/word_vector_experiment/output")
    / Path("pmc_document_vectors_300_replace.tsv.xz"),
    sep="\t",
)
pmc_embed_df.head()

pmc_journal_mapped_df = (
    pmc_articles_df[["doi", "pmcid"]]
    .merge(pmc_embed_df, left_on="pmcid", right_on="document")
    .drop("pmcid", axis=1)
)
pmc_journal_mapped_df.head()

# # Calculate Distances

# ## biorxiv -> published versions

biorxiv_published = (
    biorxiv_journal_mapped_df.query("pmcid.notnull()")
    .query("pmcoa == True")
    .sort_values("pmcid", ascending=True)
    .drop_duplicates("pmcid")
    .set_index("pmcid")
)
biorxiv_published.head()

PMC_published = (
    pmc_journal_mapped_df.query(f"document in {biorxiv_published.index.tolist()}")
    .sort_values("document", ascending=True)
    .set_index("document")
)
PMC_published.head()

article_distances = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()].drop(
        ["document", "published_doi", "pmcoa"], axis=1
    ),
    PMC_published.drop(["doi", "journal"], axis=1),
    "euclidean",
)
article_distances.shape

articles_distance_df = (
    biorxiv_published.loc[PMC_published.index.tolist()]
    .reset_index()[["document", "pmcid"]]
    .assign(
        distance=np.diag(article_distances, k=0), journal=PMC_published.journal.tolist()
    )
)
articles_distance_df.head()

# ## biorxiv -> random paper same journal

PMC_off_published = (
    pmc_journal_mapped_df.drop("doi", axis=1)
    .query(f"document not in {biorxiv_published.index.tolist()}")
    .query(f"journal in {articles_distance_df.journal.unique().tolist()}")
    .groupby("journal", group_keys=False)
    .apply(lambda x: x.sample(1, random_state=100))
)
PMC_off_published.head()

journal_mapper = {
    journal: col for col, journal in enumerate(PMC_off_published.journal.tolist())
}
list(journal_mapper.items())[0:10]

off_article_dist = cdist(
    biorxiv_published.loc[PMC_published.index.tolist()]
    .drop(["document", "published_doi", "pmcoa"], axis=1)
    .values,
    PMC_off_published.drop(["document", "journal"], axis=1).values,
    "euclidean",
)
off_article_dist.shape

data = []
for idx, row in tqdm.tqdm(articles_distance_df.iterrows()):
    if row["journal"] in journal_mapper:
        data.append(
            {
                "document": row["document"],
                "pmcid": (
                    PMC_off_published.query(f"journal=='{row['journal']}'")
                    .reset_index()
                    .document.values[0]
                ),
                "journal": row["journal"],
                "distance": off_article_dist[idx, journal_mapper[row["journal"]]],
            }
        )

final_df = articles_distance_df.assign(label="pre_vs_published").append(
    pd.DataFrame.from_records(data).assign(label="pre_vs_random")
)
final_df.head()

final_df = biorxiv_journal_df[["document", "preprint_doi"]].merge(final_df)
final_df.to_csv("output/annotated_links/article_distances.tsv", sep="\t", index=False)
final_df.head()

# # Distribution plot

g = (
    p9.ggplot(
        final_df.replace(
            {
                "pre_vs_published": "preprint-published",
                "pre_vs_random": "preprint-random",
            }
        )
    )
    + p9.aes(x="label", y="distance")
    + p9.geom_violin(fill="#a6cee3")
    + p9.labs(x="Document Pair Groups", y="Euclidean Distance")
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=2)
)
g.save("output/figures/biorxiv_article_distance.svg")
g.save("output/figures/biorxiv_article_distance.png")
print(g)

# # Logistic Regression bioRxiv preprints -> published PMC articles

model = LogisticRegressionCV(
    Cs=5,
    cv=10,
    random_state=100,
    penalty="elasticnet",
    solver="saga",
    l1_ratios=[0.1, 0.5, 0.8],
    verbose=1,
)

retained_ids = list(
    set(PMC_published.index.tolist()) & set(biorxiv_published.index.tolist())
)

training_dataset = (
    biorxiv_published.dropna()
    .drop(["document", "published_doi", "pmcoa"], axis=1)
    .loc[retained_ids]
    - PMC_published.loc[retained_ids].dropna().drop(["journal", "doi"], axis=1)
).assign(
    biorxiv_document=biorxiv_published.loc[retained_ids].document.values, true_link=1
)
training_dataset.head()

journals = (
    PMC_published.loc[retained_ids]
    .query(f"journal in {PMC_off_published.journal.tolist()}")
    .journal.values.tolist()
)

off_documents = (
    PMC_published.loc[retained_ids]
    .query(f"journal in {PMC_off_published.journal.tolist()}")
    .index.tolist()
)

training_dataset = (
    training_dataset.append(
        pd.DataFrame(
            biorxiv_published.loc[off_documents]
            .drop(["document", "published_doi", "pmcoa"], axis=1)
            .values
            - PMC_off_published.iloc[list(map(lambda x: journal_mapper[x], journals))]
            .set_index("journal")
            .drop("document", axis=1)
            .values,
            columns=[f"feat_{idx}" for idx in range(300)],
        ).assign(true_link=-1)
    )
    .reset_index(drop=True)
    .drop("biorxiv_document", axis=1)
    .dropna()
)
training_dataset.head()

fit_model = model.fit(
    training_dataset.sample(frac=1, random_state=100).drop("true_link", axis=1),
    training_dataset.sample(frac=1, random_state=100).true_link,
)

fit_model.scores_

pickle.dump(fit_model, open("output/optimized_model.pkl", "wb"))

# # Find bioRxiv unpublished ->  published PMC articles

biorxiv_unpublished = biorxiv_journal_mapped_df.query("published_doi.isnull()").drop(
    ["published_doi", "pmcid", "pmcoa"], axis=1
)
print(biorxiv_unpublished.shape)
biorxiv_unpublished.head()

PMC_unlinked = pmc_journal_mapped_df.query(
    f"""
        document not in {
            biorxiv_published
            .reset_index()
            .pmcid
            .tolist()
        }
        """
)
print(PMC_unlinked.shape)
PMC_unlinked.head()

cutoff_score = final_df.query("label=='pre_vs_random'").distance.min()
cutoff_score

chunksize = 100
chunk_iterator = range(0, biorxiv_unpublished.shape[0], chunksize)

for idx, chunk in tqdm.tqdm(enumerate(chunk_iterator)):

    # Chunk the documents so memory doesn't break
    biorxiv_subset = biorxiv_unpublished.iloc[chunk : chunk + chunksize]

    # Calculate distances
    paper_distances = cdist(
        biorxiv_subset.drop(["document"], axis=1),
        PMC_unlinked.drop(["journal", "document", "doi"], axis=1),
        "euclidean",
    )

    # Get elements less than threshold
    sig_indicies = np.where(paper_distances < cutoff_score)
    results = zip(
        sig_indicies[0],
        sig_indicies[1],
        paper_distances[paper_distances < cutoff_score],
    )

    # Map the results into records for pandas to parse
    results = list(
        map(
            lambda x: dict(
                document=biorxiv_subset.iloc[x[0]].document,
                pmcid=PMC_unlinked.iloc[x[1]].document,
                distance=x[2],
            ),
            results,
        )
    )

    # There may be cases where there are no matches
    if len(results) > 0:
        # Generate pandas dataframe
        potential_matches_df = (
            biorxiv_journal_df[["document", "preprint_doi"]]
            .merge(pd.DataFrame.from_records(results))
            .sort_values("distance")
        )

        # Write to file
        if idx == 0:
            potential_matches_df.to_csv(
                "output/potential_biorxiv_pmc_links_rerun.tsv", sep="\t", index=False
            )

        else:
            potential_matches_df.to_csv(
                "output/potential_biorxiv_pmc_links_rerun.tsv",
                sep="\t",
                index=False,
                mode="a",
                header=False,
            )


# # Bin Potential Matches

potential_matches_df = pd.read_csv(
    "output/potential_biorxiv_pmc_links_rerun.tsv", sep="\t"
)
potential_matches_df.head()

potential_matches_df = (
    potential_matches_df.rename(index=str, columns={"preprint_doi": "biorxiv_doi"})
    .drop_duplicates(["document", "biorxiv_doi", "pmcid"])
    .assign(
        pmcid_url=lambda x: (
            x.pmcid.apply(lambda paper: f"https://www.ncbi.nlm.nih.gov/pmc/{paper}")
        ),
        biorxiv_doi_url=lambda x: (
            x.biorxiv_doi.apply(lambda paper: f"https://doi.org/{paper}")
        ),
    )
)
potential_matches_df.head()

# +
distance_bins = np.squeeze(
    final_df.query("label=='pre_vs_published'")
    .describe()
    .loc[["25%", "50%", "75%"]]
    .values
)

distance_bins = np.append(
    [0.0],
    distance_bins,
)

distance_bins = np.append(
    distance_bins, (final_df.query("label=='pre_vs_random'").distance.min())
)

distance_bins

# +
potential_matches_df = potential_matches_df.assign(
    distance_bin=(
        pd.cut(
            potential_matches_df.distance,
            distance_bins,
            right=False,
            labels=[
                "[0, 25%ile)",
                "[25%ile, 50%ile)",
                "[50%ile, 75%ile)",
                "[75%, min(same-journal-no-known-link))",
            ],
        )
    )
)[
    [
        "document",
        "biorxiv_doi",
        "biorxiv_doi_url",
        "pmcid",
        "pmcid_url",
        "distance",
        "distance_bin",
    ]
]

potential_matches_df.to_csv(
    "output/potential_biorxiv_pmc_links_rerun.tsv", sep="\t", index=False
)

potential_matches_df.head()
