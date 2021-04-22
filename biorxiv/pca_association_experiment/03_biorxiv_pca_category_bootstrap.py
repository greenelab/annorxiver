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

# # Determine Word to PCA Associations

# This notebook is designed to run PCA over the document embeddings and calculate category-pca associations with each principal component.

# +
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import plotnine as p9
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from tqdm import tqdm_notebook
# -

journal_map_df = (
    pd.read_csv(
        Path("..")
        / Path("exploratory_data_analysis")
        / Path("output/biorxiv_article_metadata.tsv"),
        sep="\t",
    )
    .groupby("doi")
    .agg(
        {
            "doi": "last",
            "document": "first",
            "category": "last",
        }
    )
)
journal_map_df.head()

# # PCA the Documents

# Run PCA over the documents. Generates 50 principal components, but can generate more or less.

n_components = 50
random_state = 100

biorxiv_articles_df = pd.read_csv(
    Path("..")
    / Path("word_vector_experiment")
    / Path("output/word2vec_output")
    / Path("biorxiv_all_articles_300_fixed.tsv.xz"),
    sep="\t",
)
biorxiv_articles_df = biorxiv_articles_df.dropna()
biorxiv_articles_df.head()

# +
reducer = PCA(n_components=n_components, random_state=random_state)

reducer.fit(biorxiv_articles_df[[f"feat_{idx}" for idx in range(300)]].values)
# -

document_categories_df = journal_map_df[["document", "category"]].merge(
    biorxiv_articles_df, on="document"
)
document_categories_df.head()

# # Bootstrap 95% Confidence Intervals Cosine sim

# +
boostraped_iterations = 10000
pc_1 = []
pc_2 = []

# The for loop performs the following algorithm to estimate 95% confidence intervals:
#
# 1. Group document embeddings by category
# 2. Randomly sample a document from each category
# 3. Calculate the similarity scores between the documents and the first two principal components
# 4. Repeat the above steps for 10000 iterations.
# 5. Finally take the 25th percentile and the 97.5th percentile to make up the interval

for iteration in tqdm_notebook(range(boostraped_iterations)):
    sampled_df = (
        document_categories_df.groupby("category")
        .apply(lambda x: x.sample(1, random_state=iteration))
        .reset_index(drop=True)
        .sort_values("category")
    )

    document_distance = 1 - cdist(
        (sampled_df.drop(["document", "category"], axis=1).values),
        reducer.components_[0:2],
        "cosine",
    )

    pc_1.append(document_distance[:, 0])
    pc_2.append(document_distance[:, 1])

pc_1 = np.stack(pc_1)
pc_2 = np.stack(pc_2)
# -

mean_similarity_df = pd.DataFrame(
    (
        1
        - cdist(
            document_categories_df.drop(["document", "category"], axis=1).values,
            reducer.components_[0:2],
            "cosine",
        )
    ),
    columns=["pca1_cossim", "pca2_cossim"],
).assign(
    category=document_categories_df.category.tolist(),
    document=document_categories_df.document.tolist(),
)
mean_similarity_df.head()

category_sim_df = (
    mean_similarity_df.groupby("category")
    .agg({"pca1_cossim": "mean", "pca2_cossim": "mean"})
    .assign(
        pca1_cossim_upper=np.percentile(pc_1, 97.5, axis=0),
        pca1_cossim_lower=np.percentile(pc_1, 25, axis=0),
        pca2_cossim_upper=np.percentile(pc_2, 97.5, axis=0),
        pca2_cossim_lower=np.percentile(pc_2, 25, axis=0),
    )
    .reset_index()[
        [
            "category",
            "pca1_cossim_lower",
            "pca1_cossim",
            "pca1_cossim_upper",
            "pca2_cossim_lower",
            "pca2_cossim",
            "pca2_cossim_upper",
        ]
    ]
)
category_sim_df.head()

category_sim_df.to_csv("output/category_cossim_95_ci.tsv", sep="\t", index=False)

g = (
    p9.ggplot(category_sim_df)
    + p9.aes(
        x="category",
        y="pca1_cossim",
        ymin="pca1_cossim_lower",
        ymax="pca1_cossim_upper",
    )
    + p9.geom_pointrange()
    + p9.coord_flip()
    + p9.theme_bw()
    + p9.scale_x_discrete(limits=category_sim_df.category.tolist()[::-1])
    + p9.theme(
        figure_size=(11, 7),
        text=p9.element_text(size=12),
        panel_grid_major_y=p9.element_blank(),
    )
    + p9.labs(y="PC1 Cosine Similarity")
)
g.save("output/pca_plots/figures/category_pca1_95_ci.svg", dpi=500)
g.save("output/pca_plots/figures/category_pca1_95_ci.png", dpi=500)
print(g)

g = (
    p9.ggplot(category_sim_df)
    + p9.aes(
        x="category",
        y="pca2_cossim",
        ymax="pca2_cossim_upper",
        ymin="pca2_cossim_lower",
    )
    + p9.geom_pointrange()
    + p9.coord_flip()
    + p9.theme_bw()
    + p9.scale_x_discrete(limits=category_sim_df.category.tolist()[::-1])
    + p9.theme(
        figure_size=(11, 7),
        text=p9.element_text(size=12),
        panel_grid_major_y=p9.element_blank(),
    )
    + p9.labs(y="PC2 Cosine Similarity")
)
g.save("output/pca_plots/figures/category_pca2_95_ci.svg", dpi=500)
g.save("output/pca_plots/figures/category_pca2_95_ci.png", dpi=500)
print(g)

# # Plot Documents Projected on PCs Grouped by Category

projected_documents = reducer.transform(
    document_categories_df[[f"feat_{idx}" for idx in range(300)]]
)
projected_documents.shape

projected_documents_df = pd.DataFrame(
    projected_documents, columns=[f"PC_{dim+1}" for dim in range(n_components)]
).assign(
    category=document_categories_df.category.tolist(),
    document=document_categories_df.document.tolist(),
)
projected_documents_df

g = (
    p9.ggplot(projected_documents_df)
    + p9.aes(x="factor(category)", y="PC_1")
    + p9.geom_boxplot(
        fill="#a6cee3",
        outlier_size=1,
        outlier_alpha=0.65,
        fatten=1.5,
    )
    + p9.coord_flip()
    + p9.scale_x_discrete(
        limits=(
            projected_documents_df.groupby("category")
            .agg({"PC_1": "median"})
            .sort_values("PC_1", ascending=False)
            .reset_index()
            .category.tolist()[::-1]
        )
    )
    + p9.labs(x="Article Category", y="PC1")
    + p9.theme(figure_size=(6.66, 5))
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1)
)
g.save("output/pca_plots/figures/category_box_plot_pc1.png", dpi=250)
g.save(
    "output/pca_plots/svg_files/category_box_plot/category_box_plot_pc1.svg", dpi=250
)
print(g)

g = (
    p9.ggplot(projected_documents_df)
    + p9.aes(x="factor(category)", y="PC_2")
    + p9.geom_boxplot(
        fill="#a6cee3",
        outlier_size=1,
        outlier_alpha=0.65,
        fatten=1.5,
    )
    + p9.coord_flip()
    + p9.scale_x_discrete(
        limits=(
            projected_documents_df.groupby("category")
            .agg({"PC_2": "median"})
            .sort_values("PC_2", ascending=False)
            .reset_index()
            .category.tolist()[::-1]
        )
    )
    + p9.labs(x="Article Category", y="PC2")
    + p9.theme(figure_size=(6.66, 5))
    + p9.theme_seaborn(context="paper", style="ticks", font="Arial", font_scale=1)
)
g.save("output/pca_plots/figures/category_box_plot_pc2.png", dpi=250)
g.save(
    "output/pca_plots/svg_files/category_box_plot/category_box_plot_pc2.svg", dpi=250
)
print(g)


# # Tables with Figures

def dump_figures_and_table(
    table_df, figure_selector, output_path="output/table", column_value="PC_1"
):

    # Output figures to folder
    for idx, row in table_df.iterrows():
        subprocess.Popen(
            [
                "unzip",
                "-j",
                "-o",
                Path("..") / Path(row["hash"]),
                Path("content") / Path(figure_selector[row["document"]]),
                "-d",
                Path("output/table_figures"),
            ]
        )

    # Output table in Manubot format
    # to make incorporating this figure much easier in text
    # http://greenelab.github.io/annorxiver_manuscript
    (
        table_df[["document", "doi", column_value]]
        # doi - a manubot formatted doi link. Manubot creaes citations via [@doi:doi_link]
        # figure - a markdown formmated link to show figure images within a given table.
        .assign(
            doi=lambda x: x.doi.apply(lambda link: f"[@doi:{link}]"),
            figure=lambda x: x.document.apply(
                lambda doc: f"![](table_figures/{document_figure_selector[doc]})"
            ),
        )
        .drop("document", axis=1)
        .to_csv(f"{output_path}.tsv", sep="\t", index=False)
    )


document_hash_df = pd.read_csv("../biorxiv_doc_hash_mapper.tsv", sep="\t")
document_hash_df.head()

# ## Top PC1

# 035014_v1.xml does not have a figure.
# Use replacement document instead
top_pc1_documents = (
    journal_map_df[["document", "doi"]]
    .merge(
        projected_documents_df.query("category=='systems biology'")
        .sort_values("PC_1", ascending=False)
        .head(5),
        on="document",
    )
    .merge(document_hash_df, left_on="document", right_on="doc_number")
)
top_pc1_documents

document_figure_selector = {
    "044818_v1.xml": "044818_fig1.tif",
    "107250_v1.xml": "107250_fig1.tif",
    "197400_v1.xml": "197400_fig1.tif",
    "769299_v1.xml": "769299v1_fig1.tif",
    "266775_v1.xml": "266775_fig1.tif",
}

dump_figures_and_table(
    top_pc1_documents,
    document_figure_selector,
    output_path="output/tables/top_pc1_table",
    column_value="PC_1",
)

# ## Bottom PC1

bottom_pc1_documents = (
    journal_map_df[["document", "doi"]]
    .merge(
        projected_documents_df.query("category=='systems biology'")
        .sort_values("PC_1", ascending=True)
        .head(),
        on="document",
    )
    .merge(document_hash_df, left_on="document", right_on="doc_number")
)
bottom_pc1_documents

document_figure_selector = {
    "872887_v1.xml": "872887v1_fig1.tif",
    "455048_v1.xml": "455048_fig1.tif",
    "733162_v1.xml": "733162v1_fig1.tif",
    "745943_v1.xml": "745943v1_fig1.tif",
    "754572_v1.xml": "754572v1_fig1.tif",
}

dump_figures_and_table(
    bottom_pc1_documents,
    document_figure_selector,
    output_path="output/tables/bottom_pc1_table",
    column_value="PC_1",
)

# ## Top PC2

top_pc2_documents = (
    journal_map_df[["document", "doi"]]
    .merge(
        projected_documents_df.query("category=='systems biology'")
        .sort_values("PC_2", ascending=False)
        .head(),
        on="document",
    )
    .merge(document_hash_df, left_on="document", right_on="doc_number")
)
top_pc2_documents

document_figure_selector = {
    "220152_v1.xml": "220152_fig3.tif",
    "328591_v1.xml": "328591_fig1.tif",
    "595819_v1.xml": "595819_fig1.tif",
    "484204_v2.xml": "484204v2_fig1.tif",
    "781328_v1.xml": "781328v1_fig3.tif",
}

dump_figures_and_table(
    top_pc2_documents,
    document_figure_selector,
    output_path="output/tables/top_pc2_table",
    column_value="PC_2",
)

# ## Bottom PC2

bottom_pc2_documents = (
    journal_map_df[["document", "doi"]]
    .merge(
        projected_documents_df.query("category=='systems biology'")
        .sort_values("PC_2", ascending=True)
        .head(),
        on="document",
    )
    .merge(document_hash_df, left_on="document", right_on="doc_number")
)
bottom_pc2_documents

document_figure_selector = {
    "019687_v1.xml": "019687_fig1.tif",
    "301051_v1.xml": "301051_fig3.tif",
    "357939_v1.xml": "357939_fig4.tif",
    "386367_v3.xml": "386367v3_fig1.tif",
    "840280_v1.xml": "840280v1_fig1.tif",
}

dump_figures_and_table(
    bottom_pc2_documents,
    document_figure_selector,
    output_path="output/tables/bottom_pc2_table",
    column_value="PC_2",
)
