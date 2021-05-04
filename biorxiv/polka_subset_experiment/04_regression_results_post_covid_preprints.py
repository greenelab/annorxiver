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

# # Re-Run Analyses with Polka et. al. Subset

# This notebook was created in response to Polka et al. Group's inquiry on training a logistic regression model on preprints posted recently rather than preprints from 2019 and below.
# Overall their subset can be separated with a few features.

# +
from pathlib import Path
import sys

from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import plotnine as p9
import requests
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import spacy
import tqdm

from annorxiver_modules.document_helper import generate_doc_vector

mpl.rcParams["figure.dpi"] = 250
# -

# # Random BioRxiv Sample

manual_papers_df = pd.read_csv(str(Path("output/all_pairs_2021-02-11.csv")))
manual_papers_df.head().T

api_url = "https://api.biorxiv.org/details/biorxiv/2020-01-01/2020-04-30"
response = requests.get(api_url)
content = response.json()
total_papers = content["messages"][0]["total"]
total_papers

np.random.seed(100)
selected_biorxiv_papers = np.random.randint(0, total_papers, 100)
selected_biorxiv_papers.sort()
selected_biorxiv_papers

paper_cursor = {}
for paper in selected_biorxiv_papers:
    cursor = int(np.ceil(int(paper / 100)))
    if cursor not in paper_cursor:
        paper_cursor[cursor] = []
    paper_cursor[cursor].append(paper)
paper_cursor

published_doi_map = []
for paper in tqdm.tqdm(paper_cursor):
    api_url = f"https://api.biorxiv.org/details/biorxiv/2020-01-01/2020-04-30/{paper}"
    response = requests.get(api_url)
    content = response.json()
    collection = content["collection"]

    for paper_idx in paper_cursor[paper]:
        user_doi = collection[paper_idx % 100]["doi"]
        file_name = user_doi.split("/")[-1]

        api_url = f"https://api.biorxiv.org/details/biorxiv/{user_doi}"
        response = requests.get(api_url)
        content = response.json()

        latest_paper = content["collection"][-1]
        version_count = len(content["collection"])

        doc_url = "http://biorxiv.org/content"
        file_url = f"{doc_url}/early/{latest_paper['date'].replace('-', '/')}/{file_name}.source.xml"
        response = requests.get(file_url)

        with open(
            f"output/biorxiv_xml_files_recent/{file_name}_v{version_count}.xml", "wb"
        ) as outfile:
            outfile.write(response.content)

# # Document Embeddings

# ## Convert New biorxiv subset

biorxiv_documents = [
    Path(x.name) for x in list(Path("output/biorxiv_xml_files_recent").rglob("*xml"))
]

biorxiv_xpath_str = "//abstract/p|//abstract/title|//body/sec//p|//body/sec//title"
word_model = Word2Vec.load(
    str(Path("../word_vector_experiment/output/word2vec_models/300/biorxiv_300.model"))
)

biorxiv_document_map = {
    document: generate_doc_vector(
        word_model,
        document_path=str(Path("output/biorxiv_xml_files_recent") / document),
        xpath=biorxiv_xpath_str,
    )
    for document in tqdm.tqdm_notebook(biorxiv_documents)
}

# +
biorxiv_vec_df = (
    pd.DataFrame.from_dict(biorxiv_document_map, orient="index")
    .rename(columns={col: f"feat_{col}" for col in range(int(300))})
    .rename_axis("document")
    .reset_index()
)

biorxiv_vec_df.to_csv(
    "output/random_recent_biorxiv_subset_embeddings.tsv", sep="\t", index=False
)

biorxiv_vec_df.head().T
# -

# ## Load the Documents

polka_preprints_df = pd.read_csv("output/polka_et_al_biorxiv_embeddings.tsv", sep="\t")
polka_preprints_df.head()

pca_components = pd.read_csv(
    Path("../pca_association_experiment/output/word_pca_similarity/pca_components.tsv"),
    sep="\t",
)
pca_components.head()

# ## PCA Components

# This section aims to see which principal components have a high association with Polka et al's subset. Furthermore, we also aim to see if we can use linear models to explain which PCs affect preprint prediction.

document_pca_sim = 1 - cdist(
    polka_preprints_df.drop("document", axis=1).values, pca_components.values, "cosine"
)
print(document_pca_sim.shape)
document_pca_sim

document_to_pca_map = {
    document: document_pca_sim[idx, :]
    for idx, document in enumerate(polka_preprints_df.document.tolist())
}

polka_pca_sim_df = (
    pd.DataFrame.from_dict(document_to_pca_map, orient="index")
    .rename(index=str, columns={col: f"pc{col+1}" for col in range(int(300))})
    .reset_index()
    .rename(index=str, columns={"index": "document"})
)
# polka_pca_sim_df.to_csv("output/polka_pca_enrichment.tsv", sep="\t")
polka_pca_sim_df = polka_pca_sim_df.assign(label="polka")
polka_pca_sim_df.head()

document_pca_sim = 1 - cdist(
    biorxiv_vec_df.drop("document", axis=1).values,
    pca_components.values,
    "cosine",
)
print(document_pca_sim.shape)
document_pca_sim

document_to_pca_map = {
    document: document_pca_sim[idx, :]
    for idx, document in enumerate(biorxiv_vec_df.document.tolist())
}

biorxiv_pca_sim_df = (
    pd.DataFrame.from_dict(document_to_pca_map, orient="index")
    .rename(index=str, columns={col: f"pc{col+1}" for col in range(int(300))})
    .reset_index()
    .rename(index=str, columns={"index": "document"})
    .assign(label="biorxiv")
)
biorxiv_pca_sim_df.head()

# ## PC Regression

# ### Logistic Regression

# Goal here is to determine if we can figure out which PCs separate the bioRxiv subset from Polka et al.'s subset. Given that their dataset is only 60 papers we downsampled our dataset to contain only 60 papers.

dataset_df = biorxiv_pca_sim_df.append(polka_pca_sim_df)
dataset_df.head()

model = LogisticRegressionCV(
    cv=10, Cs=100, max_iter=1000, penalty="l1", solver="liblinear"
)
model.fit(
    StandardScaler().fit_transform(dataset_df[[f"pc{idx+1}" for idx in range(50)]]),
    dataset_df["label"],
)

best_result = list(filter(lambda x: x[1] == model.C_, enumerate(model.Cs_)))[0]
print(best_result)

print("Best CV Fold")
print(model.scores_["polka"][:, best_result[0]])
model.scores_["polka"][:, best_result[0]].mean()

model_weights_df = pd.DataFrame.from_dict(
    {
        "weight": model.coef_[0],
        "pc": list(range(1, 51)),
    }
)
model_weights_df["pc"] = pd.Categorical(model_weights_df["pc"])
model_weights_df.head()

g = (
    p9.ggplot(model_weights_df, p9.aes(x="pc", y="weight"))
    + p9.geom_col(position=p9.position_dodge(width=5), fill="#253494")
    + p9.coord_flip()
    + p9.scale_x_discrete(limits=list(sorted(range(1, 51), reverse=True)))
    + p9.theme_seaborn(context="paper", style="ticks", font_scale=1.1, font="Arial")
    + p9.theme(figure_size=(10, 8))
    + p9.labs(
        title="Regression Model Weights", x="Princpial Component", y="Model Weight"
    )
)
# g.save("output/figures/pca_log_regression_weights.svg")
# g.save("output/figures/pca_log_regression_weights.png", dpi=250)
print(g)

fold_features = model.coefs_paths_["polka"].transpose(1, 0, 2)
model_performance_df = pd.DataFrame.from_dict(
    {
        "feat_num": ((fold_features.astype(bool).sum(axis=1)) > 0).sum(axis=1),
        "C": model.Cs_,
        "score": model.scores_["polka"].mean(axis=0),
    }
)
model_performance_df.head()

# +
fig, ax1 = plt.subplots()
ax1.set_xscale("log")
ax2 = plt.twinx()

ax1.plot(
    model_performance_df.C.tolist(),
    model_performance_df.feat_num.tolist(),
    label="Features",
    marker=".",
)
ax1.set_ylabel("# of Features")
ax1.set_xlabel("Inverse Regularization (C)")
ax1.legend(loc=0)

ax2.plot(
    model_performance_df.C.tolist(),
    model_performance_df.score.tolist(),
    label="Score",
    marker=".",
    color="green",
)
ax2.set_ylabel("Score (Accuracy %)")
ax2.legend(loc=4)
# plt.savefig("output/preprint_classifier_results.png")
# -

plot_path = list(
    zip(
        model.Cs_,
        model.scores_["polka"].transpose(),
        model.coefs_paths_["polka"].transpose(1, 0, 2),
    )
)

data_records = []
for cs in plot_path[33:40]:
    model = LogisticRegression(C=cs[0], max_iter=1000, penalty="l1", solver="liblinear")
    model.fit(
        StandardScaler().fit_transform(dataset_df[[f"pc{idx+1}" for idx in range(50)]]),
        dataset_df["label"],
    )
    data_records.append(
        {
            "C": cs[0],
            "PCs": ",".join(map(str, model.coef_.nonzero()[1] + 1)),
            "feat_num": len(model.coef_.nonzero()[1]),
            "accuracy": cs[1].mean(),
        }
    )

model_coefs_df = pd.DataFrame.from_records(data_records)
model_coefs_df
