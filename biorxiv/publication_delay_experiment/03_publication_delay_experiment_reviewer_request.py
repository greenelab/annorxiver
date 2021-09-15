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

# # Measure the Difference between Preprint-Published similarity and Published Articles

# This notebook is designed to answer the question how much do preprints change with each new version.
# Based on results within my manuscript a review wanted to know the answer to the above question.
# Also this notebook outputs an excel file that contains all preprint published pairs and their respective publication information.
# Allows people to manually inspect practical consequences (if any) for preprints that take longer to publish.

from datetime import timedelta
import numpy as np
import pandas as pd
import plotnine as p9
from scipy.stats import linregress

# # Load the Document Distances

published_date_distances = pd.read_csv(
    "output/preprint_published_distances.tsv", sep="\t"
)
for col in ["preprint_date", "published_date"]:
    published_date_distances[col] = pd.to_datetime(published_date_distances[col])
published_date_distances["time_to_published"] = pd.to_timedelta(
    published_date_distances["time_to_published"]
)
print(published_date_distances.shape)
published_date_distances.head()

# # Plot Version count against Doc Distances

# Reviewer wanted to see if there is an association between version count and document distances. (i.e. if preprints with more versions have more text changes).

# +
x = (published_date_distances["version_count"].values.tolist(),)
y = published_date_distances["doc_distances"].values.tolist()

results = linregress(x, y)
print(results)
# -

published_date_distances["version_count"] = pd.Categorical(
    published_date_distances["version_count"].tolist()
)
g = (
    p9.ggplot(published_date_distances, p9.aes(x="version_count", y="doc_distances"))
    + p9.geom_boxplot(fill="#b2df8a")
    + p9.geom_line(
        data=pd.DataFrame(
            dict(
                version_count=np.arange(1, 13),
                doc_distances=np.arange(1, 13) * 0.02669 + 0.8697,
            )
        ),
        linetype="dashed",
        color="#1f78b4",
        size=1,
    )
    + p9.annotate(
        "text",
        label=f"y={results.slope:0.4f}*X + {results.intercept:0.4f}",
        x=9,
        y=7.5,
        size=13,
        color="#1f78b4",
    )
    + p9.labs(
        x="# of Preprint Versions",
        y="Euclidean Distance of Preprint-Published Versions",
    )
    + p9.theme_seaborn(style="white", context="notebook")
)
g.save("output/version_count_doc_distances.svg")
g.save("output/version_count_doc_distances.png", dpi=600)
print(g)

# Overall, preprints change with each new version; however, based on the magnitude of the slope I'd argue that these changes are minor compared to substantial changes (~6 distance units)

# # Output published dates to Excel

# Reviewer asked if manually pursuing preprints that take longer to publish would produce any interesting results. Great question, but not enough time to go into that; however, providing a supplementary file for others to look into could provide an in depth answer.

excel_print_df = published_date_distances.drop(
    ["document", "category", "pmcoa"], axis=1
).rename(
    index=str,
    columns={
        "preprint_date": "posted_date",
        "time_to_published": "days_till_published",
        "doc_distances": "preprint_published_distance",
    },
)[
    [
        "preprint_doi",
        "posted_date",
        "pmcid",
        "published_doi",
        "journal",
        "published_date",
        "days_till_published",
        "preprint_published_distance",
        "version_count",
    ]
]
excel_print_df

excel_print_df = excel_print_df[excel_print_df["days_till_published"] > pd.Timedelta(0)]
excel_print_df["posted_date"] = excel_print_df.posted_date.dt.date
excel_print_df["published_date"] = excel_print_df.published_date.dt.date

(
    excel_print_df.sort_values("days_till_published", ascending=False).to_excel(
        "output/published_preprints_information.xlsx", engine="xlsxwriter"
    )
)
