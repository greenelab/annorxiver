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

# # Grab the bioRxiv subset from Polka et al 2021

# This notebook is designed to download preprints that have been manually annotated by [Polka et al](https://www.biorxiv.org/content/10.1101/2021.02.20.432090v1).
# Once these preprints have been downloaded they will be processed by my constructed pipeline.

# +
from pathlib import Path
import tarfile

import pandas as pd
import requests
import tqdm
import urllib.request as request
# -

manual_papers_df = pd.read_csv(str(Path("output/all_pairs_2021-02-11.csv")))
manual_papers_df.head().T

manual_papers_df.exclude.unique()

papers_to_download = manual_papers_df.fillna("").query(
    "exclude.str.contains('keep')&source=='biorxiv'"
)
print(papers_to_download.shape)
papers_to_download.head()

parsed_files = [
    str(x.stem).split("_")[0]
    for x in list(Path("output/biorxiv_xml_files").rglob("*xml"))
]

published_doi_map = []
for idx, paper in tqdm.tqdm(papers_to_download.iterrows()):
    user_doi = paper["doi"]
    file_name = user_doi.split("/")[-1]

    if file_name in parsed_files:
        continue

    api_url = f"https://api.biorxiv.org/details/biorxiv/{user_doi}"
    response = requests.get(api_url)
    content = response.json()
    latest_paper = content["collection"][-1]
    version_count = len(content["collection"])
    published_doi_map.append(
        {"biorxiv_doi": user_doi, "published_doi": latest_paper["published"]}
    )

    doc_url = "http://biorxiv.org/content"
    file_url = f"{doc_url}/early/{latest_paper['date'].replace('-', '/')}/{file_name}.source.xml"

    response = requests.get(file_url)

    with open(
        f"output/biorxiv_xml_files/{file_name}_v{version_count}.xml", "wb"
    ) as outfile:
        outfile.write(response.content)

if not Path("output/polka_et_al_mapped_subset.tsv").exists():
    mapped_papers_df = pd.DataFrame.from_records(published_doi_map)
    mapped_papers_df.to_csv(
        "output/polka_et_al_mapped_subset.tsv", sep="\t", index=False
    )
else:
    mapped_papers_df = pd.read_csv("output/polka_et_al_mapped_subset.tsv", sep="\t")
mapped_papers_df.head()

# # Perform DOI to PM(C)ID Conversion

# Copy and paste the list into the text box on this online conversion website: https://www.ncbi.nlm.nih.gov/pmc/pmctopmid/. Download the csv results and continue parsing the file.

for doi in mapped_papers_df.published_doi.tolist():
    print(doi)

mapped_doi_pmcids = pd.read_csv("output/mapped_doi_to_pmc.csv")
mapped_doi_pmcids.head()

# +
if not Path("output/polka_et_al_pmc_mapped_subset.tsv").exists():
    pmcid_mapped_papers_df = (
        mapped_papers_df.merge(
            mapped_doi_pmcids, left_on="published_doi", right_on="DOI"
        )
        .query("PMCID.notnull()")
        .drop("DOI", axis=1)
    )
    pmcid_mapped_papers_df.to_csv(
        "output/polka_et_al_pmc_mapped_subset.tsv", sep="\t", index=False
    )
else:
    pmcid_mapped_papers_df = pd.read_csv(
        "output/polka_et_al_pmc_mapped_subset.tsv", sep="\t"
    )

print(pmcid_mapped_papers_df.shape)
pmcid_mapped_papers_df.head()
# -

# # Download Files from PMCOA's FTP server

pmc_open_access_url = "ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"
response = request.urlopen(f"{pmc_open_access_url}")
files = response.read().decode("utf-8").splitlines()
tar_files = [f.split(" ")[-1] for f in files]
pmcid_list = pmcid_mapped_papers_df.PMCID.tolist()

if not any(Path("output/pmcoa_xml_files").iterdir()):
    # Cycle through each tar file on the server
    for tar_file in tqdm.tqdm(tar_files):

        # If not xml files skip
        if all(suffix != ".xml" for suffix in Path(tar_file).suffixes):
            continue

        # If temp file skip
        if Path(tar_file).suffix == ".tmp":
            continue

        # Grab the file from the tarfile
        print(f"Requesting {pmc_open_access_url}{tar_file}....")
        requested_file_stream = request.urlopen(f"{pmc_open_access_url}{tar_file}")
        open_stream = tarfile.open(fileobj=requested_file_stream, mode="r:gz")

        while True:
            try:
                pmc_paper = open_stream.next()

                if pmc_paper is None:
                    break

                if pmc_paper.isdir():
                    continue

                paper_pathlib = Path("output/pmcoa_xml_files") / Path(pmc_paper.name)
                if paper_pathlib.stem in pmcid_list:

                    new_paper = open_stream.extractfile(pmc_paper)
                    paper_pathlib.parent.mkdir(exist_ok=True)

                    with open(f"{str(paper_pathlib)}", "wb") as outfile:
                        outfile.write(new_paper.read())
            except tarfile.ReadError:
                print(f"There is an error in {requested_file_stream}.")
                break

        open_stream.close()

