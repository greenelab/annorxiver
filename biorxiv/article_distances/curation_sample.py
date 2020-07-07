import pandas as pd
import numpy as np

data_df = (
    pd.read_csv(
        "output/annotated_links/potential_biorxiv_pmc_links.tsv", 
        sep="\t"
    )
)

sampled_df = (
    data_df
    .groupby("distance_bin")
    .apply(lambda x: x.sample(50, random_state=100))
)

final_df = (
    sampled_df
    [["biorxiv_doi_url", "pmcid_url"]]
    .assign(is_same_paper=np.NaN)
    .sample(frac=1, random_state=100)
    .to_csv(
        "output/annotated_links/biorxiv_pmc_links_curation.tsv", 
        sep="\t", index=False
    )
)
