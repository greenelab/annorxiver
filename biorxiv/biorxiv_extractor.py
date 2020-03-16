import os
from pathlib import Path
import re
import subprocess
import tqdm

import pandas as pd


files = (
    list(Path("Back_Content").rglob("*.meca"))
    +
    list(Path("Current_Content").rglob("*.meca"))
)

doc_file_hash_mapper = []
already_seen = set()

for file_name in tqdm.tqdm(files):
    doc_hash = file_name.name

    result = (
        subprocess.Popen(
            f"unzip -l {file_name}",
            shell=True, stdout=subprocess.PIPE
        )
        .communicate()
    )

    match = re.search(r'content/([\d]+)\.xml', str(result[0]))
    content_file_name = match.group(1)
    version = 1
    updated_file_name = f"{content_file_name}_v{version}"
    
    while updated_file_name in already_seen:
        version += 1
        updated_file_name = f"{content_file_name}_v{version}"

    already_seen.add(updated_file_name)

    if match is None:
        print(f"{file_name} did not match the file pattern [\d]+")
        continue

    doc_file_hash_mapper.append(
        {
            "hash": str(file_name),
            "doc_number": f"{updated_file_name}.xml"
        }
    )

    result = (
        subprocess
        .Popen(
            f"unzip -jo {file_name} content/{content_file_name}.xml -d biorxiv_articles/.", 
            shell=True, stdout=subprocess.PIPE
        )
        .communicate()
    )

    rename_result = (
        subprocess
        .Popen(
            f"mv biorxiv_articles/{content_file_name}.xml biorxiv_articles/{updated_file_name}.xml",
            shell=True, stdout=subprocess.PIPE
        )
        .communicate()
    )

(
    pd.DataFrame
    .from_records(doc_file_hash_mapper)
    .to_csv("biorxiv_doc_hash_mapper.tsv", sep="\t", index=False)
)
