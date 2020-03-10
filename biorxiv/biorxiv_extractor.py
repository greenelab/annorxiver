import os
from pathlib import Path
import re
import subprocess
import tqdm

import pandas as pd


files = list(Path("Back_Content").rglob("*.meca")) + list(Path("Current_Content").rglob("*.meca"))
doc_file_hash_mapper = []

for file_name in tqdm.tqdm(files):
	doc_hash = file_name.name
	result = subprocess.Popen(f"unzip -l {file_name}", shell=True, stdout=subprocess.PIPE).communicate()
	match = re.search(r'content/([\d]+\.xml)',str(result[0]))
	
	if match is None:
		print(f"{file_name} did not match the file pattern [\d]+")
		continue
	
	doc_file_hash_mapper.append({"hash":str(file_name), "doc_number":match.group(1)})
	result = subprocess.Popen(f"unzip -jo {file_name} content/{match.group(1)} -d biorxiv_articles/.", shell=True, stdout=subprocess.PIPE).communicate()

(
	pd.DataFrame
	.from_records(doc_file_hash_mapper)
	.to_csv("biorxiv_doc_hash_mapper.tsv", sep="\t", index=False)
)