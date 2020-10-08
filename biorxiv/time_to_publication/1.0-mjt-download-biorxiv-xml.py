#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
from zipfile import ZipFile, BadZipFile
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import boto3


# In[2]:


bucket_url = "biorxiv-src-monthly"


# In[3]:


get_ipython().run_cell_magic('time', '', 'client = boto3.client("s3")\n\nresult = client.list_objects_v2(Bucket=bucket_url, RequestPayer="requester")\nkey_list = [r["Key"] for r in result["Contents"]]\n\nwhile result["IsTruncated"]:\n    result = client.list_objects_v2(\n        Bucket=bucket_url,\n        RequestPayer="requester",\n        ContinuationToken=result["NextContinuationToken"],\n    )\n    key_list += [r["Key"] for r in result["Contents"]]\n\nprint(len(key_list))')


# In[4]:


def get_xml(key):
    p = Path(key)
    if p.suffix == ".meca":
        target_folder = p.with_suffix("")
        if not target_folder.exists():
            target_folder.mkdir(exist_ok=True, parents=True)
            client = boto3.client("s3")
            client.download_file(
                bucket_url, key, key, ExtraArgs={"RequestPayer": "requester"}
            )
            try:
                with ZipFile(key, "r") as zipObj:
                    for sub_file in zipObj.namelist():
                        if sub_file.endswith(".xml"):
                            zipObj.extract(sub_file, target_folder)
            except BadZipFile:
                pass
            p.unlink()


# In[5]:


get_ipython().run_cell_magic('time', '', 'PARALLEL = True\nif PARALLEL:\n    Parallel(n_jobs=32)(delayed(get_xml)(key) for key in tqdm(key_list))\nelse:\n    [get_xml(key) for key in tqdm(key_list)]')


# In[ ]:




