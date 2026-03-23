# Code to push free-format csv datasets to dvc_pandas repository
from __future__ import annotations

import os
import sys

import polars as pl
from dotenv import load_dotenv
from dvc_pandas import Dataset, DatasetMeta, Repository, RepositoryCredentials

load_dotenv()

dataset_id = sys.argv[1]

settings = {
    'pks_rakennukset_lammitys': {
        'identifier': 'helsinki/aluesarjat/02um_rakennukset_lammitys',
        'metadata': {
            'url_description': 'https://stat.hel.fi/pxweb/fi/Aluesarjat/Aluesarjat__asu__rakan__umkun/02UM_Rakennukset_lammitys.px/table/tableViewLayout1/',
            'url_download': 'https://stat.hel.fi:443/sq/9ee19d39-17d4-439c-8ae0-0f227abbe326',
            'download_type': 'pxweb',
            'local_path': '/Users/jouni/Downloads/02UM_20241206-130609.csv',
            'skip_rows': 2,
            'encoding': 'iso-8859-1',
            'quote_char': '"',
            'value_column': 'Rakennukset',
            'previous_version': {
                'path': 'https://s3.kausal.tech/datasets/',
                'etag': 'a5aab9b6851f1faaff9bfa7dce89c7e9',  # (from dvctest repo)
                'key': 'a5/aab9b6851f1faaff9bfa7dce89c7e9',
            },
            'column_types': {
                'Alue': 'Categorical',
                'Rakennuksen käyttötarkoitus': 'Categorical',
                'Rakennuksen lämmitystapa': 'Categorical',
                'Rakennuksen lämmitysaine': 'Categorical',
                'Tiedot': 'Categorical',
                'Vuosi': 'String',
                'Value': 'Float64',
            },
        },
    },
}

dataset_settings = settings[dataset_id]
df = pl.read_csv(
    str(dataset_settings['metadata']['local_path']),
    skip_rows=int(dataset_settings['metadata']['skip_rows']),
    encoding=str(dataset_settings['metadata']['encoding']),
    quote_char=str(dataset_settings['metadata']['quote_char']),
)

metas = DatasetMeta(identifier=str(dataset_settings['identifier']), metadata=dict(dataset_settings['metadata']))

creds = RepositoryCredentials(
    git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
    git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
    git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
    git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
)
# Store edited dataset with proper DatasetMeta object
ds = Dataset(df, meta=metas)
repo = Repository(
    repo_url='https://github.com/kausaltech/dvctest.git',
    dvc_remote='kausal-s3',
    repo_credentials=creds,
)
repo.push_dataset(ds)

# ------------------ Possibly useful code ------------------

# def download_pxweb_data(url): # I haven't checked if this actually works.
#     """
#     Download data from a PX-Web API endpoint and return as a Polars DataFrame.

#     Args:
#         url: URL to the PX-Web API endpoint

#     Returns:
#         Polars DataFrame containing the downloaded data

#     """
#     import json

#     import polars as pl
#     import requests
#     from pyjstat import pyjstat

#     # Get metadata about the dataset
#     meta_resp = requests.get(url)
#     meta = json.loads(meta_resp.text)

#     # Construct query to get all data
#     query = {
#         "query": [],
#         "response": {
#             "format": "json-stat2"
#         }
#     }

#     # Add selection for each dimension
#     for dim in meta["variables"]:
#         query["query"].append({
#             "code": dim["code"],
#             "selection": {
#                 "filter": "all",
#                 "values": ["*"]
#             }
#         })

#     # Get the actual data
#     data_resp = requests.post(url, json=query)

#     # Convert JSON-stat to pandas then to polars
#     df_pd = pyjstat.from_json_stat(data_resp.json())[0]
#     return pl.from_pandas(df_pd)

# import pandas as pd
# import settings
# from utils.dvc import load_datasets
# df = load_datasets('syke/alas_emissions')
# settings.DVC_PANDAS_REPOSITORY
