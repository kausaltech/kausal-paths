import pandas as pd
import os
from dvc_pandas import Dataset, Repository

os.chdir('/Users/jouni/devel/climate4cast')

dfmain = pd.read_csv('Potsdam-Parquet.csv')
pdindex = ['Sector', 'Scope', 'Quantity', 'Energiträger', 'GHG', 'Year']
outdvcpath = 'gpc/potsdam'

pdframe = dfmain.set_index(pdindex)
ds = Dataset(pdframe, identifier = outdvcpath)
repo = Repository(repo_url = 'git@github.com:kausaltech/dvctest.git', dvc_remote = 'kausal-s3')
repo.push_dataset(ds)
