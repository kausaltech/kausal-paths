from __future__ import annotations

import os
import sys

import dvc_pandas
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from dvc_pandas import DatasetMeta

load_dotenv()

incsvpath = sys.argv[1]
incsvsep = sys.argv[2]
outcsvpath = sys.argv[3]
outdvcpath = sys.argv[4]

unitreplace = [['tCO2e', 't'],
               ['p-km', 'pkm'], ['Mkm', 'Gm'],
               ['€', 'EUR']]

# ---------------------------------------------------------------------------------------
df = pl.read_csv(incsvpath, separator = incsvsep, infer_schema_length = 1000)

# Drop columns from 'droplist' if present, and empty columns.
droplist = ['Description']
for col in droplist:
    if col in df.columns:
        df = df.drop(col)

for col in df.columns:
    if df.select(col).unique().to_series(0).to_list() == [None]:
        df = df.drop(col)

# Find context, value, and dimension columns.
context = []
values = []
for c in df.columns:
    if c.isdigit():
        values.append(c)
    else:
        context.append(c)

dims = [c for c in context if c not in ['Quantity', 'Unit', 'UUID', 'Is_action']]

if 'UUID' in df.columns:
    groups = dims + ['UUID']
else:
    groups = dims

# Check input dataset for duplicated dimension sets.
duplicates = df.group_by(groups).agg(pl.len()).filter(pl.col('len') > 1)
if len(duplicates) > 0:
    print('There are duplicate values. Remove them and try again.')
    print(duplicates)
    exit()
else:
    print('No duplicates, continuing...')

# Check input dataset for rows containing null values in required columns.
missing_data = df.filter((pl.col('Sector') + pl.col('Quantity') + pl.col('Unit')).is_null())
if missing_data.is_empty():
    print('No missing data in columns Sector, Quantity, Unit. Continuing...')
else:
    print('Missing data in obligatory cells. Fill in and try again.')
    print(missing_data.select(['Sector', 'Quantity', 'Unit']))
    exit()

# Replace units as specified in the 'unitreplace' list.
unitcol = df.select('Unit').to_series(0).to_list()
for ur in unitreplace:
    unitcol = [x.replace(ur[0], ur[1]) for x in unitcol]
df = df.with_columns(pl.Series(name = 'Unit', values = unitcol))

# Replace scope numbers with labels. Detailed actions have no 'Scope' column.
if 'Scope' in df.columns:
    scopecol = df.select('Scope').to_series(0).to_list()
    labels = []
    for x in scopecol:
        if x:
            labels.append('Scope %i' % x)
        else:
            labels.append(x)
    df = df.with_columns(pl.Series(name = 'Scope', values = labels))

# ---------------------------------------------------------------------------------------
if 'Value' in df.columns and 'Year' in df.columns:  # If dataset is already in long format
    dfmain = df
    dims = [d for d in dims if d not in ['Year', 'Value']]
else:
    dfmain = df.head(1).select(context).with_columns([(pl.lit(0.0).alias('Value').cast(pl.Float64)),
                                                    (pl.lit(0).alias('Year').cast(pl.Int64))]).clear()

    df = df.with_row_index(name = 'Index')
    for i in range(len(df)):
        print('Row %i of %i' % ((i + 1), len(df)))
        for y in values:
            mcols = list(context)
            mcols.extend([y])

            mframe = df.filter(pl.col('Index') == i).select(mcols).with_columns(pl.lit(y).cast(pl.Int64))
            mframe.columns = dfmain.columns
            # Ignore empty cells and possible empty values from statfi.
            if mframe['Value'][0] is not None and mframe['Value'][0] not in ['.', '-']:
                mframe = mframe.with_columns(pl.col('Value').cast(pl.Float64))
                dfmain = pl.concat([dfmain, mframe], rechunk=False)
    dfmain = dfmain.rechunk()   # This helped speed a bit.

if 'Is_action' in dfmain.columns:
    dfmain = dfmain.with_columns([pl.when(
        (pl.col('Is_action')) &
        (pl.col('Year').eq(0))
    ).then(pl.lit(None)).otherwise(pl.col('UUID')).alias('UUID')])
    dfmain = dfmain.drop('Is_action')

if outcsvpath.upper() not in ['N', 'NONE']:
    dfmain.write_csv(outcsvpath)

if outdvcpath.upper() not in ['N', 'NONE']:
    from dvc_pandas import Dataset, DatasetMeta, Repository

    indexcols = list(dims)
    indexcols.extend(['Year'])
    if 'Quantity' in dfmain.columns:   # Detailed actions have no 'Quantity' column.
        indexcols.extend(['Quantity'])
    pdindex = pd.MultiIndex.from_frame(pd.DataFrame(dfmain.select(indexcols).fill_null('.'), columns = indexcols))

    valuecols = list(set(dfmain.columns) - set(indexcols))
    pdframe = pd.DataFrame(dfmain.select(valuecols), index = pdindex, columns = valuecols)

    pl_df = pl.from_pandas(pdframe.reset_index())
    meta = DatasetMeta(identifier=outdvcpath, index_columns=indexcols)
    ds = Dataset(pl_df, meta=meta)
    creds = dvc_pandas.RepositoryCredentials(
        git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
        git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
        git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
        git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
    )
    repo = Repository(repo_url='https://github.com/kausaltech/dvctest.git', dvc_remote='kausal-s3', repo_credentials=creds)
    repo.push_dataset(ds)
