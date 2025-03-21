from __future__ import annotations

import os
import re
import sys

import dvc_pandas
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from dvc_pandas import Dataset, DatasetMeta, Repository


def to_snake_case(string):
    """Convert a string to snake_case."""
    if not isinstance(string, str):
        return string

    # Replace special characters and convert to lowercase
    s = string.replace('-', ' ').replace('ä', 'a').replace('ö', 'o').lower().replace(' ', '_')

    # Remove non-alphanumeric characters and normalize underscores
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s)

    return s


def load_data(file_path: str, separator: str, slice_name: str) -> pl.DataFrame:
    """Load CSV data and filter for specified slice."""
    df = pl.read_csv(file_path, separator=separator, infer_schema_length=1000)
    return df.filter(pl.col('Sector') == slice_name).drop(['Slice'])


def clean_dataframe(df: pl.DataFrame, drop_columns: list[str]) -> pl.DataFrame:
    """Remove specified columns and empty columns."""
    # Drop columns from droplist if present
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(col)

    # Drop empty columns
    for col in df.columns:
        if df.select(col).unique().to_series(0).to_list() == [None]:
            df = df.drop(col)

    return df


def identify_column_types(df: pl.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Identify context, value, and dimension columns."""
    context = []
    values = []

    for c in df.columns:
        if c.isdigit():
            values.append(c)
        else:
            context.append(c)

    dims = [c for c in context if c not in ['Quantity', 'Unit', 'UUID', 'Is_action']]

    return context, values, dims


def check_duplicates(df: pl.DataFrame, groups: list[str]) -> bool:
    """Check for duplicated dimension sets."""
    duplicates = df.group_by(groups).agg(pl.len()).filter(pl.col('len') > 1)
    if len(duplicates) > 0:
        print('There are duplicate values. Remove them and try again.')
        print(duplicates)
        return False
    print('No duplicates, continuing...')
    return True


def check_missing_data(df: pl.DataFrame) -> bool:
    """Check for missing data in required columns."""
    missing_data = df.filter((pl.col('Sector') + pl.col('Quantity') + pl.col('Unit')).is_null())
    if missing_data.is_empty():
        print('No missing data in columns Sector, Quantity, Unit. Continuing...')
        return True
    print('Missing data in obligatory cells. Fill in and try again.')
    print(missing_data.select(['Sector', 'Quantity', 'Unit']))
    return False


def standardize_units(df: pl.DataFrame, unit_replacements: list[list[str]]) -> pl.DataFrame:
    """Replace units according to standardization rules."""
    unitcol = df.select('Unit').to_series(0).to_list()
    for ur in unit_replacements:
        unitcol = [x.replace(ur[0], ur[1]) for x in unitcol]
    return df.with_columns(pl.Series(name='Unit', values=unitcol))


def standardize_scopes(df: pl.DataFrame) -> pl.DataFrame:
    """Replace scope numbers with labels."""
    if 'Scope' in df.columns:
        scopecol = df.select('Scope').to_series(0).to_list()
        labels = []
        for x in scopecol:
            if x:
                labels.append(f'Scope {x}')
            else:
                labels.append(x)
        return df.with_columns(pl.Series(name='Scope', values=labels))
    return df


def transform_to_long_format(df: pl.DataFrame, context: list[str], values: list[str]) -> pl.DataFrame:
    """Transform data from wide to long format if needed."""
    if 'Value' in df.columns and 'Year' in df.columns:
        # Already in long format
        return df
    # Initialize empty dataframe for long format
    dfmain = df.head(1).select(context).with_columns([
        (pl.lit('0.0').alias('Value').cast(pl.String)),
        (pl.lit(0).alias('Year').cast(pl.Int64))
    ]).clear()

    # Convert numeric columns to strings for probabilistic data
    df = df.with_columns([pl.col(col).cast(pl.String) for col in values])
    df = df.with_row_index(name='Index')

    # Process each row and value column
    for i in range(len(df)):
        print(f'Row {i+1} of {len(df)}')
        for y in values:
            mcols = list(context)
            mcols.extend([y])

            mframe = df.filter(pl.col('Index') == i).select(mcols).with_columns(pl.lit(y).cast(pl.Int64))
            mframe.columns = dfmain.columns

            # Ignore empty cells and statfi empty values
            if mframe['Value'][0] is not None and mframe['Value'][0] not in ['.', '-']:
                dfmain = pl.concat([dfmain, mframe], rechunk=False)

    return dfmain.rechunk()


def process_action_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Process Is_action flags if present."""
    if 'Is_action' in df.columns:
        df = df.with_columns([pl.when(
            (pl.col('Is_action')) &
            (pl.col('Year').eq(0))
        ).then(pl.lit(None)).otherwise(pl.col('UUID')).alias('UUID')])
        return df.drop('Is_action')
    return df


def convert_value_types(df: pl.DataFrame) -> pl.DataFrame:
    """Try to convert Value column to Float64."""
    try:
        df_test = df.with_columns(pl.col('Value').cast(pl.Float64))
        print('Values are stored as Float64.')
        return df_test  # noqa: TRY300
    except pl.exceptions.InvalidOperationError as e:
        print('Value column contains probabilistic values and is stored as String.')
        print(f'Conversion details: {e}')
        return df


def save_to_csv(df: pl.DataFrame, file_path: str) -> None:
    """Save data to CSV if requested."""
    if file_path.upper() not in ['N', 'NONE']:
        df.write_csv(file_path)
        print(f'Data saved to {file_path}')


def standardize_column_names(df: pl.DataFrame, index_cols: list[str]) -> tuple[pl.DataFrame, list[str]]:
    """Standardize column names to snake_case."""
    new_cols = [col if col == 'Year' else to_snake_case(col) for col in index_cols]
    df = df.rename(dict(zip(index_cols, new_cols, strict=False)))

    for col in new_cols:
        if df[col].dtype == pl.Utf8 or df[col].dtype == pl.String:
            df = df.with_columns(pl.col(col).map_elements(to_snake_case, return_dtype=str))

    return df, new_cols


def push_to_dvc(df: pl.DataFrame, output_path: str, slice_name: str, units: dict[str, str],
                index_cols: list[str], new_cols: list[str]) -> None:
    """Push dataset to DVC repository."""
    meta = DatasetMeta(
        identifier=output_path,
        index_columns=new_cols,
        units=units,
        metadata={
            'name': {'fi': slice_name},
        }
    )

    ds = Dataset(df, meta=meta)

    creds = dvc_pandas.RepositoryCredentials(
        git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
        git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
        git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
        git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
    )

    repo = Repository(
        repo_url='https://github.com/kausaltech/dvctest.git',
        dvc_remote='kausal-s3',
        repo_credentials=creds
    )

    repo.push_dataset(ds)
    print(f'Dataset pushed to DVC at {output_path}')


def prepare_dvc_data(df: pl.DataFrame, dims: list[str]) -> tuple[pd.MultiIndex, list[str], list[str]]:
    """Prepare data structure for DVC upload."""
    indexcols = list(dims)
    indexcols.extend(['Year'])

    if 'Quantity' in df.columns:
        indexcols.extend(['Quantity'])

    pdindex = pd.MultiIndex.from_frame(pd.DataFrame(df.select(indexcols).fill_null('.'),
                                                   columns=indexcols))

    valuecols = list(set(df.columns) - set(indexcols))
    pdframe = pd.DataFrame(df.select(valuecols), index=pdindex, columns=valuecols)

    pl_df = pl.from_pandas(pdframe.reset_index())

    # Remove columns that shouldn't be in the final dataset
    for col in ['Unit', 'Sector', 'Quantity']:
        if col in pl_df.columns:
            pl_df = pl_df.drop(col)

    # Keep only existing index columns
    indexcols = [c for c in indexcols if c in pl_df.columns]

    return pl_df, indexcols


def main():
    """Process and convert data."""
    load_dotenv()

    # Get command line arguments
    incsvpath = sys.argv[1]
    incsvsep = sys.argv[2]
    outcsvpath = sys.argv[3]
    outdvcpath = sys.argv[4]
    slicename = sys.argv[5]

    # Unit standardization rules
    unitreplace = [
        ['tCO2e', 't'],
        ['p-km', 'pkm'],
        ['Mkm', 'Gm'],
        ['€', 'EUR']
    ]

    # Step 1: Load data
    df = load_data(incsvpath, incsvsep, slicename)

    # Step 2: Get unit for metadata
    units = {'Value': df['Unit'][0]}

    # Step 3: Clean dataframe
    df = clean_dataframe(df, drop_columns=['Description'])

    # Step 4: Identify column types
    context, values, dims = identify_column_types(df)

    # Determine grouping columns
    if 'UUID' in df.columns:
        groups = dims + ['UUID']
    else:
        groups = dims

    # Step 5: Validate data
    if not check_duplicates(df, groups):
        return

    if not check_missing_data(df):
        return

    # Step 6: Standardize values
    df = standardize_units(df, unitreplace)
    df = standardize_scopes(df)

    # Step 7: Transform to long format if needed
    dfmain = transform_to_long_format(df, context, values)

    # Update dimensions list to exclude Year and Value if data was already in long format
    if 'Value' in df.columns and 'Year' in df.columns:
        dims = [d for d in dims if d not in ['Year', 'Value']]

    # Step 8: Process action flags
    dfmain = process_action_flags(dfmain)

    # Step 9: Convert value types if possible
    dfmain = convert_value_types(dfmain)

    # Step 10: Save to CSV if requested
    save_to_csv(dfmain, outcsvpath)

    # Step 11: Prepare and push to DVC if requested
    if outdvcpath.upper() not in ['N', 'NONE']:
        # Format outdvcpath with snake_case slice name
        outdvcpath = f"{outdvcpath}/{to_snake_case(slicename)}"
        print(f"DVC path: {outdvcpath}")

        # Prepare data for DVC
        pl_df, indexcols = prepare_dvc_data(dfmain, dims)

        # Standardize column names
        pl_df, new_cols = standardize_column_names(pl_df, indexcols)

        print(pl_df)

        # Push to DVC
        push_to_dvc(pl_df, outdvcpath, slicename, units, indexcols, new_cols)


if __name__ == "__main__":
    main()
