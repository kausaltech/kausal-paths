from __future__ import annotations

import os
import re
import sys

import dvc_pandas
import polars as pl
from dotenv import load_dotenv
from dvc_pandas import Dataset, DatasetMeta, Repository


def to_snake_case(string):
    """Convert a string to snake_case."""
    if not isinstance(string, str):
        return string

    s = string.replace('-', ' ').replace('ä', 'a').replace('ö', 'o').lower().replace(' ', '_')
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s)

    return s


def load_data(file_path: str, separator: str) -> pl.DataFrame:
    """Load CSV data from file."""
    return pl.read_csv(file_path, separator=separator, infer_schema_length=1000)


def split_by_slice(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split the dataframe into separate dataframes by Slice."""
    slices = {}

    unique_slices = df.select('Slice').unique().to_series(0).to_list()

    for slice_name in unique_slices:
        if slice_name is not None:
            slice_df = df.filter(pl.col('Slice') == slice_name).drop('Slice')
            slices[slice_name] = slice_df
            print(f"Created dataset for slice: {slice_name} with {len(slice_df)} rows")

    return slices


def extract_units(df: pl.DataFrame, slice_name: str) -> dict:
    """Extract units from the dataframe."""
    units = {}
    if 'Sector' in df.columns and 'Unit' in df.columns:
        # Get unique sector-unit pairs
        unique_sectors = df.select(['Sector', 'Unit']).unique()

        for row_idx in range(len(unique_sectors)):
            row = unique_sectors.row(row_idx, named=True)
            sector = row['Sector']
            unit = row['Unit']

            if sector and unit:
                units[sector] = unit

    # If no Sector column or no valid sector-unit pairs, use "Value" as the key
    if not units and 'Unit' in df.columns:
        first_unit = df.select('Unit')[0, 0]
        if first_unit:
            units['Value'] = first_unit
    return units


def extract_description(df: pl.DataFrame, slice_name: str) -> str:
    """Extract description from the dataframe."""
    description = None # FIXME Does not show up in admin UI.
    if 'Description' in df.columns:
        descriptions = []
        rows_with_description = df.filter(pl.col('Description').is_not_null())

        for row_idx in range(len(rows_with_description)):
            row = rows_with_description.row(row_idx, named=True)

            parts = []
            if row.get('Sector'):
                parts.append(row['Sector'])

            # Add values from dimension columns
            for col in df.columns:
                if (
                    (df[col].dtype == pl.Utf8 or df[col].dtype == pl.String) and
                    (col not in ['Sector', 'Unit', 'Value', 'Description', 'Quantity', 'Year']) and
                    (row.get(col))
                ):
                    parts.append(f"{col}: {row[col]}")  # noqa: PERF401

            if row['Description']:
                parts.append(row['Description'])

            if parts:
                descriptions.append(" - ".join(parts))

        if descriptions:
            description = "<br/>".join(descriptions)
    return description


def extract_metrics(df: pl.DataFrame, slice_name: str) -> list:
    """Extract metrics from the dataframe."""
    # 3. Extract metrics from Quantity and Sector columns
    metrics = []
    if 'Sector' in df.columns and 'Quantity' in df.columns:
        unique_metrics = df.select(['Sector', 'Quantity']).unique()

        for row_idx in range(len(unique_metrics)):
            row = unique_metrics.row(row_idx, named=True)
            sector = row['Sector']
            quantity = row['Quantity']

            if sector and quantity:
                metrics.append({
                    "id": to_snake_case(sector),
                    "quantity": quantity, # FIXME Does not enter the database properly
                    "label": sector
                })

    return metrics


def clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Remove metadata columns and empty columns."""
    # 1. Drop metadata columns that are now in metadata
    metadata_columns = ['Unit', 'Description', 'Quantity']
    for col in metadata_columns:
        if col in df.columns:
            df = df.drop(col)

    # 2. Drop any columns that only contain null values
    for col in df.columns:
        if df.select(col).unique().to_series(0).to_list() == [None]:
            df = df.drop(col)

    return df


def convert_to_standard_format(df: pl.DataFrame) -> pl.DataFrame:
    """Convert dataframe to standard format with Year column if needed."""
    # Check if already in standard format
    if 'Year' in df.columns:
        return df

    # Identify year columns (columns with numeric names)
    year_columns = [col for col in df.columns if col.isdigit()]

    if not year_columns:
        raise ValueError("No year columns found and no Year column exists")

    # Get non-year columns
    context_columns = [col for col in df.columns if not col.isdigit()]

    # Initialize result dataframe
    result_df = df.head(1).select(context_columns).with_columns([
        (pl.lit(0).alias('Year').cast(pl.Int64)),
        (pl.lit('0.0').alias('Value').cast(pl.String))
    ]).clear()

    # For each row and year, create a new row in the result
    df = df.with_row_index(name='__row_idx')

    for i in range(len(df)):
        row = df.filter(pl.col('__row_idx') == i)

        for year in year_columns:
            value = row.select(year)[0, 0]

            # Skip empty values
            if value is None or value in ['.', '-']:
                continue

            # Create new row with context values and year/value
            new_row = row.select(context_columns).with_columns([
                pl.lit(int(year)).cast(pl.Int64).alias('Year'),
                pl.lit(str(value)).alias('Value')
            ])

            result_df = pl.concat([result_df, new_row], rechunk=False)

    return result_df.drop('__row_idx') if '__row_idx' in result_df.columns else result_df


def pivot_by_sector(df: pl.DataFrame) -> pl.DataFrame:
    """Convert dataframe to have sectors as columns."""
    # Check if we need to pivot (do we have a Sector column?)
    if 'Sector' not in df.columns:
        return df

    # Get unique sectors
    unique_sectors = df.select('Sector').unique().to_series(0).to_list()
    unique_sectors = [s for s in unique_sectors if s is not None]

    # If only one sector, just rename Value column to that sector
    if len(unique_sectors) == 1:
        sector = unique_sectors[0]
        return df.with_columns(
            pl.col('Value').alias(sector)
        ).drop(['Sector', 'Value'])

    # Get dimension columns (all except Sector and Value)
    dim_cols = [col for col in df.columns if col not in ['Sector', 'Value']]

    # Ensure no null sectors
    df = df.with_columns(
        pl.when(pl.col('Sector').is_null())
          .then(pl.lit("unknown"))
          .otherwise(pl.col('Sector'))
          .alias('Sector')
    )

    # Pivot to have sectors as columns
    result_df = df.pivot(
        values="Value",
        index=dim_cols,
        on="Sector"
    )

    return result_df


def check_for_duplicates(df: pl.DataFrame) -> bool:
    """Check if the dataframe has any duplicate rows by all columns."""
    if len(df) == df.unique().shape[0]:
        return False  # No duplicates
    return True  # Has duplicates


def prepare_for_dvc(df: pl.DataFrame, units: dict) -> pl.DataFrame:
    """Prepare dataframe for DVC by standardizing column names."""
    # Standardize column names (except 'Year')
    columns = df.columns
    sectors = list(units.keys())
    new_columns = [col if col in sectors + ['Year'] else to_snake_case(col) for col in columns]

    # Rename columns
    df = df.rename(dict(zip(columns, new_columns, strict=False)))

    # Convert string values in columns to snake case
    for col in new_columns:
        if col not in sectors + ['Year']:
            df = df.with_columns(
                pl.col(col).map_elements(to_snake_case, return_dtype=str).alias(col)
            )

    return df


def save_to_csv(df: pl.DataFrame, file_path: str, slice_name: str) -> None:
    """Save dataframe to CSV if a path is provided."""
    if file_path.upper() not in ['N', 'NONE']:
        # Create a unique filename for each slice
        file_name, file_ext = os.path.splitext(file_path)  # noqa: PTH122
        slice_file_path = f"{file_name}_{to_snake_case(slice_name)}{file_ext}"

        df.write_csv(slice_file_path)
        print(f'Data saved to {slice_file_path}')


def push_to_dvc(df: pl.DataFrame, output_path: str, slice_name: str,
                units: dict, description: str | None, metrics: list,
                language: str) -> None:
    """Push dataset to DVC repository."""
    if output_path.upper() in ['N', 'NONE']:
        return

    # Get index columns (excluding sector value columns)
    index_columns = [col for col in df.columns if col not in units.keys()]

    # Build metadata
    metadata = {'name': {language: slice_name}}
    if description:
        metadata['description'] = {language: description}
    if metrics:
        metadata['metrics'] = metrics

    # Create dataset metadata
    meta = DatasetMeta(
        identifier=output_path,
        index_columns=index_columns,
        units=units,
        metadata=metadata
    )

    # Create dataset
    ds = Dataset(df, meta=meta)

    # Set up credentials
    creds = dvc_pandas.RepositoryCredentials(
        git_username=os.getenv('DVC_PANDAS_GIT_USERNAME'),
        git_token=os.getenv('DVC_PANDAS_GIT_TOKEN'),
        git_ssh_public_key_file=os.getenv('DVC_SSH_PUBLIC_KEY_FILE'),
        git_ssh_private_key_file=os.getenv('DVC_SSH_PRIVATE_KEY_FILE'),
    )

    # Initialize repository
    repo = Repository(
        repo_url='https://github.com/kausaltech/dvctest.git',
        dvc_remote='kausal-s3',
        repo_credentials=creds
    )

    # Add timestamp to force update
    import time
    ds.meta.metadata['updated_at'] = str(int(time.time()))

    repo.push_dataset(ds)
    print(f'Dataset pushed to DVC at {output_path}')


def process_slice(df: pl.DataFrame, slice_name: str, outcsvpath: str, outdvcpath: str, language: str) -> None:
    """Process a single slice of data."""
    print(f"\n==== Processing slice: {slice_name} ====")

    # 1. Extract metadata before manipulating dataframe
    units = extract_units(df, slice_name)
    metrics = extract_metrics(df, slice_name)
    description = extract_description(df, slice_name)
    print(f"Units: {units}")
    print(f"Metrics: {len(metrics)} entries")
    if description:
        print("Description extracted")

    # 2. Clean dataframe (remove metadata columns and empty columns)
    df = clean_dataframe(df)

    # 3. Convert to standard format with Year column if needed
    df = convert_to_standard_format(df)
    print(f"Data converted to standard format with {len(df)} rows")

    # 4. Pivot by sector to have sectors as columns
    df = pivot_by_sector(df)
    dim_ids = [s for s in df.columns if s not in units.keys()]
    print(f"Data pivoted by sector with dimension columns: {dim_ids}")

    # 5. Check for issues
    if check_for_duplicates(df):
        print("Warning: Dataframe contains duplicate rows")

    # 6. Prepare for DVC (standardize column names)
    df = prepare_for_dvc(df, units)

    # 7. Save to CSV if requested
    save_to_csv(df, outcsvpath, slice_name)

    # 8. Push to DVC if requested
    if outdvcpath.upper() not in ['N', 'NONE']:
        slice_dvc_path = f"{outdvcpath}/{to_snake_case(slice_name)}"
        push_to_dvc(df, slice_dvc_path, slice_name, units, description, metrics, language)


def main():
    """Process and convert data for all slices."""
    load_dotenv()

    # Get command line arguments
    incsvpath = sys.argv[1]
    incsvsep = sys.argv[2]
    outcsvpath = sys.argv[3]
    outdvcpath = sys.argv[4]
    language = sys.argv[5]
    specific_slice = sys.argv[6] if len(sys.argv) > 6 else None

    # Load data
    full_df = load_data(incsvpath, incsvsep)

    # Process slices
    if specific_slice:
        # Process only the specified slice
        print(f"Processing only slice: {specific_slice}")
        slice_df = full_df.filter(pl.col('Slice') == specific_slice).drop('Slice')
        process_slice(slice_df, specific_slice, outcsvpath, outdvcpath, language)
    else:
        # Process all slices
        slice_dfs = split_by_slice(full_df)
        print(f"Found {len(slice_dfs)} slices to process")

        for slice_name, slice_df in slice_dfs.items():
            process_slice(slice_df, slice_name, outcsvpath, outdvcpath, language)


if __name__ == "__main__":
    main()
