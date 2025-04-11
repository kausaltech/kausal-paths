from __future__ import annotations

import os
import re
import sys
from typing import Any

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


def validate_required_columns(df: pl.DataFrame) -> None:
    """Validate that required columns exist in the dataframe."""
    required_columns = ['Quantity', 'Unit']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    if not any(col in df.columns for col in ['MetricGroup', 'Metric', 'Sector']):
        raise ValueError("Missing metric group column. Dataset must contain either 'MetricGroup', 'Metric', or 'Sector' column.")


def determine_metric_column(df: pl.DataFrame) -> str:
    """Determine which column to use for metrics."""
    if 'MetricGroup' in df.columns:
        return 'MetricGroup'
    if 'Metric' in df.columns:
        return 'Metric'  # backward compatibility
    if 'Sector' in df.columns:
        return 'Sector'  # legacy support
    raise ValueError("No metric group column found. DataFrame must contain 'MetricGroup', 'Metric', or 'Sector' column.")


def create_compound_identifiers(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    """Create compound identifiers from metric group and quantity."""
    if len(df) == len(df['Quantity'].unique()):
        df = df.with_columns(
            pl.col('Quantity').map_elements(to_snake_case, return_dtype=pl.Utf8).alias('CompoundID')
        )
    else:
        df = df.with_columns([
            (
                pl.concat_str([
                    pl.col(metric_col).map_elements(to_snake_case, return_dtype=pl.Utf8),
                    pl.lit('_'),
                    pl.when(pl.col('Quantity').is_null())
                    .then(pl.lit('value'))
                    .otherwise(pl.col('Quantity').map_elements(to_snake_case, return_dtype=pl.Utf8))
                ])
            ).alias('CompoundID')
        ])
    return df


def split_by_slice(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split the dataframe into separate dataframes by Slice."""
    slices = {}

    if 'Slice' in df.columns:
        unique_slices = df.select('Slice').unique().to_series(0).to_list()
    else:
        raise ValueError("If specific_slice is not defined, there must be column Slice.")

    for slice_name in unique_slices:
        if slice_name is not None:
            slice_df = df.filter(pl.col('Slice') == slice_name).drop('Slice')
            slices[slice_name] = slice_df
            print(f"Created dataset for slice: {slice_name} with {len(slice_df)} rows")

    return slices


def extract_units(df: pl.DataFrame, slice_name: str) -> dict:
    """Extract units from the dataframe using compound identifiers."""
    units = {}

    # Use the already created compound IDs
    unique_metrics = df.select(['CompoundID', 'Unit']).unique()

    for row_idx in range(len(unique_metrics)):
        row = unique_metrics.row(row_idx, named=True)
        compound_id = row['CompoundID']
        unit = row['Unit']

        if compound_id and unit:
            units[compound_id] = unit

    return units


def extract_description(df: pl.DataFrame, slice_name: str, metric_col: str) -> str | None:
    """Extract description from the dataframe."""
    description = None # FIXME Does not show up in admin UI.
    if 'Description' in df.columns:
        descriptions = []
        rows_with_description = df.filter(pl.col('Description').is_not_null())

        for row_idx in range(len(rows_with_description)):
            row = rows_with_description.row(row_idx, named=True)

            parts = []
            if row.get(metric_col):
                parts.append(row[metric_col])

            # Add values from dimension columns
            for col in df.columns:
                if (
                    (df[col].dtype == pl.Utf8 or df[col].dtype == pl.String) and
                    (col not in [metric_col, 'Unit', 'Value', 'Description', 'Quantity', 'Year', 'CompoundID']) and
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


def extract_metrics(df: pl.DataFrame, language: str, metric_col: str) -> list:
    """
    Extract metrics from the dataframe using compound identifiers.

    The metric_col parameter should be 'MetricGroup', 'Metric', or 'Sector',
    in order of preference.
    """
    metrics = []

    # Use the already created compound IDs
    unique_metrics = df.select([metric_col, 'CompoundID', 'Quantity']).unique()

    for row_idx in range(len(unique_metrics)):
        row = unique_metrics.row(row_idx, named=True)
        metric_name = row[metric_col]
        compound_id = row['CompoundID']
        quantity = row['Quantity']

        if metric_name and quantity and compound_id:
            metrics.append({
                "id": compound_id,
                "quantity": to_snake_case(quantity),
                "label": {language: f"{metric_name} ({quantity})"},
            })

    return metrics


def clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Remove metadata columns and empty columns."""
    # 1. Drop metadata columns that are now in metadata but keep CompoundID
    metadata_columns = ['Unit', 'Description']
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


def pivot_by_compound_id(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    """Convert dataframe to have compound identifiers as columns."""
    # Get unique compound IDs
    unique_ids = df.select('CompoundID').unique().to_series(0).to_list()
    unique_ids = [s for s in unique_ids if s is not None]
    # If only one compound ID, just rename Value column to that ID
    if len(unique_ids) == 1:
        compound_id = unique_ids[0]
        return df.with_columns(
            pl.col('Value').alias(compound_id)
        ).drop([metric_col, 'Quantity', 'Value', 'CompoundID'])

    # Get dimension columns (all except those used for pivoting and the value)
    dim_cols = [col for col in df.columns if col not in [metric_col, 'Quantity', 'Value', 'CompoundID']]

    # Ensure no null compound IDs
    df = df.with_columns(
        pl.when(pl.col('CompoundID').is_null())
          .then(pl.lit("unknown"))
          .otherwise(pl.col('CompoundID'))
          .alias('CompoundID')
    )


    # Pivot to have compound IDs as columns
    try:
        df = df.with_columns(pl.col('Value').cast(pl.Float64))
    finally:
        pass
    result_df = df.pivot(
        values="Value",
        index=dim_cols,
        on="CompoundID"
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
    metrics = list(units.keys())
    new_columns = [col if col in metrics + ['Year'] else to_snake_case(col) for col in columns]

    # Rename columns
    df = df.rename(dict(zip(columns, new_columns, strict=False)))

    # Convert string values in columns to snake case
    for col in new_columns:
        if col not in metrics + ['Year']:
            df = df.with_columns(
                pl.col(col).map_elements(to_snake_case, return_dtype=pl.Utf8).alias(col)
            )

    return df


def save_to_csv(df: pl.DataFrame, file_path: str, slice_name: str) -> None:
    """Save dataframe to CSV if a path is provided."""
    if file_path.upper() not in ['N', 'NONE']:
        # Create a unique filename for each slice
        file_name, file_ext = os.path.splitext(file_path)  # noqa: PTH122
        slice_file_path = f"{file_name}_{to_snake_case(slice_name)}{file_ext}"

        df.write_csv(slice_file_path)
        print(df)
        print(f'Data saved to {slice_file_path}')


def push_to_dvc(df: pl.DataFrame, output_path: str, slice_name: str,
                units: dict, description: str | None, metrics: list,
                language: str) -> None:
    """Push dataset to DVC repository."""
    if output_path.upper() in ['N', 'NONE']:
        return

    # Get index columns (excluding metric value columns)
    index_columns = [col for col in df.columns if col not in units.keys()]

    # Build metadata
    metadata: dict[str, Any] = {
        'name': {language: slice_name},
        'identifier': to_snake_case(slice_name),
    }
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

    # 1. Validate required columns
    validate_required_columns(df)

    # 2. Determine which column to use for metrics
    metric_col = determine_metric_column(df)
    print(f"Using '{metric_col}' as the metric group column")

    # 3. Create compound identifiers early
    df = create_compound_identifiers(df, metric_col)

    # 4. Extract metadata using compound identifiers
    units = extract_units(df, slice_name)
    metrics = extract_metrics(df, language, metric_col)
    description = extract_description(df, slice_name, metric_col)
    print(f"Units: {units}")
    print(f"Metrics: {len(metrics)} entries")
    if description:
        print("Description extracted")

    # 5. Clean dataframe (remove metadata columns)
    df = clean_dataframe(df)

    # 6. Convert to standard format with Year column if needed
    df = convert_to_standard_format(df)
    print(f"Data converted to standard format with {len(df)} rows")

    # 7. Pivot by compound ID
    df = pivot_by_compound_id(df, metric_col)
    dim_ids = [s for s in df.columns if s not in units.keys()]
    print(f"Data pivoted by compound identifiers with dimension columns: {dim_ids}")

    # 8. Check for issues
    if check_for_duplicates(df):
        print("Warning: Dataframe contains duplicate rows")

    # 9. Prepare for DVC (standardize column names)
    df = prepare_for_dvc(df, units)

    # 10. Save to CSV if requested
    save_to_csv(df, outcsvpath, slice_name)

    # 11. Push to DVC if requested
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
        slice_df = full_df.filter(pl.col('Slice') == specific_slice).drop('Slice') if 'Slice' in full_df.columns else full_df
        process_slice(slice_df, specific_slice, outcsvpath, outdvcpath, language)
    else:
        # Process all slices
        slice_dfs = split_by_slice(full_df)
        print(f"Found {len(slice_dfs)} slices to process")

        for slice_name, slice_df in slice_dfs.items():
            process_slice(slice_df, slice_name, outcsvpath, outdvcpath, language)


if __name__ == "__main__":
    main()
