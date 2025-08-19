from __future__ import annotations

import argparse
import os
import re
import warnings

# from pathlib import Path
from typing import Any

# import chardet
import dvc_pandas
import polars as pl
from dotenv import load_dotenv
from dvc_pandas import Dataset, DatasetMeta, Repository
from use_ids import load_yaml_mappings, replace_labels_with_ids  # type: ignore


def to_snake_case(string: str) -> str:
    """Convert a string to snake_case."""
    if not isinstance(string, str):
        return string

    s = string.replace('-', ' ').replace('ä', 'a').replace('ö', 'o').lower().replace(' ', '_')
    s = re.sub(r'[^a-z0-9_]', '', s)
    s = re.sub(r'_+', '_', s)

    return s


def load_data(file_path: str, separator: str, encoding: str) -> pl.DataFrame:
    """Load CSV data from file."""
    # return pl.read_csv(file_path, separator=separator, infer_schema_length=1000)

    try:
        print(f"trying {encoding}")
        return pl.read_csv(file_path, separator=separator,
                            infer_schema_length=1000, encoding=encoding)
    except pl.exceptions.ComputeError as e:
        print(f"encooding {encoding} failed.")
        print(e)

    raise Exception(f"Could not read {file_path} with any encoding")

def validate_required_columns(df: pl.DataFrame) -> None:
    """Validate that required columns exist in the dataframe."""
    required_columns = ['Quantity', 'Unit']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    if not any(col in df.columns for col in ['Metric', 'Sector']):
        raise ValueError("Missing metric column. Dataset must contain either 'Metric', or 'Sector' column.")


def determine_metric_column(df: pl.DataFrame) -> str:
    """Determine which column to use for metrics."""
    if 'Metric' in df.columns:
        return 'Metric'  # backward compatibility
    if 'Sector' in df.columns:
        return 'Sector'  # legacy support
    if 'sector' in df.columns:
        return 'sector'  # legacy support
    raise ValueError("No metric column found. DataFrame must contain 'Metric', or 'Sector' column.")


def create_metric_col(df: pl.DataFrame, metric_col: str) -> pl.DataFrame:
    """
    Create metric labels from metric and quantity.

    Uses the simplest combination of columns that maintains uniqueness.
    """
    # Get the full unique count with all columns
    if metric_col != 'Metric':
        df.rename({metric_col: 'Metric'})
    unique_metrics = df.select(['Metric', 'Quantity', 'Unit']).unique()
    if len(unique_metrics) != len(df.select('Metric').unique()):
        raise ValueError(f"Column {metric_col} contains duplicate values. Please check the data.")

    # # Try combinations in order of simplicity
    # cols_to_use = [metric_col]
    # if len(df.select(cols_to_use).unique()) != full_count:
    #     cols_to_use.append('Quantity')
    #     if len(df.select(cols_to_use).unique()) != full_count:
    #         cols_to_use.append('Unit')

    # # Create the label by joining the selected columns with spaces
    # df = df.with_columns(
    #     pl.concat_str([
    #         pl.col(col)
    #         for col in cols_to_use
    #     ], separator=" ").alias('metric_col')
    # ).drop(metric_col)

    # Convert metric_col to snake_case
    df = df.with_columns([
        pl.col('Metric').map_elements(to_snake_case, return_dtype=pl.Utf8).alias('metric_col')
    ])

    return df


def split_by_dataset(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split the dataframe into separate dataframes by Dataset."""
    datasets = {}

    if 'Dataset' in df.columns:
        unique_datasets = df.select('Dataset').unique().to_series(0).to_list()
    else:
        raise ValueError("If specific_dataset is not defined, there must be column Dataset.")

    for dataset_name in unique_datasets:
        if dataset_name is not None:
            dataset_df = df.filter(pl.col('Dataset') == dataset_name).drop('Dataset')
            datasets[dataset_name] = dataset_df
            print(f"Created dataset for dataset: {dataset_name} with {len(dataset_df)} rows")

    return datasets


def extract_units(df: pl.DataFrame, dataset_name: str) -> dict:
    """Extract units from the dataframe using metric labels."""
    units = {}

    # Use the already created metric labels
    unique_metrics = df.select(['metric_col', 'Unit']).unique()

    for row in unique_metrics.iter_rows(named=True):
        metric_col = row['metric_col']
        unit = row['Unit']

        if metric_col and unit:
            units[metric_col] = unit

    return units

def extract_units_from_row(df: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    """Extract units from the first row if it contains only strings, otherwise treat as data."""
    units: dict[str, str] = {}

    if df.height == 0:
        return units, df

    first_row = df.row(0, named=True)

    # Check if the first row contains any numeric values, i.e. is data row
    has_numeric = False
    for value in first_row.values():
        if value is not None:
            try:
                float(value)
                has_numeric = True
                break
            except (ValueError, TypeError):
                continue
    if has_numeric:
        return units, df
    for col_name, unit_value in first_row.items():
        if unit_value is not None and str(unit_value).strip():
            units[col_name] = str(unit_value).strip()

    df_cleaned = df.slice(1)
    df_cleaned = df_cleaned.with_columns([
        pl.col(col).cast(pl.Float64, strict=False) for col in units.keys()
    ])
    return units, df_cleaned

def extract_description(df: pl.DataFrame, dataset_name: str) -> str | None:
    """Extract description from the dataframe."""
    description = None # FIXME Does not show up in admin UI.
    if 'Description' in df.columns:
        descriptions = []
        rows_with_description = df.filter(pl.col('Description').is_not_null())

        for row_idx in range(len(rows_with_description)):
            row = rows_with_description.row(row_idx, named=True)

            parts = []
            if row.get('metric_col'):
                parts.append(row['metric_col'])

            # Add values from dimension columns
            for col in df.columns:
                if (
                    (df[col].dtype == pl.Utf8 or df[col].dtype == pl.String) and
                    (col not in ['metric_col', 'Unit', 'Value', 'Description', 'Quantity', 'Year']) and
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


def extract_metrics(df: pl.DataFrame, language: str) -> list:
    """Extract metrics from the dataframe using metric labels."""
    metrics = []

    # Use the already created metric labels
    unique_metrics = df.select(['metric_col', 'Metric', 'Quantity']).unique()

    for row in unique_metrics.iter_rows(named=True):
        metric_id = row['metric_col']
        metric_name = row['Metric']
        quantity = row['Quantity']

        if metric_name and quantity:
            metrics.append({
                "id": to_snake_case(metric_id),
                "quantity": to_snake_case(quantity),
                "label": {language: metric_name},
            })

    return metrics


def clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Remove metadata columns and empty columns."""
    # 1. Drop metadata columns that are now in metadata but keep CompoundID
    metadata_columns = ['Unit', 'Description', 'Metric']
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


def pivot_by_compound_id(df: pl.DataFrame) -> pl.DataFrame:
    """Convert dataframe to have metric labels as columns."""
    # Get unique metric labels
    unique_ids = df.select('metric_col').unique().to_series(0).to_list()
    unique_ids = [s for s in unique_ids if s is not None]

    # Get dimension columns (all except those used for pivoting and the value)
    dim_cols = [col for col in df.columns if col not in ['Quantity', 'Value', 'metric_col']]

    # Ensure no null metric labels
    df = df.with_columns(
        pl.when(pl.col('metric_col').is_null())
          .then(pl.lit('Value'))
          .otherwise(pl.col('metric_col'))
          .alias('metric_col')
    )

    # Pivot to have metric labels as columns
    try:
        df = df.with_columns(pl.col('Value').cast(pl.Float64))
    except ValueError:
        warnings.warn(
            "Some values could not be converted to float and will be kept as strings. "
            "This is normal for some datasets that contain non-numeric values.",
            UserWarning,
            stacklevel=2
        )

    # Check for duplicates in the combination of index + pivot columns
    duplicate_check_cols = dim_cols + ["metric_col"]
    duplicates_mask = df.select(duplicate_check_cols).is_duplicated()

    if duplicates_mask.any():
        print("Duplicates found:")
        duplicates = df.filter(duplicates_mask)
        print(duplicates)
        print(duplicates.columns)
        raise ValueError("Stopping execution due to unexpected duplicates")

    result_df = df.pivot(
        values="Value",
        index=dim_cols,
        on="metric_col"
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
    cols = [col for col in new_columns if col not in ['metric']]
    for col in cols:
        if col not in metrics + ['Year']:
            if df[col].dtype != pl.Utf8:
                print(df)
                print(units)
                raise ValueError(f"Column {col} does not contain strings.")
            print('column to convert', col)
            df = df.with_columns(
                pl.col(col).map_elements(to_snake_case, return_dtype=pl.Utf8).alias(col)
            )

    return df


def save_to_csv(df: pl.DataFrame, file_path: str, dataset_name: str) -> None:
    """Save dataframe to CSV if a path is provided."""
    if file_path.upper() not in ['N', 'NONE']:
        # Create a unique filename for each dataset
        file_name, file_ext = os.path.splitext(file_path)  # noqa: PTH122
        dataset_file_path = f"{file_name}_{to_snake_case(dataset_name)}{file_ext}"

        df.write_csv(dataset_file_path)
        print(df)
        print(f'Data saved to {dataset_file_path}')


def push_to_dvc(df: pl.DataFrame, output_path: str, dataset_name: str,
                units: dict, description: str | None, metrics: list,
                language: str) -> None:
    """Push dataset to DVC repository."""
    if output_path.upper() in ['N', 'NONE']:
        return

    # Get index columns (excluding metric value columns)
    index_columns = [col for col in df.columns if col not in units.keys()]

    # Build metadata
    metadata: dict[str, Any] = {
        'name': {language: dataset_name},
        'identifier': to_snake_case(dataset_name),
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
    if ds.meta.metadata is None:
        ds.meta.metadata = {}
    ds.meta.metadata['updated_at'] = str(int(time.time()))

    # TODO If pushing fails and you end up having a local commit, you cannot push again.
    # Then, you can just remove the local cache, e.g. rm -rf /Users/jouni/Library/Caches/dvc-pandas/
    repo.push_dataset(ds)
    print(f'Dataset pushed to DVC at {output_path}')


def process_dataset(df: pl.DataFrame, dataset_name: str, outcsvpath: str, outdvcpath: str, language: str) -> None:
    """Process a single dataset of data."""
    print(f"\n==== Processing dataset: {dataset_name} ====\n")

    if len(df) == 0:
        print(f"Dataset {dataset_name} has no data. Skipping.")
        return

    # 1. Validate required columns
    validate_required_columns(df)

    # 2. Determine which column to use for metrics
    metric_col = determine_metric_column(df)
    print(f"Using '{metric_col}' as the metric column.")

    # 3. Create metric labels
    df = create_metric_col(df, metric_col)

    # 4. Extract metadata using metric labels
    units = extract_units(df, dataset_name)
    metrics = extract_metrics(df, language)
    description = extract_description(df, dataset_name)
    print(f"Units: {units}")
    print(f"Metrics: {len(metrics)} entries")
    print(metrics)
    if description:
        print("Description extracted")

    # 5. Clean dataframe (remove metadata columns)
    df = clean_dataframe(df)

    # 6. Convert to standard format with Year column if needed
    df = convert_to_standard_format(df)
    print(f"Data converted to standard format with {len(df)} rows")

    # 7. Pivot by compound ID
    df = pivot_by_compound_id(df)
    dim_ids = [s for s in df.columns if s not in units.keys()]
    print(f"Data pivoted by compound identifiers with dimension columns: {dim_ids}")

    # 8. Check for issues
    if check_for_duplicates(df):
        print("Warning: Dataframe contains duplicate rows")

    # 9. Prepare for DVC (standardize column names)
    df = prepare_for_dvc(df, units)

    # 10. Save to CSV if requested
    save_to_csv(df, outcsvpath, dataset_name)

    # 11. Push to DVC if requested
    dataset_dvc_path = f"{outdvcpath}/{to_snake_case(dataset_name)}"
    push_to_dvc(df, dataset_dvc_path, dataset_name, units, description, metrics, language)


def main():
    """Process and convert data for all datasets."""
    load_dotenv()

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and convert data for all datasets")

    # Required arguments
    parser.add_argument('--input-csv', '-i',
                       required=True,
                       help='Input CSV file path')

    parser.add_argument('--output-dvc', '-o',
                       required=True,
                       help='Output DVC file path')

    # Arguments with defaults
    parser.add_argument('--output-csv', '-c',
                       required=False,
                       default='NONE',
                       help='Output CSV file path')

    parser.add_argument('--csv-separator', '-s',
                       default=',',
                       choices=[',', ';', '\t', '|'],
                       help='CSV separator (default: comma)')

    parser.add_argument('--encoding', '-e',
                        default='utf-8',
                        choices=['utf-8', 'cp1252', 'latin-1'],
                        help='CSV file encoding (default: utf-8)')

    parser.add_argument('--language', '-l',
                       default='en',
                       help='Language code (default: en)')

    parser.add_argument('--dataset', '-d',
                       default=None,
                       help='Process only specific dataset (optional)')

    # Parse arguments
    args = parser.parse_args()

    # Use the parsed arguments
    incsvpath = args.input_csv
    incsvsep = args.csv_separator
    outcsvpath = args.output_csv
    outdvcpath = args.output_dvc
    language = args.language
    specific_dataset = args.dataset
    encoding = args.encoding

    # Load data
    full_df = load_data(incsvpath, incsvsep, encoding)

    # Process datasets
    if specific_dataset:
        if specific_dataset == 'plain_csv':
            print("Uploading the csv file as is, but checking for units.")
            units, full_df = extract_units_from_row(full_df)
            push_to_dvc(full_df, outdvcpath, '', units, None, [], language)
        elif specific_dataset == 'csv_w_standard_dims': # TODO Metadata gets lost
            print("Uploading the csv file with standard dimensions.")
            mappings = load_yaml_mappings()
            full_df = replace_labels_with_ids(full_df, mappings)
            save_to_csv(full_df, outcsvpath, '')
            push_to_dvc(full_df, outdvcpath, '', {}, None, [], language)
        else:
            # Process only the specified
            print(f"Processing only dataset: {specific_dataset}")
            d = 'Dataset'
            dataset_df = full_df.filter(pl.col(d) == specific_dataset).drop(d) if d in full_df.columns else full_df
            process_dataset(dataset_df, specific_dataset, outcsvpath, outdvcpath, language)
    else:
        # Process all datasets
        dataset_dfs = split_by_dataset(full_df)
        print(f"Found {len(dataset_dfs)} datasets to process")

        for dataset_name, dataset_df in dataset_dfs.items():
            process_dataset(dataset_df, dataset_name, outcsvpath, outdvcpath, language)


if __name__ == "__main__":
    main()
