from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml

try:
    from openpyxl import load_workbook
    _has_openpyxl = True
except ImportError:
    _has_openpyxl = False
    load_workbook = None

try:
    import pandas as pd
    _has_pandas = True
except ImportError:
    _has_pandas = False
    pd = None


@dataclass
class ColumnSpec:
    """Describes what a column contains."""

    # For data columns:
    quantity: str = ''
    unit: str = ''
    year: int | None = None

    # For dimension columns (if this column contains dimension values):
    dimension_name: str = ''  # If set, this column contains values for this dimension

    # Additional dimension values (for wide format):
    dimensions: dict[str, str] = field(default_factory=dict)  # dimension_name -> value

    def is_dimension_column(self) -> bool:
        """Check if this column contains dimension values (long format)."""
        return bool(self.dimension_name)

    def is_data_column(self) -> bool:
        """Check if this column contains data values (wide format)."""
        return bool(self.quantity)


@dataclass
class DatasetSchema:
    """Generic schema for peculiar datasets with implicit structure."""

    row_identifier_name: str
    identifier_mappings: dict[str, dict[str, Any]]  # identifier -> additional attributes
    dimension_names: list[str]  # All dimension names used in this dataset
    column_specs: dict[int, ColumnSpec]  # column_number -> what it contains
    skip_rows: int = 1  # Skip explanation/header rows
    identifier_column: int = 0  # Column with row identifiers (0-based)
    has_header: bool = False  # Skip header row by default, use technical col names.
    value_column_name: str = 'Value'  # Name for the value column in output

    def __post_init__(self):
        if self.identifier_mappings is None:
            self.identifier_mappings = {}
        if self.column_specs is None:
            self.column_specs = {}


@dataclass
class DatasetConfig:
    """Complete dataset configuration including file paths and schema."""

    input_file_path: str
    output_file_path: str
    schema: DatasetSchema
    file_type: str = 'csv'  # 'csv' or 'excel'
    sheet_name: str | None = None  # For Excel files: specific sheet name, or None for all sheets
    sheet_year_mapping: dict[str, int] | None = None  # Map sheet names to years


class DatasetTransformer:
    """Transforms peculiar CSV and Excel datasets into normalized format."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    def _read_excel_sheet(self, file_path: str, sheet_name: str | None = None) -> pl.DataFrame:
        """Read an Excel file sheet into a Polars DataFrame."""
        if _has_openpyxl and load_workbook is not None:
            wb = load_workbook(file_path, data_only=True, read_only=True)
            if sheet_name:
                ws = wb[sheet_name]
            else:
                ws = wb.active

            if ws is None:
                wb.close()
                return pl.DataFrame()

            # Convert to list of lists
            data = [list(row) for row in ws.iter_rows(values_only=True)]

            wb.close()

            if not data:
                return pl.DataFrame()

            # Determine max columns
            max_cols = max(len(row) for row in data) if data else 0

            # Pad shorter rows
            padded_data = [list(row) + [None] * (max_cols - len(row)) for row in data]

            # Apply skip_rows first
            if self.schema.skip_rows > 0:
                padded_data = padded_data[self.schema.skip_rows:]

            if not padded_data:
                return pl.DataFrame()

            # Create column names
            if self.schema.has_header and len(padded_data) > 0:
                # Use first row as header
                header_row = padded_data[0]
                col_names = [
                    f"column_{i+1}" if (not h or str(h).strip() == '') else str(h).strip()
                    for i, h in enumerate(header_row[:max_cols])
                ]
                data_rows = padded_data[1:]
            else:
                col_names = [f"column_{i+1}" for i in range(max_cols)]
                data_rows = padded_data

            if not data_rows:
                return pl.DataFrame()

            # Create DataFrame
            df = pl.DataFrame(data_rows, schema=col_names[:max_cols], orient="row")

            return df

        if _has_pandas and pd is not None:
            df_pd = pd.read_excel(
                file_path,
                sheet_name=sheet_name or 0,
                header=0 if self.schema.has_header else None,
                skiprows=self.schema.skip_rows
            )
            return pl.from_pandas(df_pd)

        raise ImportError(
            "Excel support requires either 'openpyxl' or 'pandas' package. " +
            "Install with: pip install openpyxl (recommended) or pip install pandas openpyxl"
        )

    def load_and_transform(self, file_path: str, sheet_name: str | None = None, year: int | None = None) -> pl.DataFrame:  # noqa: C901, PLR0912
        """Load CSV or Excel and transform to normalized format."""
        # Determine file type
        file_path_obj = Path(file_path)
        is_excel = file_path_obj.suffix.lower() in ['.xlsx', '.xls']

        if is_excel:
            df = self._read_excel_sheet(file_path, sheet_name)
        else:
            df = pl.read_csv(
                file_path,
                skip_rows=self.schema.skip_rows,
                has_header=self.schema.has_header,
            )

        # Get column name for identifier column
        identifier_col_name = df.columns[self.schema.identifier_column]

        # Extract identifiers from specified column
        identifiers = df.select(
            pl.col(identifier_col_name).alias('identifier')
        ).to_series()

        # Transform to long format
        result_rows = []

        for row_idx, identifier in enumerate(identifiers):
            if identifier is None or str(identifier).strip() == '':
                continue

            # Get additional attributes for this identifier
            identifier_attrs = self.schema.identifier_mappings.get(str(identifier), {})

            # First, collect dimension values from dimension columns (long format)
            row_dimensions = {}
            for col_num, col_spec in self.schema.column_specs.items():
                if col_spec.is_dimension_column():
                    try:
                        dim_value = df.item(row_idx, col_num)
                        if dim_value is not None and str(dim_value).strip() != '':
                            row_dimensions[col_spec.dimension_name] = str(dim_value)
                    except (IndexError, ValueError, TypeError):
                        continue

            # Process each data column (wide format)
            for col_num, col_spec in self.schema.column_specs.items():
                if not col_spec.is_data_column():
                    continue  # Skip dimension columns

                try:
                    value = df.item(row_idx, col_num)
                    if value is not None and str(value).strip() != '':

                        # Build result row with identifier
                        row_data = {
                            self.schema.row_identifier_name: str(identifier),
                            'Quantity': col_spec.quantity,
                            'Unit': col_spec.unit,
                        }

                        # Add year if specified (from column spec or parameter)
                        if year is not None:
                            row_data['Year'] = year  # type: ignore
                        elif col_spec.year is not None:
                            row_data['Year'] = col_spec.year  # type: ignore

                        # Add dimensions from long format (from dimension columns)
                        row_data.update(row_dimensions)

                        # Add dimensions from wide format (from column spec)
                        row_data.update(col_spec.dimensions)

                        # Add value
                        row_data[self.schema.value_column_name] = float(value)  # type: ignore

                        # Add any additional attributes from identifier mapping
                        row_data.update(identifier_attrs)

                        result_rows.append(row_data)

                except (IndexError, ValueError, TypeError):
                    continue  # Skip invalid values

        result_df = pl.DataFrame(result_rows)

        return result_df

    def get_summary(self) -> dict[str, Any]:
        """Get summary of schema configuration."""
        data_columns = [spec for spec in self.schema.column_specs.values() if spec.is_data_column()]
        dimension_columns = [spec for spec in self.schema.column_specs.values() if spec.is_dimension_column()]

        summary = {
            'total_columns': len(self.schema.column_specs),
            'data_columns': len(data_columns),
            'dimension_columns': len(dimension_columns),
            'dimensions': self.schema.dimension_names,
            'identifiers': len(self.schema.identifier_mappings)
        }

        if data_columns:
            summary['quantities'] = list(set(spec.quantity for spec in data_columns if spec.quantity))
            summary['units'] = list(set(spec.unit for spec in data_columns if spec.unit))
            years = [spec.year for spec in data_columns if spec.year is not None]
            if years:
                summary['years'] = sorted(set(years))  # type: ignore

        return summary


def list_excel_sheets(file_path: str | Path) -> list[str]:
    """List all sheet names in an Excel file."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() not in ['.xlsx', '.xls']:
        raise ValueError(f"File is not an Excel file: {file_path}")

    if _has_openpyxl and load_workbook is not None:
        wb = load_workbook(file_path, read_only=True)
        sheet_names = wb.sheetnames
        wb.close()
        return sheet_names

    if _has_pandas:
        xl_file = pd.ExcelFile(file_path)
        return [str(name) for name in xl_file.sheet_names]

    raise ImportError(
        "Excel support requires either 'openpyxl' or 'pandas' package. " +
        "Install with: pip install openpyxl (recommended) or pip install pandas openpyxl"
    )


def load_config(yaml_file_path: str | Path) -> DatasetConfig:
    """
    Load complete dataset configuration from YAML file.

    Args:
        yaml_file_path: Path to the YAML configuration file

    Returns:
        DatasetConfig object with schema and file paths

    """
    yaml_file_path = Path(yaml_file_path)

    if not yaml_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_file_path}")

    with yaml_file_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Parse column specs
    column_specs = {}
    for col_num_str, spec_dict in config.get('column_specs', {}).items():
        col_num = int(col_num_str)

        # Create ColumnSpec based on whether it's a dimension or data column
        if spec_dict.get('dimension_name'):
            # This is a dimension column (long format)
            column_specs[col_num] = ColumnSpec(
                dimension_name=spec_dict['dimension_name']
            )
        else:
            # This is a data column (wide format)
            column_specs[col_num] = ColumnSpec(
                quantity=spec_dict.get('quantity', ''),
                unit=spec_dict.get('unit', ''),
                year=spec_dict.get('year'),
                dimensions=spec_dict.get('dimensions', {})
            )

    # Create DatasetSchema
    schema = DatasetSchema(
        row_identifier_name=config['row_identifier_name'],
        skip_rows=config.get('skip_rows', 1),
        has_header=config.get('has_header', False),
        identifier_column=config.get('identifier_column', 0),
        identifier_mappings=config.get('identifier_mappings', {}),
        column_specs=column_specs,
        dimension_names=config.get('dimension_names', []),
        value_column_name=config.get('value_column_name', 'Value')
    )

    # Determine file type
    input_path = Path(config['input_file_path'])
    file_type = 'excel' if input_path.suffix.lower() in ['.xlsx', '.xls'] else 'csv'

    # Create and return complete config
    return DatasetConfig(
        input_file_path=config['input_file_path'],
        output_file_path=config['output_file_path'],
        schema=schema,
        file_type=config.get('file_type', file_type),
        sheet_name=config.get('sheet_name'),
        sheet_year_mapping=config.get('sheet_year_mapping')
    )


def process_dataset(config_path: str | Path, verbose: bool = True) -> pl.DataFrame:  # noqa: C901, PLR0912
    """
    Process a dataset using configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print progress information

    Returns:
        Transformed DataFrame

    """
    # Load configuration
    if verbose:
        print(f"Loading configuration from: {config_path}")

    config = load_config(config_path)

    # Validate input file exists
    input_path = Path(config.input_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Transform the data
    if verbose:
        print(f"Processing input file: {config.input_file_path}")
        if config.file_type == 'excel':
            print("  File type: Excel")
            if config.sheet_name:
                print(f"  Sheet: {config.sheet_name}")
            elif config.sheet_year_mapping:
                print("  Processing multiple sheets with year mapping")

    transformer = DatasetTransformer(config.schema)

    # Handle Excel files with multiple sheets
    if config.file_type == 'excel' and config.sheet_year_mapping:
        # Process multiple sheets, one per year
        all_results = []

        for sheet_name, year in config.sheet_year_mapping.items():
            if verbose:
                print(f"\nProcessing sheet '{sheet_name}' (year: {year})")

            sheet_df = transformer.load_and_transform(
                config.input_file_path,
                sheet_name=sheet_name,
                year=year
            )

            if len(sheet_df) > 0:
                all_results.append(sheet_df)
                if verbose:
                    print(f"  Rows from this sheet: {len(sheet_df)}")

        if all_results:
            result_df = pl.concat(all_results, rechunk=False)
        else:
            result_df = pl.DataFrame()

    elif config.file_type == 'excel' and config.sheet_name:
        # Process single specified sheet
        result_df = transformer.load_and_transform(
            config.input_file_path,
            sheet_name=config.sheet_name
        )
    else:
        # Process single file (CSV or Excel default sheet)
        result_df = transformer.load_and_transform(config.input_file_path)

    if verbose:
        print("\nTransformation Summary:")
        summary = transformer.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"\nTotal rows in output: {len(result_df)}")

    # Save result
    output_path = Path(config.output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Writing output to: {config.output_file_path}")

    result_df.write_csv(config.output_file_path)

    if verbose:
        print("✓ Processing complete!")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Transform peculiar CSV datasets into normalized format using YAML configuration.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s potsdam_scenario_dataset.yaml
  %(prog)s config/dataset_schema.yaml --quiet
  %(prog)s my_config.yaml --show-data
        """
    )

    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    parser.add_argument(
        '-s', '--show-data',
        action='store_true',
        help='Display the transformed data'
    )

    parser.add_argument(
        '--head',
        type=int,
        metavar='N',
        help='Show first N rows of output (implies --show-data)'
    )

    parser.add_argument(
        '--list-sheets',
        action='store_true',
        help='List available sheets in Excel file and exit'
    )

    args = parser.parse_args()

    # Handle list-sheets option
    if args.list_sheets:
        try:
            config = load_config(args.config)
            if config.file_type == 'excel':
                sheets = list_excel_sheets(config.input_file_path)
                print(f"\nAvailable sheets in {config.input_file_path}:")
                for i, sheet in enumerate(sheets, 1):
                    print(f"  {i}. {sheet}")
                return 0
            print(f"File is not an Excel file: {config.input_file_path}")
            return 1  # noqa: TRY300
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    try:
        result_df = process_dataset(args.config, verbose=not args.quiet)

        # Display data if requested
        if args.show_data or args.head:
            print("\n" + "="*80)
            print("Transformed Data:")
            print("="*80)
            if args.head:
                print(result_df.head(args.head))
            else:
                print(result_df)

        return 0  # noqa: TRY300

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
