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
    output_file_path: str | None = None  # Optional, can be set at top level
    schema: DatasetSchema | None = None
    file_type: str | None = None  # Auto-detected if None: 'csv', 'excel', or 'parquet'
    sheet_name: str | None = None  # For Excel files: specific sheet name
    sheet_year_mapping: dict[str, int] | None = None  # Map sheet names to years


class FileLoader:
    """Handles opening and reading files in different formats."""

    @staticmethod
    def detect_file_type(file_path: str | Path) -> str:
        """Detect file type from extension."""
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in ['.xlsx', '.xls']:
            return 'excel'
        if suffix == '.parquet':
            return 'parquet'
        return 'csv'

    @staticmethod
    def load_csv(file_path: str | Path, skip_rows: int = 0, has_header: bool = True) -> pl.DataFrame:
        """Load a CSV file into a Polars DataFrame."""
        return pl.read_csv(
            file_path,
            skip_rows=skip_rows,
            has_header=has_header,
        )

    @staticmethod
    def load_parquet(file_path: str | Path) -> pl.DataFrame:
        """Load a parquet file into a Polars DataFrame."""
        return pl.read_parquet(file_path)

    @staticmethod
    def load_excel(
        file_path: str | Path,
        sheet_name: str | None = None,
        skip_rows: int = 0,
        has_header: bool = False
    ) -> pl.DataFrame:
        """Load an Excel file sheet into a Polars DataFrame."""
        if not _has_openpyxl or load_workbook is None:
            raise ImportError(
                "Excel support requires 'openpyxl' package. Install with: pip install openpyxl"
            )

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
        if skip_rows > 0:
            padded_data = padded_data[skip_rows:]

        if not padded_data:
            return pl.DataFrame()

        # Create column names
        if has_header and len(padded_data) > 0:
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

    @classmethod
    def load_file(
        cls,
        file_path: str | Path,
        file_type: str | None = None,
        sheet_name: str | None = None,
        skip_rows: int = 0,
        has_header: bool = False
    ) -> pl.DataFrame:
        """Load a file based on its type."""
        if file_type is None:
            file_type = cls.detect_file_type(file_path)

        if file_type == 'parquet':
            return cls.load_parquet(file_path)
        if file_type == 'excel':
            return cls.load_excel(file_path, sheet_name, skip_rows, has_header)
        # Default to CSV
        return cls.load_csv(file_path, skip_rows, has_header)


class DataCollector:
    """Handles collecting and extracting data from loaded DataFrames."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    def collect_data(self, df: pl.DataFrame, year: int | None = None) -> list[dict[str, Any]]:  # noqa: C901, PLR0912
        """Collect data from DataFrame and return as list of row dictionaries."""
        if len(df) == 0:
            return []

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

        return result_rows


class DataTransformer:
    """Handles converting collected data into normalized format."""

    @staticmethod
    def transform_to_dataframe(rows: list[dict[str, Any]]) -> pl.DataFrame:
        """Convert list of row dictionaries to normalized DataFrame."""
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)


class OutputWriter:
    """Handles writing output files."""

    @staticmethod
    def write_csv(df: pl.DataFrame, output_path: str | Path, verbose: bool = True) -> None:
        """Write DataFrame to CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"Writing output to: {output_path}")

        df.write_csv(output_path)

        if verbose:
            print(f"✓ Written {len(df)} rows to {output_path}")


class DatasetProcessor:
    """Orchestrates the complete dataset processing pipeline."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.file_loader = FileLoader()
        self.data_collector = DataCollector(config.schema) if config.schema else None
        self.data_transformer = DataTransformer()
        self.output_writer = OutputWriter()

    def process(self, verbose: bool = True) -> pl.DataFrame:  # noqa: C901, PLR0912
        """Process a single dataset configuration."""
        if self.config.schema is None:
            # For parquet files without schema, just load and return as-is
            if verbose:
                print(f"Loading file: {self.config.input_file_path}")
            df = self.file_loader.load_file(
                self.config.input_file_path,
                self.config.file_type
            )
            return df

        # Full processing pipeline
        if verbose:
            print(f"Processing: {self.config.input_file_path}")
            if self.config.file_type == 'excel':
                print("  File type: Excel")
                if self.config.sheet_name:
                    print(f"  Sheet: {self.config.sheet_name}")
                elif self.config.sheet_year_mapping:
                    print("  Processing multiple sheets with year mapping")

        if self.data_collector is None:
            raise ValueError("Schema is required for data collection")

        # Step 1: Open file(s)
        if self.config.file_type == 'excel' and self.config.sheet_year_mapping:
            # Process multiple sheets
            all_rows = []
            for sheet_name, year in self.config.sheet_year_mapping.items():
                if verbose:
                    print(f"\n  Processing sheet '{sheet_name}' (year: {year})")

                df = self.file_loader.load_file(
                    self.config.input_file_path,
                    file_type='excel',
                    sheet_name=sheet_name,
                    skip_rows=self.config.schema.skip_rows,
                    has_header=self.config.schema.has_header
                )

                # Step 2: Collect data
                rows = self.data_collector.collect_data(df, year=year)
                all_rows.extend(rows)

                if verbose:
                    print(f"    Collected {len(rows)} rows from this sheet")

            # Step 3: Transform to DataFrame
            result_df = self.data_transformer.transform_to_dataframe(all_rows)

        elif self.config.file_type == 'excel' and self.config.sheet_name:
            # Process single sheet
            df = self.file_loader.load_file(
                self.config.input_file_path,
                file_type='excel',
                sheet_name=self.config.sheet_name,
                skip_rows=self.config.schema.skip_rows,
                has_header=self.config.schema.has_header
            )
            rows = self.data_collector.collect_data(df)
            result_df = self.data_transformer.transform_to_dataframe(rows)

        else:
            # Process single file (CSV, parquet, or Excel default sheet)
            df = self.file_loader.load_file(
                self.config.input_file_path,
                file_type=self.config.file_type,
                skip_rows=self.config.schema.skip_rows,
                has_header=self.config.schema.has_header
            )
            rows = self.data_collector.collect_data(df)
            result_df = self.data_transformer.transform_to_dataframe(rows)

        return result_df

    def get_summary(self) -> dict[str, Any]:
        """Get summary of schema configuration."""
        if self.config.schema is None:
            return {}

        data_columns = [spec for spec in self.config.schema.column_specs.values() if spec.is_data_column()]
        dimension_columns = [spec for spec in self.config.schema.column_specs.values() if spec.is_dimension_column()]

        summary = {
            'total_columns': len(self.config.schema.column_specs),
            'data_columns': len(data_columns),
            'dimension_columns': len(dimension_columns),
            'dimensions': self.config.schema.dimension_names,
            'identifiers': len(self.config.schema.identifier_mappings)
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

    if not _has_openpyxl or load_workbook is None:
        raise ImportError(
            "Excel support requires 'openpyxl' package. Install with: pip install openpyxl"
        )

    wb = load_workbook(file_path, read_only=True)
    sheet_names = wb.sheetnames
    wb.close()
    return sheet_names


def parse_column_specs(config_dict: dict[str, Any]) -> dict[int, ColumnSpec]:
    """Parse column specifications from config dictionary."""
    column_specs = {}
    for col_num_str, spec_dict in config_dict.get('column_specs', {}).items():
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
    return column_specs


def parse_schema(config_dict: dict[str, Any]) -> DatasetSchema:
    """Parse dataset schema from config dictionary."""
    column_specs = parse_column_specs(config_dict)

    return DatasetSchema(
        row_identifier_name=config_dict['row_identifier_name'],
        skip_rows=config_dict.get('skip_rows', 1),
        has_header=config_dict.get('has_header', False),
        identifier_column=config_dict.get('identifier_column', 0),
        identifier_mappings=config_dict.get('identifier_mappings', {}),
        column_specs=column_specs,
        dimension_names=config_dict.get('dimension_names', []),
        value_column_name=config_dict.get('value_column_name', 'Value')
    )


def parse_dataset_config(config_dict: dict[str, Any], default_output: str | None = None) -> DatasetConfig:
    """Parse a single dataset configuration from dictionary."""
    # Determine file type
    input_path = Path(config_dict['input_file_path'])
    file_type = FileLoader.detect_file_type(input_path)

    # Parse schema if present
    schema = None
    if 'row_identifier_name' in config_dict:
        schema = parse_schema(config_dict)

    return DatasetConfig(
        input_file_path=config_dict['input_file_path'],
        output_file_path=config_dict.get('output_file_path', default_output),
        schema=schema,
        file_type=config_dict.get('file_type', file_type),
        sheet_name=config_dict.get('sheet_name'),
        sheet_year_mapping=config_dict.get('sheet_year_mapping')
    )


def load_config(yaml_file_path: str | Path) -> tuple[list[DatasetConfig], str | None]:
    """
    Load dataset configurations from YAML file.

    Supports both single config and list of configs.

    Args:
        yaml_file_path: Path to the YAML configuration file

    Returns:
        Tuple of (list of DatasetConfig objects, output_file_path)

    """
    yaml_file_path = Path(yaml_file_path)

    if not yaml_file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {yaml_file_path}")

    with yaml_file_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Handle top-level output path
    default_output = config.get('output_file_path')

    # Check if config is a list of datasets
    if 'datasets' in config:
        # Multiple dataset configurations
        dataset_configs = [
            parse_dataset_config(dataset_dict, default_output)
            for dataset_dict in config['datasets']
        ]
        return dataset_configs, default_output

    # Single dataset configuration (backward compatibility)
    dataset_config = parse_dataset_config(config, default_output)
    return [dataset_config], default_output


def process_datasets(config_path: str | Path, verbose: bool = True) -> pl.DataFrame:  # noqa: C901, PLR0912
    """
    Process one or more datasets using configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file
        verbose: Whether to print progress information

    Returns:
        Transformed DataFrame (concatenated if multiple datasets)

    """
    # Load configuration
    if verbose:
        print(f"Loading configuration from: {config_path}")

    dataset_configs, output_path = load_config(config_path)

    if verbose:
        print(f"Found {len(dataset_configs)} dataset configuration(s)")

    # Process each dataset
    all_results = []
    for idx, config in enumerate(dataset_configs, 1):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Processing dataset {idx}/{len(dataset_configs)}")

        # Validate input file exists
        input_path = Path(config.input_file_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Process dataset
        processor = DatasetProcessor(config)
        result_df = processor.process(verbose=verbose)

        if verbose and config.schema:
            print("\nTransformation Summary:")
            summary = processor.get_summary()
            for key, value in summary.items():
                print(f"  {key}: {value}")

        if len(result_df) > 0:
            all_results.append(result_df)
            if verbose:
                print(f"  Collected {len(result_df)} rows")
        elif verbose:
            print("  No data collected")

    # Concatenate all results
    if all_results:
        final_df = pl.concat(all_results, rechunk=False)
        if verbose:
            print(f"\n{'='*80}")
            print(f"Total rows after concatenation: {len(final_df)}")
    else:
        final_df = pl.DataFrame()
        if verbose:
            print("\nNo data collected from any dataset")

    # Save result if output path is specified
    if output_path:
        output_writer = OutputWriter()
        output_writer.write_csv(final_df, output_path, verbose=verbose)

    return final_df


def main():
    """
    Start the command-line usage from here.

    Returns:
        Exit code (0 for success, 1 for error)

    """
    parser = argparse.ArgumentParser(
        description='Transform datasets into normalized format using YAML configuration.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.yaml
  %(prog)s config.yaml --quiet
  %(prog)s config.yaml --show-data
  %(prog)s config.yaml --list-sheets
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
            dataset_configs, _ = load_config(args.config)
            config = dataset_configs[0]  # Use first config
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
        result_df = process_datasets(args.config, verbose=not args.quiet)

        # Display data if requested
        if args.show_data or args.head:
            print("\n" + "="*80)
            print("Transformed Data:")
            print("="*80)
            if args.head:
                print(result_df.head(args.head))
            else:
                print(result_df)
            return 0
        return 0  # noqa: TRY300

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
