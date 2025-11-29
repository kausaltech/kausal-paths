from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml


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

    # Forward fill option
    forward_fill: bool = False  # If True, empty cells are filled with value above

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
    skip_row_ranges: list[tuple[int, int]] = field(default_factory=list)  # [(start, end), ...] inclusive ranges
    identifier_column: int = 0  # Column with row identifiers (0-based)
    has_header: bool = False  # Skip header row by default, use technical col names.
    value_column_name: str = 'Value'  # Name for the value column in output

    def __post_init__(self):
        # Convert skip_row_ranges to list of tuples if needed
        if self.skip_row_ranges:
            self.skip_row_ranges = [tuple(r) if isinstance(r, list) else r
                                   for r in self.skip_row_ranges]

    def should_skip_row(self, row_idx: int) -> bool:
        """Check if a row index should be skipped."""
        for start, end in self.skip_row_ranges:
            if start <= row_idx <= end:
                return True
        return False


@dataclass
class DatasetConfig:
    """Complete dataset configuration including file paths and schema."""

    input_file_path: str
    output_file_path: str
    schema: DatasetSchema


class DatasetTransformer:
    """Transforms peculiar CSV datasets into normalized format."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    def _apply_forward_fill(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply forward fill to specified columns."""
        forward_fill_cols = [
            col_num for col_num, col_spec in self.schema.column_specs.items()
            if col_spec.forward_fill
        ]

        if not forward_fill_cols:
            return df

        # Convert to list of column names
        col_names = [f'column_{col_num + 1}' for col_num in forward_fill_cols]

        # Apply forward fill to each column
        for col_name in col_names:
            if col_name in df.columns:
                df = df.with_columns(
                    pl.col(col_name).fill_null(strategy='forward').alias(col_name)
                )

        return df

    def load_and_transform(self, file_path: str) -> pl.DataFrame:
        """Load CSV and transform to normalized format."""

        # Load all rows first (no skip_rows parameter)
        df = pl.read_csv(
            file_path,
            has_header=self.schema.has_header,
        )

        # Apply forward fill before filtering rows
        df = self._apply_forward_fill(df)

        # Extract identifiers from specified column
        identifiers = df.select(
            pl.col(f'column_{self.schema.identifier_column + 1}').alias('identifier')
        ).to_series()

        # Transform to long format
        result_rows = []

        for row_idx, identifier in enumerate(identifiers):
            # Check if this row should be skipped
            if self.schema.should_skip_row(row_idx):
                continue

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

                        # Add year if specified
                        if col_spec.year is not None:
                            row_data['Year'] = col_spec.year

                        # Add dimensions from long format (from dimension columns)
                        row_data.update(row_dimensions)

                        # Add dimensions from wide format (from column spec)
                        row_data.update(col_spec.dimensions)

                        # Add value
                        row_data[self.schema.value_column_name] = float(value)

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
            'identifiers': len(self.schema.identifier_mappings),
            'skip_row_ranges': self.schema.skip_row_ranges
        }

        if data_columns:
            summary['quantities'] = list(set(spec.quantity for spec in data_columns if spec.quantity))
            summary['units'] = list(set(spec.unit for spec in data_columns if spec.unit))
            years = [spec.year for spec in data_columns if spec.year is not None]
            if years:
                summary['years'] = sorted(set(years))

        return summary


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

    with Path.open(yaml_file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Parse column specs
    column_specs = {}

    # First, handle column_spec_templates if they exist
    templates = config.get('column_spec_templates', {})

    # Then process column_specs
    for key, spec_dict in config.get('column_specs', {}).items():
        # Handle column ranges (e.g., "4-6")
        if isinstance(key, str) and '-' in key:
            start, end = map(int, key.split('-'))
            col_range = range(start, end + 1)
        else:
            col_range = [int(key)]

        # Create ColumnSpec for each column in range
        for col_num in col_range:
            # Check if this spec references a template
            if 'template' in spec_dict:
                template_name = spec_dict['template']
                if template_name not in templates:
                    raise ValueError(f"Template '{template_name}' not found")
                base_spec = templates[template_name].copy()
                # Merge with any overrides in spec_dict
                base_spec.update({k: v for k, v in spec_dict.items() if k != 'template'})
                spec_dict = base_spec  # noqa: PLW2901

            if spec_dict.get('dimension_name'):
                # This is a dimension column (long format)
                column_specs[col_num] = ColumnSpec(
                    dimension_name=spec_dict['dimension_name'],
                    forward_fill=spec_dict.get('forward_fill', False)
                )
            else:
                # This is a data column (wide format)
                column_specs[col_num] = ColumnSpec(
                    quantity=spec_dict.get('quantity', ''),
                    unit=spec_dict.get('unit', ''),
                    year=spec_dict.get('year'),
                    dimensions=spec_dict.get('dimensions', {}),
                    forward_fill=spec_dict.get('forward_fill', False)
                )

    # Parse skip_row_ranges
    skip_row_ranges = []
    if 'skip_row_ranges' in config:
        for range_spec in config['skip_row_ranges']:
            if isinstance(range_spec, list) and len(range_spec) == 2:
                skip_row_ranges.append((range_spec[0], range_spec[1]))
            elif isinstance(range_spec, int):
                skip_row_ranges.append((range_spec, range_spec))
    # Backward compatibility with old skip_rows
    elif 'skip_rows' in config:
        skip_rows = config['skip_rows']
        if skip_rows > 0:
            skip_row_ranges.append((0, skip_rows - 1))

    # Create DatasetSchema
    schema = DatasetSchema(
        row_identifier_name=config['row_identifier_name'],
        skip_row_ranges=skip_row_ranges,
        has_header=config.get('has_header', False),
        identifier_column=config.get('identifier_column', 0),
        identifier_mappings=config.get('identifier_mappings', {}),
        column_specs=column_specs,
        dimension_names=config.get('dimension_names', []),
        value_column_name=config.get('value_column_name', 'Value')
    )

    # Create and return complete config
    return DatasetConfig(
        input_file_path=config['input_file_path'],
        output_file_path=config['output_file_path'],
        schema=schema
    )


def process_dataset(config_path: str | Path, verbose: bool = True) -> pl.DataFrame:
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

    transformer = DatasetTransformer(config.schema)
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

    args = parser.parse_args()

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
