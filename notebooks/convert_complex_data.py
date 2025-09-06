from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class ColumnSpec:
    """Describes what a column contains."""

    quantity: str
    unit: str
    year: int
    dimension1: str = ''  # Some dimensions, define in schema
    dimension2: str = ''
    dimension3: str = ''

@dataclass
class DatasetSchema:
    """Generic schema for peculiar datasets with implicit structure."""

    row_identifier_name: str
    identifier_mappings: dict[str, dict[str, Any]]  # identifier -> additional attributes
    dimension_names: list[str] # Names for 0-3 categories that are actually used
    column_specs: dict[int, ColumnSpec]  # column_number -> what it contains
    skip_rows: int = 1  # Skip explanation/header rows
    identifier_column: int = 0  # Column with row identifiers (0-based)
    has_header: bool = False # Skip header row by default, use technical col names.

    def __post_init__(self):
        if self.identifier_mappings is None:
            self.identifier_mappings = {}
        if self.column_specs is None:
            self.column_specs = {}

class DatasetTransformer:
    """Transforms peculiar CSV datasets into normalized format."""

    def __init__(self, schema: DatasetSchema):
        self.schema = schema

    def load_and_transform(self, file_path: str) -> pl.DataFrame:
        """Load CSV and transform to normalized format."""

        df = pl.read_csv(
            file_path,
            skip_rows=self.schema.skip_rows,
            has_header=self.schema.has_header,
        )

        # Extract identifiers from specified column
        identifiers = df.select(
            pl.col(f'column_{self.schema.identifier_column + 1}').alias('identifier')
        ).to_series()

        # Transform to long format
        result_rows = []

        for row_idx, identifier in enumerate(identifiers):
            if identifier is None or str(identifier).strip() == '':
                continue

            # Get additional attributes for this identifier
            identifier_attrs = self.schema.identifier_mappings.get(str(identifier), {})

            # Process each data column
            for col_num, col_spec in self.schema.column_specs.items():
                try:
                    value = df.item(row_idx, col_num)
                    if value is not None and str(value).strip() != '':

                        # Build result row
                        row_data = {
                            self.schema.row_identifier_name: str(identifier),
                            'Quantity': col_spec.quantity,
                            'Unit': col_spec.unit,
                            'Year': col_spec.year,
                            'Dimension1': col_spec.dimension1,
                            'Dimension2': col_spec.dimension2,
                            'Dimension3': col_spec.dimension3,
                            'Value': float(value)
                        }

                        # Add any additional attributes from identifier mapping
                        row_data.update(identifier_attrs)

                        result_rows.append(row_data)

                except (IndexError, ValueError, TypeError):
                    continue  # Skip invalid values

        result_df = pl.DataFrame(result_rows)

        dim_placeholders = ['Dimension1', 'Dimension2', 'Dimension3']
        new_dims = self.schema.dimension_names
        drop_dims = dim_placeholders[len(new_dims):]
        old_dims = dim_placeholders[:len(new_dims)]
        result_df = result_df.drop(drop_dims)
        result_df = result_df.rename(dict(zip(old_dims, new_dims, strict=False)))

        return result_df

    def get_summary(self) -> dict[str, Any]:
        """Get summary of schema configuration."""
        return {
            'total_columns': len(self.schema.column_specs),
            'quantities': list(set(spec.quantity for spec in self.schema.column_specs.values())),
            'dimension1': list(set(spec.dimension1 for spec in self.schema.column_specs.values())),
            'dimension2': list(set(spec.dimension2 for spec in self.schema.column_specs.values())),
            'dimension3': list(set(spec.dimension3 for spec in self.schema.column_specs.values())),
            'years': sorted(set(spec.year for spec in self.schema.column_specs.values())),
            'identifiers': len(self.schema.identifier_mappings)
        }

# Adjust this function to your case-specific needs.
# In addition to input and output file paths, there should not be a need to adjust anything else.
# Move these case-specific parts to ../netzeroplanner-framework-config/potsdam_scenario_dataset.schema
def create_dataset_schema() -> DatasetSchema:
    """Create schema for energy dataset - but using generic structure."""

    # Map the row identifiers to whatever other columns you may need
    identifier_mappings = {
        # Strom (Electricity)
        'Strom': {'Sector': 'Strom'},
        # Wärme (Heat/Heating)
        'Heizstrom': {'Sector': 'Wärme'},
        'Fernwärme': {'Sector': 'Wärme'},
        'Nahwärme': {'Sector': 'Wärme'},
        'Gas': {'Sector': 'Wärme'},
        'Biogas': {'Sector': 'Wärme'},
        'Heizöl': {'Sector': 'Wärme'},
        'Kohle': {'Sector': 'Wärme'},
        'Biomasse': {'Sector': 'Wärme'},
        'Solarthermie': {'Sector': 'Wärme'},
        'Umweltwärme': {'Sector': 'Wärme'},
        # Verkehr (Transport)
        'Fahrstrom': {'Sector': 'Verkehr'},
        'Benzin fossil': {'Sector': 'Verkehr'},
        'Benzin biogen': {'Sector': 'Verkehr'},
        'Diesel fossil': {'Sector': 'Verkehr'},
        'Diesel biogen': {'Sector': 'Verkehr'},
        'CNG fossil': {'Sector': 'Verkehr'},
        'LPG': {'Sector': 'Verkehr'},
        'Wasserstoff': {'Sector': 'Verkehr'},
    }

    # Define categories in columns by column number (0-based)
    column_specs = {
        1: ColumnSpec('Energy', 'MWh/a', 1995, 'Aktual'),
        2: ColumnSpec('Energy', 'MWh/a', 1999, 'Aktual'),
        3: ColumnSpec('Energy', 'MWh/a', 2003, 'Aktual'),
        4: ColumnSpec('Energy', 'MWh/a', 2008, 'Aktual'),
        5: ColumnSpec('Energy', 'MWh/a', 2012, 'Aktual'),
        6: ColumnSpec('Energy', 'MWh/a', 2014, 'Aktual'),
        7: ColumnSpec('Energy', 'MWh/a', 2017, 'Aktual'),
        8: ColumnSpec('Energy', 'MWh/a', 2020, 'Aktual'),
        9: ColumnSpec('Energy', 'MWh/a', 2021, 'Aktual'),
        10: ColumnSpec('Energy', 'MWh/a', 2022, 'Aktual'),
        11: ColumnSpec('Energy', 'MWh/a', 2023, 'Aktual'),
        # 12: ColumnSpec('Energy', 'MWh/a', 2020, 'Trend-Szenario'), # To avoid duplicate data
        13: ColumnSpec('Energy', 'MWh/a', 2030, 'Trend-Szenario'),
        14: ColumnSpec('Energy', 'MWh/a', 2040, 'Trend-Szenario'),
        15: ColumnSpec('Energy', 'MWh/a', 2050, 'Trend-Szenario'),
        # 16: ColumnSpec('Energy', 'MWh/a', 2020, 'Masterplan-Szenario'),
        17: ColumnSpec('Energy', 'MWh/a', 2030, 'Masterplan-Szenario'),
        18: ColumnSpec('Energy', 'MWh/a', 2040, 'Masterplan-Szenario'),
        19: ColumnSpec('Energy', 'MWh/a', 2050, 'Masterplan-Szenario'),
        20: ColumnSpec('Emissions', 't/a', 1995, 'Aktual'),
        21: ColumnSpec('Emissions', 't/a', 1999, 'Aktual'),
        22: ColumnSpec('Emissions', 't/a', 2003, 'Aktual'),
        23: ColumnSpec('Emissions', 't/a', 2008, 'Aktual'),
        24: ColumnSpec('Emissions', 't/a', 2012, 'Aktual'),
        25: ColumnSpec('Emissions', 't/a', 2014, 'Aktual'),
        26: ColumnSpec('Emissions', 't/a', 2017, 'Aktual'),
        27: ColumnSpec('Emissions', 't/a', 2020, 'Aktual'),
        28: ColumnSpec('Emissions', 't/a', 2021, 'Aktual'),
        # 29: ColumnSpec('Emissions', 't/a', 2020, 'Trend-Szenario'),
        30: ColumnSpec('Emissions', 't/a', 2030, 'Trend-Szenario'),
        31: ColumnSpec('Emissions', 't/a', 2040, 'Trend-Szenario'),
        32: ColumnSpec('Emissions', 't/a', 2050, 'Trend-Szenario'),
        # 33: ColumnSpec('Emissions', 't/a', 2020, 'Masterplan-Szenario'),
        34: ColumnSpec('Emissions', 't/a', 2030, 'Masterplan-Szenario'),
        35: ColumnSpec('Emissions', 't/a', 2040, 'Masterplan-Szenario'),
        36: ColumnSpec('Emissions', 't/a', 2050, 'Masterplan-Szenario'),
    }

    return DatasetSchema(
        skip_rows=1,
        has_header=False,
        identifier_column=0,  # (0-based)
        identifier_mappings=identifier_mappings,
        column_specs=column_specs,
        row_identifier_name='Energy carrier',
        dimension_names=['Scenario']
    )

# Usage example
def main():
    # Create schema for your dataset
    schema = create_dataset_schema()

    # Transform the data
    transformer = DatasetTransformer(schema)
    file_path = '/Users/jouni/Downloads/Zielwerte Masterplan V3.4.xlsx - potsdam.csv'
    result_df = transformer.load_and_transform(file_path)

    print("Summary:", transformer.get_summary())
    print(result_df)

    # Save result
    result_df.write_csv('normalized_data.csv')

if __name__ == "__main__":
    main()
