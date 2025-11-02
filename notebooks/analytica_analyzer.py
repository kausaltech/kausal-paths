from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import polars as pl

try:
    from defusedxml import ElementTree as ET  # noqa: N817
except ImportError:
    print("Installing defusedxml is recommended: pip install defusedxml")
    import xml.etree.ElementTree as ET


@dataclass
class AnalyticaDimension:
    name: str
    definition: list[str]
    description: str | None


@dataclass
class AnalyticaTable:
    name: str
    dimension_names: list[str]
    dimensions: list[AnalyticaDimension]
    dimension_sizes: list[int]
    data_flat: list[float]
    data_length: int


class AnalyzerXMLParser:
    def __init__(self, xml_file):
        """Initialize with XML file path or string."""
        try:
            self.tree = ET.parse(xml_file)  # noqa: S314
            self.root = self.tree.getroot()
        except (FileNotFoundError, TypeError, ET.ParseError, OSError):
            # If it's a string or parsing fails, try parsing as XML string
            self.root = ET.fromstring(xml_file)  # noqa: S314
        assert self.root is not None
        # Store all dimensions upon initialization
        self.dimensions = self.find_all_dimensions()

    def find_dimension_definition(self, dim_name):
        """
        Find the definition of a specific dimension.

        Merges parsing of values into this function.
        """
        # Search for index element with matching name
        for index in self.root.findall(f'.//index[@name="{dim_name}"]'):  # pyright: ignore[reportOptionalMemberAccess]
            definition_elem = index.find('definition')
            description_elem = index.find('description')

            if definition_elem is None or not definition_elem.text:
                continue

            definition_text = definition_elem.text.strip()

            # Parse the definition to get list of values
            values = None

            # Handle list format: [value1,value2,...]
            if definition_text.startswith('['):
                # Remove brackets and split
                content = definition_text[1:-1]
                # Handle both numeric and text values
                values = [v.strip().strip("'\"") for v in content.split(',')]

            # Handle sequence format: sequence(start,end,step)
            else:
                seq_match = re.search(r'sequence\((.*?)\)', definition_text)
                if seq_match:
                    params = [float(x.strip()) for x in seq_match.group(1).split(',')]
                    start, end, step = params[0], params[1], params[2]
                    values = [str(v) for v in np.arange(start, end + step/2, step)]

            if values:
                result = AnalyticaDimension(
                    name=dim_name,
                    definition=values,
                    description=description_elem.text if description_elem is not None else None,
                )
                return result

        return None


    def find_all_dimensions(self):
        """Find all dimension definitions and store them."""
        dimensions = {}
        for index in self.root.findall('.//index'):  # pyright: ignore[reportOptionalMemberAccess]
            name = index.get('name')
            if name:
                dim = self.find_dimension_definition(name)
                if dim:
                    dimensions[name] = dim
        dimensions['Zone'] = AnalyticaDimension(
            name='Zone',
            definition=['Downtown','Centre','Suburb'],
            description='Urban zones'
        )
        return dimensions

    # Parse data values - all are floats
    def parse_value(self, v: str):
        v = v.strip()
        if not v or v == '':
            return 0.0
        # Handle K suffix (thousands)
        if v.endswith('K'):
            return float(v[:-1]) * 1000.0
        if v.endswith('M'):
            return float(v[:-1]) * 1000000.0
        if v.upper() in ["NAN", "'NAN'"]:
            return np.nan
        return float(v)

    def extract_table_data(self, table_name):
        """Extract table data and parse it into structured format."""
        # Find the variable/table
        for variable in self.root.findall(f'.//variable[@name="{table_name}"]'):  # pyright: ignore[reportOptionalMemberAccess]
            definition = variable.find('definition')

            if definition is None or not definition.text:
                continue

            # Extract Table(dimensions)(data)
            match = re.search(r'Table\((.*?)\)\((.*)\)', definition.text, re.DOTALL)
            if not match:
                continue

            dimensions_str = match.group(1)
            data_str = match.group(2)

            # Parse dimension names
            dimension_names = [d.strip() for d in dimensions_str.split(',')]

            # Split by comma and parse
            data_values = [self.parse_value(v) for v in data_str.split(',')]

            # Get dimension info from stored dimensions
            table_dimensions = []
            dim_sizes = []

            for dim_name in dimension_names:
                if dim_name in self.dimensions:
                    dim = self.dimensions[dim_name]
                    table_dimensions.append(dim)
                    dim_sizes.append(len(dim.definition))
                else:
                    print(f"Warning: Dimension '{dim_name}' not found in stored dimensions")
                    return None

            result = AnalyticaTable(
                name=table_name,
                dimension_names=dimension_names,
                dimensions=table_dimensions,
                dimension_sizes=dim_sizes,
                data_flat=data_values,
                data_length=len(data_values)
            )

            return result

        return None


    def table_to_polars(self, table_name):
        """
        Convert table to polars DataFrame (flattened with dimension columns).

        Uses C-order: rightmost dimension varies fastest.
        """
        table_data = self.extract_table_data(table_name)

        if not table_data:
            print(f"Could not extract table data for {table_name}")
            return None

        # Reshape using C-order (rightmost varies fastest)
        shape = table_data.dimension_sizes
        try:
            data_array = np.array(table_data.data_flat, dtype=float).reshape(shape, order='C')
        except Exception as e:
            print(f"Error reshaping: {e}")
            print(f"Expected elements: {np.prod(shape)}, Got: {len(table_data.data_flat)}")
            return None

        # Generate all index combinations (C-order: rightmost varies fastest)
        import itertools
        indices_lists = [dim.definition for dim in table_data.dimensions]
        combinations = list(itertools.product(*indices_lists))

        # Create polars DataFrame
        df_dict = {}
        for i, dim_name in enumerate(table_data.dimension_names):
            df_dict[dim_name] = [combo[i] for combo in combinations]

        df_dict['Value'] = data_array.flatten(order='C').tolist()

        df = pl.DataFrame(df_dict)

        return df


if __name__ == "__main__":
    # Collect important data from an Analytica XML file
    file = '../../Downloads/12889_2005_278_MOESM1_ESM(1).xml'

    # Initialize parser
    parser = AnalyzerXMLParser(file)
    table_name = 'Scen1_0'

    # Extract Scen1_0 table
    print(f"Extracting {table_name} table data...")
    print("=" * 70)

    # Get table info
    table_data = parser.extract_table_data(table_name)

    if table_data:
        print(f"Table: {table_data.name}")
        print(f"Dimensions: {table_data.dimensions}")
        print(f"Dimension sizes: {table_data.dimension_sizes}")
        print(f"Total data points: {table_data.data_length}")
        print(f"Expected points: {np.prod([s for s in table_data.dimension_sizes if s])}")

    # Convert to polars DataFrame
    print("\n\nConverting to Polars DataFrame...")

    df = parser.table_to_polars(table_name)

    if df is not None:
        print(f"\nDataFrame shape: {df.shape}")

        out = f'./model-outputs/{table_name}_data.csv'
        df.write_csv(out)
        print(f"\nData saved to '{out}'.")

    print(df)
