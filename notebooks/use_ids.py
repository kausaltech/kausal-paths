from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import yaml


def load_yaml_mappings(yaml_file_path: str | None = None) -> dict[str, dict[str, str]]:
    """Load YAML file and create label-to-ID mappings for dimensions."""
    if yaml_file_path is None:
        yaml_path = Path("../configs/frameworks/standard_dims.yaml")
    else:
        yaml_path = Path(yaml_file_path)

    with yaml_path.open(mode='r', encoding='utf-8') as file:
        data = yaml.safe_load(file)['dimensions']

    mappings: dict[str, dict[str, str]] = {}
    langs = ['', '_en', '_fi', '_de', '_sv']

    for dim in data:
        dim_id = dim.get('id')
        label_to_id = {}
        for lang in langs:
            label = 'label' + lang
            if label in dim:
                label_to_id[dim[label]] = dim_id

        # Process categories
        for category in dim.get('categories', []):
            cat_id = category.get('id')
            for lang in langs:
                label = 'label' + lang
                if label in category:
                    label_to_id[category[label]] = cat_id

            if 'aliases' in category:
                aliases = category['aliases']
                for alias in aliases:
                    label_to_id[alias] = cat_id

        mappings[dim_id] = label_to_id

    return mappings

def replace_labels_with_ids(df: pl.DataFrame, mappings: dict[str, dict[str, str]],
                               column_mappings: dict[str, str] | None = None) -> pl.DataFrame:
    """More comprehensive replacement using map_dict for better performance."""
    if column_mappings is None:
        dim_cols = [(dim, col) for dim in mappings.keys() for col in mappings[dim].keys() if col in df.columns]
    else:
        # dim_id = column_mappings.keys()
        raise KeyError("Don't define column_mappings. Functionality has not been developed.")

    df_updated = df.clone()

    for dim_id, col_name in dim_cols:
        print("Replacing labels with mappings on column:", col_name)
        # for label, id_val in mappings[dim_id].items():
        #     print(f"  '{label}' -> '{id_val}'")

        def _mapping_fun(value) -> str:
            label_map = mappings[dim_id]  # noqa: B023
            return label_map.get(value, value)

        # Use map_elements for direct mapping
        df_updated = df_updated.with_columns(
            pl.col(col_name).map_elements(_mapping_fun, return_dtype=pl.Utf8).alias(col_name)
        )
        df_updated = df_updated.rename({col_name: dim_id})

    return df_updated

def main():
    if len(sys.argv) != 3:
        print("Usage: python use_ids.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load your dataframe
    df = pl.read_csv(input_file)

    # Load mappings
    mappings = load_yaml_mappings()

    # Apply replacements
    df = replace_labels_with_ids(df, mappings)

    df.write_csv(output_file)
    print(f"Processed {input_file} -> {output_file}")

if __name__ == "__main__":
    main()
