from __future__ import annotations

import argparse
import asyncio
import itertools
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: PTH100, PTH120
import aiohttp
import polars as pl
import yaml
from dotenv import load_dotenv

from common import polars as ppl
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Collect city data from GraphQL API')
    parser.add_argument('--input', required=True, help="""
        YAML file containing instances, nodes with target units, and postprocessor function (optional)""")
    parser.add_argument('--output', required=True, help='Base name for output CSV files')
    return parser.parse_args()

def read_config(yaml_file):
    config = yaml.safe_load(Path(yaml_file).open('r'))  # noqa: SIM115
    return config

async def fetch_node_data(session, url, instance_id, node_id):
    print(f"Processing data for {instance_id}, node {node_id}")

    session_token = os.getenv('AUTHJS_SESSION_TOKEN')
    csrf_token = os.getenv('AUTHJS_CSRF_TOKEN')
    csrf_token_django = os.getenv('CSRFTOKEN')

    query = """
    query GetNodeValues($nodeId: ID!, $instanceId: ID!)
        @instance(identifier: $instanceId) {
        instance {
            referenceYear
            targetYear
        }
        node(id: $nodeId) {
            id
            unit {
                short
                # standard # TODO After these changes have been deployed
            }
            metricDim {
                forecastFrom
                dimensions {
                    originalId
                    kind
                    categories {
                        originalId
                    }
                }
                years
                values
            }
        }
    }
    """

    base_url = '/'.join(url.split('/')[:-2])  # Get base URL without /v1/graphql/
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:134.0) Gecko/20100101 Firefox/134.0',
        'Accept': 'application/json, multipart/mixed',
        'Accept-Language': 'en-US,en;q=0.5',
        'content-type': 'application/json',
        'Origin': base_url,
        'Referer': url,
        'Connection': 'keep-alive',
        'Cookie': f'csrftoken={csrf_token_django}; authjs.session-token={session_token}; authjs.csrf-token={csrf_token}',
        'X-CSRFToken': f'{csrf_token}'
    }

    payload = {
        'query': query,
        'variables': {
            'nodeId': node_id,
            'instanceId': instance_id,
        },
        'operationName': 'GetNodeValues'
    }

    async with session.post(
        url,
        json=payload,
        headers=headers
    ) as response:
        if response.status != 200:
            print(f"Error {response.status}")
            print(await response.text())
            return None
        return await response.json()

async def fetch_all_instances(url, instances, node_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_node_data(session, url, instance, node_id)
            for instance in instances
            for node_id in node_ids
        ]

        results = await asyncio.gather(*tasks)

    return dict(zip(
        [(instance, node_id) for instance in instances for node_id in node_ids],
        results,
        strict=False
    ))

def create_dataframe(data, processor, node_str, target_unit):
    if data is None:
        print("No data received")
        return None

    print(f"Node {node_str} has {len(data['data']['node']['metricDim']['values'])} values")

    dims = data['data']['node']['metricDim']['dimensions']
    years = data['data']['node']['metricDim']['years']
    values = data['data']['node']['metricDim']['values']
    forecast_from = data['data']['node']['metricDim']['forecastFrom']
    unit_str = data['data']['node']['unit']['short']
    reference_year = data['data']['instance']['referenceYear']
    target_year = data['data']['instance']['targetYear']

    map_units = { # Needed until the standard unit is available
        '1.0 kt/v': '1.0 kt/a',
        'kt/v': 'kt/a',
        'Einw.': 'cap',
        'as.': 'cap',
        'tCO₂e/yr': 't/a',
    }
    unit_str = map_units.get(unit_str, unit_str)
    unit = unit_registry(unit_str)

    if forecast_from is None:
        forecast_from = max(years) if years else None  # Only historical values
    if forecast_from is not None:
        forecast_from -= 1
    if reference_year is not None:
        forecast_from = reference_year

    dim_names = [dim['originalId'] for dim in dims]
    # Create lists of categories for each dimension
    dim_categories = [
        [cat['originalId'] for cat in dim['categories']]
        for dim in dims
    ]

    # Create all combinations of dimension categories
    combinations = list(itertools.product(*dim_categories))

    # Create rows for DataFrame
    rows = []
    value_index = 0

    for combo in combinations:
        for year in years:
            if value_index < len(values):
                row = {
                    **{dim_names[i]: combo[i] for i in range(len(dims))},  # Using actual dimension names
                    YEAR_COLUMN: year,
                    VALUE_COLUMN: values[value_index],
                }
                rows.append(row)
                value_index += 1

    meta = ppl.DataFrameMeta(
        units={VALUE_COLUMN: unit},
        primary_keys=dim_names + [YEAR_COLUMN]
        )
    df = ppl.to_ppdf(pl.DataFrame(rows), meta)
    df = df.ensure_unit(VALUE_COLUMN, target_unit)
    df = postprocess_data[processor](df, forecast_from, target_year)

    return df

def emission_targets(df, reference_year, target_year):
    meta = df.get_meta()
    df = (
        df.filter(pl.col(YEAR_COLUMN).is_in([reference_year, target_year]))
        .group_by(pl.col([YEAR_COLUMN])).agg(pl.col(VALUE_COLUMN).sum())
        .sort(by=[YEAR_COLUMN])
    )
    df = df.with_columns(
            pl.when(pl.col(YEAR_COLUMN) == reference_year)
                    .then(pl.lit('newest'))
                    .otherwise(pl.lit('target'))
                    .alias('param')
        )
    df = ppl.to_ppdf(df, meta)
    return df

def no_processing(df, reference_year, target_year):
    return df

postprocess_data = {
    'emission_targets': emission_targets,
    'none': no_processing,
}

async def main():
    """
    Collect data from several Kausal Paths instances.

    Typical command:
    python ./notebooks/collect_city_data.py
        --input ../netzeroplanner-framework-config/emission_potential.yaml
        --output notebooks/emission_data
    """
    args = parse_args()
    config = read_config(args.input)
    processor = config.get('processor', 'none')
    instances = config['instances']
    node_ids = [node['id'] for node in config['nodes']]
    url = config['url']
    output_base = args.output

    results = await fetch_all_instances(url, instances, node_ids)

    # Create a dictionary to store DataFrames by node
    node_dfs: dict[str, list[ppl.PathsDataFrame]] = {}

    for (instance, node_id), data in results.items():
        # Find the target unit for this node
        target_unit = next((node['target_unit'] for node in config['nodes'] if node['id'] == node_id), None)

        if data is None:
            print(f"    WARNING: Data cannot be collected for {node_id} in instance {instance}.")
            continue
        if data['data']['node'] is None:
            print(f"    WARNING: Node {node_id} does not exist in instance {instance}.")
            continue

        instance_df = create_dataframe(data, processor, f'{instance} {node_id}', target_unit)

        # Add instance column
        if instance_df is not None:
            # Add instance and node columns
            instance_df = instance_df.with_columns([
                pl.lit(instance).alias('instance'),
                pl.lit(node_id).alias('node'),
                pl.lit(target_unit).alias('unit')
            ])

            # Add to the node_dfs dictionary
            if node_id not in node_dfs:
                node_dfs[node_id] = []
            node_dfs[node_id].append(instance_df)

    # Process each node separately
    for node_id, dfs in node_dfs.items():
        if dfs:
            try:
                # Concatenate all instances for this node
                node_df = pl.concat(dfs, how='vertical')

                # Create summary rows for this node (sum across all instances)
                summary_rows = (
                    node_df
                    .group_by([col for col in node_df.columns if col not in ['instance', VALUE_COLUMN, YEAR_COLUMN]]
                    )
                    .agg([pl.col(VALUE_COLUMN).sum().alias(VALUE_COLUMN),
                        pl.lit(None).alias(YEAR_COLUMN)])
                    .with_columns(pl.lit('ALL').alias('instance'))
                ).select(node_df.columns)

                final_node_df = pl.concat([node_df, summary_rows], how='vertical')

                # Save to a node-specific CSV file
                output_file = f"{output_base}_{node_id.replace('.', '_')}.csv"
                print(f"Saving node {node_id} to {output_file}")
                final_node_df.write_csv(output_file)

            except Exception as e:
                print(f"Error processing node {node_id}: {e}")
        else:
            print(f"No data for node {node_id}")

    return True

if __name__ == "__main__":
    asyncio.run(main())
