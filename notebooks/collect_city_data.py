from __future__ import annotations

import argparse
import asyncio
import itertools
import os
from pathlib import Path

import aiohttp
import polars as pl
import yaml
from dotenv import load_dotenv

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Collect city data from GraphQL API')
    parser.add_argument('--input', required=True, help="""
        YAML file containing instances, nodes, and postprocessor function (optional)""")
    parser.add_argument('--output', required=True, help='Output CSV file name')
    return parser.parse_args()

def read_config(yaml_file):
    config = yaml.safe_load(Path(yaml_file).open('r'))  # noqa: SIM115
    return config

async def fetch_node_data(session, instance_id, node_id):
    print(f"Processing data for {instance_id}, node {node_id}")
    # url = "http://localhost:8000/v1/graphql/"
    url = "https://api.paths.kausal.dev/v1/graphql/"

    session_token = os.getenv('AUTHJS_SESSION_TOKEN')
    csrf_token = os.getenv('AUTHJS_CSRF_TOKEN')
    csrf_token_django = os.getenv('CSRFTOKEN')

    query = """
    query GetNodeValues($nodeId: ID!, $instanceId: ID!) @locale(lang: "en")
        @instance(identifier: $instanceId) {
        instance {
            referenceYear
            targetYear
        }
        node(id: $nodeId) {
            id
            unit {
                short
            }
            metricDim {
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

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:134.0) Gecko/20100101 Firefox/134.0',
        'Accept': 'application/json, multipart/mixed',
        'Accept-Language': 'en-US,en;q=0.5',
        'content-type': 'application/json',
        'Origin': 'https://api.paths.kausal.dev',
        'Referer': 'https://api.paths.kausal.dev/v1/graphql/',
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

async def fetch_all_instances(instances, node_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_node_data(session, instance, node_id)
            for instance in instances
            for node_id in node_ids
        ]

        results = await asyncio.gather(*tasks)

    return dict(zip(
        [(instance, node_id) for instance in instances for node_id in node_ids],
        results,
        strict=False
    ))

def create_dataframe(data, processor):
    if data is None:
        print("No data received")
        return None
    print(f"Received data with {len(data['data']['node']['metricDim']['values'])} values")

    dims = data['data']['node']['metricDim']['dimensions']
    years = data['data']['node']['metricDim']['years']
    values = data['data']['node']['metricDim']['values']
    unit = data['data']['node']['unit']['short']
    reference_year = data['data']['instance']['referenceYear']
    target_year = data['data']['instance']['targetYear']

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
                    **{f"dim_{i+1}": combo[i] for i in range(len(dims))},  # Using dim_1, dim_2, etc.
                    'year': year,
                    'value': values[value_index],
                    'unit': unit,
                }
                rows.append(row)
                value_index += 1

    df = pl.DataFrame(rows)
    df = postprocess_data[processor](df, reference_year, target_year)

    return df

def emission_targets(df, reference_year, target_year):
    df = (df.filter(pl.col('year').is_in([reference_year, target_year]))
          .group_by(pl.col(['year', 'unit'])).agg(pl.col('value').sum()))
    return df

def no_processing(df, reference_year, target_year):
    return df

postprocess_data = {
    'emission_targets': emission_targets,
    'none': no_processing,
}

async def main():
    args = parse_args()
    config = read_config(args.input)
    processor = config.get('processor', 'none')
    instances = config['instances']
    node_ids = config['node_ids']
    output_file = args.output

    results = await fetch_all_instances(instances, node_ids)
    dfs = []  # List to store DataFrames from each instance

    for (instance, node_id), data in results.items():
        # Create DataFrame from the instance data
        instance_df = create_dataframe(data, processor)
        # Add instance column
        if instance_df is not None:
            instance_df = instance_df.with_columns([
                pl.lit(instance).alias('instance'),
                pl.lit(node_id).alias('node')
            ])
            dfs.append(instance_df)

    # Concatenate all DataFrames vertically
    if dfs:
        try:
            final_df = pl.concat(dfs, how='vertical')
            print(final_df)
            final_df.write_csv(output_file)
            return final_df  # noqa: TRY300
        except Exception as e:
            print(f"Error creating final DataFrame: {e}")
            return None
    print("No data to process")
    return None

if __name__ == "__main__":
    asyncio.run(main())
