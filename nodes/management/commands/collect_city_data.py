from __future__ import annotations

import asyncio
import itertools
import os
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

import aiohttp
import polars as pl
import yaml
from dotenv import load_dotenv

from common import polars as ppl
from nodes.constants import VALUE_COLUMN, YEAR_COLUMN
from nodes.units import unit_registry

load_dotenv()

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
    # Limit concurrent connections
    connector = aiohttp.TCPConnector(
        limit=5,  # Total connection pool size
        limit_per_host=2,  # Max connections per host
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True,
    )

    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
    ) as session:

        # Process in smaller batches instead of all at once
        batch_size = 5  # Process 5 requests at a time
        all_results = {}

        combinations = [(instance, node_id) for instance in instances for node_id in node_ids]

        for i in range(0, len(combinations), batch_size):
            batch = combinations[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} of {len(combinations)//batch_size + 1}")

            tasks = [
                fetch_node_data_with_retry(session, url, instance, node_id)
                for instance, node_id in batch
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for (instance, node_id), result in zip(batch, batch_results, strict=False):
                all_results[(instance, node_id)] = result if not isinstance(result, Exception) else None

            # Small delay between batches
            await asyncio.sleep(1)

        return all_results

async def fetch_node_data_with_retry(session, url, instance_id, node_id, max_retries=3):
    """Fetch node data with retry logic."""
    for attempt in range(max_retries):
        try:
            result = await fetch_node_data(session, url, instance_id, node_id)
            return result  # noqa: TRY300
        except (TimeoutError, aiohttp.ClientError) as e:
            print(f"Attempt {attempt + 1} failed for {instance_id}/{node_id}: {e}")
            if attempt == max_retries - 1:
                print(f"All retries failed for {instance_id}/{node_id}")
                return None
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    return None

def validate_data_structure(data, node_str) -> bool:
    if data is None:
        print("No data received")
        return False

    # Check for missing key OR None value
    if data.get('data') is None:
        print(f"Missing or null 'data' key in response for {node_str}")
        return False

    # Chain the gets safely
    node_data = data.get('data', {}).get('node')
    if node_data is None:
        print(f"Missing or null 'node' key for {node_str}")
        return False

    metric_dim = node_data.get('metricDim')
    if metric_dim is None:
        print(f"Missing or null 'metricDim' for {node_str}")
        return False

    # Check for empty values list
    if not metric_dim.get('values'):  # This catches None, missing key, and empty list
        print(f"No values found for {node_str}")
        return False
    return True

def create_dataframe(data, processor, node_str, target_unit):
    if data is None:
        print("No data received")
        return None

    if not validate_data_structure(data, node_str):
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
        'ktCO₂e': 'kt/a',
        '1.0 ktCO₂e/a': 'kt/a',
        'ktCO₂e/yr': 'kt/a',
        'ktCO₂e/v': 'kt/a',
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

class Command(BaseCommand):
    help = 'Collect city data from GraphQL API and save to CSV files'

    def add_arguments(self, parser):
        parser.add_argument(
            '--input',
            required=True,
            help='YAML file containing instances, nodes with target units, and postprocessor function (optional)'
        )
        parser.add_argument(
            '--output',
            required=True,
            help='Base name for output CSV files'
        )

    def handle(self, *args, **options):
        """
        Collect data from several Kausal Paths instances.

        Typical command:
        python manage.py collect_city_data
            --input ../netzeroplanner-framework-config/emission_potential.yaml
            --output notebooks/emission_data
        """
        input_file = options['input']
        output_base = options['output']

        try:
            config = read_config(input_file)
            processor = config.get('processor', 'none')
            instances = config['instances']
            node_ids = [node['id'] for node in config['nodes']]
            url = config['url']

            self.stdout.write(f"Processing {len(instances)} instances with {len(node_ids)} nodes")

            # Run the async function
            results = asyncio.run(self._fetch_and_process_data(url, instances, node_ids, config, processor, output_base))

            if results:
                self.stdout.write(
                    self.style.SUCCESS('Successfully collected and saved city data')
                )
            else:
                self.stdout.write(
                    self.style.WARNING('Data collection completed with some issues')
                )

        except Exception as e:
            raise CommandError(f'Error during data collection: {e}') from e

    async def _fetch_and_process_data(self, url, instances, node_ids, config, processor, output_base) -> bool:
        """Async method containing the main data processing logic."""
        results = await fetch_all_instances(url, instances, node_ids)

        # Create a dictionary to store DataFrames by node
        node_dfs: dict[str, list[ppl.PathsDataFrame]] = {}

        for (instance, node_id), data in results.items():
            # Find the target unit for this node
            target_unit = next((node['target_unit'] for node in config['nodes'] if node['id'] == node_id), None)

            if data is None:
                self.stdout.write(f"    WARNING: Data cannot be collected for {node_id} in instance {instance}.")
                continue
            if data['data']['node'] is None:
                self.stdout.write(f"    WARNING: Node {node_id} does not exist in instance {instance}.")
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
                    self.stdout.write(f"Saving node {node_id} to {output_file}")
                    final_node_df.write_csv(output_file)

                except Exception as e:
                    self.stdout.write(f"Error processing node {node_id}: {e}")
            else:
                self.stdout.write(f"No data for node {node_id}")

        return True
