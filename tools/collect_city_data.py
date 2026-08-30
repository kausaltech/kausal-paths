# ruff: noqa: INP001  # tools/ is an implicit namespace package by design; run with `-m`.
# A previous version collected the data directly from the production instances via API.
# https://github.com/kausaltech/kausal-paths/blob/6471f1c860aa86e177290f80bced9435113e4ea6/nodes/management/commands/collect_city_data.py

# ruff: noqa: F401
from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import django
from django.conf import settings  # pyright: ignore[reportUnusedImport]

# import altair as alt
import yaml

# from great_tables import GT

# # Allow Django to run in async environments (like Jupyter)
# os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

# Configure Django
django.setup()

import polars as pl  # noqa: E402

from common import polars as ppl  # noqa: E402
from common.polars import DataFrameMeta  # noqa: E402
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN  # noqa: E402
from nodes.exceptions import NodeError  # noqa: E402
from nodes.units import unit_registry  # noqa: E402
from tools.instance_support import get_context, get_nodes  # noqa: E402

# initialize_notebook_env()

if TYPE_CHECKING:
    from datetime import date

    from common.polars import PathsDataFrame
    from nodes.context import Context
    from nodes.units import Quantity

# config_file = '../netzeroplanner-framework-config/emission_potential.yaml'


SUMMARY_ID = 'sum_over_instances'
"""Id of the cross-instance summary; also the prefix of every saved CSV."""

COLLECT_ONLY_PROCESSORS = frozenset({'sum_over_dims', 'find_target_values', 'convert_to_target_units', 'sum_over_instances'})
"""Processors that read the database; --merge replaces them with `merge_regions`."""

SUMMARY_ROW_IDS = ('TOTAL', 'NUMBER')
"""Synthetic `Instance` values that `aggregate_instances` appends to a summary."""


def created_at_date(context: Context, instance_id: str) -> date:
    """
    Return the creation date on the instance's database row.

    `Instance.config` is None for an instance loaded from YAML that has no row in this
    database, and such an instance has no creation date to report. Raising here rather
    than reaching through the None gives `collect_instance`'s handler something to log --
    it skips the instance and names it -- instead of an AttributeError on NoneType.
    """
    config = context.instance.config
    if config is None:
        raise ValueError(f'Instance {instance_id} has no database row, so no creation date.')
    return config.created_at.date()


def aggregate_instances(df: PathsDataFrame, topic: str) -> PathsDataFrame:
    """Collapse the per-instance rows of a summary into one `topic` row per param."""

    return (
        df.paths
        .sum_over_dims(['Instance', YEAR_COLUMN])
        .with_columns([pl.lit(topic).alias('Instance'), pl.lit(0).alias(YEAR_COLUMN), pl.lit(0).alias('CreatedAt')])
        .add_to_index(['Instance', YEAR_COLUMN])
    )


@dataclass
class NodeData:
    """Individual node with its dataframe."""

    id: str
    df: ppl.PathsDataFrame


@dataclass
class InstanceData:
    """Instance containing multiple nodes."""

    id: str
    target_year: int
    created_at: date
    nodes: list[NodeData] = field(default_factory=list)

    def add_node(self, node_id: str, df: ppl.PathsDataFrame) -> NodeData:
        """Add a node to this instance."""

        node = NodeData(id=node_id, df=df)
        self.nodes.append(node)
        return node

    def get_node_df(self, node_id: str) -> ppl.PathsDataFrame | None:
        """Get a specific node df by id."""

        node = next((node for node in self.nodes if node.id == node_id), None)
        if node is None:
            return None
        return node.df

    def update_node_df(self, node_id: str, df: ppl.PathsDataFrame) -> InstanceData:
        node = next((node for node in self.nodes if node.id == node_id), None)
        assert node is not None
        node.df = df
        return self


@dataclass
class DataCollection:
    """Main container for all dc."""

    output_path: str
    output_date: str
    processors: list[str]
    logs: list[str]
    instances: list[InstanceData]
    summaries: list[InstanceData]
    target_units: dict[str, str]

    def add_instance(self, instance_id: str, target_year: int, created_at: date) -> InstanceData:
        """Add a new instance."""
        instance = InstanceData(id=instance_id, target_year=target_year, created_at=created_at)
        self.instances.append(instance)
        return instance

    def get_instance(self, instance_id: str) -> InstanceData | None:
        """Get a specific instance by id."""

        return next((inst for inst in self.instances if inst.id == instance_id), None)

    def read_config(self, yaml_file):
        config = yaml.safe_load(Path(yaml_file).open('r'))  # noqa: SIM115
        return config

    def find_target_values(self) -> DataCollection:
        for instance in self.instances:
            for node in instance.nodes:
                df: ppl.PathsDataFrame = node.df
                meta = df.get_meta()
                target_year = instance.target_year
                obs_year = df.filter(~pl.col(FORECAST_COLUMN))[YEAR_COLUMN].max()
                if obs_year is None:
                    obs_year = df[YEAR_COLUMN].min()
                    self.logs.append(f'No historical data found for node {instance.id}/{node.id}. Using first year {obs_year!r}.')
                df = df.filter(pl.col(YEAR_COLUMN).is_in([obs_year, target_year])).sort(by=[YEAR_COLUMN])
                df = df.with_columns(
                    pl.when(pl.col(YEAR_COLUMN) == obs_year).then(pl.lit('newest')).otherwise(pl.lit('target')).alias('param')
                )
                df = ppl.to_ppdf(df, meta).add_to_index('param')
                instance.update_node_df(node.id, df)
        return self

    def convert_to_target_units(self) -> DataCollection:
        multipliers: dict[str, Quantity] = {
            'kt_co2e/a': unit_registry.parse_expression('1 * kt/kt_co2e'),
        }
        for instance in self.instances:
            for node in instance.nodes:
                df: PathsDataFrame = node.df
                df_unit = df.get_meta().units[VALUE_COLUMN]
                for from_unit, to_unit in multipliers.items():
                    if df_unit.is_compatible_with(from_unit):
                        df = df.multiply_quantity(VALUE_COLUMN, to_unit)
                df = df.ensure_unit(VALUE_COLUMN, self.target_units[node.id])
                instance.update_node_df(node.id, df)
        return self

    def sum_over_dims(self) -> DataCollection:
        for instance in self.instances:
            for node in instance.nodes:
                df = node.df
                dropcols = [dim for dim in df.primary_keys if dim != YEAR_COLUMN]
                df = df.paths.sum_over_dims(dropcols)
                instance.update_node_df(node_id=node.id, df=df)
        return self

    def sum_over_instances(self) -> DataCollection:

        # node_ids = list({node.id for instance in dc.instances for node in instance.nodes})
        summary = InstanceData(id=SUMMARY_ID, target_year=0, created_at=datetime.now().date())  # noqa: DTZ005
        for instance in self.instances:
            for node in instance.nodes:
                df: PathsDataFrame = node.df
                df = df.with_columns([
                    pl.lit(instance.id).alias('Instance'),
                    pl.lit(instance.created_at).alias('CreatedAt'),
                ]).add_to_index('Instance')
                sum_df: PathsDataFrame | None = summary.get_node_df(node.id)
                if sum_df is None:
                    summary.add_node(node.id, df)
                elif set(sum_df.primary_keys) == set(df.primary_keys):
                    summary.update_node_df(node.id, sum_df.paths.concat_vertical(df))
                else:
                    print(df.head())
                    print(sum_df.head())
                    self.logs.append(
                        ''.join([
                            f'Node {node.id} has primary keys {df.primary_keys} in instance {instance.id}',
                            f' but expected {sum_df.primary_keys}. Ignore the node in sum.',
                        ])
                    )
        for node in summary.nodes:
            number = aggregate_instances(node.df.with_columns(pl.lit(1.0).alias(VALUE_COLUMN)), 'NUMBER')
            total = aggregate_instances(node.df, 'TOTAL')
            total = total.paths.concat_vertical(number)
            assert set(node.df.columns) == set(total.columns)
            total = total.select(node.df.columns)
            summary.update_node_df(node.id, node.df.paths.concat_vertical(total))
        self.summaries.append(summary)

        return self

    def round_summaries(self) -> DataCollection:
        self.logs.append('Rounding summaries to 8 decimal places.')
        for summary in self.summaries:
            for node in summary.nodes:
                node.df = node.df.with_columns(pl.col(VALUE_COLUMN).round(8))
        return self

    def calculate_difference(self) -> DataCollection:
        self.logs.append('Calculating difference between the newest and target values.')
        for summary in self.summaries:
            new_nodes: list[NodeData] = []
            for node in summary.nodes:
                df = node.df
                if 'param' not in df.columns:
                    continue
                diff_df = (
                    df
                    .filter(pl.col('param').is_in(['newest', 'target']))
                    .group_by(['Instance', 'CreatedAt'])
                    .agg([
                        pl.col(VALUE_COLUMN).filter(pl.col('param') == 'newest').max().alias('newest_value'),
                        pl.col(VALUE_COLUMN).filter(pl.col('param') == 'target').max().alias('target_value'),
                        pl.col(YEAR_COLUMN).filter(pl.col('param') == 'newest').max().alias('newest_year'),
                        pl.col(YEAR_COLUMN).filter(pl.col('param') == 'target').max().alias('target_year'),
                    ])
                    .with_columns([
                        (pl.col('target_value') - pl.col('newest_value')).alias(VALUE_COLUMN),
                        (pl.col('target_year') - pl.col('newest_year')).alias(YEAR_COLUMN),
                        pl.lit(value=True).alias(FORECAST_COLUMN),
                        pl.lit('difference').alias('param'),
                    ])
                    .select(df.columns)
                    .sort('CreatedAt')
                )
                diff_df = ppl.to_ppdf(diff_df, df.get_meta())
                new_nodes.append(NodeData(id=f'{node.id}_difference', df=diff_df))
            summary.nodes.extend(new_nodes)
        return self

    def summary_file(self, node_id: str, region: str | None, date: str) -> str:
        """
        Return the path of the summary CSV for one node.

        `region` tags the file so the three regional runs of the same config land
        side by side; a region-less run keeps the historical single-server name.
        """
        unit_id = node_id
        if unit_id not in self.target_units and unit_id.endswith('_difference'):
            unit_id = unit_id.removesuffix('_difference')
        unit = self.target_units[unit_id].replace('/', '-')
        tag = f'_{region}' if region else ''
        return f'{self.output_path}{SUMMARY_ID}_{node_id}_{unit}{tag}_{date}.csv'

    def report_log(self, file_name: str) -> None:
        date = str(datetime.now().strftime('%Y-%m-%d'))  # noqa: DTZ005
        tag = f'_{self.region}' if self.region else ''
        log_file = f'{self.output_path}log_{file_name}{tag}_{date}.txt'
        self.logs.append(f'Saving log file to {log_file}')
        out = ['During processing, the following things happened:']
        out.extend(self.logs)
        outtext = '\n'.join(out)
        with open(log_file, 'w') as f:  # noqa: PTH123
            f.write(outtext)
        print(outtext)

    def save_summaries(self) -> DataCollection:
        self.logs.append('Saving summaries about:')
        date = str(datetime.now().strftime('%Y-%m-%d'))  # noqa: DTZ005
        for summary in self.summaries:
            self.logs.append(f'- {summary.id}:')
            for node in summary.nodes:
                output_file = self.summary_file(node.id, self.region, date)
                node.df.write_csv(output_file)
                self.logs.append(f'  - Saved nodes {node.id} in {output_file}.')
        return self

    def db_identifiers(self) -> set[str]:
        """Return the identifiers held by the database this run is pointed at (i.e. one server)."""
        from nodes.models import InstanceConfig

        return set(InstanceConfig.objects.values_list('identifier', flat=True))

    def read_summary(self, node_id: str, region: str, date: str) -> ppl.PathsDataFrame | None:
        """
        Read one regional summary CSV back as a PathsDataFrame.

        Falls back to the newest file for that region when `date` has none, so a
        merge still works when the three restores happened on different days.
        """
        wanted = Path(self.summary_file(node_id, region, date))
        if not wanted.exists():
            pattern = Path(self.summary_file(node_id, region, '*')).name
            candidates = sorted(Path(self.output_path).glob(pattern))
            if not candidates:
                self.logs.append(f'No {node_id} summary found for region {region}. Skipping the region.')
                return None
            wanted = candidates[-1]
            self.logs.append(f'No {node_id} summary for {date} in region {region}; using {wanted.name} instead.')
        df = pl.read_csv(wanted, try_parse_dates=True)
        df = df.filter(~pl.col('Instance').is_in(SUMMARY_ROW_IDS))
        unit_id = node_id.removesuffix('_difference') if node_id not in self.target_units else node_id
        meta = DataFrameMeta(
            units={VALUE_COLUMN: unit_registry.parse_units(self.target_units[unit_id])},
            primary_keys=[YEAR_COLUMN, 'param', 'Instance'],
        )
        self.logs.append(f'  - Read {df.height} rows of {node_id} for region {region} from {wanted.name}.')
        return ppl.to_ppdf(df, meta)

    def merge_regions(self) -> DataCollection:
        """
        Combine the per-region summary CSVs into one, recomputing TOTAL and NUMBER.

        The regional files each carry their own TOTAL/NUMBER rows over their own
        subset of cities; those are dropped on read and re-derived here, so the
        merged file means the same thing a single-server run used to mean.
        """
        assert self.merge is not None
        date = self.merge_date or str(datetime.now().strftime('%Y-%m-%d'))  # noqa: DTZ005
        node_ids = self.node_ids

        summary = InstanceData(id=SUMMARY_ID, target_year=0, created_at=datetime.now().date())  # noqa: DTZ005
        for node_id in node_ids:
            self.logs.append(f'Merging {node_id}:')
            seen: dict[str, str] = {}
            merged: ppl.PathsDataFrame | None = None
            for region in self.merge:
                df = self.read_summary(node_id, region, date)
                if df is None:
                    continue
                duplicates = sorted({i for i in df['Instance'].unique() if i in seen})
                if duplicates:
                    self.logs.append(
                        f'  - {len(duplicates)} instance(s) already collected from region {seen[duplicates[0]]} '
                        f'also appear on {region}; keeping the first and NOT double-counting: {", ".join(duplicates)}'
                    )
                    df = df.filter(~pl.col('Instance').is_in(duplicates))
                seen.update(dict.fromkeys(df['Instance'].unique(), region))
                merged = df if merged is None else merged.paths.concat_vertical(df)
            if merged is None:
                self.logs.append(f'  - No data for {node_id} in any region. Skipping.')
                continue
            number = aggregate_instances(merged.with_columns(pl.lit(1.0).alias(VALUE_COLUMN)), 'NUMBER')
            total = aggregate_instances(merged, 'TOTAL')
            total = total.paths.concat_vertical(number).select(merged.columns)
            self.logs.append(f'  - Merged {len(seen)} instances from {len(self.merge)} region(s).')
            summary.add_node(node_id, merged.paths.concat_vertical(total))
        self.summaries.append(summary)
        return self

    def no_processing(self) -> DataCollection:
        return self

    def resolve_instances(self, config: dict, region: str | None) -> list[str]:
        """
        Pick the instance list for `region`, tolerating the old flat-list configs.

        `instances:` is either a plain list (one production server, historical
        shape) or a mapping of region -> list. The two are not interchangeable:
        asking for a region from a flat list, or omitting one for a region-keyed
        config, would silently collect the wrong set of cities.
        """
        instances = config['instances']
        if isinstance(instances, list):
            if region is not None:
                raise SystemExit(
                    f'--region {region} given, but {config.get("_config_file", "the config")} lists instances as a '
                    'flat list. Convert `instances:` to a region -> list mapping, or drop --region.'
                )
            self.regional = False
            return list(instances)

        self.regional = True
        known = sorted(instances)
        if region is None:
            raise SystemExit(f'This config lists instances per region ({", ".join(known)}). Pass --region.')
        if region not in instances:
            raise SystemExit(f'Unknown region {region!r}; the config knows {", ".join(known)}.')
        return list(instances[region] or [])

    def report_coverage(self, config: dict, region: str, listed: list[str], db_ids: set[str]) -> None:
        """
        Compare the region's list against the identifiers this database actually holds.

        This is what makes a multi-server run trustworthy. `get_context()` falls
        back to `configs/<id>.yaml` when an instance is missing from the database,
        so a city hosted on another server would otherwise be quietly computed
        from repo YAML in every regional run, and counted three times in the sum.
        """
        listed_here = set(listed)
        by_region: dict[str, set[str]] = {r: set(ids or []) for r, ids in config['instances'].items()}
        listed_anywhere = set().union(*by_region.values()) if by_region else set()

        missing = sorted(listed_here - db_ids)
        if missing:
            self.logs.append(
                f'{len(missing)} instance(s) listed under region {region} are absent from this database '
                f'and are skipped (moved, renamed or deleted?): {", ".join(missing)}'
            )
        misplaced = {other: sorted(db_ids & ids) for other, ids in by_region.items() if other != region and (db_ids & ids)}
        for other, ids in misplaced.items():
            self.logs.append(
                f'{len(ids)} instance(s) listed under region {other} are hosted on {region}. '
                f'Move them in the config: {", ".join(ids)}'
            )
        unlisted = sorted(db_ids - listed_anywhere)
        if unlisted:
            block = '\n'.join(f'  - {i}' for i in unlisted)
            self.logs.append(
                f'{len(unlisted)} instance(s) exist on {region} but are in no region list. '
                f'Check whether they are real cities, then add under `{region}:`\n{block}'
            )
        self.logs.append(
            f'Region {region}: {len(listed_here & db_ids)} of {len(listed_here)} listed instances '
            f'found in a database holding {len(db_ids)} instances.'
        )

    def __init__(self, config_file: str, region: str | None = None, merge: list[str] | None = None):
        config = self.read_config(config_file)
        config['_config_file'] = config_file
        output_path = config.get('output_path', '')
        original_instance_map: dict[str, str] = config.get('original_instance', {})
        output_date: str = str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # noqa: DTZ005

        self.output_path = output_path
        self.output_date = output_date
        self.region = region
        self.merge = merge
        self.merge_date: str | None = None
        self.regional = False
        self.instances = []
        self.summaries = []
        self.target_units = {node['id']: node['target_unit'] for node in config['nodes']}
        self.node_ids = [node['id'] for node in config['nodes']]
        self.collect_processors: list[str] = config.get('processors', [])
        self.logs = [f'Collect data from {config_file}.']

        if merge is not None:
            self.processors = ['merge_regions'] + [p for p in self.collect_processors if p not in COLLECT_ONLY_PROCESSORS]
            self.logs.append(f'Merging regions {", ".join(merge)}; no database is read.')
            return

        self.processors = self.collect_processors
        instances = self.resolve_instances(config, region)
        # instances = instances[0:10] # Used to simplify testing

        db_ids: set[str] | None = None
        if self.regional:
            assert region is not None
            db_ids = self.db_identifiers()
            self.report_coverage(config, region, instances, db_ids)

        for instance_id in instances:
            if db_ids is not None and instance_id not in db_ids:
                continue  # already reported by report_coverage; never fall back to repo YAML
            try:
                self.collect_instance(instance_id, original_instance_map, db_ids)
            except Exception:
                self.logs.append(f'Instance {instance_id} could not be collected and is skipped:')
                self.logs.append(f'    {traceback.format_exc().strip().splitlines()[-1]}')

    def creation_date(
        self, instance_id: str, context: Context, original_instance_map: dict[str, str], db_ids: set[str] | None
    ) -> date:
        """
        Return the date the instance was first created, following `original_instance` when set.

        The original may live on a different server than its successor, in which
        case its creation date is simply not reachable from this run.
        """
        original_id = original_instance_map.get(instance_id) or instance_id
        if original_id == instance_id:
            return created_at_date(context, instance_id)
        if db_ids is not None and original_id not in db_ids:
            self.logs.append(
                f'Original instance {original_id} for {instance_id} is not on this server. '
                + 'Using the current instance created_at.'
            )
            return created_at_date(context, instance_id)
        try:
            original_context = get_context(original_id)
        except FileNotFoundError:
            self.logs.append(
                f'Original instance {original_id} not found for {instance_id}. Using the current instance created_at.'
            )
            return created_at_date(context, instance_id)
        return created_at_date(original_context, original_id)

    def collect_instance(self, instance_id: str, original_instance_map: dict[str, str], db_ids: set[str] | None) -> None:
        """Compute every configured node for one instance and record the results."""
        try:
            context = get_context(instance_id)
        except FileNotFoundError:
            self.logs.append(f'Instance {instance_id} not found. Skipping.')
            return

        nodes = get_nodes(instance_id)
        created_at = self.creation_date(instance_id, context, original_instance_map, db_ids)
        instance = self.add_instance(instance_id=instance_id, target_year=context.target_year, created_at=created_at)
        for node_id in self.node_ids:
            node = nodes.get(node_id)
            if node is None:
                self.logs.append(f'Node {node_id} not found in instance {instance.id}.')
                continue
            try:
                df = node.get_output_pl()
                instance.add_node(node_id=node_id, df=df)
            except ValueError, NodeError:
                self.logs.append(f'Node {node_id} in instance {instance.id} gave an error and is skipped:')
                self.logs.append(f'    {traceback.format_exc().strip().splitlines()[-1]}')
                continue

    def process_data(self) -> DataCollection:

        PROCESS_DATA = {
            'convert_to_target_units': self.convert_to_target_units,
            'calculate_difference': self.calculate_difference,
            'find_target_values': self.find_target_values,
            'save_summaries': self.save_summaries,
            'sum_over_dims': self.sum_over_dims,
            'sum_over_instances': self.sum_over_instances,
            'merge_regions': self.merge_regions,
            'round_summaries': self.round_summaries,
            'none': self.no_processing,
        }
        dc = self
        for processor in dc.processors:
            if processor not in PROCESS_DATA.keys():
                dc.logs.append(f'Processor {processor} is not defined. Ignoring.')
                continue
            dc.logs.append(f'Processing {processor} ...')
            dc = PROCESS_DATA[processor]()
        return dc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Collect node outputs across instances. The models are computed locally, so one run sees exactly '
            'one production database: collect once per region, then combine with --merge.'
        ),
    )
    parser.add_argument('config_file', help='Collector config, e.g. ../scripts/paths/collectors/emission_potential.yaml')
    parser.add_argument(
        '--region',
        help='Region whose instance list to collect. Must match the database this run is pointed at (DATABASE_URL).',
    )
    parser.add_argument(
        '--merge',
        nargs='+',
        metavar='REGION',
        help='Combine already-saved regional summaries into one, instead of collecting. Reads no database.',
    )
    parser.add_argument(
        '--date',
        help='Date stamp of the regional summaries to merge (default: today). Only meaningful with --merge.',
    )
    args = parser.parse_args(argv)
    if args.region and args.merge:
        parser.error('--region collects from one database and --merge reads saved files; use one or the other.')
    if args.date and not args.merge:
        parser.error('--date only applies to --merge.')
    return args


def main():
    args = parse_args()
    config_file = args.config_file

    dc = DataCollection(config_file=config_file, region=args.region, merge=args.merge)
    dc.merge_date = args.date
    dc = dc.process_data()
    file_name = config_file.split('/')[-1].split('.')[0]

    dc.report_log(file_name)


if __name__ == '__main__':
    main()
