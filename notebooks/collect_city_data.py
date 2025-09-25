# A previous version collected the data directly from the production instances via API.
# https://github.com/kausaltech/kausal-paths/blob/6471f1c860aa86e177290f80bced9435113e4ea6/nodes/management/commands/collect_city_data.py

# ruff: noqa: F401
from __future__ import annotations

# import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import django
from django.conf import settings  # pyright: ignore[reportUnusedImport]

# import altair as alt
import polars as pl
import yaml

# from great_tables import GT

# # Allow Django to run in async environments (like Jupyter)
# os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

# Set the Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'paths.settings')

# Configure Django
django.setup()

from common import polars as ppl  # noqa: E402
from nodes.constants import FORECAST_COLUMN, VALUE_COLUMN, YEAR_COLUMN  # noqa: E402
from nodes.exceptions import NodeComputationError  # noqa: E402
from nodes.units import Quantity, unit_registry  # noqa: E402
from notebooks.notebook_support import get_context, get_nodes  # noqa: E402

# initialize_notebook_env()

if TYPE_CHECKING:
    from common.polars import PathsDataFrame

# config_file = '../netzeroplanner-framework-config/emission_potential.yaml'

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

    def add_instance(self, instance_id: str, target_year: int) -> InstanceData:
        """Add a new instance."""
        instance = InstanceData(id=instance_id, target_year=target_year)
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
                df = (
                    df.filter(pl.col(YEAR_COLUMN).is_in([obs_year, target_year]))
                    .sort(by=[YEAR_COLUMN])
                )
                df = df.with_columns(
                    pl.when(pl.col(YEAR_COLUMN) == obs_year)
                    .then(pl.lit('newest'))
                    .otherwise(pl.lit('target'))
                    .alias('param')
                )
                df = ppl.to_ppdf(df, meta).add_to_index('param')
                instance.update_node_df(node.id, df)
        return self

    def convert_to_target_units(self) -> DataCollection:
        multipliers: dict[str, Quantity] = {
            'kt_co2e/a': unit_registry('1 * kt/kt_co2e'),
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
        summary = InstanceData(id='sum_over_instances', target_year=0)
        for instance in self.instances:
            for node in instance.nodes:
                df: PathsDataFrame = node.df
                df = (df
                    .with_columns(pl.lit(instance.id).alias('Instance'))
                    .add_to_index('Instance')
                )
                sum_df: PathsDataFrame | None = summary.get_node_df(node.id)
                if sum_df is None:
                    summary.add_node(node.id, df)
                elif set(sum_df.primary_keys) == set(df.primary_keys):
                    summary.update_node_df(node.id, sum_df.paths.concat_vertical(df))
                else:
                    print(df.head())
                    print(sum_df.head())
                    self.logs.append("".join([
                        f"Node {node.id} has primary keys {df.primary_keys} in instance {instance.id}",
                        f" but expected {sum_df.primary_keys}. Ignore the node in sum."]))
        for node in summary.nodes:
            total = node.df.paths.sum_over_dims(['Instance', YEAR_COLUMN])
            total = total.with_columns([
                pl.lit('TOTAL').alias('Instance'),
                pl.lit(0).alias(YEAR_COLUMN)]).add_to_index(['Instance', YEAR_COLUMN])
            assert set(node.df.columns) == set(total.columns)
            total = total.select(node.df.columns)
            summary.update_node_df(node.id, node.df.paths.concat_vertical(total))
        self.summaries.append(summary)

        return self

    def report_log(self) -> None:
        self.logs.append(f"Saving log file to {self.output_path}log.txt")
        out = ["During processing, the following things happened:"]
        out.extend(self.logs)
        outtext = '\n'.join(out)
        with open(f'{self.output_path}log.txt', 'w') as f:  # noqa: PTH123
            f.write(outtext)
        print(outtext)

    def save_summaries(self) -> DataCollection:
        self.logs.append("Saving summaries about:")
        output_path = self.output_path
        for summary in self.summaries:
            self.logs.append(f"- {summary.id}:")
            for node in summary.nodes:
                unit = self.target_units[node.id].replace('/', '-')
                output_file = f"{output_path}{summary.id}_{node.id}_{unit}.csv"
                node.df.write_csv(output_file)
                self.logs.append(f"  - Saved nodes {node.id} in {output_file}.")
        return self

    def no_processing(self) -> DataCollection:
        return self

    def __init__(self, config_file: str):
        config = self.read_config(config_file)
        processors = config.get('processors', [])
        output_path = config.get('output_path', '')
        output_date: str = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))  # noqa: DTZ005

        self.output_path = output_path
        self.output_date = output_date
        self.processors = processors
        self.instances = []
        self.summaries = []
        self.target_units = {node['id']: node['target_unit'] for node in config['nodes'] }
        self.logs = [f"Collect data from {config_file}."]

        instances = config['instances']
        # instances = instances[0:10] # Used to simplify testing
        node_ids = [node['id'] for node in config['nodes']]

        for instance_id in instances:
            try:
                context = get_context(instance_id)
            except FileNotFoundError:
                self.logs.append(f"Instance {instance_id} not found. Skipping.")
                continue

            nodes = get_nodes(instance_id)
            target_year = context.target_year
            instance = self.add_instance(instance_id=instance_id, target_year=target_year)
            for node_id in node_ids:
                node = nodes.get(node_id)
                if node is None:
                    self.logs.append(f"Node {node_id} not found in instance {instance.id}.")
                    continue
                try:
                    df = node.get_output_pl()
                    instance.add_node(node_id=node_id, df=df)
                except (ValueError, NodeComputationError):
                    self.logs.append(f"Node {node_id} in instance {instance.id} gave and error and is skipped.")
                    continue

    def process_data(self) -> DataCollection:

        PROCESS_DATA = {
            'convert_to_target_units': self.convert_to_target_units,
            'find_target_values': self.find_target_values,
            'save_summaries': self.save_summaries,
            'sum_over_dims': self.sum_over_dims,
            'sum_over_instances': self.sum_over_instances,
            'none': self.no_processing,
        }
        dc = self
        for processor in dc.processors:
            if processor not in PROCESS_DATA.keys():
                dc.logs.append(f"Processor {processor} is not defined. Ignoring.")
                continue
            dc.logs.append(f"Processing {processor} ...")
            dc = PROCESS_DATA[processor]()
        return dc


def main():

    config_file = sys.argv[1]

    dc = DataCollection(config_file=config_file)
    dc = dc.process_data()

    dc.report_log()

if __name__ == "__main__":
    main()
