from __future__ import annotations

import ctypes
import gc
import json
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from django.core.management.base import BaseCommand
from pydantic import BaseModel, Field, PrivateAttr

import loguru
import polars as pl
import psutil
from deepdiff import DeepDiff
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from rich import print
from rich.console import Console

from kausal_common.logging.errors import print_exception
from kausal_common.logging.warnings import register_warning_handler
from kausal_common.perf.perf_context import estimate_size_bytes

from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.datasets import JSONDataset
from nodes.exceptions import NodeError
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from django.core.management.base import CommandParser

    from nodes.actions.action import ActionNode
    from nodes.context import Context
    from nodes.instance import Instance
    from nodes.node import Node

console = Console()


def sort_schema(schema: dict[str, Any], ignore_dim_enums: bool = False):
    pks = schema['primaryKey']
    pks.sort()

    fields: list[dict[str, Any]] = schema['fields']
    fields.sort(key=lambda x: x['name'])
    for field in fields:
        constraints = field.get('constraints')
        if field['type'] == 'any' and 'constraints' in field and ignore_dim_enums:
            field['type'] = 'string'
            field['extDtype'] = 'str'
            field.pop('constraints', None)
            field.pop('ordered', None)
            continue
        if constraints and 'enum' in constraints:
            constraints['enum'].sort()
    return schema


def get_sorted_rows(table: dict[str, Any]):
    pks: list[str] = table['schema']['primaryKey']
    rows: list[dict[str, Any]] = table['data']
    rows.sort(key=lambda x: tuple([x[key] or '' for key in pks]))
    return rows


def dir_path(str_path: str) -> Path:
    path = Path(str_path)
    if not path.exists():
        path.mkdir(parents=True)
    return existing_dir(str_path)


def existing_dir(str_path: str) -> Path:
    path = Path(str_path)
    if path.is_dir():
        return path
    raise NotADirectoryError(str_path)


def file_path(str_path: str) -> Path:
    path = Path(str_path)
    if not path.parent.is_dir():
        raise NotADirectoryError(path.parent)
    if path.is_file() or not path.exists():
        return path
    raise ValueError(f'{str_path} is not a file')


type NodeFailReason = Literal['output', 'compare', 'check']


class NodeDetail(BaseModel):
    node_id: str
    failure_at: NodeFailReason | None = None
    check_time_ms: float = 0

    def mark_success(self, check_time_ms: float):
        self.failure_at = None
        self.check_time_ms = check_time_ms

    def mark_failed(self, reason: NodeFailReason):
        self.failure_at = reason


type InstanceFailReason = Literal['init', 'nodes']
type ScenarioId = Literal['default', 'baseline']


class InstanceDetail(BaseModel):
    instance_id: str
    failure_at: InstanceFailReason | None = None
    nodes: list[NodeDetail] = Field(default_factory=list)
    baseline_nodes: list[NodeDetail] = Field(default_factory=list)

    _active_scenario_id: ScenarioId = PrivateAttr(default='default')

    def add_node(self, node: Node, fail_reason: NodeFailReason | None, check_time_ms: float):
        nodes = self.baseline_nodes if self._active_scenario_id == 'baseline' else self.nodes
        for details in nodes:
            if details.node_id == node.id:
                break
        else:
            details = NodeDetail(node_id=node.id)
            nodes.append(details)
        if not fail_reason:
            details.mark_success(check_time_ms)
        else:
            details.mark_failed(fail_reason)
        details.check_time_ms = check_time_ms

    def get_node_details(self, node: Node) -> NodeDetail | None:
        nodes = self.baseline_nodes if self._active_scenario_id == 'baseline' else self.nodes
        for details in nodes:
            if details.node_id == node.id:
                return details
        return None

    def set_active_scenario_id(self, scenario_id: ScenarioId):
        assert scenario_id in ['default', 'baseline']
        self._active_scenario_id = scenario_id


class CheckState(BaseModel):
    checked_instances: set[str] = Field(default_factory=set)
    failed_instances: set[str] = Field(default_factory=set)
    instance_details: list[InstanceDetail] = Field(default_factory=list)

    _compare_mode: bool = PrivateAttr(default=False)
    _output_file: Path | None = PrivateAttr(default=None)

    def add_node(self, node: Node, fail_reason: NodeFailReason | None, check_time_ms: float):
        for details in self.instance_details:
            if details.instance_id == node.context.instance.id:
                break
        else:
            details = InstanceDetail(instance_id=node.context.instance.id)
            self.instance_details.append(details)
        details.add_node(node, fail_reason, check_time_ms)

    def get_node_details(self, node: Node) -> NodeDetail | None:
        details = self.get_details_for_instance(node.context.instance.id)
        if details:
            return details.get_node_details(node)
        return None

    def add_instance(self, instance_id: str) -> InstanceDetail:
        details = self.get_details_for_instance(instance_id)
        if not details:
            details = InstanceDetail(instance_id=instance_id)
            self.instance_details.append(details)
        return details

    def get_details_for_instance(self, instance_id: str) -> InstanceDetail | None:
        for details in self.instance_details:
            if details.instance_id == instance_id:
                return details
        details = InstanceDetail(instance_id=instance_id)
        self.instance_details.append(details)
        return details

    def has_instance(self, instance_id: str) -> bool:
        return any(details.instance_id == instance_id for details in self.instance_details)

    def mark_failed(self, instance_id: str, reason: InstanceFailReason):
        self.checked_instances.add(instance_id)
        self.failed_instances.add(instance_id)
        details = self.add_instance(instance_id)
        details.failure_at = reason

    def mark_success(self, instance: Instance):
        self.checked_instances.add(instance.id)
        self.failed_instances.discard(instance.id)
        details = self.add_instance(instance.id)
        details.failure_at = None

    def set_output_file(self, output_file: Path):
        self._output_file = output_file

    def set_compare_mode(self):
        self._compare_mode = True

    def save(self):
        if not self._output_file or self._compare_mode:
            return
        with self._output_file.open('w') as f:
            f.write(self.model_dump_json(indent=2))


class Command(BaseCommand):
    help = 'Validate computation models and store/compare results'

    store: bool
    compare: bool
    spec_only: bool
    dry_run: bool
    logger: loguru.Logger
    state: CheckState
    maxfail: int
    nr_fails: int = 0
    state_dir: Path | None
    state_file: Path | None
    check_perf: bool
    model_output_dir: Path | None
    action_impact_output_dir: Path | None
    trace_rss: bool
    gc_after_instance: bool
    malloc_trim_after_instance: bool
    process: psutil.Process
    rss_start_bytes: int | None = None
    rss_prev_bytes: int | None = None
    malloc_trim: Any | None = None
    all_nodes: bool
    graph_dir: Path | None
    manifest_dir: Path | None
    trace_object_graph: bool
    trace_object_limit: int
    trace_tracemalloc: bool
    trace_tracemalloc_limit: int
    trace_new_objects: bool
    trace_new_object_limit: int
    limit: int

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('instances', metavar='INSTANCE_ID', type=str, nargs='*')
        parser.add_argument('--skip', dest='skip', metavar='INSTANCE_ID', action='append')
        parser.add_argument(
            '--store',
            dest='store',
            action='store_true',
            help='Store outputs to state directory for later comparison',
        )
        parser.add_argument(
            '--compare',
            dest='compare',
            action='store_true',
            help='Compare outputs to previous run for each instance',
            default=None,
        )
        parser.add_argument(
            '--spec-only',
            dest='spec_only',
            action='store_true',
            help='Only initialize each instance; skip output comparison and node execution',
        )
        parser.add_argument(
            '--dry-run',
            dest='dry_run',
            action='store_true',
            help='Do not update the state file',
        )
        parser.add_argument(
            '--start-from', dest='start_from', metavar='INSTANCE_ID', action='store', help='Instance ID to start from'
        )
        parser.add_argument('--only', dest='only', metavar='INSTANCE_ID', action='store', help='Check only the given instance')
        parser.add_argument(
            '--maxfail',
            dest='maxfail',
            metavar='NUM',
            type=int,
            action='store',
            default=1,
            help='Maximum number of failures before stopping (0 means no limit)',
        )
        parser.add_argument(
            '--state-dir',
            dest='state_dir',
            metavar='DIR',
            type=dir_path,
            action='store',
            default='./model-outputs',
            help='Directory for test state files',
        )
        parser.add_argument(
            '--state-resume', dest='state_resume', action='store_true', help='Resume after the last checked instance'
        )
        parser.add_argument(
            '--check-perf', action='store_true', default=False, help='Check node computation times for performance regressions'
        )
        parser.add_argument(
            '--trace-rss',
            action='store_true',
            default=False,
            help='Print process RSS before and after each instance run',
        )
        parser.add_argument(
            '--gc-after-instance',
            action='store_true',
            default=False,
            help='Force garbage collection after each instance and print the post-GC RSS in --trace-rss mode',
        )
        parser.add_argument(
            '--malloc-trim-after-instance',
            action='store_true',
            default=False,
            help='Call malloc_trim(0) after each instance and print the post-trim RSS in --trace-rss mode',
        )
        parser.add_argument(
            '--all-nodes',
            action='store_true',
            default=False,
            help='Store/compare outputs for all nodes instead of only outcome nodes',
        )
        parser.add_argument(
            '--trace-object-graph',
            action='store_true',
            default=False,
            help='After each instance, force GC and log surviving Instance/Context/Node/Dataset/PathsDataFrame objects',
        )
        parser.add_argument(
            '--trace-object-limit',
            type=int,
            default=8,
            help='Maximum number of retained datasets/dataframes to print in --trace-object-graph mode',
        )
        parser.add_argument(
            '--trace-tracemalloc',
            action='store_true',
            default=False,
            help='Take tracemalloc snapshots before and after each instance and log the largest Python allocation deltas',
        )
        parser.add_argument(
            '--trace-tracemalloc-limit',
            type=int,
            default=8,
            help='Maximum number of tracemalloc diff entries to print in --trace-tracemalloc mode',
        )
        parser.add_argument(
            '--trace-new-objects',
            action='store_true',
            default=False,
            help='Log newly surviving GC-tracked objects after each instance, with allocation traceback when available',
        )
        parser.add_argument(
            '--trace-new-object-limit',
            type=int,
            default=8,
            help='Maximum number of new surviving objects to print in --trace-new-objects mode',
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=0,
            help='Maximum number of instances to check (0 means no limit)',
        )

    def load_state(self) -> CheckState:
        if self.state_file and self.state_file.exists():
            with self.state_file.open('r') as f:
                state = CheckState.model_validate_json(f.read())
        else:
            state = CheckState()
        if self.state_file:
            state.set_output_file(self.state_file)
        return state

    def save_state(self):
        if self.dry_run:
            return
        self.state.save()

    def should_dump_debug_artifacts(self) -> bool:
        return bool(self.store and self.state_dir and not self.dry_run)

    @staticmethod
    def serialize_edge_dimensions(edge_dims: dict[str, Any] | None) -> dict[str, Any] | None:
        if edge_dims is None:
            return None
        out: dict[str, Any] = {}
        for dim_id, edge_dim in edge_dims.items():
            out[dim_id] = {
                'categories': [cat.id for cat in edge_dim.categories],
                'exclude': edge_dim.exclude,
                'flatten': edge_dim.flatten,
            }
        return out

    def dump_instance_graph(self, instance: Instance):
        if not self.should_dump_debug_artifacts() or self.graph_dir is None:
            return

        ctx = instance.context
        nodes = []
        edges = []
        seen_edges: set[tuple[str, str]] = set()
        for node in sorted(ctx.nodes.values(), key=lambda n: n.id):
            nodes.append({
                'id': node.id,
                'class': f'{type(node).__module__}.{type(node).__qualname__}',
                'quantity': node.quantity,
                'unit': str(node.unit) if node.unit is not None else None,
                'is_outcome': node.is_outcome,
                'tags': sorted(node.tags),
                'input_dimensions': sorted(node.input_dimensions.keys()),
                'output_dimensions': sorted(node.output_dimensions.keys()),
                'input_nodes': sorted(input_node.id for input_node in node.input_nodes),
                'output_nodes': sorted(output_node.id for output_node in node.output_nodes),
                'input_datasets': [
                    {
                        'id': ds.id,
                        'type': type(ds).__name__,
                        'input_dataset': getattr(ds, 'input_dataset', None),
                        'output_dimensions': ds.output_dimensions,
                    }
                    for ds in node.input_dataset_instances
                ],
            })
            for edge in node.edges:
                if edge.output_node != node:
                    continue
                edge_key = (edge.input_node.id, edge.output_node.id)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edges.append({
                    'input_node': edge.input_node.id,
                    'output_node': edge.output_node.id,
                    'tags': list(edge.tags),
                    'metrics': list(edge.metrics),
                    'from_dimensions': self.serialize_edge_dimensions(edge.from_dimensions),
                    'to_dimensions': self.serialize_edge_dimensions(edge.to_dimensions),
                })

        payload = {
            'instance_id': instance.id,
            'nodes': nodes,
            'edges': sorted(edges, key=lambda edge: (edge['input_node'], edge['output_node'])),
        }
        path = self.graph_dir / f'{instance.id}.json'
        with path.open('w') as f:
            json.dump(payload, f, indent=2)

    def make_node_manifest(self, node: Node) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            'node_id': node.id,
            'class': f'{type(node).__module__}.{type(node).__qualname__}',
            'quantity': node.quantity,
            'unit': str(node.unit) if node.unit is not None else None,
            'input_dimensions': sorted(node.input_dimensions.keys()),
            'output_dimensions': sorted(node.output_dimensions.keys()),
            'is_outcome': node.is_outcome,
        }
        try:
            df = node.get_output_pl()
        except Exception as exc:
            manifest['error'] = {
                'type': type(exc).__name__,
                'message': str(exc),
            }
            return manifest

        meta = df.get_meta()
        manifest.update({
            'rows': len(df),
            'primary_keys': list(meta.primary_keys),
            'dimensions': list(df.dim_ids),
            'metric_columns': list(df.metric_cols),
            'schema': {col: str(dtype) for col, dtype in df.schema.items()},
            'units': {col: str(unit) for col, unit in meta.units.items()},
        })
        if YEAR_COLUMN in df.columns:
            manifest['year_min'] = df[YEAR_COLUMN].min()
            manifest['year_max'] = df[YEAR_COLUMN].max()
        if FORECAST_COLUMN in df.columns:
            manifest['forecast_true_rows'] = len(df.filter(pl.col(FORECAST_COLUMN)))
            manifest['forecast_false_rows'] = len(df.filter(~pl.col(FORECAST_COLUMN)))
            manifest['first_forecast_year'] = df.filter(pl.col(FORECAST_COLUMN))[YEAR_COLUMN].min()
        return manifest

    def dump_scenario_manifest(self, instance: Instance):
        if not self.should_dump_debug_artifacts() or self.manifest_dir is None:
            return

        ctx = instance.context
        scenario_id = ctx.active_scenario.id
        scenario_dir = self.manifest_dir / instance.id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            'instance_id': instance.id,
            'scenario_id': scenario_id,
            'nodes': [self.make_node_manifest(node) for node in sorted(ctx.nodes.values(), key=lambda n: n.id)],
        }
        path = scenario_dir / f'{scenario_id}.json'
        with path.open('w') as f:
            json.dump(payload, f, indent=2)

    # def _schema_exclude_obj(self, obj: Any, path: str) -> bool:
    #     print(path)
    #     # if 'constraints' in path:
    #     #    breakpoint()
    #     return False

    def handle_output_artifact(
        self,
        logger: loguru.Logger,
        path: Path,
        current: dict[str, Any],
        *,
        artifact_type: str,
        artifact_id: str,
        instance_id: str,
    ) -> NodeFailReason | None:
        if self.store:
            logger.info('Storing {artifact_type} to {path}', artifact_type=artifact_type, path=path)
            if self.dry_run:
                return None
            with path.open('w') as f:
                json.dump(current, f, indent=2)

        if not self.compare:
            return None
        if not path.exists():
            logger.error('No {artifact_type} file found: {path}', artifact_type=artifact_type, path=path)
            return 'output'

        logger.info('Comparing {artifact_type} to {path}', artifact_type=artifact_type, path=path)
        with path.open('r') as f:
            reference = json.load(f)

        schema_diff = DeepDiff(sort_schema(reference['schema']), sort_schema(current['schema']))
        has_diffs = False
        if schema_diff:
            logger.error('Instance {instance_id}, {artifact_id} schema differs', instance_id=instance_id, artifact_id=artifact_id)
            print(schema_diff)
            print('Reference schema:')
            print(reference['schema'])
            print('Current schema:')
            print(current['schema'])
            has_diffs = True

        ref_rows = get_sorted_rows(reference)
        cur_rows = get_sorted_rows(current)
        if len(ref_rows) != len(cur_rows):
            logger.error(
                'Instance {instance_id}, {artifact_id} row counts differ',
                instance_id=instance_id,
                artifact_id=artifact_id,
            )
            print(f'Reference rows: {len(ref_rows)}')
            print(f'Current rows: {len(cur_rows)}')
            print('Reference rows (first and last 5):\n%s\n...\n%s' % (ref_rows[:5], ref_rows[-5:]))
            print('Current rows (first and last 5):\n%s\n...\n%s' % (cur_rows[:5], cur_rows[-5:]))
            has_diffs = True

        diff = DeepDiff(ref_rows, cur_rows, math_epsilon=1e-6)
        if diff:
            logger.error('Instance {instance_id}, {artifact_id} rows differ', instance_id=instance_id, artifact_id=artifact_id)
            print(diff)
            has_diffs = True

        if has_diffs:
            return 'compare'
        return None

    def handle_node_output(self, logger: loguru.Logger, node: Node) -> NodeFailReason | None:
        if not self.model_output_dir:
            return None
        instance_path = self.model_output_dir / node.context.instance.id
        path = instance_path / ('%s-%s.json' % (node.context.active_scenario.id, node.id))
        instance_path.mkdir(parents=True, exist_ok=True)
        df = node.get_output_pl()
        out = JSONDataset.serialize_df(df)
        return self.handle_output_artifact(
            logger,
            path,
            out,
            artifact_type='output',
            artifact_id='node %s' % node.id,
            instance_id=node.context.instance.id,
        )

    def handle_action_impact_output(self, logger: loguru.Logger, action: ActionNode, node: Node) -> NodeFailReason | None:
        if not self.action_impact_output_dir:
            return None
        instance_path = self.action_impact_output_dir / action.context.instance.id
        impact_path = instance_path / 'action-impacts'
        impact_path.mkdir(parents=True, exist_ok=True)
        instance_details = self.state.get_details_for_instance(action.context.instance.id)
        assert instance_details is not None
        path = instance_path / ('%s-%s.json' % (action.id, node.id))
        instance_path.mkdir(parents=True, exist_ok=True)
        df = action.compute_impact(node)
        out = JSONDataset.serialize_df(df)
        return self.handle_output_artifact(
            logger,
            path,
            out,
            artifact_type='action impact',
            artifact_id='action %s -> node %s' % (action.id, node.id),
            instance_id=action.context.instance.id,
        )

    def check_node(self, node: Node) -> NodeFailReason | None:
        logger = self.logger.bind(node_id=node.id, instance_id=node.context.instance.id)
        logger.info('Checking node {node} (class: {node_class})', node=node.id, node_class=node.__class__.__name__)
        fail_reason: NodeFailReason | None = 'output'
        details = self.state.get_node_details(node)
        try:
            fail_reason = self.handle_node_output(logger, node)
            if fail_reason:
                return fail_reason
            fail_reason = 'check'
            node.check()
        except NodeError as err:
            if details and self.compare:
                if fail_reason == 'check' and details.failure_at == 'check':
                    logger.error(
                        'Node %s failed checks, but that happened also in the reference run; not printing traceback' % node.id
                    )
                    return fail_reason
                if fail_reason == 'output' and details.failure_at == 'output':
                    logger.error(
                        'Node %s failed to generate output, but that happened also in the reference run; not printing traceback'
                        % node.id
                    )
                    return fail_reason
            if err.__cause__:
                print_exception(err.__cause__)
            else:
                print_exception(err)
            logger.error(f'Error getting output for node {node.context.instance.id}:{node.id}')
            return fail_reason
        fail_reason = None
        return fail_reason

    def evaluate_perf(self, check_time_ms: int, details: NodeDetail | None):
        if not self.check_perf or not details:
            return
        exec_time = check_time_ms
        reference_exec_time = details.check_time_ms
        time_diff = exec_time - reference_exec_time
        if exec_time > reference_exec_time * 1.1 and time_diff > 500:
            print(
                'Execution time %sms is more than 10%% longer than reference %sms for node %s'
                % (exec_time, reference_exec_time, details.node_id)
            )

    def get_rss_bytes(self) -> int:
        return self.process.memory_info().rss

    @staticmethod
    def format_mib(num_bytes: int) -> str:
        return f'{num_bytes / (1024 * 1024):.1f} MiB'

    def log_object_graph(self, instance_id: str):
        if not self.trace_object_graph:
            return

        from common.polars import PathsDataFrame
        from nodes.context import Context
        from nodes.datasets import Dataset
        from nodes.instance import Instance
        from nodes.node import Node

        gc.collect()
        objects = gc.get_objects()
        instances = [obj for obj in objects if isinstance(obj, Instance)]
        contexts = [obj for obj in objects if isinstance(obj, Context)]
        nodes = [obj for obj in objects if isinstance(obj, Node)]
        datasets = [obj for obj in objects if isinstance(obj, Dataset)]
        dataframes = [obj for obj in objects if isinstance(obj, PathsDataFrame)]

        owner_map: dict[int, list[str]] = defaultdict(list)
        for ds in datasets:
            df = getattr(ds, 'df', None)
            if isinstance(df, PathsDataFrame):
                owner_map[id(df)].append(f'Dataset:{ds.id}')

        total_df_bytes = 0
        df_infos: list[dict[str, Any]] = []
        for df in dataframes:
            size_bytes = estimate_size_bytes(df)
            assert size_bytes is not None
            total_df_bytes += size_bytes
            units = df.get_meta().units
            owner_labels = owner_map.get(id(df), [])
            df_infos.append({
                'rows': len(df),
                'cols': len(df.columns),
                'size_bytes': size_bytes,
                'primary_keys': list(df.primary_keys),
                'units': {col: str(unit) for col, unit in units.items()},
                'owners': owner_labels,
            })

        dataset_infos: list[dict[str, Any]] = []
        for ds in datasets:
            df = getattr(ds, 'df', None)
            size_bytes = estimate_size_bytes(df) if isinstance(df, PathsDataFrame) else 0
            dataset_infos.append({
                'id': ds.id,
                'class': type(ds).__name__,
                'has_df': isinstance(df, PathsDataFrame),
                'df_bytes': size_bytes,
            })

        self.logger.info(
            'Retained objects after {instance_id}: '
            'instances={instances} contexts={contexts} nodes={nodes} '
            'datasets={datasets} dataframes={dataframes} dataframe_bytes={dataframe_bytes}',
            instance_id=instance_id,
            instances=len(instances),
            contexts=len(contexts),
            nodes=len(nodes),
            datasets=len(datasets),
            dataframes=len(dataframes),
            dataframe_bytes=self.format_mib(total_df_bytes),
        )

        top_datasets = sorted(dataset_infos, key=lambda item: item['df_bytes'], reverse=True)[: self.trace_object_limit]
        for info in top_datasets:
            if not info['has_df']:
                continue
            self.logger.info(
                'Retained dataset {dataset_id} ({dataset_class}) df={df_bytes}',
                dataset_id=info['id'],
                dataset_class=info['class'],
                df_bytes=self.format_mib(info['df_bytes']),
            )

        top_dataframes = sorted(df_infos, key=lambda item: item['size_bytes'], reverse=True)[: self.trace_object_limit]
        for index, info in enumerate(top_dataframes, start=1):
            owners = ', '.join(info['owners']) if info['owners'] else '<unknown>'
            unit_summary = ', '.join(f'{col}={unit}' for col, unit in info['units'].items()) or '<none>'
            self.logger.info(
                'Retained dataframe #{index}: rows={rows} cols={cols} '
                'size={size} owners={owners} primary_keys={primary_keys} units={units}',
                index=index,
                rows=info['rows'],
                cols=info['cols'],
                size=self.format_mib(info['size_bytes']),
                owners=owners,
                primary_keys=info['primary_keys'],
                units=unit_summary,
            )

        if contexts:
            context_ids = Counter(getattr(ctx, 'obj_id', '<unknown>') for ctx in contexts)
            self.logger.info(
                'Retained context ids after {instance_id}: {context_ids}',
                instance_id=instance_id,
                context_ids=dict(context_ids),
            )

    def log_rss(self, instance_id: str, before_rss: int, after_rss: int, after_gc_rss: int | None = None):
        if self.rss_start_bytes is None:
            self.rss_start_bytes = before_rss
        if self.rss_prev_bytes is None:
            self.rss_prev_bytes = before_rss

        delta_prev = after_rss - self.rss_prev_bytes
        delta_start = after_rss - self.rss_start_bytes
        self.logger.info(
            'RSS after {instance_id}: {rss} (delta {delta_prev:+.1f} MiB from previous, {delta_start:+.1f} MiB from start)',
            instance_id=instance_id,
            rss=self.format_mib(after_rss),
            delta_prev=delta_prev / (1024 * 1024),
            delta_start=delta_start / (1024 * 1024),
        )
        if after_gc_rss is not None:
            gc_delta = after_gc_rss - after_rss
            self.logger.info(
                'RSS after gc for {instance_id}: {rss} ({gc_delta:+.1f} MiB from pre-gc)',
                instance_id=instance_id,
                rss=self.format_mib(after_gc_rss),
                gc_delta=gc_delta / (1024 * 1024),
            )
            self.rss_prev_bytes = after_gc_rss
        else:
            self.rss_prev_bytes = after_rss

    def maybe_log_tracemalloc(
        self,
        instance_id: str,
        before_snapshot: tracemalloc.Snapshot | None,
    ):
        if not self.trace_tracemalloc or before_snapshot is None:
            return

        gc.collect()
        after_snapshot = tracemalloc.take_snapshot()
        stats = after_snapshot.compare_to(before_snapshot, 'lineno')
        positive_stats = [stat for stat in stats if stat.size_diff > 0]
        total_size_diff = sum(stat.size_diff for stat in positive_stats)
        total_count_diff = sum(stat.count_diff for stat in positive_stats)
        self.logger.info(
            'tracemalloc after {instance_id}: +{size_diff:.1f} MiB across {count_diff} Python allocations',
            instance_id=instance_id,
            size_diff=total_size_diff / (1024 * 1024),
            count_diff=total_count_diff,
        )

        for index, stat in enumerate(positive_stats[: self.trace_tracemalloc_limit], start=1):
            frame = stat.traceback[0]
            self.logger.info(
                'tracemalloc #{index} after {instance_id}: {source_path}:{source_line} '
                '+{size_diff:.1f} KiB in {count_diff:+d} blocks',
                index=index,
                instance_id=instance_id,
                source_path=frame.filename,
                source_line=frame.lineno,
                size_diff=stat.size_diff / 1024,
                count_diff=stat.count_diff,
            )

    @staticmethod
    def object_sort_key(obj: Any) -> tuple[int, str]:
        from common.polars import PathsDataFrame
        from nodes.datasets import Dataset

        if isinstance(obj, PathsDataFrame):
            size = estimate_size_bytes(obj) or 0
            return (size, 'PathsDataFrame')
        if isinstance(obj, Dataset):
            df = getattr(obj, 'df', None)
            size = estimate_size_bytes(df) if isinstance(df, PathsDataFrame) else 0
            return (size or 0, 'Dataset')
        return (sys.getsizeof(obj), type(obj).__name__)

    def log_new_object_detail(self, instance_id: str, index: int, obj: Any, after_objects: list[Any], top_objects: list[Any]):
        from common.polars import PathsDataFrame
        from nodes.datasets import Dataset

        extra: dict[str, Any]
        if isinstance(obj, PathsDataFrame):
            extra = {
                'kind': 'PathsDataFrame',
                'rows': len(obj),
                'cols': len(obj.columns),
                'size': self.format_mib(estimate_size_bytes(obj) or 0),
                'primary_keys': list(obj.primary_keys),
            }
        else:
            assert isinstance(obj, Dataset)
            df = getattr(obj, 'df', None)
            extra = {
                'kind': f'Dataset:{type(obj).__name__}',
                'dataset_id': obj.id,
                'has_df': isinstance(df, PathsDataFrame),
                'df_size': self.format_mib(estimate_size_bytes(df) or 0) if isinstance(df, PathsDataFrame) else '0.0 MiB',
            }

        tb = tracemalloc.get_object_traceback(obj) if tracemalloc.is_tracing() else None
        alloc_site = '<unknown>'
        if tb and len(tb) > 0:
            frame = tb[0]
            alloc_site = f'{frame.filename}:{frame.lineno}'

        referrers = [ref for ref in gc.get_referrers(obj) if ref is not after_objects and ref is not top_objects]
        referrer_counts = Counter(type(ref).__name__ for ref in referrers)
        self.logger.info(
            'New surviving object #{index} after {instance_id}: kind={kind} alloc_site={alloc_site} '
            'details={details} referrers={referrers}',
            index=index,
            instance_id=instance_id,
            kind=extra.pop('kind'),
            alloc_site=alloc_site,
            details=extra,
            referrers=dict(referrer_counts.most_common(5)),
        )

    def maybe_log_new_objects(self, instance_id: str, before_object_ids: set[int] | None):
        if not self.trace_new_objects or before_object_ids is None:
            return

        from common.polars import PathsDataFrame
        from nodes.datasets import Dataset

        gc.collect()
        after_objects = gc.get_objects()
        new_objects = [obj for obj in after_objects if id(obj) not in before_object_ids]
        type_counts = Counter(type(obj).__name__ for obj in new_objects)
        self.logger.info(
            'New surviving objects after {instance_id}: total={total} top_types={top_types}',
            instance_id=instance_id,
            total=len(new_objects),
            top_types=dict(type_counts.most_common(8)),
        )

        interesting_objects = [
            obj for obj in new_objects if isinstance(obj, (PathsDataFrame, Dataset, PolarsDataFrame, PandasDataFrame))
        ]

        if not interesting_objects:
            self.logger.info('No new surviving Dataset or PathsDataFrame objects after {instance_id}', instance_id=instance_id)
            return

        top_objects = sorted(interesting_objects, key=self.object_sort_key, reverse=True)[: self.trace_new_object_limit]
        for index, obj in enumerate(top_objects, start=1):
            self.log_new_object_detail(instance_id, index, obj, after_objects, top_objects)

    def run_nodes(self, logger: loguru.Logger, ctx: Context) -> bool:
        if self.all_nodes:
            logger.info('Checking all nodes')
            nodes = sorted(ctx.nodes.values(), key=lambda node: node.id)
        else:
            logger.info('Checking outcome nodes')
            nodes = ctx.get_outcome_nodes()

        statuses: list[NodeFailReason | None] = []

        instance_details = self.state.get_details_for_instance(ctx.instance.id)
        assert instance_details is not None

        failed = False

        for node in nodes:
            now = time.time()
            fail_reason = self.check_node(node)
            check_time_ms = int((time.time() - now) * 1000)
            details = instance_details.get_node_details(node)
            if fail_reason and self.compare:
                # Returns False (failure) if the same node succeeded in the previous run
                if not details:
                    continue
                if details.failure_at and details.failure_at == fail_reason:
                    continue
                self.nr_fails += 1
                if self.maxfail > 0 and self.nr_fails < self.maxfail:
                    failed = True
                    continue
                return False
            self.evaluate_perf(check_time_ms, details)

            instance_details.add_node(node, fail_reason, check_time_ms)
            statuses.append(fail_reason)
        return not any(statuses) and not failed

    def run_action_impacts(self, logger: loguru.Logger, ctx: Context) -> bool:  # noqa: C901
        if ctx.active_scenario.id == 'baseline':
            return True

        actions = sorted((action for action in ctx.get_actions() if action.is_enabled()), key=lambda action: action.id)
        if not actions:
            return True

        statuses: list[NodeFailReason | None] = []
        failed = False

        for action in actions:
            outcome_nodes = sorted(action.get_downstream_nodes(only_outcome=True), key=lambda node: node.id)
            if not outcome_nodes:
                continue

            action_logger = logger.bind(action_id=action.id)
            action_logger.info(
                'Checking action impacts for {action} against {count} outcome nodes',
                action=action.id,
                count=len(outcome_nodes),
            )

            for node in outcome_nodes:
                impact_logger = action_logger.bind(node_id=node.id)
                fail_reason: NodeFailReason | None = 'output'
                try:
                    fail_reason = self.handle_action_impact_output(impact_logger, action, node)
                except NodeError as err:
                    if err.__cause__:
                        print_exception(err.__cause__)
                    else:
                        print_exception(err)
                    impact_logger.error(
                        'Error getting action impact for {instance_id}:{action_id}->{node_id}',
                        instance_id=ctx.instance.id,
                        action_id=action.id,
                        node_id=node.id,
                    )
                except Exception as err:
                    print_exception(err)
                    impact_logger.error(
                        'Unexpected error getting action impact for {instance_id}:{action_id}->{node_id}',
                        instance_id=ctx.instance.id,
                        action_id=action.id,
                        node_id=node.id,
                    )

                if fail_reason and self.compare:
                    self.nr_fails += 1
                    if self.maxfail > 0 and self.nr_fails < self.maxfail:
                        failed = True
                        continue
                    return False

                statuses.append(fail_reason)

        return not any(statuses) and not failed

    def maybe_log_rss(self, instance_id: str, before_rss: int):
        if not self.trace_rss:
            return
        after_rss = self.get_rss_bytes()
        after_gc_rss: int | None = None
        after_trim_rss: int | None = None
        if self.gc_after_instance:
            gc.collect()
            after_gc_rss = self.get_rss_bytes()
        if self.malloc_trim_after_instance and self.malloc_trim is not None:
            self.malloc_trim(0)
            after_trim_rss = self.get_rss_bytes()
        self.log_rss(instance_id, before_rss=before_rss, after_rss=after_rss, after_gc_rss=after_gc_rss)
        if after_trim_rss is not None:
            trim_base = after_gc_rss if after_gc_rss is not None else after_rss
            trim_delta = after_trim_rss - trim_base
            self.logger.info(
                'RSS after malloc_trim for {instance_id}: {rss} ({trim_delta:+.1f} MiB from pre-trim)',
                instance_id=instance_id,
                rss=self.format_mib(after_trim_rss),
                trim_delta=trim_delta / (1024 * 1024),
            )
            self.rss_prev_bytes = after_trim_rss

    def check_instance(self, ic: InstanceConfig):  # noqa: PLR0915
        logger = self.logger.bind(instance_id=ic.identifier)
        logger.info('Checking instance %s' % ic.identifier)
        instance_id = ic.identifier
        before_rss = self.get_rss_bytes() if self.trace_rss else 0
        before_snapshot: tracemalloc.Snapshot | None = None
        before_object_ids: set[int] | None = None
        if self.trace_tracemalloc:
            gc.collect()
            before_snapshot = tracemalloc.take_snapshot()
        if self.trace_new_objects:
            gc.collect()
            before_object_ids = {id(obj) for obj in gc.get_objects()}
        instance_details = self.state.add_instance(instance_id)
        try:
            instance = ic.get_instance()
        except Exception as e:
            logger.error('Error initializing instance %s' % instance_id)
            print_exception(e)
            if self.compare and instance_details.failure_at == 'init':
                return True
            self.state.mark_failed(instance_id, 'init')
            self.save_state()
            return False

        if self.spec_only:
            self.state.mark_success(instance)
            self.save_state()
            return True

        ctx = instance.context
        ctx.cache.clear()
        self.dump_instance_graph(instance)
        baseline_scenario = ctx.scenarios.get('baseline', None)
        with ctx.run():
            succeeded = self.run_nodes(logger, ctx)
            succeeded = self.run_action_impacts(logger, ctx) and succeeded
            self.dump_scenario_manifest(instance)
            if baseline_scenario:
                logger.info('Checking baseline scenario')
                with baseline_scenario.override(set_active=True):
                    instance_details.set_active_scenario_id('baseline')
                    succeeded = self.run_nodes(logger, ctx) and succeeded
                    self.dump_scenario_manifest(instance)
                instance_details.set_active_scenario_id('default')

        if True:
            logger.info('Cleaning instance')
            instance.clean()

        if succeeded:
            self.state.mark_success(instance)
        else:
            self.state.mark_failed(instance_id, 'nodes')
        self.save_state()

        baseline_scenario = None
        ctx = None  # type: ignore[assignment]
        instance = None  # type: ignore[assignment]
        instance_details = None

        self.maybe_log_new_objects(instance_id, before_object_ids)
        self.maybe_log_tracemalloc(instance_id, before_snapshot)
        self.maybe_log_rss(instance_id, before_rss)
        return succeeded

    def handle(self, *args, **options):  # noqa: C901, PLR0912, PLR0915
        instance_ids = options['instances']
        self.state_dir = options['state_dir']
        if self.state_dir:
            self.model_output_dir = self.state_dir / 'outputs'
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            self.action_impact_output_dir = self.state_dir / 'action-impacts'
            self.action_impact_output_dir.mkdir(parents=True, exist_ok=True)
            self.graph_dir = self.state_dir / 'graphs'
            self.graph_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_dir = self.state_dir / 'manifests'
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            self.state_file = self.state_dir / 'state.json'
            self.state = self.load_state()
        else:
            self.state_file = None
            self.state = CheckState()
            self.action_impact_output_dir = None
            self.graph_dir = None
            self.manifest_dir = None

        self.logger = loguru.logger.bind(name='test_instance')
        register_warning_handler()
        self.limit = options['limit']
        self.maxfail = options['maxfail']
        self.store = bool(options['store'])
        if options['compare'] is None and not self.store:
            self.compare = True
        else:
            self.compare = bool(options['compare'])
        self.spec_only = bool(options['spec_only'])
        self.dry_run = bool(options['dry_run'])
        self.check_perf = bool(options['check_perf'])
        self.trace_rss = bool(options['trace_rss'])
        self.gc_after_instance = bool(options['gc_after_instance'])
        self.malloc_trim_after_instance = bool(options['malloc_trim_after_instance'])
        self.all_nodes = bool(options['all_nodes'])
        self.trace_object_graph = bool(options['trace_object_graph'])
        self.trace_object_limit = int(options['trace_object_limit'])
        self.trace_tracemalloc = bool(options['trace_tracemalloc'])
        self.trace_tracemalloc_limit = int(options['trace_tracemalloc_limit'])
        self.trace_new_objects = bool(options['trace_new_objects'])
        self.trace_new_object_limit = int(options['trace_new_object_limit'])
        self.process = psutil.Process()
        if (self.trace_tracemalloc or self.trace_new_objects) and not tracemalloc.is_tracing():
            tracemalloc.start(25)
        if self.malloc_trim_after_instance:
            try:
                libc = ctypes.CDLL('libc.so.6')
                self.malloc_trim = libc.malloc_trim
                self.malloc_trim.argtypes = [ctypes.c_size_t]
                self.malloc_trim.restype = ctypes.c_int
            except OSError:
                self.malloc_trim = None
                self.logger.warning('--malloc-trim-after-instance requested, but malloc_trim is not available on this platform')
        if self.compare:
            if not self.state_dir:
                self.logger.error('--compare requires --state-dir')
                exit(1)
            self.state.set_compare_mode()

        only_instance = options['only']
        if only_instance:
            instance_ids = [only_instance]
        elif not instance_ids:
            self.logger.info('No instances provided, checking all instances')
            if self.compare:
                instance_ids = self.state.checked_instances
            else:
                instance_ids = list(InstanceConfig.objects.all().order_by('identifier').values_list('identifier', flat=True))

        start_from = options['start_from']

        for iid in sorted(instance_ids):
            if start_from:
                if iid == start_from:
                    start_from = None
                else:
                    continue
            if options['state_resume'] and iid in self.state.checked_instances:
                continue
            if options['skip'] and iid in options['skip']:
                continue

            ic = InstanceConfig.objects.get(identifier=iid)
            if ic.has_framework_config():
                continue
            succeeded = self.check_instance(ic)
            if not succeeded:
                self.nr_fails += 1
                if self.maxfail > 0 and self.nr_fails >= self.maxfail:
                    self.logger.error('Maximum number of failures reached, stopping')
                    break
                continue
            del ic
            self.log_object_graph(iid)
            if self.limit > 0:
                self.limit -= 1
                if self.limit == 0:
                    self.logger.info('Maximum number of instances reached, stopping')
                    break

        if self.nr_fails:
            self.logger.error('Failed {nr_fails} instances', nr_fails=self.nr_fails)
            exit(1)
