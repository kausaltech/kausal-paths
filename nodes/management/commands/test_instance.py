from __future__ import annotations

import ctypes
import gc
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from django.core.management.base import BaseCommand
from pydantic import BaseModel, Field, PrivateAttr

import loguru
import polars as pl
import psutil
from deepdiff import DeepDiff
from rich import print
from rich.console import Console

from kausal_common.logging.errors import print_exception
from kausal_common.logging.warnings import register_warning_handler

from nodes.constants import FORECAST_COLUMN, YEAR_COLUMN
from nodes.datasets import JSONDataset
from nodes.exceptions import NodeError
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from django.core.management.base import CommandParser

    from nodes.context import Context
    from nodes.instance import Instance
    from nodes.node import Node

console = Console()


def sort_schema(schema: dict[str, Any]):
    pks = schema['primaryKey']
    pks.sort()

    fields: list[dict[str, Any]] = schema['fields']
    fields.sort(key=lambda x: x['name'])
    for field in fields:
        constraints = field.get('constraints')
        if constraints and 'enum' in constraints:
            constraints['enum'].sort()
    return schema


def get_sorted_rows(table: dict[str, Any]):
    pks: list[str] = table['schema']['primaryKey']
    rows: list[dict[str, Any]] = table['data']
    rows.sort(key=lambda x: tuple([x[key] for key in pks]))
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

    def handle_node_output(self, logger: loguru.Logger, node: Node) -> NodeFailReason | None:
        if not self.model_output_dir:
            return None
        instance_path = self.model_output_dir / node.context.instance.id
        path = instance_path / ('%s-%s.json' % (node.context.active_scenario.id, node.id))
        if self.store and self.model_output_dir:
            instance_path.mkdir(parents=True, exist_ok=True)
            df = node.get_output_pl()
            out = JSONDataset.serialize_df(df)
            logger.info('Storing output to %s' % path)
            if self.dry_run:
                return None
            with path.open('w') as f:
                json.dump(out, f, indent=2)
        if self.compare and self.model_output_dir:
            if not path.exists():
                logger.error('No output file found: %s' % path)
                return 'output'
            logger.info('Comparing output to %s' % path)
            with path.open('r') as f:
                reference = json.load(f)
            df = node.get_output_pl()
            current = JSONDataset.serialize_df(df)
            schema_diff = DeepDiff(sort_schema(reference['schema']), sort_schema(current['schema']))
            has_diffs = False
            if schema_diff:
                logger.error('Instance %s, node %s schema differs' % (node.context.instance.id, node.id))
                print(schema_diff)
                print('Reference schema:')
                print(reference['schema'])
                print('Current schema:')
                print(current['schema'])
                has_diffs = True
            ref_rows = get_sorted_rows(reference)
            cur_rows = get_sorted_rows(current)
            if len(ref_rows) != len(cur_rows):
                logger.error('Instance %s, node %s row counts' % (node.context.instance.id, node.id))
                print(f'Reference rows: {len(ref_rows)}')
                print(f'Current rows: {len(cur_rows)}')
                print('Reference rows (first and last 5):\n%s\n...\n%s' % (ref_rows[:5], ref_rows[-5:]))
                print('Current rows (first and last 5):\n%s\n...\n%s' % (cur_rows[:5], cur_rows[-5:]))
                has_diffs = True
            diff = DeepDiff(get_sorted_rows(reference), get_sorted_rows(current), math_epsilon=1e-6)
            if diff:
                logger.error('Instance %s, node %s rows differ' % (node.context.instance.id, node.id))
                print(diff)
                has_diffs = True
            if has_diffs:
                return 'compare'
        return None

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

        for node in nodes:
            now = time.time()
            fail_reason = self.check_node(node)
            check_time_ms = int((time.time() - now) * 1000)
            details = instance_details.get_node_details(node)
            if fail_reason and self.compare:
                # Returns False (failure) if the same node succeeded in the previous run
                if not details:
                    return True
                if details.failure_at and details.failure_at == fail_reason:
                    return True
                return False
            self.evaluate_perf(check_time_ms, details)

            instance_details.add_node(node, fail_reason, check_time_ms)
            statuses.append(fail_reason)
        return not any(statuses)

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

    def check_instance(self, ic: InstanceConfig):
        logger = self.logger.bind(instance_id=ic.identifier)
        logger.info('Checking instance %s' % ic.identifier)
        before_rss = self.get_rss_bytes() if self.trace_rss else 0
        instance_details = self.state.add_instance(ic.identifier)
        try:
            instance = ic.get_instance()
        except Exception as e:
            logger.error('Error initializing instance %s', ic.identifier)
            print_exception(e)
            if self.compare and instance_details.failure_at == 'init':
                return True
            self.state.mark_failed(ic.identifier, 'init')
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
            self.state.mark_failed(ic.identifier, 'nodes')
        self.save_state()
        self.maybe_log_rss(ic.identifier, before_rss)
        return succeeded

    def handle(self, *args, **options):  # noqa: C901, PLR0912, PLR0915
        instance_ids = options['instances']
        self.state_dir = options['state_dir']
        if self.state_dir:
            self.model_output_dir = self.state_dir / 'outputs'
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            self.graph_dir = self.state_dir / 'graphs'
            self.graph_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_dir = self.state_dir / 'manifests'
            self.manifest_dir.mkdir(parents=True, exist_ok=True)
            self.state_file = self.state_dir / 'state.json'
            self.state = self.load_state()
        else:
            self.state_file = None
            self.state = CheckState()
            self.graph_dir = None
            self.manifest_dir = None

        self.logger = loguru.logger.bind(name='test_instance')
        register_warning_handler()
        self.maxfail = options['maxfail']
        self.store = bool(options['store'])
        self.compare = bool(options['compare'])
        self.spec_only = bool(options['spec_only'])
        self.dry_run = bool(options['dry_run'])
        self.check_perf = bool(options['check_perf'])
        self.trace_rss = bool(options['trace_rss'])
        self.gc_after_instance = bool(options['gc_after_instance'])
        self.malloc_trim_after_instance = bool(options['malloc_trim_after_instance'])
        self.all_nodes = bool(options['all_nodes'])
        self.process = psutil.Process()
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
        if self.nr_fails:
            self.logger.error('Failed {nr_fails} instances', nr_fails=self.nr_fails)
            exit(1)
