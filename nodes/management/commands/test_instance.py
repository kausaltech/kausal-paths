from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from django.core.management.base import BaseCommand
from pydantic import BaseModel, Field, PrivateAttr

import loguru
from deepdiff import DeepDiff
from rich import print
from rich.console import Console

from kausal_common.logging.errors import print_exception
from kausal_common.logging.warnings import register_warning_handler

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
            self.nodes.append(details)
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

    def mark_failed(self, instance: Instance, reason: InstanceFailReason):
        self.failed_instances.add(instance.id)
        self.checked_instances.add(instance.id)
        details = self.get_details_for_instance(instance.id)
        if not details:
            details = self.add_instance(instance.id)
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
    model_output_dir: Path | None

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
        try:
            fail_reason = self.handle_node_output(logger, node)
            if fail_reason:
                return fail_reason
            fail_reason = 'check'
            node.check()
        except NodeError as err:
            if err.__cause__:
                print_exception(err.__cause__)
            else:
                print_exception(err)
            logger.error(f'Error getting output for node {node.context.instance.id}:{node.id}')
            return fail_reason
        fail_reason = None
        return fail_reason

    def run_nodes(self, logger: loguru.Logger, ctx: Context) -> bool:
        logger.info('Checking outcome nodes')

        statuses: list[NodeFailReason | None] = []

        for node in ctx.get_outcome_nodes():
            now = time.time()
            fail_reason = self.check_node(node)
            check_time_ms = int((time.time() - now) * 1000)
            if fail_reason and self.compare:
                details = self.state.get_node_details(node)
                # Returns False (failure) if the same node succeeded in the previous run
                if not details:
                    return True
                if details.failure_at and details.failure_at == fail_reason:
                    return True
                return False
            self.state.add_node(node, fail_reason, check_time_ms)
            statuses.append(fail_reason)
        return not any(statuses)

    def check_instance(self, ic: InstanceConfig):
        logger = self.logger.bind(instance_id=ic.identifier)
        logger.info('Checking instance %s' % ic.identifier)
        instance_details = self.state.add_instance(ic.identifier)
        try:
            instance = ic.get_instance()
        except Exception as e:
            logger.error('Error initializing instance %s', ic.identifier)
            print_exception(e)
            if self.compare and self.state.has_instance(ic.identifier):
                return False
            self.state.failed_instances.add(ic.identifier)
            self.state.checked_instances.add(ic.identifier)
            self.save_state()
            return False

        if self.spec_only:
            self.state.mark_success(instance)
            self.save_state()
            return True

        ctx = instance.context
        ctx.cache.clear()
        baseline_scenario = ctx.scenarios.get('baseline', None)
        with ctx.run():
            succeeded = self.run_nodes(logger, ctx)
            if baseline_scenario:
                instance_details.set_active_scenario_id('baseline')
                logger.info('Checking baseline scenario')
                with baseline_scenario.override(set_active=True):
                    succeeded = self.run_nodes(logger, ctx)
                instance_details.set_active_scenario_id('default')
        logger.info('Cleaning instance')
        instance.clean()

        if succeeded:
            self.state.mark_success(instance)
        else:
            self.state.mark_failed(instance, 'nodes')
        self.save_state()

        if True:
            import gc

            logger.info('Collecting garbage')
            nr_unreachable = gc.collect()
            if nr_unreachable:
                logger.warning('Garbage collection identified %d unreachable objects' % nr_unreachable)
        return succeeded

    def handle(self, *args, **options):  # noqa: C901, PLR0912, PLR0915
        instance_ids = options['instances']
        self.state_dir = options['state_dir']
        if self.state_dir:
            self.model_output_dir = self.state_dir / 'outputs'
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            self.state_file = self.state_dir / 'state.json'
            self.state = self.load_state()
        else:
            self.state_file = None
            self.state = CheckState()

        self.logger = loguru.logger.bind(name='test_instance')
        register_warning_handler()
        self.maxfail = options['maxfail']
        self.store = bool(options['store'])
        self.compare = bool(options['compare'])
        self.spec_only = bool(options['spec_only'])
        self.dry_run = bool(options['dry_run'])
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
