from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand
from pydantic import BaseModel, Field, PrivateAttr

import loguru
from deepdiff import DeepDiff
from rich import print
from rich.console import Console
from rich.traceback import Traceback

from kausal_common.logging.warnings import register_warning_handler

from nodes.datasets import JSONDataset
from nodes.exceptions import NodeError
from nodes.models import InstanceConfig
from nodes.scenario import ScenarioKind

if TYPE_CHECKING:
    from django.core.management.base import CommandParser

    from nodes.context import Context
    from nodes.instance import Instance
    from nodes.node import Node

console = Console()


def make_comparable(table: dict[str, Any]):
    schema: dict[str, Any] = table['schema']
    pks: list[str] = schema['primaryKey']
    pks.sort()
    fields: list[dict[str, Any]] = schema['fields']

    fields.sort(key=lambda x: x['name'])
    data: list[dict[str, Any]] = table['data']
    data.sort(key=lambda x: tuple([x[key] for key in pks]))
    return data


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


class NodeDetail(BaseModel):
    node_id: str
    success: bool = False
    check_time_ms: float = 0


class InstanceDetail(BaseModel):
    instance_id: str
    success: bool = False
    nodes: list[NodeDetail] = Field(default_factory=list)
    baseline_nodes: list[NodeDetail] = Field(default_factory=list)

    def add_node(self, node: Node, success: bool, check_time_ms: float, baseline: bool = False):
        nodes = self.baseline_nodes if baseline else self.nodes
        for details in nodes:
            if details.node_id == node.id:
                break
        else:
            details = NodeDetail(node_id=node.id)
            self.nodes.append(details)
        details.success = success
        details.check_time_ms = check_time_ms


class CheckState(BaseModel):
    checked_instances: set[str] = Field(default_factory=set)
    failed_instances: set[str] = Field(default_factory=set)
    instance_details: list[InstanceDetail] = Field(default_factory=list)

    _output_file: Path | None = PrivateAttr(default=None)

    def add_node(self, node: Node, success: bool, check_time_ms: float):
        for details in self.instance_details:
            if details.instance_id == node.context.instance.id:
                break
        else:
            details = InstanceDetail(instance_id=node.context.instance.id)
            self.instance_details.append(details)
        context = node.context
        details.add_node(node, success, check_time_ms, baseline=context.active_scenario.kind == ScenarioKind.BASELINE)

    def get_details_for_instance(self, instance: Instance) -> InstanceDetail:
        for details in self.instance_details:
            if details.instance_id == instance.id:
                return details
        details = InstanceDetail(instance_id=instance.id)
        self.instance_details.append(details)
        return details

    def mark_failed(self, instance: Instance):
        self.failed_instances.add(instance.id)
        self.checked_instances.add(instance.id)
        details = self.get_details_for_instance(instance)
        details.success = False

    def mark_success(self, instance: Instance):
        self.checked_instances.add(instance.id)
        self.failed_instances.discard(instance.id)
        details = self.get_details_for_instance(instance)
        details.success = True

    def set_output_file(self, output_file: Path):
        self._output_file = output_file

    def save(self):
        if not self._output_file:
            return
        with self._output_file.open('w') as f:
            f.write(self.model_dump_json(indent=2))


class Command(BaseCommand):
    help = 'Validate computation models and store/compare results'

    store: bool
    compare: bool
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

    def save_state(self, state: CheckState):
        if not self.state_file:
            return
        with self.state_file.open('w') as f:
            f.write(state.model_dump_json(indent=2))

    def handle_node_output(self, logger: loguru.Logger, node: Node) -> bool:
        if self.store and self.model_output_dir:
            df = node.get_output_pl()
            out = JSONDataset.serialize_df(df)
            store_path = self.model_output_dir
            fn = store_path / ('%s-%s-%s.json' % (node.context.instance.id, node.context.active_scenario.id, node.id))
            logger.info('Storing output to %s' % fn)
            with fn.open('w') as f:
                json.dump(out, f, indent=2)
            return True
        if self.compare and self.model_output_dir:
            fn = self.model_output_dir / ('%s-%s-%s.json' % (node.context.instance.id, node.context.active_scenario.id, node.id))
            if not fn.exists():
                logger.error('No output file found: %s' % fn)
                return False
            logger.info('Comparing output to %s' % fn)
            with fn.open('r') as f:
                data = json.load(f)
            df = node.get_output_pl()
            df_ser = JSONDataset.serialize_df(df)
            diffs = list(DeepDiff(make_comparable(data), make_comparable(df_ser), math_epsilon=1e-6))
            if diffs:
                logger.error('Instance %s, node %s differs' % (node.context.instance.id, node.id))
                print(diffs)
                return False
            return True
        return True

    def check_node(self, node: Node) -> bool:
        logger = self.logger.bind(node_id=node.id, instance_id=node.context.instance.id)
        logger.info('Checking node {node} (class: {node_class})', node=node.id, node_class=node.__class__.__name__)
        success = False
        try:
            node.check()
            success = self.handle_node_output(logger, node)
        except NodeError as e:
            if e.__cause__:
                err = e.__cause__
                tb = Traceback.from_exception(type(err), err, err.__traceback__)
                console.print(tb)
                logger.error(
                    'Error checking node {instance_id}:{node_id}\nNode dependency path: {dep_path}',
                    instance_id=node.context.instance.id,
                    node_id=node.id,
                    dep_path=e.get_dependency_path(),
                )
            success = False
        if not success:
            self.state.mark_failed(node.context.instance)
        self.state.save()
        return success

    def run_nodes(self, logger: loguru.Logger, ctx: Context) -> bool:
        logger.info('Checking outcome nodes')
        for node in ctx.get_outcome_nodes():
            now = time.time()
            success = self.check_node(node)
            check_time_ms = int((time.time() - now) * 1000)
            self.state.add_node(node, success, check_time_ms)
            if not success:
                return False
        return True

    def check_instance(self, ic: InstanceConfig):
        logger = self.logger.bind(instance_id=ic.identifier)
        logger.info('Checking instance %s' % ic.identifier)
        try:
            instance = ic.get_instance()
        except Exception as e:
            logger.error('Error initializing instance %s', ic.identifier)
            console.print(e)
            self.state.failed_instances.add(ic.identifier)
            self.state.checked_instances.add(ic.identifier)
            self.state.save()
            return False

        ctx = instance.context
        ctx.cache.clear()
        baseline_scenario = ctx.scenarios.get('baseline', None)
        with ctx.run():
            succeeded = self.run_nodes(logger, ctx)
            if succeeded and baseline_scenario:
                logger.info('Checking baseline scenario')
                with baseline_scenario.override(set_active=True):
                    succeeded = self.run_nodes(logger, ctx)
        if succeeded:
            self.state.mark_success(instance)
        else:
            self.state.mark_failed(instance)
        self.state.save()
        return succeeded

    def handle(self, *args, **options):  # noqa: C901, PLR0912
        instance_ids = options['instances']
        self.state_dir = options['state_dir']
        if self.state_dir:
            self.model_output_dir = self.state_dir / 'outputs'
            self.model_output_dir.mkdir(parents=True, exist_ok=True)
            self.state_file = self.state_dir / 'state.json'
        self.logger = loguru.logger.bind(name='test_instance')
        register_warning_handler()
        if self.state_file:
            self.state = self.load_state()
        else:
            self.state = CheckState()
        self.maxfail = options['maxfail']
        self.store = bool(options['store'])
        self.compare = bool(options['compare'])
        if self.compare and not self.state_dir:
            self.logger.error('--compare requires --state-dir')
            exit(1)

        only_instance = options['only']
        if only_instance:
            instance_ids = [only_instance]
        elif not instance_ids:
            self.logger.info('No instances provided, checking all instances')
            if self.compare:
                instance_ids = list(self.state.checked_instances - self.state.failed_instances)
            else:
                instance_ids = list(InstanceConfig.objects.all().order_by('identifier').values_list('identifier', flat=True))

        start_from = options['start_from']

        for iid in instance_ids:
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
