"""
Record every additive-vs-factor decision the unit heuristic makes.

The rule in :mod:`nodes.operands` decides an untagged input's role by comparing
units (see ``role_from_unit_comparison``). That is the implicit half of the
rule, and it is what we are removing: once every input carries an explicit
``additive`` / ``non_additive`` tag, loading a config that still needs the
heuristic can fail outright.

This command does not change anything. It loads instances, computes every node
so that every decision actually fires, and writes one JSONL record per
decision. ``tools/inject_operand_tags.py`` turns those records into authored
YAML tags.

Because the heuristic reads a *computed* output unit for node inputs
(``output_unit_of``), a node that does not compute yields no record for its
inputs. Instances that fail are reported so the coverage gap is visible rather
than silent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from django.core.management.base import BaseCommand

from loguru import logger

from common.cache import CacheKind
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Callable
    from pathlib import Path

    from nodes.context import Context
    from nodes.node import Node
    from nodes.operands import OperandRole
    from nodes.units import Unit

    type UnitRule = Callable[..., OperandRole]


class Recorder:
    """Collects one record per unit-heuristic decision, keyed to its instance."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.instance_id: str | None = None

    def record(
        self,
        node: Node,
        *,
        source_unit: Unit | None,
        source_id: str,
        source_kind: Literal['node', 'dataset'],
        role: str | None,
        error: str | None = None,
    ) -> None:
        self.records.append({
            'instance': self.instance_id,
            'node': node.id,
            'node_class': f'{type(node).__module__}.{type(node).__qualname__}',
            'node_unit': str(node.unit) if node.unit is not None else None,
            'source': source_id,
            'source_kind': source_kind,
            'source_unit': str(source_unit) if source_unit is not None else None,
            'role': role,
            'error': error,
        })


def install_recorder(recorder: Recorder) -> UnitRule:
    """Wrap ``role_from_unit_comparison`` so every decision — and every refusal — is seen."""
    from nodes import operands

    original: UnitRule = operands.role_from_unit_comparison

    def wrapper(
        node: Node,
        *,
        source_unit: Unit | None,
        source_id: str,
        source_kind: Literal['node', 'dataset'],
        default_role: OperandRole | None,
    ) -> OperandRole:
        try:
            role = original(
                node,
                source_unit=source_unit,
                source_id=source_id,
                source_kind=source_kind,
                default_role=default_role,
            )
        except Exception as error:
            recorder.record(
                node,
                source_unit=source_unit,
                source_id=source_id,
                source_kind=source_kind,
                role=None,
                error=str(error),
            )
            raise
        recorder.record(
            node,
            source_unit=source_unit,
            source_id=source_id,
            source_kind=source_kind,
            role=role,
        )
        return role

    operands.role_from_unit_comparison = wrapper  # pyright: ignore[reportAttributeAccessIssue]
    return original


def restore_recorder(original: UnitRule) -> None:
    from nodes import operands

    operands.role_from_unit_comparison = original


class Command(BaseCommand):
    help = 'Record the additive-vs-factor decisions the unit heuristic makes, as JSONL.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', metavar='INSTANCE_ID', type=str, nargs='*')
        parser.add_argument('--all', action='store_true', help='Process every instance in the database')
        parser.add_argument('--skip', metavar='INSTANCE_ID', action='append', default=[])
        parser.add_argument('--output', '-o', type=str, required=True, help='JSONL output path')
        parser.add_argument(
            '--compute',
            action='store_true',
            help='Also compute every node, to catch the compute-time heuristic that GenericNode still uses (slow)',
        )

    def record_graph_roles(self, iid: str, recorder: Recorder) -> None:
        """
        Record the graph-time half of the heuristic.

        Migrated classes (AdditiveNode, MultiplicativeNode and friends) no longer
        reach ``role_from_unit_comparison`` at all: their roles are decided once at
        graph build by ``infer_legacy_port_roles``, against *declared* port units.
        That decision needs no computation, so it is the cheap pass.

        Routing goes through ``get_instance_graph`` rather than
        ``build_instance_snapshot`` directly: most of the fleet is YAML- or
        framework-backed and has no DB spec, so building the snapshot from the
        editor tables fails for them with "node has no computation spec".
        """
        from nodes.instance_graph_cache import get_instance_graph
        from nodes.models import PreferredInstanceSource

        ic = InstanceConfig.objects.get(identifier=iid)
        try:
            graph = get_instance_graph(ic, PreferredInstanceSource.DRAFT, refresh=True)
        except ValueError:
            # YAML and framework snapshots carry DVC dataset references with no
            # catalog entry; only the loader synthesizes placeholders for them.
            # Tolerant load: unrolled bindings now fail the node, and this
            # command exists to *count* those, so it must not be stopped by one.
            graph = ic._initialize_instance(tolerate_node_failures=True).context.instance_graph
            if graph is None:
                raise
        # Refusals matter more than classifications for the "fail loudly"
        # design: an unclassified binding on a port-declaring class is the one
        # that gets dropped. Record them whether or not the node classified
        # anything else.
        for diagnostic in graph.diagnostics:
            if diagnostic.code != 'unclassified_port_role':
                continue
            node = graph.node_by_id.get(diagnostic.node_id) if diagnostic.node_id else None
            if node is None:
                continue
            declares = bool(node.node_class.input_port_declarations)
            recorder.records.append({
                'instance': iid,
                'layer': 'graph',
                'kind': 'refusal',
                'node': node.identifier,
                'node_class': node.node_class_path,
                'hook': node.node_class.infer_legacy_port_roles.__qualname__.split('.')[0],
                'declares_ports': declares,
                'port': str(diagnostic.port_id),
                'role': None,
                'basis': diagnostic.message,
                'error': None,
            })

        for meta in graph.nodes:
            roles = meta.inferred_port_roles
            if not roles:
                continue
            bases = {d.port_id: d.message for d in meta.port_role_diagnostics}
            # Which class's hook decided matters more than the message text. A
            # hook that can only ever emit 'additive'/'impute' — AdditiveNode's,
            # and everything inheriting it — is not applying a heuristic even
            # when its basis string mentions a unit: the unit changes the
            # explanation, not the answer. Only a hook that can also emit
            # 'factors' is deciding anything.
            hook = meta.node_class.infer_legacy_port_roles.__qualname__.split('.')[0]
            for port in meta.spec.input_ports:
                role = roles.get(port.id)
                if role is None:
                    continue
                sources = [
                    (b.source_node.identifier if hasattr(b, 'source_node') else str(getattr(b, 'external_dataset_id', '')))
                    for b in meta.bindings_for_port(port.id)
                ]
                recorder.records.append({
                    'instance': iid,
                    'layer': 'graph',
                    'node': meta.identifier,
                    'node_class': meta.node_class_path,
                    'hook': hook,
                    'node_unit': None,
                    'port': str(port.id),
                    'port_unit': str(port.unit) if port.unit is not None else None,
                    'source': ','.join(s for s in sources if s),
                    'source_kind': 'port',
                    'source_unit': None,
                    'role': role,
                    'basis': bases.get(port.id),
                    'error': None,
                })

    def instance_ids(self, options: dict[str, Any]) -> list[str]:
        if options['instances']:
            ids = list(options['instances'])
        elif options['all']:
            ids = list(InstanceConfig.objects.all().order_by('identifier').values_list('identifier', flat=True))
        else:
            raise SystemExit('Give instance identifiers or --all')
        return [iid for iid in ids if iid not in set(options['skip'])]

    def compute_all(self, context: Context) -> list[str]:
        """Compute every node, collecting failures rather than stopping at the first."""
        failures: list[str] = []
        for node in sorted(context.nodes.values(), key=lambda n: n.id):
            try:
                node.get_output_pl()
            except Exception as error:  # a broken node must not hide the rest
                failures.append(f'{node.id}: {error}')
        return failures

    def handle(self, *args, **options) -> None:
        from pathlib import Path

        recorder = Recorder()
        original = install_recorder(recorder)
        out_path = Path(options['output'])
        instance_failures: dict[str, str] = {}
        node_failures: dict[str, list[str]] = {}
        try:
            for iid in self.instance_ids(options):
                recorder.instance_id = iid
                logger.info(f'Classifying {iid}')
                try:
                    self.record_graph_roles(iid, recorder)
                except Exception as error:  # a broken graph is data, not a crash
                    instance_failures[iid] = f'graph: {error}'
                    logger.error(f'{iid}: could not build graph: {error}')
                    continue
                if not options['compute']:
                    continue
                try:
                    ic = InstanceConfig.objects.get(identifier=iid)
                    instance = ic.get_instance()
                    context = instance.context
                    # The heuristic only fires inside a real ``_compute``. A warm
                    # local or Redis cache serves the output instead and the run
                    # silently records nothing, so keep only within-run memoization.
                    context.cache.set_allowed_cache_kinds({CacheKind.RUN})
                    context.cache.clear()
                    context.generate_baseline_values()
                except Exception as error:  # an unloadable instance is data, not a crash
                    instance_failures[iid] = str(error)
                    logger.error(f'{iid}: could not load: {error}')
                    continue
                with context.run():
                    failures = self.compute_all(context)
                if failures:
                    node_failures[iid] = failures
        finally:
            restore_recorder(original)

        with out_path.open('w') as out:
            for record in recorder.records:
                out.write(json.dumps(record) + '\n')

        self.report(recorder, instance_failures, node_failures, out_path)

    def report(
        self,
        recorder: Recorder,
        instance_failures: dict[str, str],
        node_failures: dict[str, list[str]],
        out_path: Path,
    ) -> None:
        records = recorder.records
        decided = [r for r in records if r['role'] is not None]
        refused = [r for r in records if r['role'] is None]
        by_role: dict[str, int] = {}
        for record in decided:
            by_role[record['role']] = by_role.get(record['role'], 0) + 1

        self.stdout.write(f'\nWrote {len(records)} inferences to {out_path}')
        split = '  '.join(f'{role}={count}' for role, count in sorted(by_role.items()))
        self.stdout.write(f'  decided: {len(decided)}  {split}')
        self.stdout.write(f'  refused: {len(refused)}')

        # Split by hook, not by message: a hook that never emits 'factors'
        # applies no additive-vs-factor heuristic, however its basis reads.
        roles_by_hook: dict[str, set[str]] = {}
        count_by_hook: dict[str, int] = {}
        for record in decided:
            hook = record.get('hook') or record['node_class']
            roles_by_hook.setdefault(hook, set()).add(record['role'])
            count_by_hook[hook] = count_by_hook.get(hook, 0) + 1
        heuristic_hooks = {hook for hook, roles in roles_by_hook.items() if 'factors' in roles}
        heuristic = sum(count for hook, count in count_by_hook.items() if hook in heuristic_hooks)

        self.stdout.write(f'\n  ADDITIVE-VS-FACTOR HEURISTIC (hook can emit "factors"): {heuristic}')
        self.stdout.write(f'  fixed-role inference (hook never emits "factors"):       {len(decided) - heuristic}')
        self.stdout.write('\n  by hook:')
        for hook, count in sorted(count_by_hook.items(), key=lambda item: -item[1]):
            mark = 'HEURISTIC' if hook in heuristic_hooks else 'fixed    '
            self.stdout.write(f'    {count:6}  {mark}  {hook}  roles={sorted(roles_by_hook[hook])}')
        if instance_failures:
            self.stdout.write(f'\n  instances that did not load: {len(instance_failures)}')
            for iid, error in sorted(instance_failures.items()):
                self.stdout.write(f'    {iid}: {error[:140]}')
        if node_failures:
            total = sum(len(v) for v in node_failures.values())
            self.stdout.write(f'\n  nodes that did not compute: {total} across {len(node_failures)} instances')
