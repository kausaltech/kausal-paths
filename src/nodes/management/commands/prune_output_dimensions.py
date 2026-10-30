"""
Withdraw the output-dimension assertions that shape inference reproduces.

Usage:
    python manage.py prune_output_dimensions mainz-bisko            # plan
    python manage.py prune_output_dimensions --all-db --apply
"""

from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from nodes.constraints.assertions import OutputAssertion, classify_output_assertions
from nodes.instance_graph_cache import get_instance_graph
from nodes.models import InstanceConfig, NodeConfig, PreferredInstanceSource

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from uuid import UUID


class Command(BaseCommand):
    help = (
        "Withdraw output ports' dimension assertions where inference reproduces them, and report the ones it cannot. "
        'Writes nothing without --apply.'
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instance', nargs='*', type=str, help='Instance identifier(s)')
        parser.add_argument('--all-db', action='store_true', help='All database-sourced, non-framework instances')
        parser.add_argument('--apply', action='store_true', help='Write the pruned specs')
        parser.add_argument('--verbose-kept', action='store_true', help='List every assertion that has to stay')

    def handle(self, *args: Any, **options: Any) -> None:
        identifiers: list[str] = options['instance']
        if options['all_db']:
            if identifiers:
                raise CommandError('Name instances or pass --all-db, not both')
            configs = list(
                InstanceConfig.objects.filter(config_source='database', framework_config__isnull=True).order_by('identifier')
            )
        elif identifiers:
            configs = list(InstanceConfig.objects.filter(identifier__in=identifiers))
            missing = set(identifiers) - {ic.identifier for ic in configs}
            if missing:
                raise CommandError(f'No such instance: {", ".join(sorted(missing))}')
        else:
            raise CommandError('Name instances or pass --all-db')

        kept_by_class: Counter[str] = Counter()
        for ic in configs:
            if ic.config_source != 'database':
                self.stderr.write(
                    self.style.WARNING(f'{ic.identifier}: YAML-sourced; its YAML re-asserts every dimension, skipping')
                )
                continue
            kept_by_class.update(self.prune_instance(ic, apply=options['apply'], verbose_kept=options['verbose_kept']))

        if kept_by_class:
            self.stdout.write('\nAssertions inference cannot replace yet, by node class:')
            for node_class, count in kept_by_class.most_common():
                self.stdout.write(f'  {count:6d}  {node_class}')
        if not options['apply']:
            self.stdout.write(self.style.NOTICE('\nPlan only; pass --apply to write.'))

    def prune_instance(self, ic: InstanceConfig, *, apply: bool, verbose_kept: bool) -> Counter[str]:
        graph = get_instance_graph(ic, PreferredInstanceSource.DRAFT, refresh=True)
        assertions = classify_output_assertions(graph)
        redundant = [assertion for assertion in assertions if assertion.redundant]
        kept = [assertion for assertion in assertions if not assertion.redundant]
        outcome = sum(1 for assertion in kept if assertion.is_outcome)
        self.stdout.write(
            f'{ic.identifier}: {len(assertions)} asserted output ports, {len(redundant)} redundant, '
            f'{len(kept)} kept ({outcome} on outcome nodes)'
        )
        if verbose_kept:
            for assertion in kept:
                self.stdout.write(f'    keep {graph.describe_uuid(assertion.node_id)}: {self._why_kept(assertion)}')

        if apply and redundant:
            self._withdraw(ic, redundant)
        return Counter(assertion.node_class for assertion in kept if not assertion.is_outcome)

    @staticmethod
    def _why_kept(assertion: OutputAssertion) -> str:
        if assertion.is_outcome:
            return 'outcome'
        if assertion.inferred is None:
            return f'cannot infer; asserts {list(assertion.asserted)}'
        return f'infers {list(assertion.inferred)}, asserts {list(assertion.asserted)}'

    @staticmethod
    def _withdraw(ic: InstanceConfig, redundant: list[OutputAssertion]) -> None:
        ports_by_node: dict[UUID, set[UUID]] = defaultdict(set)
        for assertion in redundant:
            ports_by_node[assertion.node_id].add(assertion.port_id)
        with transaction.atomic():
            for nc in NodeConfig.objects.with_spec().filter(instance=ic, uuid__in=ports_by_node.keys()):
                assert nc.spec is not None
                spec = nc.spec.model_copy(deep=True)
                for port in spec.output_ports:
                    if port.id in ports_by_node[nc.uuid]:
                        port.dimensions = None
                # `save()` goes through ClusterableModel, which can revert modeltrans fields.
                NodeConfig.objects.filter(pk=nc.pk).update(spec=spec)
            if ic.live_revision_id is not None:
                ic.has_unpublished_changes = True
            ic.invalidate_cache(save=False)
            ic.save(update_fields=['has_unpublished_changes', 'cache_invalidated_at'])
