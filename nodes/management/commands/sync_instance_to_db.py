"""
Load a YAML instance via InstanceLoader and export its spec to the DB.

Usage:
    python manage.py sync_instance_to_db configs/espoo.yaml
    python manage.py sync_instance_to_db configs/espoo.yaml --dry-run
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand
from django.db import transaction

from nodes.models import InstanceConfig, NodeConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from nodes.defs import NodeSpec


class DryRunError(Exception):
    pass


class Command(BaseCommand):
    help = 'Load instance from YAML, export spec to InstanceConfig + NodeConfig'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instance', nargs='*', type=str, help='Instance identifier(s)')
        parser.add_argument('--all', action='store_true', help='Sync all instances')
        parser.add_argument('--dry-run', action='store_true', help='Load and export but do not save to DB')
        parser.add_argument('--start-from', type=str, help='Instance identifier to start from')

    def sync_one_instance(self, instance_id: str, dry_run: bool = False) -> None:
        from nodes.spec_export import sync_instance_to_db

        try:
            with transaction.atomic():
                try:
                    sync_instance_to_db(instance_id)
                except FileNotFoundError as e:
                    self.stderr.write(self.style.ERROR(f'Error loading YAML for {instance_id}: {e}'))
                    return
                self._print_summary(instance_id)
                if dry_run:
                    raise DryRunError()  # noqa: TRY301
        except DryRunError:
            return

    def handle(self, *args, **options) -> None:
        dry_run: bool = options['dry_run']

        if options['all']:
            instance_ids = InstanceConfig.objects.filter(framework_config__isnull=True).values_list('identifier', flat=True)
        else:
            instance_ids = options['instance']
        start_from = options['start_from']
        for instance_id in instance_ids:
            if start_from:
                if instance_id != start_from:
                    continue
                start_from = None
            self.sync_one_instance(instance_id, dry_run=dry_run)
        if dry_run:
            self.stdout.write(self.style.SUCCESS('Dry run complete — no changes made.'))
            return

        self.stdout.write(self.style.SUCCESS('Done. %d instances synced.' % len(instance_ids)))

    def _print_summary(self, instance_id: str) -> None:
        instance_spec = InstanceConfig.objects.get(identifier=instance_id).spec
        node_qs = NodeConfig.objects.qs.filter(instance__identifier=instance_id).active()
        node_specs: list[tuple[str, NodeSpec]] = list(node_qs.values_list('identifier', 'spec'))
        dataset_repo_url = instance_spec.dataset_repo.url if instance_spec.dataset_repo else None
        self.stdout.write('\n--- Instance Spec ---')
        self.stdout.write(f'  ID: {instance_id}')
        self.stdout.write(f'  Years: {instance_spec.years.model_dump()}')
        self.stdout.write(f'  Dataset repo: {dataset_repo_url or "(none)"}')
        self.stdout.write(f'  Features: {instance_spec.features}')
        self.stdout.write(f'  Global params: {len(instance_spec.params)}')
        self.stdout.write(f'  Action groups: {len(instance_spec.action_groups)}')
        self.stdout.write(f'  Scenarios: {len(instance_spec.scenarios)}')

        self.stdout.write(f'\n--- Node Specs ({len(node_specs)}) ---')
        for node_id, spec in node_specs[:5]:
            n_metrics = len(spec.output_ports)
            n_params = len(spec.params)
            self.stdout.write(f'  {node_id}: {n_metrics} metrics, {n_params} params, kind={spec.kind}')

        if len(node_specs) > 5:
            self.stdout.write(f'  ... and {len(node_specs) - 5} more')
