"""
Load a YAML instance via InstanceLoader and export its spec to the DB.

Usage:
    python manage.py sync_instance_to_db configs/espoo.yaml
    python manage.py sync_instance_to_db configs/espoo.yaml --dry-run
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from django.core.management.base import BaseCommand, CommandError

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Load instance from YAML, export spec to InstanceConfig + NodeConfig'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('yaml_file', type=str, help='Path to the YAML config file')
        parser.add_argument('--dry-run', action='store_true', help='Load and export but do not save to DB')

    def handle(self, *args, **options) -> None:
        from nodes.instance_loader import InstanceLoader
        from nodes.spec_export import export_instance_spec, export_node_spec

        yaml_path = Path(options['yaml_file']).resolve()
        if not yaml_path.exists():
            raise CommandError(f'File not found: {yaml_path}')

        dry_run: bool = options['dry_run']

        self.stdout.write(f'Loading instance from {yaml_path}...')
        loader = InstanceLoader.from_yaml(yaml_path)
        instance = loader.instance
        ctx = loader.context
        self.stdout.write(f'Loaded instance: {instance.id} ({len(ctx.nodes)} nodes)')

        if dry_run:
            instance_spec = export_instance_spec(instance)
            node_specs = {nid: export_node_spec(n) for nid, n in ctx.nodes.items()}
            self._print_summary(instance_spec, node_specs)
            self.stdout.write(self.style.SUCCESS('Dry run complete — no changes made.'))
            return

        from nodes.spec_export import sync_instance_to_db

        sync_instance_to_db(instance.id, yaml_path=yaml_path)
        self.stdout.write(self.style.SUCCESS('Done.'))

    def _print_summary(self, instance_spec, node_specs) -> None:
        self.stdout.write('\n--- Instance Spec ---')
        self.stdout.write(f'  Years: {instance_spec.years.model_dump()}')
        self.stdout.write(f'  Dataset repo: {instance_spec.dataset_repo.url or "(none)"}')
        self.stdout.write(f'  Features: {instance_spec.features}')
        self.stdout.write(f'  Global params: {len(instance_spec.params)}')
        self.stdout.write(f'  Action groups: {len(instance_spec.action_groups)}')
        self.stdout.write(f'  Scenarios: {len(instance_spec.scenarios)}')

        self.stdout.write(f'\n--- Node Specs ({len(node_specs)}) ---')
        for node_id, spec in list(node_specs.items())[:5]:
            n_metrics = len(spec.output_metrics)
            n_params = len(spec.params)
            self.stdout.write(f'  {node_id}: {n_metrics} metrics, {n_params} params, class={spec.node_class}')
        if len(node_specs) > 5:
            self.stdout.write(f'  ... and {len(node_specs) - 5} more')
