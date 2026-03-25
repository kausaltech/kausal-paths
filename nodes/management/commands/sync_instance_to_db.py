"""
Load a YAML instance via InstanceLoader and export its spec to the DB.

Usage:
    python manage.py sync_instance_to_db configs/espoo.yaml
    python manage.py sync_instance_to_db configs/espoo.yaml --dry-run
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from nodes.defs import NodeSpec
    from nodes.edges import Edge, EdgeDimension


def _serialize_edge_dimension(dim_id: str, ed: EdgeDimension) -> dict[str, Any]:
    """Serialize an EdgeDimension to the YAML-compatible dict format."""
    d: dict[str, Any] = {'id': dim_id}
    cat_ids = [c.id for c in ed.categories]
    if cat_ids:
        d['categories'] = cat_ids
    if ed.flatten:
        d['flatten'] = True
    if ed.exclude:
        d['exclude'] = True
    return d


def _edge_to_transforms(edge: Edge) -> dict[str, Any]:
    """Convert runtime Edge dimension mappings to a dict with from/to separated."""
    result: dict[str, Any] = {}
    if edge.from_dimensions:
        result['from_dimensions'] = [_serialize_edge_dimension(dim_id, ed) for dim_id, ed in edge.from_dimensions.items()]
    if edge.to_dimensions:
        result['to_dimensions'] = [_serialize_edge_dimension(dim_id, ed) for dim_id, ed in edge.to_dimensions.items()]
    return result


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

        # Export specs
        instance_spec = export_instance_spec(instance)

        node_specs: dict[str, tuple[str, NodeSpec]] = {}  # node_id -> (node_type, NodeSpec)
        for node_id, node in ctx.nodes.items():
            node_specs[node_id] = (type(node).__name__, export_node_spec(node))

        self.stdout.write(
            f'Exported specs: {len(instance_spec.scenarios)} scenarios, '
            f'{len(instance_spec.action_groups)} action groups, '
            f'{len(instance_spec.params)} global params',
        )

        if dry_run:
            self._print_summary(instance_spec, node_specs)
            self.stdout.write(self.style.SUCCESS('Dry run complete — no changes made.'))
            return

        # Save to DB
        from nodes.models import InstanceConfig, NodeConfig

        try:
            ic = InstanceConfig.objects.get(identifier=instance.id)
        except InstanceConfig.DoesNotExist as err:
            raise CommandError(f'InstanceConfig "{instance.id}" not found. Create it in the admin first.') from err

        with transaction.atomic():
            ic.spec = instance_spec
            ic.config_source = 'database'
            ic.save(update_fields=['spec', 'config_source'])

            saved_count = 0
            skipped: list[str] = []
            for node_id, (_node_type, spec) in node_specs.items():
                try:
                    nc = NodeConfig.objects.get(instance=ic, identifier=node_id)
                    nc.spec = spec
                    nc.save(update_fields=['spec'])
                    saved_count += 1
                except NodeConfig.DoesNotExist:
                    skipped.append(node_id)

            # Export edges
            edge_count = self._sync_edges(ic, ctx)

        if skipped:
            self.stderr.write(f'Warning: {len(skipped)} nodes not found in DB: {", ".join(skipped[:10])}')

        self.stdout.write(
            self.style.SUCCESS(
                f'Saved specs: InstanceConfig + {saved_count} NodeConfigs, {edge_count} edges',
            )
        )

    def _sync_edges(self, ic, ctx) -> int:
        from nodes.models import NodeConfig, NodeEdge

        # Clear existing edges
        NodeEdge.objects.filter(instance=ic).delete()

        nc_map: dict[str, NodeConfig] = {nc.identifier: nc for nc in NodeConfig.objects.filter(instance=ic)}

        edge_count = 0
        for node_id, node in ctx.nodes.items():
            from_nc = nc_map.get(node_id)
            if from_nc is None:
                continue
            for edge in node.edges:
                if edge.input_node.id != node_id:
                    continue  # only process outgoing edges
                to_nc = nc_map.get(edge.output_node.id)
                if to_nc is None:
                    continue
                # Build transformation list from edge dimensions
                transforms = _edge_to_transforms(edge)
                tags = list(edge.tags) if edge.tags else []
                NodeEdge.objects.create(
                    instance=ic,
                    from_node=from_nc,
                    from_port='output',
                    to_node=to_nc,
                    to_port=f'from_{node_id}',
                    transformations=transforms,
                    tags=tags,
                )
                edge_count += 1
        return edge_count

    def _print_summary(self, instance_spec, node_specs) -> None:
        self.stdout.write('\n--- Instance Spec ---')
        self.stdout.write(f'  Years: {instance_spec.years.model_dump()}')
        self.stdout.write(f'  Dataset repo: {instance_spec.dataset_repo.url or "(none)"}')
        self.stdout.write(f'  Features: {len(instance_spec.features)} keys')
        self.stdout.write(f'  Global params: {len(instance_spec.params)}')
        self.stdout.write(f'  Action groups: {len(instance_spec.action_groups)}')
        self.stdout.write(f'  Scenarios: {len(instance_spec.scenarios)}')

        self.stdout.write(f'\n--- Node Specs ({len(node_specs)}) ---')
        for node_id, (node_type, spec) in list(node_specs.items())[:5]:
            n_metrics = len(spec.output_metrics)
            n_params = len(spec.params)
            self.stdout.write(f'  {node_id} ({node_type}): {n_metrics} metrics, {n_params} params, class={spec.node_class}')
        if len(node_specs) > 5:
            self.stdout.write(f'  ... and {len(node_specs) - 5} more')
