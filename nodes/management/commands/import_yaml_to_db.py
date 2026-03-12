"""
Import a YAML instance configuration into the model editor DB models.

Usage:
    python manage.py import_yaml_to_db configs/helsinki.yaml
    python manage.py import_yaml_to_db configs/espoo.yaml --dry-run
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

if TYPE_CHECKING:
    from nodes.models import ActionGroup, InstanceConfig, NodeConfig

I18N_FIELDS = ['name', 'short_description', 'description', 'goal']


class Command(BaseCommand):
    help = 'Import a YAML instance config into model editor DB models'

    def add_arguments(self, parser):
        parser.add_argument('yaml_file', type=str, help='Path to the YAML config file')
        parser.add_argument('--dry-run', action='store_true', help='Parse and validate without saving')

    def handle(self, *args, **options):
        yaml_path = Path(options['yaml_file']).resolve()
        if not yaml_path.exists():
            raise CommandError(f'File not found: {yaml_path}')

        dry_run = options['dry_run']

        self.stdout.write(f'Loading YAML config from {yaml_path}...')
        data = self._load_yaml(yaml_path)
        instance_id = data.get('id', data.get('identifier', ''))
        self.stdout.write(f'Instance: {instance_id}')

        if dry_run:
            self._report_contents(data)
            self.stdout.write(self.style.SUCCESS('Dry run complete — no changes made.'))
            return

        with transaction.atomic():
            ic = _import_instance(data, self.stdout, self.stderr)

        self.stdout.write(self.style.SUCCESS(f'Successfully imported "{ic.identifier}" (config_source=database, pk={ic.pk})'))

    def _load_yaml(self, yaml_path: Path) -> dict[str, Any]:
        from nodes.instance_loader import InstanceYAMLConfig

        yaml_conf = InstanceYAMLConfig.from_entrypoint(yaml_path)
        yaml_conf.load()
        data = yaml_conf.data
        assert data is not None
        return data

    def _report_contents(self, data: dict[str, Any]) -> None:
        for label, key in [
            ('Nodes', 'nodes'),
            ('Actions', 'actions'),
            ('Scenarios', 'scenarios'),
            ('Action groups', 'action_groups'),
            ('Dimensions', 'dimensions'),
        ]:
            self.stdout.write(f'  {label}: {len(data.get(key, []))}')


def _import_instance(data: dict[str, Any], stdout: Any, stderr: Any) -> InstanceConfig:
    from nodes.models import InstanceConfig

    instance_id = data.get('id', data.get('identifier', ''))
    try:
        ic = InstanceConfig.objects.get(identifier=instance_id)
        stdout.write(f'Updating existing InstanceConfig: {ic.identifier}')
    except InstanceConfig.DoesNotExist as err:
        msg = f'InstanceConfig "{instance_id}" does not exist. Create it in the admin first.'
        raise CommandError(msg) from err

    _update_instance_fields(ic, data)
    ic.clear_model_editor_data()
    ag_map = _import_action_groups(ic, data)
    node_map, action_ids = _import_nodes(ic, data, ag_map)
    edge_count = _import_edges(ic, data, node_map, stderr)
    _import_scenarios(ic, data)

    stdout.write(
        f'  Imported: {len(node_map)} nodes ({len(action_ids)} actions), '
        + f'{edge_count} edges, {len(data.get("scenarios", []))} scenarios, '
        + f'{len(ag_map)} action groups'
    )
    return ic


def _update_instance_fields(ic: InstanceConfig, data: dict[str, Any]) -> None:
    ic.config_source = 'database'
    for field in ('target_year', 'reference_year', 'minimum_historical_year', 'maximum_historical_year', 'model_end_year'):
        setattr(ic, field, data.get(field))
    ic.emission_unit = data.get('emission_unit', '')
    ic.features = data.get('features', {})
    ic.parameters = data.get('parameters', [])

    dataset_repo = data.get('dataset_repo', {})
    if isinstance(dataset_repo, dict):
        ic.dataset_repo_url = dataset_repo.get('url', '')
        ic.dataset_repo_commit = dataset_repo.get('commit')
        ic.dataset_repo_dvc_remote = dataset_repo.get('dvc_remote')

    modeled_keys = {
        'id',
        'identifier',
        'default_language',
        'name',
        'owner',
        'site_url',
        'supported_languages',
        'target_year',
        'reference_year',
        'minimum_historical_year',
        'maximum_historical_year',
        'model_end_year',
        'emission_unit',
        'features',
        'parameters',
        'dataset_repo',
        'nodes',
        'actions',
        'scenarios',
        'action_groups',
        'dimensions',
        'emission_sectors',
        'include',
        'frameworks',
    }
    ic.extra = {k: v for k, v in data.items() if k not in modeled_keys}
    ic.save()


def _import_action_groups(ic: InstanceConfig, data: dict[str, Any]) -> dict[str, ActionGroup]:
    from nodes.models import ActionGroup

    ag_map: dict[str, ActionGroup] = {}
    for idx, ag_data in enumerate(data.get('action_groups', [])):
        ag = ActionGroup.objects.create(
            instance=ic,
            identifier=ag_data['id'],
            name=ag_data.get('name', ag_data['id']),
            color=ag_data.get('color', ''),
            order=ag_data.get('order', idx),
            i18n=_extract_i18n(ag_data, ['name']),
        )
        ag_map[ag.identifier] = ag
    return ag_map


def _import_nodes(
    ic: InstanceConfig,
    data: dict[str, Any],
    ag_map: dict[str, ActionGroup],
) -> tuple[dict[str, NodeConfig], set[str]]:
    from nodes.models import NodeConfig

    all_nodes = data.get('nodes', []) + data.get('actions', [])
    action_ids = {a.get('id', a.get('identifier', '')) for a in data.get('actions', [])}
    node_map: dict[str, NodeConfig] = {}

    for node_data in all_nodes:
        node_id = node_data.get('id', node_data.get('identifier', ''))
        is_action = node_id in action_ids
        nc, _created = NodeConfig.objects.update_or_create(
            instance=ic,
            identifier=node_id,
            defaults=_node_defaults(node_data, is_action, ag_map),
        )
        node_map[node_id] = nc

    # Remove nodes no longer in the YAML (skip protected ones)
    stale_nodes = NodeConfig.objects.filter(instance=ic).exclude(identifier__in=set(node_map.keys()))
    for nc in stale_nodes:
        nc.delete()
    return node_map, action_ids


def _import_edges(
    ic: InstanceConfig,
    data: dict[str, Any],
    node_map: dict[str, NodeConfig],
    stderr: Any,
) -> int:
    from nodes.models import NodeEdge

    all_nodes = data.get('nodes', []) + data.get('actions', [])
    edge_count = 0

    for node_data in all_nodes:
        node_id = node_data.get('id', node_data.get('identifier', ''))
        nc = node_map[node_id]

        for out_conf in node_data.get('output_nodes', []):
            target_id = out_conf if isinstance(out_conf, str) else out_conf.get('id', out_conf.get('identifier', ''))
            if target_id not in node_map:
                stderr.write(f'  Warning: output node "{target_id}" not found for "{node_id}", skipping edge')
                continue
            transforms = _extract_transforms(out_conf) if isinstance(out_conf, dict) else []
            tags = out_conf.get('tags', []) if isinstance(out_conf, dict) else []

            NodeEdge.objects.create(
                instance=ic,
                from_node=nc,
                from_port='output',
                to_node=node_map[target_id],
                to_port=f'from_{node_id}',
                transformations=transforms,
                tags=tags,
            )
            edge_count += 1

    return edge_count


def _import_scenarios(ic: InstanceConfig, data: dict[str, Any]) -> None:
    from nodes.models import Scenario

    for s_data in data.get('scenarios', []):
        s_id = s_data.get('id', s_data.get('identifier', ''))
        kind = ''
        if s_data.get('default'):
            kind = 'default'
        elif s_data.get('kind'):
            kind = s_data['kind']

        overrides = []
        for p in s_data.get('params', []):
            override: dict[str, Any] = {'parameter_id': p.get('id', ''), 'value': p.get('value')}
            if p.get('node'):
                override['node_id'] = p['node']
            overrides.append(override)

        Scenario.objects.create(
            instance=ic,
            identifier=s_id,
            name=s_data.get('name', s_id),
            description=s_data.get('description', ''),
            kind=kind,
            all_actions_enabled=s_data.get('all_actions_enabled', False),
            parameter_overrides=overrides,
            i18n=_extract_i18n(s_data, ['name', 'description']),
        )


def _node_defaults(node_data: dict[str, Any], is_action: bool, ag_map: dict[str, ActionGroup]) -> dict[str, Any]:
    node_modeled_keys = {
        'id',
        'identifier',
        'name',
        'type',
        'quantity',
        'unit',
        'color',
        'order',
        'node_group',
        'is_outcome',
        'is_visible',
        'params',
        'input_ports',
        'output_ports',
        'pipeline',
        'formula',
        'group',
        'decision_level',
        'output_nodes',
        'input_nodes',
        'input_datasets',
    }
    i18n_data = _extract_i18n(node_data, I18N_FIELDS)

    defaults: dict[str, Any] = {
        'name': node_data.get('name'),
        'node_type': 'action' if is_action else 'formula',
        'quantity': node_data.get('quantity', ''),
        'unit': node_data.get('unit', ''),
        'color': node_data.get('color', ''),
        'order': node_data.get('order'),
        'node_group': node_data.get('node_group', ''),
        'is_outcome': node_data.get('is_outcome', False),
        'is_visible': node_data.get('is_visible', True),
        'params': node_data.get('params'),
        'input_ports': node_data.get('input_ports', []),
        'output_ports': node_data.get('output_ports', []),
        'pipeline': node_data.get('pipeline'),
        'formula': node_data.get('formula'),
        'decision_level': node_data.get('decision_level'),
        'i18n': i18n_data or None,
    }

    # Store Python class type in extra for round-trip
    extra: dict[str, Any] = {}
    if node_data.get('type'):
        extra['type'] = node_data['type']
    extra.update({k: v for k, v in node_data.items() if k not in node_modeled_keys and not _is_i18n_key(k, I18N_FIELDS)})
    defaults['extra'] = extra

    group_id = node_data.get('group')
    if group_id and group_id in ag_map:
        defaults['action_group'] = ag_map[group_id]

    return defaults


def _extract_i18n(data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {k: v for k, v in data.items() if _is_i18n_key(k, fields)}


def _is_i18n_key(key: str, fields: list[str]) -> bool:
    return any(re.match(rf'^{field}_[a-z]{{2}}(-[A-Z]{{2}})?$', key) for field in fields)


def _extract_transforms(edge_conf: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert from_dimensions/to_dimensions to normalized transformation format."""
    transforms: list[dict[str, Any]] = [
        {'kind': 'flatten', 'dimension': dim['id']} for dim in edge_conf.get('from_dimensions', []) if dim.get('flatten')
    ]

    for dim in edge_conf.get('to_dimensions', []):
        categories = dim.get('categories', [])
        if categories and len(categories) == 1 and not dim.get('flatten') and not dim.get('exclude'):
            transforms.append({'kind': 'assign_category', 'dimension': dim['id'], 'category': categories[0]})
        elif categories or dim.get('flatten') or dim.get('exclude'):
            t: dict[str, Any] = {'kind': 'select_categories', 'dimension': dim['id']}
            if categories:
                t['categories'] = categories
            if dim.get('flatten'):
                t['flatten'] = True
            if dim.get('exclude'):
                t['exclude'] = True
            transforms.append(t)

    return transforms
