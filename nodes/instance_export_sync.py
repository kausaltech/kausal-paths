"""
Compile YAML exports and apply their authored metadata to an existing DB mirror.

This is the first, deliberately narrow slice of InstanceExport-native sync. It
replaces the runtime ``Node`` as the source for ``load_nodes --update-nodes``:
the YAML is parsed directly into an export, changes are planned without
writes, and the same plan is then applied transactionally.

Dataset bindings remain in ``InstanceExport.instance.dataset_ports``. Dataset
bodies are intentionally omitted until repeated dataset import has explicit
ownership and conflict semantics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from django.db import transaction

from kausal_common.i18n.pydantic import get_modeltrans_attrs_from_str

if TYPE_CHECKING:
    from uuid import UUID

    from nodes.instance_serialization import InstanceExport, NodeSnapshot
    from nodes.models import InstanceConfig, NodeConfig


@dataclass(frozen=True)
class InstanceExportFieldChange:
    path: str
    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        return {'path': self.path, 'before': self.before, 'after': self.after}


@dataclass(frozen=True)
class InstanceExportEntityChange:
    entity_type: Literal['node']
    operation: Literal['create', 'update', 'delete']
    uuid: UUID
    identifier: str
    fields: tuple[InstanceExportFieldChange, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'entityType': self.entity_type,
            'operation': self.operation,
            'uuid': str(self.uuid),
            'identifier': self.identifier,
            'fields': [field.to_dict() for field in self.fields],
        }


@dataclass(frozen=True)
class InstanceExportSyncPlan:
    instance_identifier: str
    update_existing: bool
    overwrite: bool
    skip_descriptions: bool
    delete_stale_nodes: bool
    changes: tuple[InstanceExportEntityChange, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.changes)

    def to_dict(self) -> dict[str, Any]:
        counts = dict.fromkeys(('create', 'update', 'delete'), 0)
        for change in self.changes:
            counts[change.operation] += 1
        return {
            'instanceIdentifier': self.instance_identifier,
            'options': {
                'updateExisting': self.update_existing,
                'overwrite': self.overwrite,
                'skipDescriptions': self.skip_descriptions,
                'deleteStaleNodes': self.delete_stale_nodes,
            },
            'summary': {
                'nodesCreated': counts['create'],
                'nodesUpdated': counts['update'],
                'nodesDeleted': counts['delete'],
            },
            'changes': [change.to_dict() for change in self.changes],
        }


@dataclass(frozen=True)
class InstanceExportApplyResult:
    plan: InstanceExportSyncPlan
    node_configs: dict[UUID, NodeConfig]


def compile_instance_export_from_yaml(
    ic: InstanceConfig,
    yaml_path: str | Path | None = None,
) -> InstanceExport:
    """
    Compile YAML into a raw ``InstanceExport`` using only target identity.

    Existing node and port UUIDs preserve legacy identity, but no display
    metadata is read back from ``NodeConfig``. Dataset bodies are omitted in
    this milestone; dataset references remain part of the structural snapshot.
    """
    from nodes.instance_loader import InstanceYAMLConfig
    from nodes.instance_parser import parse_instance_snapshot
    from nodes.instance_serialization import InstanceExport
    from nodes.yaml_port_refs import build_yaml_port_reference_catalog

    if ic.has_framework_config():
        raise ValueError(f'Raw YAML export compilation is not supported for framework-backed instance {ic.identifier!r}')

    if yaml_path is None:
        resolved_path = ic.get_yaml_config_entrypoint()
        if resolved_path is None:
            raise FileNotFoundError(f'No YAML config entrypoint found for instance {ic.identifier}')
    else:
        resolved_path = Path(yaml_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f'YAML file not found: {resolved_path}')

    yaml_conf = InstanceYAMLConfig.load_for_entrypoint(resolved_path)
    data = yaml_conf.data
    assert data is not None
    if data['id'] != ic.identifier:
        raise ValueError(f'YAML instance {data["id"]!r} does not match target {ic.identifier!r}')

    node_uuids = dict(ic.nodes.values_list('identifier', 'uuid'))
    snapshot = parse_instance_snapshot(
        data,
        instance_uuid=ic.uuid,
        node_uuids=node_uuids,
        port_references=build_yaml_port_reference_catalog(ic),
    )
    return InstanceExport(instance=snapshot)


def _node_metadata_attributes(node: NodeSnapshot, primary_language: str) -> dict[str, Any]:
    if node.identifier is None:
        raise ValueError(f'Node {node.uuid} has no identifier; DB sync still requires one')

    attributes: dict[str, Any] = {
        'identifier': node.identifier,
        'order': node.order,
        'is_visible': node.is_visible,
    }
    if node.is_editable is not None:
        attributes['is_editable'] = node.is_editable
    i18n: dict[str, str] = {}
    for field_name, value in (
        ('name', node.name),
        ('short_name', node.short_name),
        ('short_description', node.short_description),
    ):
        if value is None:
            continue
        primary_value, translations = get_modeltrans_attrs_from_str(
            value,
            field_name,
            primary_language,
            strict=False,
        )
        attributes[field_name] = primary_value
        i18n.update(translations)
    attributes['i18n'] = i18n
    if node.color:
        attributes['color'] = node.color
    return attributes


def _flatten_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    flattened = {key: value for key, value in attributes.items() if key != 'i18n'}
    flattened.update({f'i18n.{key}': value for key, value in attributes.get('i18n', {}).items()})
    return flattened


def _create_changes(attributes: dict[str, Any]) -> tuple[InstanceExportFieldChange, ...]:
    return tuple(
        InstanceExportFieldChange(path=path, before=None, after=value)
        for path, value in sorted(_flatten_attributes(attributes).items())
    )


def _target_dataset_ids_by_node(ic: InstanceConfig, export: InstanceExport) -> dict[UUID, list[str]]:
    if not export.instance.spec.features.use_datasets_from_db:
        return {}

    from kausal_common.datasets.models import Dataset

    available = set(
        Dataset.objects.qs
        .for_instance_config(ic)
        .filter(is_external_placeholder=False, identifier__isnull=False)
        .values_list('identifier', flat=True)
    )
    from nodes.instance_serialization import group_dataset_bindings

    result: dict[UUID, list[str]] = {}
    for node_uuid, node_groups in group_dataset_bindings(export.instance).items():
        for _spec, dataset_id, _rows in node_groups:
            if dataset_id not in available:
                continue
            dataset_ids = result.setdefault(node_uuid, [])
            if dataset_id not in dataset_ids:
                dataset_ids.append(dataset_id)
    return result


def _updated_attributes(  # noqa: C901
    node_config: NodeConfig,
    incoming: dict[str, Any],
    *,
    overwrite: bool,
    skip_descriptions: bool,
) -> tuple[dict[str, Any], tuple[InstanceExportFieldChange, ...]]:
    updates: dict[str, Any] = {}
    changes: list[InstanceExportFieldChange] = []

    for field_name, incoming_value in incoming.items():
        if field_name == 'i18n':
            continue
        if skip_descriptions and field_name in {'short_description', 'description'}:
            continue
        current_value = getattr(node_config, field_name, None)
        if (overwrite or current_value is None) and current_value != incoming_value:
            updates[field_name] = incoming_value
            changes.append(InstanceExportFieldChange(path=field_name, before=current_value, after=incoming_value))

    current_i18n = dict(node_config.i18n or {})
    desired_i18n = dict(current_i18n)
    for key, incoming_value in incoming.get('i18n', {}).items():
        if skip_descriptions and key.startswith(('short_description_', 'description_')):
            continue
        if overwrite:
            desired_i18n[key] = incoming_value
        else:
            desired_i18n.setdefault(key, incoming_value)
    if desired_i18n != current_i18n:
        updates['i18n'] = desired_i18n
        for key in sorted(set(current_i18n) | set(desired_i18n)):
            before = current_i18n.get(key)
            after = desired_i18n.get(key)
            if before != after:
                changes.append(InstanceExportFieldChange(path=f'i18n.{key}', before=before, after=after))

    return updates, tuple(changes)


def plan_load_nodes_instance_export_sync(  # noqa: C901
    ic: InstanceConfig,
    export: InstanceExport,
    *,
    update_existing: bool,
    overwrite: bool,
    skip_descriptions: bool,
    delete_stale_nodes: bool,
) -> InstanceExportSyncPlan:
    """Plan the NodeConfig changes matching the historical load_nodes flags."""
    existing = list(ic.nodes.defer('spec'))
    by_uuid = {node.uuid: node for node in existing}
    by_identifier = {node.identifier: node for node in existing}
    matched_pks: set[int] = set()
    changes: list[InstanceExportEntityChange] = []
    dataset_ids_by_node = _target_dataset_ids_by_node(ic, export)

    for node in export.instance.nodes:
        if node.identifier is None:
            raise ValueError(f'Node {node.uuid} has no identifier; DB sync still requires one')
        node_config = by_uuid.get(node.uuid) or by_identifier.get(node.identifier)
        incoming = _node_metadata_attributes(node, export.instance.metadata.primary_language)
        desired_dataset_ids = dataset_ids_by_node.get(node.uuid, [])
        if node_config is None:
            field_changes = list(_create_changes(incoming))
            if desired_dataset_ids:
                field_changes.append(InstanceExportFieldChange(path='datasets', before=[], after=desired_dataset_ids))
            changes.append(
                InstanceExportEntityChange(
                    entity_type='node',
                    operation='create',
                    uuid=node.uuid,
                    identifier=node.identifier,
                    fields=tuple(field_changes),
                )
            )
            continue

        matched_pks.add(node_config.pk)
        if not update_existing:
            continue
        _updates, field_changes = _updated_attributes(
            node_config,
            incoming,
            overwrite=overwrite,
            skip_descriptions=skip_descriptions,
        )
        current_dataset_ids = sorted(
            identifier for identifier in node_config.datasets.values_list('identifier', flat=True) if identifier is not None
        )
        if current_dataset_ids != desired_dataset_ids:
            field_changes = (
                *field_changes,
                InstanceExportFieldChange(path='datasets', before=current_dataset_ids, after=desired_dataset_ids),
            )
        if field_changes:
            changes.append(
                InstanceExportEntityChange(
                    entity_type='node',
                    operation='update',
                    uuid=node_config.uuid,
                    identifier=node.identifier,
                    fields=field_changes,
                )
            )

    if delete_stale_nodes:
        for node_config in existing:
            if node_config.pk in matched_pks:
                continue
            changes.append(
                InstanceExportEntityChange(
                    entity_type='node',
                    operation='delete',
                    uuid=node_config.uuid,
                    identifier=node_config.identifier,
                )
            )

    changes.sort(key=lambda change: (change.identifier, change.operation))
    return InstanceExportSyncPlan(
        instance_identifier=ic.identifier,
        update_existing=update_existing,
        overwrite=overwrite,
        skip_descriptions=skip_descriptions,
        delete_stale_nodes=delete_stale_nodes,
        changes=tuple(changes),
    )


def _changes_to_attributes(changes: tuple[InstanceExportFieldChange, ...]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    i18n: dict[str, str] = {}
    for change in changes:
        if change.path == 'datasets':
            continue
        if change.path.startswith('i18n.'):
            i18n[change.path.removeprefix('i18n.')] = change.after
        else:
            attributes[change.path] = change.after
    if i18n:
        attributes['i18n'] = i18n
    return attributes


def apply_load_nodes_instance_export_sync(
    ic: InstanceConfig,
    export: InstanceExport,
    *,
    update_existing: bool,
    overwrite: bool,
    skip_descriptions: bool,
    delete_stale_nodes: bool,
) -> InstanceExportApplyResult:
    """Plan and atomically apply load_nodes-compatible NodeConfig changes."""
    from kausal_common.datasets.models import Dataset

    from nodes.models import InstanceConfig, NodeConfig

    with transaction.atomic():
        locked = InstanceConfig.objects.select_for_update().get(pk=ic.pk)
        plan = plan_load_nodes_instance_export_sync(
            locked,
            export,
            update_existing=update_existing,
            overwrite=overwrite,
            skip_descriptions=skip_descriptions,
            delete_stale_nodes=delete_stale_nodes,
        )
        node_configs = {node.uuid: node for node in locked.nodes.defer('spec')}
        node_configs_by_identifier = {node.identifier: node for node in node_configs.values()}
        target_datasets = {
            dataset.identifier: dataset
            for dataset in Dataset.objects.qs.for_instance_config(locked).filter(
                is_external_placeholder=False, identifier__isnull=False
            )
        }

        for change in plan.changes:
            if change.operation == 'create':
                attributes = _changes_to_attributes(change.fields)
                node_config = NodeConfig.objects.create(instance=locked, uuid=change.uuid, **attributes)
                dataset_change = next((field for field in change.fields if field.path == 'datasets'), None)
                if dataset_change is not None:
                    node_config.datasets.set([target_datasets[identifier] for identifier in dataset_change.after])
                node_configs[change.uuid] = node_config
                node_configs_by_identifier[node_config.identifier] = node_config
                continue

            node_config = node_configs.get(change.uuid) or node_configs_by_identifier.get(change.identifier)
            if node_config is None:
                raise RuntimeError(f'Planned node {change.identifier!r} disappeared before apply')
            if change.operation == 'delete':
                node_config.delete()
                node_configs.pop(change.uuid, None)
                node_configs_by_identifier.pop(change.identifier, None)
                continue

            attributes = _changes_to_attributes(change.fields)
            incoming_i18n = attributes.pop('i18n', {})
            for field_name, value in attributes.items():
                setattr(node_config, field_name, value)
            if incoming_i18n:
                node_config.i18n = {**(node_config.i18n or {}), **incoming_i18n}
            node_config.save()
            dataset_change = next((field for field in change.fields if field.path == 'datasets'), None)
            if dataset_change is not None:
                node_config.datasets.set([target_datasets[identifier] for identifier in dataset_change.after])
            node_configs[change.uuid] = node_config
            node_configs_by_identifier[node_config.identifier] = node_config

    return InstanceExportApplyResult(plan=plan, node_configs=node_configs)
