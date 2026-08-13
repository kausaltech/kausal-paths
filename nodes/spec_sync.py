"""
Write half of the parse-only sync: apply a parsed ``InstanceSnapshot`` to the DB.

The parse half (``nodes/instance_parser.py``) is DB-free; everything that
requires database state — resolving dataset-metric bindings against the
dataset schemas, writing rows — lives here. The pairing itself is pure and
operates on ``DatasetSchemaInfo`` collected from the ORM, so it can also run
against captured schema state (the parse oracle does this).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid3

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Hashable
    from uuid import UUID

    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric
    from kausal_common.i18n.pydantic import TranslatedString

    from nodes.defs.node_defs import NodeSpec
    from nodes.instance_serialization import DatasetPortSnapshot, InstanceSnapshot, NodeSnapshot
    from nodes.models import InstanceConfig, NodeConfig
    from nodes.yaml_port_refs import YamlPortReferenceCatalog


@dataclass
class DatasetSchemaInfo:
    """The slice of a dataset's schema the binding resolution needs."""

    metric_keys: list[str] = field(default_factory=list)
    """Binding keys of the schema's metrics (name, else label, else uuid), in metric order."""

    metric_names: dict[str, str] = field(default_factory=dict)
    """Binding key → snapshot identity (metric name, else uuid) for each metric."""

    forecast_from: int | None = None
    """The dataset-level forecast default (``Dataset.spec.forecast_from``)."""


def collect_dataset_schema_info(ic: InstanceConfig) -> dict[str, DatasetSchemaInfo]:
    """Collect per-dataset schema info for an instance's datasets (by identifier)."""
    from kausal_common.datasets.models import DatasetMetric

    from nodes.spec_export import _dataset_metric_binding_key, _get_db_datasets

    db_datasets = _get_db_datasets(ic)
    schema_pks = {ds.schema.pk for ds in db_datasets.values() if ds.schema is not None}
    metrics_by_schema: dict[int, list[DatasetMetric]] = {}
    for metric in DatasetMetric.objects.filter(schema__pk__in=schema_pks).order_by('order'):
        metrics_by_schema.setdefault(metric.schema.pk, []).append(metric)

    result: dict[str, DatasetSchemaInfo] = {}
    for identifier, ds in db_datasets.items():
        if ds.schema is None:
            continue
        info = DatasetSchemaInfo(forecast_from=(ds.spec or {}).get('forecast_from'))
        for metric in metrics_by_schema.get(ds.schema.pk, []):
            key = _dataset_metric_binding_key(metric)
            info.metric_keys.append(key)
            info.metric_names[key] = metric.name or str(metric.uuid)
        result[identifier] = info
    return result


def _binding_columns(spec_column: str | None, node_spec: NodeSpec) -> list[str]:
    """Columns a binding exposes: the selected column, or the node's output columns."""
    if spec_column is not None:
        return [spec_column]
    columns: list[str] = []
    seen: set[str] = set()
    for port in node_spec.output_ports:
        if port.column_id is None:
            continue
        column = str(port.column_id)
        if column not in seen:
            columns.append(column)
            seen.add(column)
    return columns


def resolve_dataset_port_snapshots(  # noqa: C901, PLR0912
    snapshot: InstanceSnapshot,
    schemas: dict[str, DatasetSchemaInfo],
    *,
    port_references: YamlPortReferenceCatalog | None = None,
) -> list[DatasetPortSnapshot]:
    """
    Resolve the parse-side dataset-port entries against the dataset schemas.

    The parser emits one entry per node-side column with ``metric`` set to the
    column name (it has no schema access). This reproduces the pairing the
    runtime export does in ``_binding_pairs_for_dataset``: name-match the
    schema's metrics to the columns, fall back to keeping schema-metric-keyed
    rows when nothing pairs, and drop entries whose metric doesn't exist.
    """
    from nodes.instance_serialization import DatasetPortSnapshot
    from nodes.spec_export import pair_metrics_to_columns

    node_specs: dict[UUID, NodeSpec] = {}
    for n in snapshot.nodes:
        assert n.spec is not None
        if n.identifier is None:
            raise ValueError(f'Node {n.uuid} has no identifier; dataset port IDs still require one')
        node_specs[n.uuid] = n.spec

    # Group parse-side entries into bindings: (node, dataset_index) is binding identity.
    bindings: dict[tuple[UUID, int], list[DatasetPortSnapshot]] = {}
    for port in snapshot.dataset_ports:
        bindings.setdefault((port.node, port.dataset_index), []).append(port)

    resolved: list[DatasetPortSnapshot] = []
    for (node_id, dataset_index), ports in bindings.items():
        first = ports[0]
        spec = first.spec
        dataset_id = first.dataset
        schema = schemas.get(dataset_id)
        if schema is None:
            raise ValueError(f'No dataset object for {dataset_id} on node {node_id}')

        columns = _binding_columns(spec.column, node_specs[node_id])
        log_ctx = f'Dataset {dataset_id} on node {node_id}'
        if spec.column is not None:
            pairs = [(spec.column, spec.column)]
        elif not schema.metric_keys:
            pairs = [(column, column) for column in columns]
        else:
            pairs = pair_metrics_to_columns(columns, schema.metric_keys, log_ctx=log_ctx)
            if not pairs:
                logger.warning('%s: keeping bindings with unresolved port ids so the input dataset survives' % log_ctx)
                pairs = [(name, name) for name in schema.metric_keys]

        for port_column, metric_key in pairs:
            metric_name = schema.metric_names.get(metric_key)
            if metric_name is None:
                if spec.column is not None:
                    raise ValueError(f'No metric {metric_key} in dataset {dataset_id} for node {node_id}')
                logger.debug('No metric %s in dataset %s for node %s; skipping binding' % (metric_key, dataset_id, node_id))
                continue
            source_port = next((port for port in ports if port.metric == port_column), None)
            if source_port is not None:
                port_id = source_port.port_id
            else:
                node_identifier = next(node.identifier for node in snapshot.nodes if node.uuid == node_id)
                assert node_identifier is not None
                fallback_id = _dataset_port_uuid(snapshot.metadata.uuid, node_identifier, dataset_index, port_column)
                port_id = (
                    port_references.dataset_port_id(
                        node_id,
                        dataset_id,
                        dataset_index,
                        port_column,
                        fallback_id,
                        allow_group_fallback=True,
                        fail_on_ambiguous=True,
                    )
                    if port_references is not None
                    else fallback_id
                )
            resolved.append(
                DatasetPortSnapshot(
                    node=node_id,
                    dataset=dataset_id,
                    port_id=port_id,
                    metric=metric_name,
                    dataset_index=dataset_index,
                    spec=spec,
                )
            )
    return resolved


def _dataset_port_uuid(instance_uuid: UUID, node_id: str, dataset_index: int, column: str) -> UUID:
    return uuid3(instance_uuid, ':'.join([node_id, 'dataset', str(dataset_index), column]))


# ---------------------------------------------------------------------------
# Full parse-only sync: YAML → snapshot → DB rows (no runtime)
# ---------------------------------------------------------------------------


def _apply_metadata_columns(ic: InstanceConfig, snapshot: InstanceSnapshot) -> None:
    """Seed identity metadata while preserving values already authored in the DB."""
    meta = snapshot.metadata
    ic.update_identity_metadata(
        name=cast('str | TranslatedString', meta.name),
        owner=cast('str | TranslatedString | None', meta.owner),
        primary_language=meta.primary_language,
        other_languages=list(meta.other_languages),
    )


def _sync_dimensions_from_snapshot(ic: InstanceConfig, snapshot: InstanceSnapshot) -> None:
    """Mirror ``InstanceConfig.sync_dimensions`` from the spec's dimension configs."""
    from nodes.dimensions import Dimension

    for dim_config in snapshot.spec.dimensions:
        dim = Dimension.model_validate(dim_config)
        ic.sync_dimension(dim, update_existing=True)


def _seed_node_metadata_from_snapshot(nc: NodeConfig, n: NodeSnapshot, primary_language: str) -> None:
    """
    Seed an uninitialized NodeConfig from snapshot metadata.

    This covers both new rows and legacy rows whose NULL computation spec
    proves that they have not yet adopted YAML metadata. Once initialized,
    existing ORM metadata is authoritative.
    """
    from kausal_common.i18n.pydantic import get_modeltrans_attrs_from_str

    i18n: dict[str, str] = {}
    attributes: dict[str, object] = {
        'color': n.color,
        'order': n.order,
        'is_visible': n.is_visible,
    }
    for field_name, value in (
        ('name', n.name),
        ('short_name', n.short_name),
        ('short_description', n.short_description),
        ('description', n.description),
        ('goal', n.goal),
    ):
        if value is None:
            continue
        val, tr = get_modeltrans_attrs_from_str(value, field_name, primary_language, strict=False)
        i18n.update(tr)
        attributes[field_name] = val

    for key, value in attributes.items():
        setattr(nc, key, value)
    nc.i18n = i18n


def _write_edges(ic: InstanceConfig, snapshot: InstanceSnapshot, node_configs: dict[UUID, NodeConfig]) -> int:
    from nodes.instance_serialization import edge_match_keys, existing_edge_identities, match_preserved_uuids
    from nodes.models import NodeEdge

    # Recreating the rows keeps pk order equal to authored order, but the row
    # UUID is the durable binding identity and must survive the rewrite.
    authored_uuids = {edge.uuid for edge in snapshot.edges if edge.uuid is not None}
    existing = [item for item in existing_edge_identities(ic) if item[1] not in authored_uuids]
    NodeEdge.objects.filter(instance=ic).delete()
    preserved = match_preserved_uuids(
        existing,
        [edge_match_keys(edge.from_node, edge.from_port, edge.to_node, edge.to_port) for edge in snapshot.edges],
    )
    edge_objs = []
    for edge, matched_uuid in zip(snapshot.edges, preserved, strict=True):
        from_nc = node_configs.get(edge.from_node)
        to_nc = node_configs.get(edge.to_node)
        if from_nc is None or to_nc is None:
            raise ValueError(f'Edge references unknown node: {edge.from_node} -> {edge.to_node}')
        row_uuid = edge.uuid or matched_uuid
        identity_kwargs = {'uuid': row_uuid} if row_uuid is not None else {}
        edge_objs.append(
            NodeEdge(
                instance=ic,
                from_node=from_nc,
                from_port=edge.from_port,
                to_node=to_nc,
                to_port=edge.to_port,
                transformations=list(edge.transformations),
                tags=list(edge.tags),
                **identity_kwargs,
            )
        )
    NodeEdge.objects.bulk_create(edge_objs)
    return len(edge_objs)


def _write_dataset_ports(
    ic: InstanceConfig,
    snapshot: InstanceSnapshot,
    node_configs: dict[UUID, NodeConfig],
    *,
    port_references: YamlPortReferenceCatalog,
) -> int:
    """Resolve bindings against the DB schemas and write the DatasetPort rows."""
    from kausal_common.datasets.models import DatasetMetric

    from nodes.instance_serialization import (
        dataset_port_match_keys,
        existing_dataset_port_identities,
        match_preserved_uuids,
    )
    from nodes.models import DatasetPort
    from nodes.spec_export import _get_db_datasets

    existing = existing_dataset_port_identities(ic)
    DatasetPort.objects.filter(instance=ic).delete()
    schemas = collect_dataset_schema_info(ic)
    resolved = resolve_dataset_port_snapshots(snapshot, schemas, port_references=port_references)
    if not resolved:
        return 0

    db_datasets = _get_db_datasets(ic)
    schema_pks = {ds.schema.pk for ds in db_datasets.values() if ds.schema is not None}
    metric_by_identity: dict[tuple[int, str], DatasetMetric] = {}
    for metric in DatasetMetric.objects.filter(schema__pk__in=schema_pks):
        identity = metric.name or str(metric.uuid)
        metric_by_identity[(metric.schema.pk, identity)] = metric

    triples: list[tuple[DatasetPortSnapshot, DatasetModel, DatasetMetric]] = []
    match_keys: list[tuple[Hashable, ...]] = []
    for port in resolved:
        dataset_obj = db_datasets[port.dataset]
        assert dataset_obj.schema is not None
        metric = metric_by_identity[(dataset_obj.schema.pk, port.metric)]
        triples.append((port, dataset_obj, metric))
        match_keys.append(dataset_port_match_keys(port.node, dataset_obj.pk, port.dataset_index, metric.pk))

    # Recreated rows keep their durable UUIDs (binding identity) across the rewrite.
    port_objs: list[DatasetPort] = []
    for (port, dataset_obj, metric), matched_uuid in zip(triples, match_preserved_uuids(existing, match_keys), strict=True):
        identity_kwargs = {'uuid': matched_uuid} if matched_uuid is not None else {}
        port_objs.append(
            DatasetPort(
                instance=ic,
                node=node_configs[port.node],
                port_id=port.port_id,
                dataset=dataset_obj,
                metric=metric,
                spec=port.spec,
                dataset_index=port.dataset_index,
                **identity_kwargs,
            )
        )
    DatasetPort.objects.bulk_create(port_objs)
    return len(port_objs)


def _upsert_node_configs(
    ic: InstanceConfig,
    snapshot: InstanceSnapshot,
    existing_node_configs: list[NodeConfig] | None = None,
) -> dict[UUID, NodeConfig]:
    from nodes.models import NodeConfig

    if existing_node_configs is None:
        existing_node_configs = list(
            NodeConfig.objects.with_spec().filter(instance=ic).select_related('indicator_node', 'copy_of', 'layout')
        )
    existing_by_uuid = {nc.uuid: nc for nc in existing_node_configs}
    existing_by_identifier = {nc.identifier: nc for nc in existing_node_configs}
    node_configs: dict[UUID, NodeConfig] = {}
    touched_pks: set[int] = set()
    for n in snapshot.nodes:
        if n.identifier is None:
            raise ValueError(f'Node {n.uuid} has no identifier; DB sync still requires one')
        nc = existing_by_uuid.get(n.uuid) or existing_by_identifier.get(n.identifier)
        if nc is None:
            nc = NodeConfig(instance=ic, uuid=n.uuid, identifier=n.identifier)
        _seed_node_metadata_from_snapshot(nc, n, snapshot.metadata.primary_language)
        assert n.spec is not None
        nc.is_stale = False
        nc.save()
        # Write spec via queryset.update() to bypass ClusterableModel.save()
        # which silently reverts SchemaField values.
        NodeConfig.objects.filter(pk=nc.pk).update(spec=n.spec)
        nc.spec = n.spec
        node_configs[n.uuid] = nc
        touched_pks.add(nc.pk)

    # Stale = existing rows the snapshot didn't touch. Keyed by pk, not
    # identifier: a snapshot node can match an existing row by uuid while
    # carrying a new identifier (authored-uuid rename), and the row keeps its
    # old identifier — which must not make the row it belongs to stale.
    stale_nodes = ic.nodes.exclude(pk__in=touched_pks).defer('spec')
    if stale_nodes.exists():
        logger.warning(f'Detected {len(stale_nodes)} stale nodes: {stale_nodes.values_list("identifier", flat=True)}')
        stale_nodes.update(is_stale=True)
        delete_nodes = stale_nodes.filter(pages__isnull=True, created_by__isnull=True)
        for stale_node in delete_nodes:
            logger.info(f'Stale node {stale_node.identifier} was automatically created, removing')
            stale_node.delete()
    return node_configs


def sync_parsed_instance_to_db(
    instance_id: str,
    yaml_path: str | Path | None = None,
    *,
    promote_forecast_defaults: bool = True,
) -> None:
    """
    Parse an instance's YAML into specs and sync them to the DB — no runtime init.

    This is the parse-only replacement for ``nodes.spec_export.sync_instance_to_db``
    (which still exists as the runtime-derived baseline the parse oracle
    compares against).
    """
    from django.db import transaction

    from kausal_common.i18n.pydantic import set_i18n_context

    from nodes.dataset_placeholders import sync_dataset_placeholders_from_snapshot
    from nodes.instance_loader import InstanceYAMLConfig
    from nodes.instance_parser import parse_instance_snapshot
    from nodes.instance_serialization import reconcile_snapshot_node_metadata
    from nodes.models import InstanceConfig, NodeConfig
    from nodes.spec_export import _promote_dataset_forecast_defaults

    if yaml_path is None:
        yaml_path = Path(f'configs/{instance_id}.yaml').resolve()
    else:
        yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f'YAML file not found: {yaml_path}')

    yaml_conf = InstanceYAMLConfig.load_for_entrypoint(yaml_path)
    data = yaml_conf.data
    assert data is not None

    with transaction.atomic():
        ic, _created = InstanceConfig.objects.get_or_create(identifier=data['id'])
        existing_node_configs = list(
            NodeConfig.objects.with_spec().filter(instance=ic).select_related('indicator_node', 'copy_of', 'layout')
        )
        node_uuids = {nc.identifier: nc.uuid for nc in existing_node_configs}
        from nodes.yaml_port_refs import build_yaml_port_reference_catalog

        port_references = build_yaml_port_reference_catalog(ic)
        snapshot = parse_instance_snapshot(
            data,
            instance_uuid=ic.uuid,
            node_uuids=node_uuids,
            port_references=port_references,
        )
        snapshot.spec.features.use_datasets_from_db = True
        snapshot = reconcile_snapshot_node_metadata(snapshot, existing_node_configs)

        with set_i18n_context(snapshot.metadata.primary_language, list(snapshot.metadata.other_languages)):
            _apply_metadata_columns(ic, snapshot)
            ic.spec = snapshot.spec
            ic.config_source = 'database'
            ic.invalidate_cache(save=False)
            ic.save()

            _sync_dimensions_from_snapshot(ic, snapshot)
            node_configs = _upsert_node_configs(ic, snapshot, existing_node_configs)
            edge_count = _write_edges(ic, snapshot, node_configs)
            created_placeholder_ids = sync_dataset_placeholders_from_snapshot(ic, snapshot)
            dataset_port_count = _write_dataset_ports(
                ic,
                snapshot,
                node_configs,
                port_references=port_references,
            )
            promoted = _promote_dataset_forecast_defaults(ic) if promote_forecast_defaults else 0

            from nodes.input_bindings import sync_input_bindings

            sync_input_bindings(ic)

    logger.info(
        (
            'Synced {id} (parse-only): {nodes} nodes, {edges} edges, {placeholders} dataset placeholders created, '
            '{ports} dataset ports, {forecast_defaults} dataset forecast defaults promoted'
        ),
        id=instance_id,
        nodes=len(node_configs),
        edges=edge_count,
        placeholders=len(created_placeholder_ids),
        ports=dataset_port_count,
        forecast_defaults=promoted,
    )
