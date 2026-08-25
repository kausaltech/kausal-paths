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

from nodes.instance_serialization import DatasetPortSnapshot

if TYPE_CHECKING:
    from collections.abc import Hashable
    from uuid import UUID

    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetMetric
    from kausal_common.i18n.pydantic import TranslatedString

    from datasets.validation_rules import ValidationRule
    from nodes.defs.graph import DatasetMeta
    from nodes.defs.node_defs import NodeSpec
    from nodes.instance_serialization import InstanceSnapshot, NodeSnapshot
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
    for port in snapshot.dataset_bindings:
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


def _apply_declared_dataset_editability(
    dataset: DatasetModel,
    metadata: DatasetMeta,
    declarations: dict[int, tuple[str, bool]],
) -> None:
    """Apply one explicit schema lock, rejecting contradictory declarations for a shared schema."""
    if metadata.is_editable is None:
        return
    assert dataset.schema is not None
    assert dataset.schema_id is not None
    assert dataset.identifier is not None
    previous = declarations.get(dataset.schema_id)
    if previous is not None and previous[1] != metadata.is_editable:
        raise ValueError(
            f"datasets '{previous[0]}' and '{dataset.identifier}' share a schema but declare conflicting is_editable values"
        )
    declarations[dataset.schema_id] = (dataset.identifier, metadata.is_editable)
    if dataset.schema.is_editable != metadata.is_editable:
        dataset.schema.is_editable = metadata.is_editable
        dataset.schema.save(update_fields=['is_editable'])


def _sync_dataset_metadata_from_snapshot(ic: InstanceConfig, snapshot: InstanceSnapshot) -> None:
    """
    Reconcile schema editability and metric validation rules declared under ``datasets``.

    Only datasets named in the config are managed. Explicit ``is_editable``
    values update the shared schema; an absent value preserves its current DB
    state. Metric rule rows are replaced when the declared blob list differs
    (preserving rows — and thus rule uuids — when it does not). Datasets absent
    from the config are left untouched. Invalid or conflicting declarations
    fail the sync loudly.
    """
    from kausal_common.datasets.models import Dataset

    from nodes.dataset_materialization import refresh_dataset_materialization

    declared_schema_editability: dict[int, tuple[str, bool]] = {}
    declared_schema_domains: dict[int, tuple[str, object]] = {}
    for ds_meta in snapshot.datasets:
        ds_id = ds_meta.identifier
        if not ds_id:
            raise ValueError('datasets entry is missing an identifier')
        try:
            dataset = Dataset.objects.get_queryset().for_instance_config(ic).get(identifier=ds_id)
        except Dataset.DoesNotExist:
            # A module declares ownership for every dataset it reads, but an including
            # instance legitimately uses only a subset: overriding a node drops the datasets
            # only that node read. Warn rather than raise, so one city's override cannot
            # break the sync for a declaration that is correct for the module. Enforcement is
            # unaffected — a dataset that is not there cannot be left wrongly editable.
            logger.warning(
                f"datasets entry '{ds_id}' matches no dataset of instance {ic.identifier}; "
                'skipping (check for a typo if the instance is meant to use it)'
            )
            continue
        if dataset.schema is None:
            raise ValueError(f"dataset '{ds_id}' has no schema")
        _apply_declared_dataset_editability(dataset, ds_meta, declared_schema_editability)
        domain_changed = _apply_declared_category_domain(
            ic,
            dataset,
            ds_meta,
            declared_schema_domains,
        )
        metrics_by_name = {metric.name: metric for metric in dataset.schema.metrics.all()}
        dataset_changed = domain_changed
        for metric_meta in ds_meta.metrics:
            metric = metrics_by_name.get(metric_meta.identifier) if metric_meta.identifier else None
            if metric is None:
                # Same reasoning as the missing-dataset case above, one level down: a module
                # declares rules for every metric it may read, and an including instance
                # legitimately carries only a subset. A `dataset_replacements` entry can swap in a
                # city dataset that deliberately omits a column -- `kommune/kwk_anlagenparameter`
                # declares `t_supply`, and Mainz replaces it with a dataset that has none, because
                # its utility gives a range rather than a figure and `ChpNode` falls back to the
                # node parameter when the column is absent.
                #
                # Warning is safe because this declaration is an *edit constraint*, not a binding:
                # a rule with no metric constrains nothing and cannot leave a dataset wrongly
                # editable. A metric a node actually binds is still enforced, and still raises --
                # see `No metric ... for node ...` above and in `spec_export.py`.
                logger.warning(
                    f"datasets entry '{ds_id}' declares rules for metric "
                    f"'{metric_meta.identifier}', which the dataset of instance "
                    f'{ic.identifier} does not have; skipping those rules '
                    '(check for a typo if the metric is meant to be there)'
                )
                continue
            dataset_changed |= _apply_declared_metric_rules(metric, list(metric_meta.validation_rules))
        if dataset_changed and not dataset.is_external_placeholder:
            # Rules ride in the materialized snapshot and their violations are
            # persisted there; re-evaluate under the new rule set.
            refresh_dataset_materialization(dataset, touch=False)


def _apply_declared_category_domain(
    ic: InstanceConfig,
    dataset: DatasetModel,
    metadata: DatasetMeta,
    declarations: dict[int, tuple[str, object]],
) -> bool:
    spec = metadata.category_domain_spec
    if spec is None:
        return False

    from kausal_common.datasets.category_domain import DatasetCategoryCombination, DatasetCategoryDomain
    from kausal_common.datasets.models import DimensionScope

    schema = dataset.schema
    assert schema is not None
    schema_dimension_ids = set(schema.dimensions.values_list('dimension_id', flat=True))
    scopes = (
        DimensionScope.objects
        .for_instance_config(ic)
        .filter(dimension_id__in=schema_dimension_ids)
        .select_related('dimension')
        .prefetch_related('dimension__categories')
    )
    dimensions = {scope.identifier: scope.dimension for scope in scopes if scope.identifier}
    combinations: list[DatasetCategoryCombination] = []
    for combination_spec in spec.combinations:
        categories: dict[UUID, UUID] = {}
        for dimension_identifier, category_identifier in combination_spec.categories.items():
            dimension = dimensions.get(dimension_identifier)
            if dimension is None:
                raise ValueError(
                    f"category combination '{combination_spec.id}' on dataset '{dataset.identifier}' "
                    f"references dimension '{dimension_identifier}' outside its schema"
                )
            category = next(
                (category for category in dimension.categories.all() if category.identifier == category_identifier),
                None,
            )
            if category is None:
                raise ValueError(
                    f"category combination '{combination_spec.id}' on dataset '{dataset.identifier}' "
                    f"references unknown category '{dimension_identifier}:{category_identifier}'"
                )
            categories[dimension.uuid] = category.uuid
        if spec.mode == 'closed' and set(categories) != {dimension.uuid for dimension in dimensions.values()}:
            raise ValueError(
                f"closed category combination '{combination_spec.id}' on dataset '{dataset.identifier}' "
                'must mention every schema dimension'
            )
        combinations.append(
            DatasetCategoryCombination(
                id=uuid3(
                    ic.uuid,
                    ':'.join(['dataset', dataset.identifier or '', 'category-combination', combination_spec.id]),
                ),
                identifier=combination_spec.id,
                categories=categories,
            )
        )
    domain = DatasetCategoryDomain(mode=spec.mode, combinations=combinations)
    previous = declarations.get(schema.pk)
    if previous is not None and previous[1] != domain:
        raise ValueError(
            f"datasets '{previous[0]}' and '{dataset.identifier}' share a schema but declare conflicting category domains"
        )
    declarations[schema.pk] = (dataset.identifier or str(dataset.uuid), domain)
    if schema.category_domain == domain:
        return False
    schema.category_domain = domain
    schema.save(update_fields=['category_domain'])
    return True


def _apply_declared_metric_rules(metric: DatasetMetric, declared: list[ValidationRule]) -> bool:
    """Replace the metric's rule rows when the declared rule list differs; returns whether it did."""
    from kausal_common.datasets.models import DatasetMetricValidationRule

    blobs = [rule.model_dump(mode='json') for rule in declared]
    existing_rows = list(metric.validation_rules.order_by('order'))
    if [row.rule for row in existing_rows] == blobs:
        return False
    for row in existing_rows:
        row.delete()
    for order, blob in enumerate(blobs):
        DatasetMetricValidationRule.objects.create(metric=metric, rule=blob, order=order)
    return True


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
    if n.is_editable is not None:
        attributes['is_editable'] = n.is_editable
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


def _write_bindings(
    ic: InstanceConfig,
    snapshot: InstanceSnapshot,
    node_configs: dict[UUID, NodeConfig],
    *,
    port_references: YamlPortReferenceCatalog,
) -> tuple[int, int]:
    """
    Resolve snapshot bindings and reconcile the ``NodeInputPortBinding`` rows.

    Returns (edge count, dataset-port count). The row UUID is the durable
    binding identity: authored UUIDs (stamped by the parser from the port
    reference catalog) win, and rows the catalog missed are matched to
    surviving structural identities so a re-sync never churns identity.
    """
    from kausal_common.datasets.models import DatasetMetric

    from nodes.input_bindings import reconcile_input_bindings
    from nodes.instance_serialization import (
        dataset_port_match_keys,
        edge_match_keys,
        existing_dataset_port_identities,
        existing_edge_identities,
        match_preserved_uuids,
        ordered_binding_snapshots,
    )
    from nodes.models import NodeInputPortBinding
    from nodes.spec_export import _get_db_datasets

    edges = snapshot.edge_bindings
    authored_uuids = {edge.uuid for edge in edges if edge.uuid is not None}
    existing_edges = [item for item in existing_edge_identities(ic) if item[1] not in authored_uuids]
    preserved_edge_uuids = match_preserved_uuids(
        existing_edges,
        [edge_match_keys(edge.from_node, edge.from_port, edge.to_node, edge.to_port) for edge in edges],
    )

    schemas = collect_dataset_schema_info(ic)
    resolved = resolve_dataset_port_snapshots(snapshot, schemas, port_references=port_references)

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
    preserved_port_uuids = match_preserved_uuids(existing_dataset_port_identities(ic), match_keys)

    edge_uuid_by_id = {id(edge): edge.uuid or matched for edge, matched in zip(edges, preserved_edge_uuids, strict=True)}
    port_row_by_id = {
        id(port): (dataset_obj, metric, matched)
        for (port, dataset_obj, metric), matched in zip(triples, preserved_port_uuids, strict=True)
    }

    desired: list[NodeInputPortBinding] = []
    for item, position in ordered_binding_snapshots(edges, [port for port, _ds, _m in triples]):
        if isinstance(item, DatasetPortSnapshot):
            dataset_obj, metric, matched_uuid = port_row_by_id[id(item)]
            identity_kwargs = {'uuid': matched_uuid} if matched_uuid is not None else {}
            desired.append(
                NodeInputPortBinding(
                    instance=ic,
                    node=node_configs[item.node],
                    port_id=item.port_id,
                    position=position,
                    dataset=dataset_obj,
                    metric=metric,
                    transformations=list(item.spec.transformations),
                    tags=list(item.spec.tags),
                    dataset_spec=item.spec,
                    dataset_index=item.dataset_index,
                    **identity_kwargs,
                )
            )
            continue
        from_nc = node_configs.get(item.from_node)
        to_nc = node_configs.get(item.to_node)
        if from_nc is None or to_nc is None:
            raise ValueError(f'Edge references unknown node: {item.from_node} -> {item.to_node}')
        row_uuid = edge_uuid_by_id[id(item)]
        identity_kwargs = {'uuid': row_uuid} if row_uuid is not None else {}
        desired.append(
            NodeInputPortBinding(
                instance=ic,
                node=to_nc,
                port_id=item.to_port,
                position=position,
                source_node=from_nc,
                source_port_id=item.from_port,
                transformations=list(item.transformations),
                tags=list(item.tags),
                **identity_kwargs,
            )
        )

    reconcile_input_bindings(ic, desired)
    return len(edges), len(triples)


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
            created_placeholder_ids = sync_dataset_placeholders_from_snapshot(ic, snapshot)
            _sync_dataset_metadata_from_snapshot(ic, snapshot)
            edge_count, dataset_port_count = _write_bindings(
                ic,
                snapshot,
                node_configs,
                port_references=port_references,
            )
            promoted = _promote_dataset_forecast_defaults(ic) if promote_forecast_defaults else 0

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
