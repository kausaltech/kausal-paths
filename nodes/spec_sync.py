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
from typing import TYPE_CHECKING
from uuid import UUID, uuid3

from loguru import logger

if TYPE_CHECKING:
    from nodes.defs.node_defs import NodeSpec
    from nodes.instance_serialization import DatasetPortSnapshot, InstanceSnapshot
    from nodes.models import InstanceConfig


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


def resolve_dataset_port_snapshots(  # noqa: C901
    snapshot: InstanceSnapshot,
    schemas: dict[str, DatasetSchemaInfo],
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

    node_specs: dict[str, NodeSpec] = {}
    for n in snapshot.nodes:
        assert n.spec is not None
        node_specs[n.identifier] = n.spec

    instance_uuid = snapshot.metadata.uuid

    # Group parse-side entries into bindings: (node, dataset_index) is binding identity.
    bindings: dict[tuple[str, int], list[DatasetPortSnapshot]] = {}
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
            resolved.append(
                DatasetPortSnapshot(
                    node=node_id,
                    dataset=dataset_id,
                    port_id=_dataset_port_uuid(instance_uuid, node_id, dataset_index, port_column),
                    metric=metric_name,
                    dataset_index=dataset_index,
                    spec=spec,
                )
            )
    return resolved


def _dataset_port_uuid(instance_uuid: UUID, node_id: str, dataset_index: int, column: str) -> UUID:
    return uuid3(instance_uuid, ':'.join([node_id, 'dataset', str(dataset_index), column]))
