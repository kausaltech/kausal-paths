"""Serialize and deserialize DB-sourced instance configurations."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from nodes.defs.edge_def import EdgeTransformation
from nodes.defs.instance_defs import InstanceSpec
from nodes.defs.node_defs import NodeSpec

if TYPE_CHECKING:
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import (
        Dataset as DatasetModel,
        DatasetMetric,
        DimensionCategory,
    )

    from nodes.models import DatasetPort, InstanceConfig, NodeConfig, NodeEdge


# ---------------------------------------------------------------------------
# Export models
# ---------------------------------------------------------------------------

# The `i18n` field on these models stores the raw modeltrans JSON dict
# (e.g. {"label_en": "...", "label_da": "..."}).  This allows lossless
# round-tripping of translations through export/import.


class DatasetMetricExport(BaseModel):
    identifier: str
    label: str
    unit: str
    i18n: dict[str, Any] | None = None


class DatasetExport(BaseModel):
    identifier: str
    name: str
    time_resolution: str = 'yearly'
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[DatasetMetricExport] = Field(default_factory=list)
    is_placeholder: bool = False
    external_ref: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    i18n: dict[str, Any] | None = None


class NodeExport(BaseModel):
    identifier: str
    name: str | None = None
    color: str = ''
    order: int | None = None
    is_visible: bool = True
    indicator_node: str | None = None
    spec: NodeSpec | None = None
    i18n: dict[str, Any] | None = None


class EdgeExport(BaseModel):
    from_node: str
    to_node: str
    from_port: str
    to_port: str
    transformations: list[EdgeTransformation] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class DatasetPortExport(BaseModel):
    node: str
    dataset: str
    port_id: str
    metric: str
    forecast_from: int | None = None


class InstanceExport(BaseModel):
    spec: InstanceSpec
    nodes: list[NodeExport] = Field(default_factory=list)
    edges: list[EdgeExport] = Field(default_factory=list)
    datasets: list[DatasetExport] = Field(default_factory=list)
    dataset_ports: list[DatasetPortExport] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}


# ---------------------------------------------------------------------------
# Export (to_dict)
# ---------------------------------------------------------------------------


def _export_dataset(
    ds: DatasetModel,
    ic_ct: ContentType,
    ic_pk: int,
) -> DatasetExport:
    from kausal_common.datasets.models import DatasetSchemaDimension, DimensionScope

    schema = ds.schema
    assert schema is not None

    metrics = [
        DatasetMetricExport(
            identifier=m.name or str(m.uuid),
            label=m.label,
            unit=m.unit,
            i18n=m.i18n,
        )
        for m in schema.metrics.all().order_by('order')
    ]

    # Dimensions — resolve identifiers via DatasetSchemaDimension → Dimension → DimensionScope
    dimensions: list[str] = []
    for dsd in DatasetSchemaDimension.objects.filter(schema=schema).select_related('dimension').order_by('order'):
        scope = DimensionScope.objects.filter(
            dimension=dsd.dimension,
            scope_content_type=ic_ct,
            scope_id=ic_pk,
        ).first()
        if scope and scope.identifier:
            dimensions.append(scope.identifier)

    data: dict[str, Any] | None = None
    if not ds.is_external_placeholder:
        data = _export_dataset_data(ds)

    return DatasetExport(
        identifier=ds.identifier or str(ds.uuid),
        name=schema.name,
        time_resolution=schema.time_resolution,
        dimensions=dimensions,
        metrics=metrics,
        is_placeholder=ds.is_external_placeholder,
        external_ref=ds.external_ref,
        data=data,
        i18n=schema.i18n,
    )


def _export_dataset_data(ds: DatasetModel) -> dict[str, Any]:
    """Serialize dataset DataPoints into JSON Table Schema format."""
    from nodes.datasets import DBDataset, JSONDataset

    df = DBDataset.deserialize_df(ds)
    return JSONDataset.serialize_df(df)


def _export_node(nc: NodeConfig, indicator_map: dict[int, str]) -> NodeExport:
    indicator_id: int | None = getattr(nc, 'indicator_node_id', None)
    return NodeExport(
        identifier=nc.identifier,
        name=nc.name,
        color=nc.color,
        order=nc.order,
        is_visible=nc.is_visible,
        indicator_node=indicator_map.get(indicator_id) if indicator_id else None,
        spec=nc.spec,
        i18n=nc.i18n,
    )


def _export_edge(edge: NodeEdge) -> EdgeExport:
    return EdgeExport(
        from_node=edge.from_node.identifier,
        to_node=edge.to_node.identifier,
        from_port=str(edge.from_port),
        to_port=str(edge.to_port),
        transformations=edge.transformations or [],
        tags=edge.tags or [],
    )


def _export_dataset_port(port: DatasetPort) -> DatasetPortExport:
    return DatasetPortExport(
        node=port.node.identifier,
        dataset=port.dataset.identifier or str(port.dataset.uuid),
        port_id=str(port.port_id),
        metric=port.metric.name or str(port.metric.uuid),
        forecast_from=port.forecast_from,
    )


def export_instance(ic: InstanceConfig) -> InstanceExport:
    """Serialize a DB-sourced InstanceConfig into an InstanceExport."""
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset as DatasetModel

    from nodes.models import DatasetPort, NodeEdge

    if ic.spec is None:
        msg = f'Instance {ic.identifier} has no spec — run sync_instance_to_db first'
        raise ValueError(msg)

    ic_ct = ContentType.objects.get_for_model(ic)

    # Nodes
    node_qs = ic.nodes.get_queryset().for_serialization()
    nodes_by_pk: dict[int, NodeConfig] = {}
    for nc in node_qs:
        nodes_by_pk[nc.pk] = nc
    indicator_map = {pk: nc.identifier for pk, nc in nodes_by_pk.items()}
    nodes = [_export_node(nc, indicator_map) for nc in nodes_by_pk.values()]

    # Edges
    edge_qs = NodeEdge.objects.filter(instance=ic).select_related('from_node', 'to_node')
    edges = [_export_edge(e) for e in edge_qs]

    # Datasets
    datasets_qs = DatasetModel.objects.filter(
        scope_content_type=ic_ct,
        scope_id=ic.pk,
    ).select_related('schema')
    datasets = [_export_dataset(ds, ic_ct, ic.pk) for ds in datasets_qs]

    # Dataset ports
    port_qs = DatasetPort.objects.filter(instance=ic).select_related('node', 'dataset', 'metric')
    dataset_ports = [_export_dataset_port(p) for p in port_qs]

    return InstanceExport(
        spec=ic.spec,
        nodes=nodes,
        edges=edges,
        datasets=datasets,
        dataset_ports=dataset_ports,
    )


# ---------------------------------------------------------------------------
# Import (from_dict)
# ---------------------------------------------------------------------------


def _import_dimensions(
    ic: InstanceConfig,
    export: InstanceExport,
    ic_ct: ContentType,
) -> dict[str, DimensionCategory]:
    """
    Create Dimension + DimensionCategory + DimensionScope ORM objects.

    Returns a lookup: (dimension_id, category_id) → DimensionCategory.
    The lookup key is flattened as "dim_id/cat_id" for convenience.
    """
    from kausal_common.datasets.models import (
        Dimension,
        DimensionCategory as DimensionCategoryModel,
        DimensionScope,
    )

    cat_lookup: dict[str, DimensionCategoryModel] = {}

    for dim_dict in export.spec.dimensions:
        dim_id = dim_dict['id']
        label = dim_dict.get('label', dim_id)
        if isinstance(label, dict):
            name = next(iter(label.values()), dim_id)
        else:
            name = str(label)

        dim_obj = Dimension.objects.create(name=name)
        DimensionScope.objects.create(
            dimension=dim_obj,
            identifier=dim_id,
            scope_content_type=ic_ct,
            scope_id=ic.pk,
        )

        for cat_dict in dim_dict.get('categories', []):
            cat_id = cat_dict['id']
            cat_label = cat_dict.get('label', cat_id)
            if isinstance(cat_label, dict):
                cat_name = next(iter(cat_label.values()), cat_id)
            else:
                cat_name = str(cat_label)

            cat_obj = DimensionCategoryModel.objects.create(
                dimension=dim_obj,
                identifier=cat_id,
                label=cat_name,
            )
            cat_lookup[f'{dim_id}/{cat_id}'] = cat_obj

    return cat_lookup


def _import_dataset(
    ic: InstanceConfig,
    ds_export: DatasetExport,
    ic_ct: ContentType,
    dim_lookup: dict[str, DimensionCategory],
) -> DatasetModel:
    """Create DatasetSchema, Dataset, DatasetMetric, DatasetSchemaDimension, and DataPoints."""
    from kausal_common.datasets.models import (
        Dataset as DatasetModel,
        DatasetMetric as DatasetMetricModel,
        DatasetSchema as DatasetSchemaModel,
        DatasetSchemaDimension,
        DatasetSchemaScope,
        DimensionScope,
    )

    # Create schema
    schema = DatasetSchemaModel.objects.create(
        name=ds_export.name,
        time_resolution=ds_export.time_resolution,
        i18n=ds_export.i18n or {},
    )
    DatasetSchemaScope.objects.create(
        schema=schema,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
    )

    # Create metrics
    metrics_by_id: dict[str, DatasetMetricModel] = {}
    for idx, m_export in enumerate(ds_export.metrics):
        metric = DatasetMetricModel.objects.create(
            schema=schema,
            name=m_export.identifier,
            label=m_export.label,
            unit=m_export.unit,
            order=idx,
            i18n=m_export.i18n or {},
        )
        metrics_by_id[m_export.identifier] = metric

    # Link dimensions to schema
    for idx, dim_id in enumerate(ds_export.dimensions):
        dim_scope = DimensionScope.objects.filter(
            identifier=dim_id,
            scope_content_type=ic_ct,
            scope_id=ic.pk,
        ).first()
        if dim_scope:
            DatasetSchemaDimension.objects.create(
                schema=schema,
                dimension=dim_scope.dimension,
                order=idx,
            )

    # Create dataset
    dataset = DatasetModel(
        identifier=ds_export.identifier,
        is_external_placeholder=ds_export.is_placeholder,
        external_ref=ds_export.external_ref,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        schema=schema,
    )
    dataset.save()

    # Create data points
    if ds_export.data is not None:
        _import_data_points(dataset, ds_export, metrics_by_id, dim_lookup)

    return dataset


def _import_data_points(
    dataset: DatasetModel,
    ds_export: DatasetExport,
    metrics_by_id: dict[str, DatasetMetric],
    dim_lookup: dict[str, DimensionCategory],
) -> None:
    from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory

    assert ds_export.data is not None
    dim_ids = ds_export.dimensions

    data_points: list[DataPoint] = []
    # (data_point_index, category) pairs for bulk M2M creation
    dp_categories: list[tuple[int, DimensionCategory]] = []

    for row in ds_export.data['data']:
        year_val = row.get('Year')
        if year_val is None:
            continue
        dp_date = date(year=int(year_val), month=1, day=1)

        # Resolve dimension categories for this row
        row_cats: list[DimensionCategory] = []
        for dim_id in dim_ids:
            cat_id = row.get(dim_id)
            if cat_id:
                cat = dim_lookup.get(f'{dim_id}/{cat_id}')
                if cat:
                    row_cats.append(cat)

        for metric_id, metric in metrics_by_id.items():
            value = row.get(metric_id)
            if value is None:
                continue
            dp_idx = len(data_points)
            data_points.append(
                DataPoint(
                    dataset=dataset,
                    date=dp_date,
                    metric=metric,
                    value=value,
                )
            )
            dp_categories.extend((dp_idx, cat) for cat in row_cats)

    # Bulk create data points
    created_dps = DataPoint.objects.bulk_create(data_points)

    # Bulk create M2M links
    if dp_categories:
        m2m_objs = [
            DataPointDimensionCategory(
                data_point=created_dps[dp_idx],
                dimension_category=cat,
            )
            for dp_idx, cat in dp_categories
        ]
        DataPointDimensionCategory.objects.bulk_create(m2m_objs)


def _import_nodes(
    ic: InstanceConfig,
    export: InstanceExport,
) -> dict[str, NodeConfig]:
    """Create NodeConfig objects. Returns identifier → NodeConfig map."""
    from nodes.models import NodeConfig

    nodes_by_id: dict[str, NodeConfig] = {}
    for n in export.nodes:
        nc = NodeConfig.objects.create(
            instance=ic,
            identifier=n.identifier,
            name=n.name,
            color=n.color,
            order=n.order,
            is_visible=n.is_visible,
            i18n=n.i18n or {},
        )
        # Write spec via queryset.update() to bypass ClusterableModel.save()
        if n.spec is not None:
            NodeConfig.objects.filter(pk=nc.pk).update(spec=n.spec)
            nc.spec = n.spec
        nodes_by_id[n.identifier] = nc

    # Resolve indicator_node references
    for n in export.nodes:
        if n.indicator_node and n.indicator_node in nodes_by_id:
            nc = nodes_by_id[n.identifier]
            indicator = nodes_by_id[n.indicator_node]
            NodeConfig.objects.filter(pk=nc.pk).update(indicator_node=indicator)

    return nodes_by_id


def _import_edges(
    ic: InstanceConfig,
    export: InstanceExport,
    nodes_by_id: dict[str, NodeConfig],
) -> None:
    from nodes.models import NodeEdge

    for e in export.edges:
        from_node = nodes_by_id.get(e.from_node)
        to_node = nodes_by_id.get(e.to_node)
        if from_node is None or to_node is None:
            continue
        NodeEdge.objects.create(
            instance=ic,
            from_node=from_node,
            to_node=to_node,
            from_port=e.from_port,
            to_port=e.to_port,
            transformations=e.transformations,
            tags=e.tags,
        )


def _import_dataset_ports(
    ic: InstanceConfig,
    export: InstanceExport,
    nodes_by_id: dict[str, NodeConfig],
    datasets_by_id: dict[str, DatasetModel],
) -> None:
    from nodes.models import DatasetPort

    for p in export.dataset_ports:
        node = nodes_by_id.get(p.node)
        dataset = datasets_by_id.get(p.dataset)
        if node is None or dataset is None:
            continue
        # Resolve metric by name within the dataset's schema
        assert dataset.schema is not None
        metric = dataset.schema.metrics.filter(name=p.metric).first()
        if metric is None:
            continue
        DatasetPort.objects.create(
            instance=ic,
            node=node,
            dataset=dataset,
            port_id=p.port_id,
            metric=metric,
            forecast_from=p.forecast_from,
        )


def import_instance(ic: InstanceConfig, export: InstanceExport) -> None:
    """
    Populate an InstanceConfig with computation model objects from an InstanceExport.

    The InstanceConfig must already exist (with identifier, org, etc.).
    This function creates all related objects: nodes, edges, datasets, ports.
    """
    from django.contrib.contenttypes.models import ContentType

    ic_ct = ContentType.objects.get_for_model(ic)

    # Store the spec
    ic.spec = export.spec
    ic.config_source = 'database'
    ic.save(update_fields=['spec', 'config_source'])

    # Dimensions first — datasets and data points reference them
    dim_lookup = _import_dimensions(ic, export, ic_ct)

    # Datasets (with data points)
    datasets_by_id: dict[str, DatasetModel] = {}
    for ds_export in export.datasets:
        ds = _import_dataset(ic, ds_export, ic_ct, dim_lookup)
        datasets_by_id[ds_export.identifier] = ds

    # Nodes
    nodes_by_id = _import_nodes(ic, export)

    # Edges
    _import_edges(ic, export, nodes_by_id)

    # Dataset ports
    _import_dataset_ports(ic, export, nodes_by_id, datasets_by_id)
