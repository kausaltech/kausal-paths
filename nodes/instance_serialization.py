"""
Serialize and deserialize DB-sourced instance configurations.

Two related Pydantic models define the serialization layers:

- ``InstanceSnapshot`` — structural state of an instance (spec + nodes +
  edges + dataset ports). Dataset references are pinned by identifier;
  dataset *bodies* are not included. This is the unit of revisioning.
- ``InstanceExport`` — ``InstanceSnapshot`` plus the dataset bodies as
  ``DatasetExport`` objects. Used for portable export/import (e.g. when
  cloning a framework template into a new instance).

Individual ref-only types inherit ``ModelSnapshot``, which provides a
``from_model`` classmethod bridging ORM rows to snapshot objects. Data-
carrying types (``DatasetExport``, ``DatasetMetricExport``) keep their
``Export`` names because they genuinely carry data, not just references.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Self
from uuid import UUID

from pydantic import BaseModel, Field

from kausal_common.i18n.pydantic import (
    I18nBaseModel,
    TranslatedString,
    get_modeltrans_attrs_from_str,
    get_translated_string_from_modeltrans,
)

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

    from nodes.models import InstanceConfig, NodeConfig


# Current schema version for ``InstanceSnapshot`` and ``InstanceExport``.
# Bump when making non-backwards-compatible changes to the snapshot layout.
SNAPSHOT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Snapshot base + models
# ---------------------------------------------------------------------------

# The ``i18n`` field on these models stores the raw modeltrans JSON dict
# (e.g. {"label_en": "...", "label_da": "..."}).  This allows lossless
# round-tripping of translations through export/import.


class ModelSnapshot(I18nBaseModel):
    """
    Base for Pydantic types that mirror ORM-row state of editable children.

    Subclasses declare their fields; ``from_model`` maps an ORM instance to
    this snapshot shape (default: attribute access via
    ``model_validate(obj, from_attributes=True)``). Override when a field
    needs dereferencing (e.g. FK → string identifier).

    Inherits ``I18nBaseModel`` so ``TranslatedString``-typed fields are
    handled uniformly; snapshots without i18n fields pay no runtime cost.
    """

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        return cls.model_validate(obj, from_attributes=True)


def _ts_from_modeltrans(obj: Any, field_name: str, primary_language: str) -> TranslatedString | None:
    """
    Read a modeltrans-backed field into a ``TranslatedString``.

    Returns ``None`` when the field is empty across all languages.
    """
    val = getattr(obj, field_name, None)
    i18n: dict[str, Any] = getattr(obj, 'i18n', None) or {}
    has_translation = any(k.startswith(f'{field_name}_') and v for k, v in i18n.items())
    if not val and not has_translation:
        return None
    return get_translated_string_from_modeltrans(obj, field_name, primary_language)


class DatasetMetricSnapshot(ModelSnapshot):
    identifier: str
    label: TranslatedString | None = None
    unit: str

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        # Metrics live under a DatasetSchema; primary language is the
        # schema's parent scope's instance-config primary language. For the
        # nested path the caller resolves the language and passes via
        # ``from_model_with_language`` below — the default path assumes
        # i18n-less data.
        return cls(
            identifier=obj.name or str(obj.uuid),
            label=_ts_from_modeltrans(obj, 'label', 'en') if obj.label or obj.i18n else None,
            unit=obj.unit,
        )

    @classmethod
    def from_model_with_language(cls, obj: Any, primary_language: str) -> Self:
        return cls(
            identifier=obj.name or str(obj.uuid),
            label=_ts_from_modeltrans(obj, 'label', primary_language),
            unit=obj.unit,
        )


class DatasetSnapshot(ModelSnapshot):
    """
    Pydantic representation of a ``Dataset`` ORM row.

    Includes its DataPoints. Used both as the Wagtail revision payload for Dataset
    (via ``Dataset.serializable_data`` bridged in Paths) and as the
    dataset-body carrier inside ``InstanceExport``.
    """

    schema_version: int = 1
    identifier: str | None = None
    name: TranslatedString | None = None
    is_external_placeholder: bool = False
    external_ref: dict[str, Any] | None = None
    time_resolution: str = 'yearly'
    dimensions: list[str] = Field(default_factory=list)
    metrics: list[DatasetMetricSnapshot] = Field(default_factory=list)
    data: dict[str, Any] | None = None

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        from kausal_common.datasets.models import DatasetSchemaDimension, DimensionScope

        schema = obj.schema
        metrics: list[DatasetMetricSnapshot] = []
        dimensions: list[str] = []
        name_ts: TranslatedString | None = None
        time_resolution = 'yearly'
        primary_language = _primary_language_for_dataset(obj)

        if schema is not None:
            time_resolution = schema.time_resolution
            # Schema name is a plain CharField + an i18n TranslationField.
            name_ts = _ts_from_modeltrans(schema, 'name', primary_language)
            metrics = [
                DatasetMetricSnapshot.from_model_with_language(m, primary_language)
                for m in schema.metrics.all().order_by('order')
            ]
            if obj.scope_content_type is not None and obj.scope_id is not None:
                for dsd in DatasetSchemaDimension.objects.filter(schema=schema).select_related('dimension').order_by('order'):
                    scope = DimensionScope.objects.filter(
                        dimension=dsd.dimension,
                        scope_content_type=obj.scope_content_type,
                        scope_id=obj.scope_id,
                    ).first()
                    if scope and scope.identifier:
                        dimensions.append(scope.identifier)

        data: dict[str, Any] | None = None
        if not obj.is_external_placeholder:
            data = _export_dataset_data_safe(obj)

        return cls(
            identifier=obj.identifier,
            name=name_ts,
            is_external_placeholder=obj.is_external_placeholder,
            external_ref=obj.external_ref,
            time_resolution=time_resolution,
            dimensions=dimensions,
            metrics=metrics,
            data=data,
        )


def _primary_language_for_dataset(obj: Any) -> str:
    """Resolve the primary language for a Dataset via its scope's InstanceConfig."""
    scope = getattr(obj, 'scope', None)
    if scope is not None:
        lang = getattr(scope, 'primary_language', None)
        if lang:
            return lang
    return 'en'


class NodeSnapshot(ModelSnapshot):
    identifier: str
    name: TranslatedString | None = None
    short_description: TranslatedString | None = None
    description: TranslatedString | None = None
    goal: TranslatedString | None = None
    color: str = ''
    order: int | None = None
    is_visible: bool = True
    indicator_node: str | None = None
    spec: NodeSpec | None = None

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        indicator_id: int | None = getattr(obj, 'indicator_node_id', None)
        indicator_identifier: str | None = None
        if indicator_id:
            indicator = getattr(obj, 'indicator_node', None)
            indicator_identifier = indicator.identifier if indicator else None
        primary_language: str = obj.instance.primary_language
        return cls(
            identifier=obj.identifier,
            name=_ts_from_modeltrans(obj, 'name', primary_language),
            short_description=_ts_from_modeltrans(obj, 'short_description', primary_language),
            description=_ts_from_modeltrans(obj, 'description', primary_language),
            goal=_ts_from_modeltrans(obj, 'goal', primary_language),
            color=obj.color,
            order=obj.order,
            is_visible=obj.is_visible,
            indicator_node=indicator_identifier,
            spec=obj.spec,
        )


class EdgeSnapshot(ModelSnapshot):
    from_node: str
    to_node: str
    from_port: UUID
    to_port: UUID
    transformations: list[EdgeTransformation] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        return cls(
            from_node=obj.from_node.identifier,
            to_node=obj.to_node.identifier,
            from_port=obj.from_port,
            to_port=obj.to_port,
            transformations=obj.transformations or [],
            tags=obj.tags or [],
        )


class DatasetPortSnapshot(ModelSnapshot):
    node: str
    dataset: str
    port_id: UUID
    metric: str
    forecast_from: int | None = None
    # Populated once Dataset acquires RevisionMixin (see paths/dataset_pydantic.py
    # and kausal_common/datasets/models.py bridge).
    dataset_revision: int | None = None

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        # Pin to the dataset's current revision so the snapshot is
        # deterministically reconstructible even if the dataset later
        # changes. ``None`` if the dataset has never been saved as a
        # revision yet (common for fresh datasets).
        dataset_revision_id: int | None = None
        latest_rev = getattr(obj.dataset, 'latest_revision_id', None)
        if latest_rev is not None:
            dataset_revision_id = latest_rev
        return cls(
            node=obj.node.identifier,
            dataset=obj.dataset.identifier or str(obj.dataset.uuid),
            port_id=obj.port_id,
            metric=obj.metric.name or str(obj.metric.uuid),
            forecast_from=obj.forecast_from,
            dataset_revision=dataset_revision_id,
        )


class InstanceSnapshot(BaseModel):
    """
    Structural state of an instance; unit of revisioning.

    Contains spec + nodes + edges + dataset ports. Dataset references are
    identifier-pinned; dataset bodies live in ``DatasetExport`` alongside
    (see ``InstanceExport``).
    """

    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    spec: InstanceSpec
    nodes: list[NodeSnapshot] = Field(default_factory=list)
    edges: list[EdgeSnapshot] = Field(default_factory=list)
    dataset_ports: list[DatasetPortSnapshot] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}


class InstanceExport(BaseModel):
    """
    Self-contained export: snapshot + dataset bodies.

    Used for cloning template instances and any standalone import/export
    flow where dataset data needs to travel with the model structure.
    Each ``DatasetSnapshot`` carries its DataPoints in its ``data`` field.
    """

    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    instance: InstanceSnapshot
    datasets: list[DatasetSnapshot] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}


# ---------------------------------------------------------------------------
# Export / Import helpers
# ---------------------------------------------------------------------------


def _export_dataset_data(ds: DatasetModel) -> dict[str, Any]:
    """Serialize dataset DataPoints into JSON Table Schema format."""
    from nodes.datasets import DBDataset, JSONDataset

    df = DBDataset.deserialize_df(ds)
    return JSONDataset.serialize_df(df)


def _export_dataset_data_safe(ds: DatasetModel) -> dict[str, Any] | None:
    """
    Serialize DataPoints.

    Returns ``None`` when the dataset has no data or the deserialization
    fails (e.g. empty / mis-seeded datasets during tests). Robustness
    matters here because ``serializable_data()`` is
    called on every ``save_revision`` and must not crash on edge cases.
    """
    if not ds.data_points.exists():
        return None
    return _export_dataset_data(ds)


def build_instance_snapshot(ic: InstanceConfig) -> InstanceSnapshot:
    """
    Structural snapshot of a DB-sourced InstanceConfig.

    Dataset references are pinned by identifier; dataset bodies are not
    included. Use ``export_instance`` when the bodies are also needed.
    """
    from nodes.models import NodeEdge

    if ic.spec is None:
        msg = f'Instance {ic.identifier} has no spec — run sync_instance_to_db first'
        raise ValueError(msg)

    node_qs = ic.nodes.get_queryset().for_serialization().select_related('indicator_node')
    nodes = [NodeSnapshot.from_model(nc) for nc in node_qs]

    edge_qs = NodeEdge.objects.filter(instance=ic).select_related('from_node', 'to_node')
    edges = [EdgeSnapshot.from_model(e) for e in edge_qs]

    port_qs = _dataset_port_qs_for(ic)
    dataset_ports = [DatasetPortSnapshot.from_model(p) for p in port_qs]

    return InstanceSnapshot(
        spec=ic.spec,
        nodes=nodes,
        edges=edges,
        dataset_ports=dataset_ports,
    )


def _dataset_port_qs_for(ic: InstanceConfig) -> Any:
    from nodes.models import DatasetPort

    return DatasetPort.objects.filter(instance=ic).select_related('node', 'dataset', 'metric')


def export_instance(ic: InstanceConfig) -> InstanceExport:
    """Serialize a DB-sourced InstanceConfig with dataset bodies included."""
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset as DatasetModel

    snapshot = build_instance_snapshot(ic)

    ic_ct = ContentType.objects.get_for_model(ic)
    datasets_qs = DatasetModel.objects.filter(
        scope_content_type=ic_ct,
        scope_id=ic.pk,
    ).select_related('schema')
    datasets = [DatasetSnapshot.from_model(ds) for ds in datasets_qs]

    return InstanceExport(instance=snapshot, datasets=datasets)


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

    for dim_dict in export.instance.spec.dimensions:
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
    ds_snapshot: DatasetSnapshot,
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

    primary_lang = ic.primary_language

    # Resolve schema name from the TranslatedString snapshot.
    schema_fields: dict[str, Any] = {'time_resolution': ds_snapshot.time_resolution}
    schema_i18n: dict[str, str] = {}
    _apply_translated(schema_fields, schema_i18n, ds_snapshot.name, 'name', primary_lang)
    if schema_fields.get('name') is None:
        # DatasetSchema.name is required; fall back to empty if the source
        # snapshot had no name in any language.
        schema_fields['name'] = ''

    schema = DatasetSchemaModel.objects.create(
        i18n=schema_i18n,
        **schema_fields,
    )
    DatasetSchemaScope.objects.create(
        schema=schema,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
    )

    # Create metrics (metric.label is TranslatedString in the snapshot)
    metrics_by_id: dict[str, DatasetMetricModel] = {}
    for idx, m_snap in enumerate(ds_snapshot.metrics):
        metric_fields: dict[str, Any] = {}
        metric_i18n: dict[str, str] = {}
        _apply_translated(metric_fields, metric_i18n, m_snap.label, 'label', primary_lang)
        if metric_fields.get('label') is None:
            metric_fields['label'] = m_snap.identifier
        metric = DatasetMetricModel.objects.create(
            schema=schema,
            name=m_snap.identifier,
            unit=m_snap.unit,
            order=idx,
            i18n=metric_i18n,
            **metric_fields,
        )
        metrics_by_id[m_snap.identifier] = metric

    # Link dimensions to schema
    for idx, dim_id in enumerate(ds_snapshot.dimensions):
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
        identifier=ds_snapshot.identifier,
        is_external_placeholder=ds_snapshot.is_external_placeholder,
        external_ref=ds_snapshot.external_ref,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        schema=schema,
    )
    dataset.save()

    # Create data points
    if ds_snapshot.data is not None:
        _import_data_points(dataset, ds_snapshot, metrics_by_id, dim_lookup)

    return dataset


def _import_data_points(
    dataset: DatasetModel,
    ds_snapshot: DatasetSnapshot,
    metrics_by_id: dict[str, DatasetMetric],
    dim_lookup: dict[str, DimensionCategory],
) -> None:
    from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory

    assert ds_snapshot.data is not None
    dim_ids = ds_snapshot.dimensions

    data_points: list[DataPoint] = []
    # (data_point_index, category) pairs for bulk M2M creation
    dp_categories: list[tuple[int, DimensionCategory]] = []

    for row in ds_snapshot.data['data']:
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


def _apply_translated(
    fields: dict[str, Any],
    i18n: dict[str, str],
    ts: TranslatedString | None,
    field_name: str,
    default_lang: str,
) -> None:
    """
    Split a TranslatedString into its modeltrans parts.

    The primary-language value goes into ``fields[field_name]`` and the
    non-primary translations into ``i18n`` (modeltrans keys like
    ``{field}_{lang}``). No-op on ``None``.
    """
    if ts is None:
        fields[field_name] = None
        return
    primary_val, translations = get_modeltrans_attrs_from_str(ts, field_name, default_lang, strict=False)
    fields[field_name] = primary_val
    i18n.update(translations)


def _import_nodes(
    ic: InstanceConfig,
    export: InstanceExport,
) -> dict[str, NodeConfig]:
    """Create NodeConfig objects. Returns identifier → NodeConfig map."""
    from nodes.models import NodeConfig

    primary_lang = ic.primary_language
    nodes_by_id: dict[str, NodeConfig] = {}
    for n in export.instance.nodes:
        fields: dict[str, Any] = {}
        i18n_dict: dict[str, str] = {}
        _apply_translated(fields, i18n_dict, n.name, 'name', primary_lang)
        _apply_translated(fields, i18n_dict, n.short_description, 'short_description', primary_lang)
        _apply_translated(fields, i18n_dict, n.description, 'description', primary_lang)
        _apply_translated(fields, i18n_dict, n.goal, 'goal', primary_lang)

        nc = NodeConfig.objects.create(
            instance=ic,
            identifier=n.identifier,
            color=n.color,
            order=n.order,
            is_visible=n.is_visible,
            i18n=i18n_dict,
            **fields,
        )
        # Write spec via queryset.update() to bypass ClusterableModel.save()
        if n.spec is not None:
            NodeConfig.objects.filter(pk=nc.pk).update(spec=n.spec)
            nc.spec = n.spec
        nodes_by_id[n.identifier] = nc

    # Resolve indicator_node references
    for n in export.instance.nodes:
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

    for e in export.instance.edges:
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

    for p in export.instance.dataset_ports:
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

    # Store the spec. Also copy the spec's language fields onto the
    # InstanceConfig row so i18n-bearing data (ActionGroup names, etc.)
    # stays loadable — the spec's TranslatedStrings are authored under
    # the template's primary_language and would be filtered out if the
    # InstanceConfig used a different language.
    ic.spec = export.instance.spec
    ic.primary_language = export.instance.spec.primary_language
    ic.other_languages = list(export.instance.spec.other_languages)
    ic.config_source = 'database'
    ic.save(update_fields=['spec', 'primary_language', 'other_languages', 'config_source'])

    # Dimensions first — datasets and data points reference them
    dim_lookup = _import_dimensions(ic, export, ic_ct)

    # Datasets (with data points)
    datasets_by_id: dict[str, DatasetModel] = {}
    for ds_snapshot in export.datasets:
        ds = _import_dataset(ic, ds_snapshot, ic_ct, dim_lookup)
        # ``identifier`` may be None for datasets keyed only by uuid; skip
        # those here since node→dataset wiring goes through identifier.
        if ds_snapshot.identifier is not None:
            datasets_by_id[ds_snapshot.identifier] = ds

    # Nodes
    nodes_by_id = _import_nodes(ic, export)

    # Edges
    _import_edges(ic, export, nodes_by_id)

    # Dataset ports
    _import_dataset_ports(ic, export, nodes_by_id, datasets_by_id)
