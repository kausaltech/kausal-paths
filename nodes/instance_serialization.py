"""
Serialize and deserialize DB-sourced instance configurations.

Two related Pydantic models define the serialization layers:

- ``InstanceSnapshot`` — structural state of an instance (spec + nodes +
  edges + dataset ports), plus the UUID catalogs needed to resolve those
  references without loading dataset bodies. This is the unit of revisioning.
- ``InstanceExport`` — ``InstanceSnapshot`` plus the dataset bodies as
  ``DatasetExport`` objects. Used for portable export/import (e.g. when
  cloning a framework template into a new instance).

Individual ref-only types inherit ``ModelSnapshot``, which provides a
``from_model`` classmethod bridging ORM rows to snapshot objects. Data-
carrying types (``DatasetExport``, ``DatasetMetricExport``) keep their
``Export`` names because they genuinely carry data, not just references.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, Self, cast
from uuid import UUID

from django.db import transaction
from django.db.models import F
from pydantic import BaseModel, Field, field_validator

from markdown_it import MarkdownIt

from kausal_common.i18n.pydantic import (
    I18nBaseModel,
    TranslatedString,
    get_modeltrans_attrs_from_str,
    get_translated_string_from_modeltrans,
)

from nodes.defs.graph import (
    DatasetMeta,
    DatasetMetricMeta,
    DimensionCategoryMeta,
    DimensionMeta,
)
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec
from nodes.defs.node_defs import DatasetPortSpec, NodeSpec
from nodes.defs.transform_def import EdgeTransformOp, PortTransformOp
from nodes.page_snapshot import PageSnapshot

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    from django.contrib.contenttypes.models import ContentType
    from django.db.models import Model, QuerySet

    from kausal_common.datasets.models import (
        Dataset as DatasetModel,
        DatasetMetric,
        DimensionCategory,
    )

    from frameworks.models import FrameworkConfig
    from nodes.models import DatasetPort, InstanceConfig, NodeConfig, NodeEdge, NodeInputPortBinding, NodeLayout
    from nodes.node import Node


# Current schema version for ``InstanceSnapshot`` and ``InstanceExport``.
# Bump when making non-backwards-compatible changes to the snapshot layout.
#   v2: split identity metadata out of the embedded spec into a dedicated
#       ``metadata`` field (``InstanceMetadata``); ``spec`` is now the
#       computation-only ``InstanceModelSpec``.
#   v3: node references use UUIDs instead of identifiers.
#   v4: node identity/display metadata lives only on ``NodeSnapshot``;
#       ``NodeSpec`` contains computation configuration only.
#   v5: optional shared model-editor layout stored on each ``NodeSnapshot``.
#   v6: instance lead title/paragraph and node StreamField body are revisioned.
#   v7: published DB datasets have a normalized immutable revision manifest.
#   v8: structural dimension and dataset catalogs carry canonical UUIDs.
SNAPSHOT_SCHEMA_VERSION = 8

_MARKDOWN = MarkdownIt('commonmark', {'html': True})


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


def _ts_from_modeltrans(obj: Model, field_name: str, primary_language: str) -> TranslatedString | None:
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


class DataPointKey(BaseModel):
    """Natural key locating a DataPoint within its dataset (id-free, restore-stable)."""

    year: int
    metric: str  # metric identifier (name or uuid)
    categories: list[str] = Field(default_factory=list)  # sorted dimension-category ids


class DataSourceSnapshot(BaseModel):
    """A published data source referenced by a dataset or its data points."""

    uuid: str  # source DataSource uuid; the join key for references within the snapshot
    name: str
    edition: str | None = None
    authority: str | None = None
    description: str | None = None
    url: str | None = None


class SourceReferenceSnapshot(BaseModel):
    """Links a data source to the dataset (``point`` is None) or to one data point."""

    data_source: str  # DataSourceSnapshot.uuid
    point: DataPointKey | None = None


class DataPointCommentSnapshot(BaseModel):
    """A (non-soft-deleted) comment on a data point. Users are referenced by uuid."""

    point: DataPointKey
    text: str
    is_sticky: bool = False
    is_review: bool = False
    review_state: str | None = None
    resolved_at: str | None = None  # ISO 8601
    created_by: str | None = None  # user uuid
    last_modified_by: str | None = None  # user uuid
    resolved_by: str | None = None  # user uuid


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
    forecast_from: int | None = None
    is_external_placeholder: bool = False
    external_ref: dict[str, Any] | None = None
    time_resolution: str = 'yearly'
    dimensions: list[str] = Field(default_factory=list)
    dimension_columns: dict[str, str] = Field(default_factory=dict)
    metrics: list[DatasetMetricSnapshot] = Field(default_factory=list)
    data: dict[str, Any] | None = None
    data_sources: list[DataSourceSnapshot] = Field(default_factory=list)
    source_references: list[SourceReferenceSnapshot] = Field(default_factory=list)
    comments: list[DataPointCommentSnapshot] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        return cls.from_model_for_instance(obj, None)

    @classmethod
    def from_model_for_instance(cls, obj: Any, instance_config: InstanceConfig | None) -> Self:
        from kausal_common.datasets.models import DatasetSchemaDimension, DimensionScope

        schema = obj.schema
        metrics: list[DatasetMetricSnapshot] = []
        dimensions: list[str] = []
        dimension_columns: dict[str, str] = {}
        name_ts: TranslatedString | None = None
        time_resolution = 'yearly'
        primary_language = instance_config.primary_language if instance_config is not None else _primary_language_for_dataset(obj)

        if schema is not None:
            time_resolution = schema.time_resolution
            # Schema name is a plain CharField + an i18n TranslationField.
            name_ts = _ts_from_modeltrans(schema, 'name', primary_language)
            metrics = [
                DatasetMetricSnapshot.from_model_with_language(m, primary_language)
                for m in schema.metrics.all().order_by('order')
            ]
            if instance_config is not None:
                from django.contrib.contenttypes.models import ContentType

                scope_content_type = ContentType.objects.get_for_model(instance_config)
                scope_id = instance_config.pk
            else:
                scope_content_type = obj.scope_content_type
                scope_id = obj.scope_id
            if scope_content_type is not None and scope_id is not None:
                for dsd in DatasetSchemaDimension.objects.filter(schema=schema).select_related('dimension').order_by('order'):
                    scope = DimensionScope.objects.filter(
                        dimension=dsd.dimension,
                        scope_content_type=scope_content_type,
                        scope_id=scope_id,
                    ).first()
                    if scope and scope.identifier:
                        dimensions.append(scope.identifier)
                        if dsd.column_name and dsd.column_name != scope.identifier:
                            dimension_columns[scope.identifier] = dsd.column_name

        data: dict[str, Any] | None = None
        data_sources: list[DataSourceSnapshot] = []
        source_references: list[SourceReferenceSnapshot] = []
        comments: list[DataPointCommentSnapshot] = []
        if not obj.is_external_placeholder:
            data = _export_dataset_data_safe(obj)
            data_sources, source_references, comments = _export_dataset_provenance(obj)

        return cls(
            identifier=obj.identifier,
            name=name_ts,
            forecast_from=(obj.spec or {}).get('forecast_from'),
            is_external_placeholder=obj.is_external_placeholder,
            external_ref=obj.external_ref,
            time_resolution=time_resolution,
            dimensions=dimensions,
            dimension_columns=dimension_columns,
            metrics=metrics,
            data=data,
            data_sources=data_sources,
            source_references=source_references,
            comments=comments,
        )


def _primary_language_for_dataset(obj: Any) -> str:
    """Resolve the primary language for a Dataset via its scope's InstanceConfig."""
    scope = getattr(obj, 'scope', None)
    if scope is not None:
        lang = getattr(scope, 'primary_language', None)
        if lang:
            return lang
    return 'en'


def _label_from_identifier(identifier: str) -> str:
    return identifier.replace('_', ' ').replace('-', ' ').title()


class NodeLayoutSnapshot(ModelSnapshot):
    x: float
    y: float
    source: Literal['auto', 'user'] = 'auto'

    @classmethod
    def from_model(cls, obj: NodeLayout) -> Self:
        return cls(x=obj.x, y=obj.y, source=cast("Literal['auto', 'user']", obj.source))


class NodeSnapshot(ModelSnapshot):
    uuid: UUID
    identifier: str | None = None
    name: TranslatedString | None = None
    short_name: TranslatedString | None = None
    short_description: TranslatedString | None = None
    """Translated Wagtail database HTML, normalized from authored Markdown at the snapshot boundary."""
    description: TranslatedString | None = None
    goal: TranslatedString | None = None
    color: str = ''
    order: int | None = None
    is_visible: bool = True
    indicator_node: UUID | None = None
    copy_of: UUID | None = None
    body: list[Any] | None = None
    """Raw StreamField data of ``NodeConfig.body``. Admin-authored only, so
    parse-side snapshots never carry it; row-side snapshots preserve it so
    published serving doesn't lose (or leak drafts of) body content."""
    spec: NodeSpec | None = None
    layout: NodeLayoutSnapshot | None = None

    @field_validator('short_description')
    @classmethod
    def render_short_description(cls, value: TranslatedString | None) -> TranslatedString | None:
        """Keep every snapshot producer on the RichTextField-compatible HTML contract."""
        if value is None:
            return None
        return TranslatedString(
            default_language=value.default_language,
            **{language: _MARKDOWN.render(text) for language, text in value.i18n.items()},
        )

    @classmethod
    def from_model(cls, obj: NodeConfig, primary_language: str | None = None) -> Self:
        indicator_id: int | None = getattr(obj, 'indicator_node_id', None)
        indicator_uuid: UUID | None = None
        if indicator_id:
            indicator = getattr(obj, 'indicator_node', None)
            indicator_uuid = indicator.uuid if indicator else None
        if primary_language is None:
            primary_language = obj.instance.primary_language
        layout = getattr(obj, 'layout', None)
        return cls(
            uuid=obj.uuid,
            identifier=obj.identifier,
            name=_ts_from_modeltrans(obj, 'name', primary_language),
            short_name=_ts_from_modeltrans(obj, 'short_name', primary_language),
            short_description=_ts_from_modeltrans(obj, 'short_description', primary_language),
            description=_ts_from_modeltrans(obj, 'description', primary_language),
            goal=_ts_from_modeltrans(obj, 'goal', primary_language),
            color=obj.color,
            order=obj.order,
            is_visible=obj.is_visible,
            indicator_node=indicator_uuid,
            copy_of=obj.copy_of.uuid if obj.copy_of else None,
            body=list(obj.body.raw_data) if obj.body else None,
            spec=obj.spec,
            layout=NodeLayoutSnapshot.from_model(layout) if layout is not None else None,
        )

    @classmethod
    def from_runtime_node(cls, obj: Node, uuid: UUID, primary_language: str) -> Self:
        """Capture the YAML/runtime-owned metadata before applying ORM overrides."""

        def translated(value: str | TranslatedString | None) -> TranslatedString | None:
            if value is None or isinstance(value, TranslatedString):
                return value
            return TranslatedString(value, default_language=primary_language)

        return cls(
            uuid=uuid,
            identifier=obj.id,
            name=translated(obj.name),
            short_name=translated(obj.short_name),
            short_description=translated(obj.description),
            color=obj.color or '',
            order=obj.order,
            is_visible=obj.is_visible,
            spec=obj._spec,
        )


def _merge_translated_metadata(
    source: TranslatedString | None,
    stored: TranslatedString | None,
) -> TranslatedString | None:
    """Merge translations with non-empty ORM values taking precedence."""
    if stored is None:
        return source
    if source is None:
        return stored
    translations = dict(source.i18n)
    translations.update(stored.i18n)
    return TranslatedString(
        default_language=stored.default_language or source.default_language,
        **translations,
    )


def reconcile_node_snapshot_metadata(
    source: NodeSnapshot,
    node_config: NodeConfig,
    primary_language: str,
) -> NodeSnapshot:
    """
    Overlay ORM-owned metadata on a YAML/runtime node snapshot.

    Empty optional ORM values retain the YAML fallback used by the legacy
    runtime. Boolean visibility is never treated as missing, so an authored
    ``False`` survives the transition to DB-backed snapshots.
    """
    stored = NodeSnapshot.from_model(node_config, primary_language=primary_language)
    return source.model_copy(
        update={
            'name': _merge_translated_metadata(source.name, stored.name),
            'short_name': _merge_translated_metadata(source.short_name, stored.short_name),
            'short_description': _merge_translated_metadata(source.short_description, stored.short_description),
            'description': _merge_translated_metadata(source.description, stored.description),
            'goal': _merge_translated_metadata(source.goal, stored.goal),
            'color': stored.color or source.color,
            'order': stored.order if stored.order is not None else source.order,
            'is_visible': stored.is_visible,
            'indicator_node': stored.indicator_node,
            'copy_of': stored.copy_of,
            'body': stored.body,
            'layout': stored.layout,
        },
    )


def _upgrade_node_metadata_v4(nodes: list[Any]) -> None:
    metadata_keys = {'uuid', 'identifier', 'name', 'short_name', 'description', 'color', 'order', 'is_visible'}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        spec = node.get('spec')
        if not isinstance(spec, dict):
            continue
        if node.get('short_name') is None and spec.get('short_name') is not None:
            node['short_name'] = spec['short_name']
        if node.get('short_description') is None and spec.get('description') is not None:
            node['short_description'] = spec['description']
        for key in metadata_keys:
            spec.pop(key, None)
        # ``kind`` duplicated the discriminator already stored in type_config.
        spec.pop('kind', None)


def _upgrade_node_references_v3(data: dict[str, Any], nodes: list[Any]) -> None:
    node_uuids: dict[str, UUID] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_uuid = node.get('uuid') or (node.get('spec') or {}).get('uuid')
        identifier = node.get('identifier')
        if node_uuid is None or identifier is None:
            raise ValueError('Legacy node snapshots require an identifier and spec UUID')
        node['uuid'] = node_uuid
        node_uuids[identifier] = UUID(str(node_uuid))

    for node in nodes:
        indicator = node.get('indicator_node')
        if indicator is not None:
            node['indicator_node'] = node_uuids[indicator]
    for edge in data.get('edges', []):
        edge['from_node'] = node_uuids[edge['from_node']]
        edge['to_node'] = node_uuids[edge['to_node']]
    for port in data.get('dataset_ports', []):
        port['node'] = node_uuids[port['node']]


class EdgeSnapshot(ModelSnapshot):
    uuid: UUID | None = None
    from_node: UUID
    to_node: UUID
    from_port: UUID
    to_port: UUID
    transformations: list[EdgeTransformOp] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: NodeEdge) -> Self:
        from_node = getattr(obj, '_from_node_uuid', None)
        to_node = getattr(obj, '_to_node_uuid', None)
        return cls(
            uuid=obj.uuid,
            from_node=from_node if from_node is not None else obj.from_node.uuid,
            to_node=to_node if to_node is not None else obj.to_node.uuid,
            from_port=obj.from_port,
            to_port=obj.to_port,
            transformations=obj.transformations or [],
            tags=obj.tags or [],
        )


class DatasetPortSnapshot(ModelSnapshot):
    uuid: UUID | None = None
    node: UUID
    dataset: str
    dataset_uuid: UUID | None = None
    port_id: UUID
    metric: str
    metric_uuid: UUID | None = None
    # Position of this binding in the node's input_dataset_instances list;
    # preserves ordering when a node has multiple dataset inputs.
    dataset_index: int = 0
    spec: DatasetPortSpec = Field(default_factory=DatasetPortSpec)
    # Populated once Dataset acquires RevisionMixin (see paths/dataset_pydantic.py
    # and kausal_common/datasets/models.py bridge).
    dataset_revision: int | None = None

    @classmethod
    def from_model(cls, obj: DatasetPort) -> Self:
        # Pin to the dataset's current revision so the snapshot is
        # deterministically reconstructible even if the dataset later
        # changes. ``None`` if the dataset has never been saved as a
        # revision yet (common for fresh datasets).
        dataset_revision_id: int | None = None
        latest_rev = getattr(obj.dataset, 'latest_revision_id', None)
        if latest_rev is not None:
            dataset_revision_id = latest_rev
        return cls(
            uuid=obj.uuid,
            node=obj.node.uuid,
            dataset=obj.dataset.identifier or str(obj.dataset.uuid),
            dataset_uuid=obj.dataset.uuid,
            port_id=obj.port_id,
            metric=obj.metric.name or str(obj.metric.uuid),
            metric_uuid=obj.metric.uuid,
            dataset_index=obj.dataset_index,
            spec=obj.spec,
            dataset_revision=dataset_revision_id,
        )


class NodePortSource(BaseModel):
    """An input-binding source: another node's output port."""

    kind: Literal['node'] = 'node'
    node_id: UUID
    port_id: UUID


class DatasetMetricSource(BaseModel):
    """
    An input-binding source: one metric of a dataset.

    Natural references keep portable exports restore-stable, matching the
    dataset snapshot boundary (``DatasetPortSnapshot.dataset`` / ``metric``).
    """

    kind: Literal['dataset'] = 'dataset'
    dataset: str
    metric: str


type InputBindingSource = NodePortSource | DatasetMetricSource


class InputBindingSnapshot(ModelSnapshot):
    """
    Snapshot form of one ``NodeInputPortBinding`` row.

    Not yet part of ``InstanceSnapshot`` — the snapshot schema still carries
    the legacy ``edges`` + ``dataset_ports`` arrays. This exists as the
    row-level ``snapshot_model`` (change history, revisions) until the
    unified-binding snapshot upgrade lands.
    """

    uuid: UUID | None = None
    node_id: UUID
    port_id: UUID
    position: int = 0
    source: InputBindingSource = Field(discriminator='kind')
    transformations: list[PortTransformOp] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: NodeInputPortBinding) -> Self:
        source: NodePortSource | DatasetMetricSource
        source_node = obj.source_node
        if source_node is not None:
            assert obj.source_port_id is not None
            source = NodePortSource(node_id=source_node.uuid, port_id=obj.source_port_id)
        else:
            assert obj.dataset is not None
            assert obj.metric is not None
            source = DatasetMetricSource(
                dataset=obj.dataset.identifier or str(obj.dataset.uuid),
                metric=obj.metric.name or str(obj.metric.uuid),
            )
        return cls(
            uuid=obj.uuid,
            node_id=obj.node.uuid,
            port_id=obj.port_id,
            position=obj.position,
            source=source,
            transformations=list(obj.transformations or []),
            tags=list(obj.tags or []),
        )


def ordered_binding_snapshots(
    edges: Sequence[EdgeSnapshot],
    dataset_ports: Sequence[DatasetPortSnapshot],
) -> list[tuple[EdgeSnapshot | DatasetPortSnapshot, int]]:
    """
    Interleave a snapshot's two legacy binding arrays in canonical delivery order.

    This is the single ordering authority for per-port ``position`` values:
    ``build_instance_graph()`` and the ``NodeInputPortBinding`` mirror must
    assign identical positions, because floating-point addition makes the
    order values are delivered to a port observable in computation results.

    On a shared port, edges come first in their snapshot order (for DB
    snapshots: the ``NodeEdge`` queryset order), then dataset ports sorted by
    ``(node, dataset_index, port, metric)``.
    """
    from collections import defaultdict

    positions: defaultdict[tuple[UUID, UUID], int] = defaultdict(int)
    result: list[tuple[EdgeSnapshot | DatasetPortSnapshot, int]] = []
    for edge in edges:
        key = (edge.to_node, edge.to_port)
        result.append((edge, positions[key]))
        positions[key] += 1
    sorted_ports = sorted(
        dataset_ports,
        key=lambda item: (item.node, item.dataset_index, str(item.port_id), item.metric),
    )
    for port in sorted_ports:
        key = (port.node, port.port_id)
        result.append((port, positions[key]))
        positions[key] += 1
    return result


def match_preserved_uuids(
    existing: Sequence[tuple[tuple[Hashable, ...], UUID]],
    replacements: Sequence[tuple[Hashable, ...]],
) -> list[UUID | None]:
    """
    Match replacement rows to existing rows through successive structural keys.

    The sync paths preserve authored order by deleting and recreating the
    ``NodeEdge`` / ``DatasetPort`` rows (pk order is the authored order), but
    the rebuilt rows must keep their durable UUIDs: the row UUID is the
    binding identity ``NodeInputPortBinding`` mirrors, and it must survive a
    re-sync. Each row supplies one key per matching pass, most specific
    first; within a pass, unmatched rows sharing a key pair up in their given
    (authored) orders, so parallel duplicates match deterministically.
    Returns one preserved UUID (or ``None``) per replacement.
    """
    from collections import defaultdict, deque

    result: list[UUID | None] = [None] * len(replacements)
    if not existing or not replacements:
        return result
    pass_count = len(replacements[0])
    assert all(len(keys) == pass_count for keys in replacements)
    assert all(len(keys) == pass_count for keys, _uuid in existing)

    free_existing = list(range(len(existing)))
    for pass_idx in range(pass_count):
        pool: defaultdict[Hashable, deque[int]] = defaultdict(deque)
        for e_idx in free_existing:
            pool[existing[e_idx][0][pass_idx]].append(e_idx)
        matched: set[int] = set()
        for r_idx, keys in enumerate(replacements):
            if result[r_idx] is not None:
                continue
            queue = pool.get(keys[pass_idx])
            if not queue:
                continue
            e_idx = queue.popleft()
            result[r_idx] = existing[e_idx][1]
            matched.add(e_idx)
        free_existing = [e_idx for e_idx in free_existing if e_idx not in matched]
        if not free_existing:
            break
    return result


def edge_match_keys(from_node: UUID, from_port: UUID, to_node: UUID, to_port: UUID) -> tuple[Hashable, ...]:
    """Structural match keys for one edge, most specific first (loose pass survives port changes)."""
    return ((from_node, from_port, to_node, to_port), (from_node, to_node))


def dataset_port_match_keys(node: UUID, dataset_pk: int, dataset_index: int, metric_pk: int) -> tuple[Hashable, ...]:
    """Structural match keys for one dataset-port row (loose pass survives dataset reordering)."""
    return ((node, dataset_pk, dataset_index, metric_pk), (node, dataset_pk, metric_pk))


def existing_edge_identities(ic: InstanceConfig) -> list[tuple[tuple[Hashable, ...], UUID]]:
    """Capture edge match keys and UUIDs, in authored (pk) order, before a sync rewrite."""
    from nodes.models import NodeEdge

    return [
        (edge_match_keys(from_node, from_port, to_node, to_port), row_uuid)
        for from_node, from_port, to_node, to_port, row_uuid in (
            NodeEdge.objects
            .filter(instance=ic)
            .order_by('pk')
            .values_list('from_node__uuid', 'from_port', 'to_node__uuid', 'to_port', 'uuid')
        )
    ]


def existing_dataset_port_identities(ic: InstanceConfig) -> list[tuple[tuple[Hashable, ...], UUID]]:
    """Capture dataset-port match keys and UUIDs, in pk order, before a sync rewrite."""
    from nodes.models import DatasetPort

    return [
        (dataset_port_match_keys(node_uuid, dataset_pk, dataset_index, metric_pk), row_uuid)
        for node_uuid, dataset_pk, dataset_index, metric_pk, row_uuid in (
            DatasetPort.objects
            .filter(instance=ic)
            .order_by('pk')
            .values_list('node__uuid', 'dataset_id', 'dataset_index', 'metric_id', 'uuid')
        )
    ]


class DatasetRevisionPinSnapshot(BaseModel):
    dataset_uuid: UUID
    identifier: str | None = None
    revision_id: int
    content_hash: str
    generation: int
    forecast_from: int | None = None


class InstanceSnapshot(BaseModel):
    """
    Structural state of an instance; unit of revisioning.

    Contains metadata + spec + nodes + edges + dataset ports.
    Structural references and their dimension/dataset catalogs are UUID-pinned.
    Dataset bodies live in ``DatasetExport`` alongside (see ``InstanceExport``).
    """

    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    # Identity metadata, projected from the InstanceConfig columns. Defaulted
    # so that pre-v2 revision blobs (which embedded metadata inside ``spec``)
    # still deserialize.
    metadata: InstanceMetadata = Field(default_factory=InstanceMetadata)
    spec: InstanceModelSpec
    copy_of: str | None = None  # uuid of the InstanceConfig this was copied from
    nodes: list[NodeSnapshot] = Field(default_factory=list)
    edges: list[EdgeSnapshot] = Field(default_factory=list)
    dataset_ports: list[DatasetPortSnapshot] = Field(default_factory=list)
    dataset_revisions: list[DatasetRevisionPinSnapshot] = Field(default_factory=list)
    dimensions: list[DimensionMeta] = Field(default_factory=list)
    datasets: list[DatasetMeta] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}

    @classmethod
    def from_serialized_data(cls, data: dict[str, Any]) -> Self:
        """Load persisted snapshot data, upgrading older node metadata and references."""
        schema_version = data.get('schema_version', 1)
        if schema_version >= SNAPSHOT_SCHEMA_VERSION:
            return cls.model_validate(data)

        data = deepcopy(data)
        nodes = data.get('nodes', [])

        if schema_version < 3:
            _upgrade_node_references_v3(data, nodes)
        if schema_version < 4:
            _upgrade_node_metadata_v4(nodes)

        data['schema_version'] = SNAPSHOT_SCHEMA_VERSION
        return cls.model_validate(data)


def reconcile_snapshot_node_metadata(
    snapshot: InstanceSnapshot,
    node_configs: Iterable[NodeConfig],
) -> InstanceSnapshot:
    """Return the desired snapshot after applying authoritative ORM metadata."""
    by_uuid = {node.uuid: node for node in node_configs}
    by_identifier = {node.identifier: node for node in node_configs}
    nodes: list[NodeSnapshot] = []
    for source in snapshot.nodes:
        node_config = by_uuid.get(source.uuid)
        if node_config is None and source.identifier is not None:
            node_config = by_identifier.get(source.identifier)
        if node_config is None:
            nodes.append(source)
            continue
        nodes.append(
            reconcile_node_snapshot_metadata(
                source,
                node_config,
                primary_language=snapshot.metadata.primary_language,
            )
        )
    return snapshot.model_copy(update={'nodes': nodes})


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
    # Wagtail page tree, for verification only (not used on import — pages are
    # copied/restored via Wagtail's own machinery). Node references are by identifier.
    pages: list[PageSnapshot] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}


# ---------------------------------------------------------------------------
# Export / Import helpers
# ---------------------------------------------------------------------------


def _data_point_key(dp: Any) -> DataPointKey:
    """Natural key for a DataPoint: (year, metric identifier, sorted category ids)."""
    metric = dp.metric
    metric_id = metric.name or str(metric.uuid)
    categories = sorted((c.identifier or str(c.uuid)) for c in dp.dimension_categories.all())
    return DataPointKey(year=dp.date.year, metric=metric_id, categories=categories)


def _export_dataset_provenance(
    ds: DatasetModel,
) -> tuple[list[DataSourceSnapshot], list[SourceReferenceSnapshot], list[DataPointCommentSnapshot]]:
    """Serialize a dataset's source references and (non-soft-deleted) data-point comments."""
    from kausal_common.datasets.models import DataPointComment, DatasetSourceReference

    sources: dict[str, DataSourceSnapshot] = {}

    def add_source(src: Any) -> str:
        key = str(src.uuid)
        if key not in sources:
            sources[key] = DataSourceSnapshot(
                uuid=key,
                name=src.name,
                edition=src.edition,
                authority=src.authority,
                description=src.description,
                url=src.url,
            )
        return key

    dataset_refs = DatasetSourceReference.objects.filter(dataset=ds).select_related('data_source')
    dp_refs = (
        DatasetSourceReference.objects
        .filter(data_point__dataset=ds)
        .select_related('data_source', 'data_point__metric')
        .prefetch_related('data_point__dimension_categories')
    )
    references = [SourceReferenceSnapshot(data_source=add_source(r.data_source), point=None) for r in dataset_refs]
    references += [
        SourceReferenceSnapshot(data_source=add_source(r.data_source), point=_data_point_key(r.data_point)) for r in dp_refs
    ]

    comment_qs = (
        DataPointComment.objects  # default manager excludes soft-deleted
        .filter(data_point__dataset=ds)
        .select_related('data_point__metric', 'created_by', 'last_modified_by', 'resolved_by')
        .prefetch_related('data_point__dimension_categories')
    )
    comments = [
        DataPointCommentSnapshot(
            point=_data_point_key(c.data_point),
            text=c.text,
            is_sticky=c.is_sticky,
            is_review=c.is_review,
            review_state=c.review_state,
            resolved_at=c.resolved_at.isoformat() if c.resolved_at else None,
            created_by=str(c.created_by.uuid) if c.created_by else None,
            last_modified_by=str(c.last_modified_by.uuid) if c.last_modified_by else None,
            resolved_by=str(c.resolved_by.uuid) if c.resolved_by else None,
        )
        for c in comment_qs
    ]
    return list(sources.values()), references, comments


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


def build_instance_snapshot(
    ic: InstanceConfig,
    dataset_revision_pins: dict[int, DatasetRevisionPinSnapshot] | None = None,
) -> InstanceSnapshot:
    """
    Structural snapshot of a DB-sourced InstanceConfig.

    Structural references are pinned by UUID; dataset bodies are not included.
    Use ``export_instance`` when the bodies are also needed.
    """
    if ic.spec is None:
        msg = f'Instance {ic.identifier} has no spec — run sync_instance_to_db first'
        raise ValueError(msg)

    node_qs = (
        ic.nodes.get_queryset().active().with_spec().select_related('indicator_node', 'copy_of', 'layout').order_by('order', 'pk')
    )
    nodes = [NodeSnapshot.from_model(nc, primary_language=ic.primary_language) for nc in node_qs]

    edges = [EdgeSnapshot.from_model(e) for e in edge_qs_for(ic)]

    port_qs = dataset_port_qs_for(ic)
    dataset_ports: list[DatasetPortSnapshot] = []
    for port in port_qs:
        snapshot = DatasetPortSnapshot.from_model(port)
        if dataset_revision_pins is not None:
            pin = dataset_revision_pins.get(port.dataset_id)
            snapshot.dataset_revision = pin.revision_id if pin is not None else None
        dataset_ports.append(snapshot)

    dimensions = _dimension_catalog_for(ic)
    datasets = _dataset_catalog_for(
        ic,
        dataset_ids={port.dataset_id for port in port_qs},
        dataset_revision_pins=dataset_revision_pins,
    )

    return InstanceSnapshot(
        metadata=InstanceMetadata.from_model(ic),
        spec=ic.spec,
        copy_of=str(ic.copy_of.uuid) if ic.copy_of else None,
        nodes=nodes,
        edges=edges,
        dataset_ports=dataset_ports,
        dataset_revisions=list(dataset_revision_pins.values()) if dataset_revision_pins is not None else [],
        dimensions=dimensions,
        datasets=datasets,
    )


def _dimension_catalog_for(ic: InstanceConfig) -> list[DimensionMeta]:
    from kausal_common.datasets.models import DimensionScope

    scopes = (
        DimensionScope.objects
        .for_instance_config(ic)
        .select_related('dimension')
        .prefetch_related('dimension__categories')
        .order_by('order')
    )
    dimensions: list[DimensionMeta] = []
    for scope in scopes:
        dimension = scope.dimension
        if scope.identifier is None:
            raise ValueError(f'Dimension {dimension.uuid} has no identifier in instance {ic.identifier}')
        categories = tuple(
            DimensionCategoryMeta(
                id=category.uuid,
                identifier=category.identifier,
                label=_ts_from_modeltrans(category, 'label', ic.primary_language),
                order=category.order,
                spec=dict(category.spec or {}),
            )
            for category in dimension.categories.all()
        )
        dimensions.append(
            DimensionMeta(
                id=dimension.uuid,
                identifier=scope.identifier,
                label=_ts_from_modeltrans(dimension, 'name', ic.primary_language),
                order=scope.order,
                spec=dict(dimension.spec or {}),
                categories=categories,
            )
        )
    return dimensions


def dataset_meta_from_model(
    dataset: DatasetModel,
    *,
    primary_language: str,
    pinned_revision_id: int | None = None,
) -> DatasetMeta:
    """Build the graph catalog entry for one dataset, exactly as snapshots record it."""
    schema = dataset.schema
    if schema is None:
        raise ValueError(f'Dataset {dataset.uuid} has no schema')
    metrics = tuple(
        DatasetMetricMeta(
            id=metric.uuid,
            identifier=metric.name,
            label=_ts_from_modeltrans(metric, 'label', primary_language),
            unit=metric.unit,
            order=metric.order,
        )
        for metric in schema.metrics.all()
    )
    declared_dimension_ids = tuple(schema_dimension.dimension.uuid for schema_dimension in schema.dimensions.all())
    return DatasetMeta(
        id=dataset.uuid,
        identifier=dataset.identifier,
        schema_id=schema.uuid,
        metrics=metrics,
        declared_dimension_ids=declared_dimension_ids,
        is_external_placeholder=dataset.is_external_placeholder,
        external_ref=dataset.external_ref,
        revision_id=pinned_revision_id if pinned_revision_id is not None else dataset.latest_revision_id,
    )


def _dataset_catalog_for(
    ic: InstanceConfig,
    *,
    dataset_ids: set[int],
    dataset_revision_pins: dict[int, DatasetRevisionPinSnapshot] | None,
) -> list[DatasetMeta]:
    from kausal_common.datasets.models import Dataset as DatasetModel

    datasets = (
        DatasetModel.objects
        .filter(pk__in=dataset_ids)
        .select_related('schema')
        .prefetch_related('schema__metrics', 'schema__dimensions__dimension')
        .order_by('pk')
    )
    result: list[DatasetMeta] = []
    for dataset in datasets:
        pin = dataset_revision_pins.get(dataset.pk) if dataset_revision_pins is not None else None
        result.append(
            dataset_meta_from_model(
                dataset,
                primary_language=ic.primary_language,
                pinned_revision_id=pin.revision_id if pin is not None else None,
            )
        )
    return result


def edge_qs_for(ic: InstanceConfig) -> QuerySet[NodeEdge]:
    """
    Edges in canonical snapshot order: creation (pk) order.

    Creation order is the authored order — the parser mirrors the YAML
    runtime's edge-creation sequence and the sync writes rows in that
    sequence, so pk order is what the YAML runtime observes; editor-created
    edges append at the end. ``NodeEdge.Meta.ordering`` (source-node pk)
    must not be used here: it reorders inputs by an accident of node
    creation and made DB-sourced sector breakdowns and additive summation
    order diverge from the same instance served from YAML. Position
    assignment (``ordered_binding_snapshots``) and the
    ``NodeInputPortBinding`` mirror both depend on this order.
    """
    from nodes.models import NodeEdge

    return (
        NodeEdge.objects
        .filter(instance=ic)
        .annotate(
            _from_node_uuid=F('from_node__uuid'),
            _to_node_uuid=F('to_node__uuid'),
        )
        .order_by('pk')
    )


def dataset_port_qs_for(ic: InstanceConfig) -> QuerySet[DatasetPort]:
    from nodes.models import DatasetPort

    return (
        DatasetPort.objects
        .filter(instance=ic)
        .select_related('node', 'dataset', 'metric')
        .only(
            'uuid',
            'dataset_index',
            'port_id',
            'spec',
            'node__uuid',
            'dataset__identifier',
            'dataset__uuid',
            'dataset__latest_revision_id',
            'metric__name',
            'metric__uuid',
        )
        .order_by('node__identifier', 'dataset_index', 'metric__order', 'port_id')
    )


def _dataset_export_key(ds: DatasetModel) -> str:
    return ds.identifier or str(ds.uuid)


def _dataset_export_rank(ds: DatasetModel, ic_ct_id: int, ic_id: int) -> tuple[bool, bool, int]:
    is_direct = ds.scope_content_type_id == ic_ct_id and ds.scope_id == ic_id
    return (not is_direct, ds.is_external_placeholder, ds.pk)


def _datasets_for_instance_export(ic: InstanceConfig, ic_ct: ContentType) -> list[DatasetModel]:
    from django.db.models import Q

    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetSchemaScope

    schema_scope_ids = DatasetSchemaScope.objects.filter(
        scope_content_type=ic_ct,
        scope_id=ic.pk,
    ).values('schema_id')
    qs = (
        DatasetModel.objects
        .filter(Q(scope_content_type=ic_ct, scope_id=ic.pk) | Q(schema_id__in=schema_scope_ids))
        .select_related('schema', 'scope_content_type')
        .distinct()
    )

    # During CADS bootstrapping an instance may temporarily have both
    # schema-scoped external placeholders and direct real datasets with the
    # same identifier. Export the real direct dataset in that case so clones
    # receive datapoints and their ports can be reconstructed.
    datasets_by_key: dict[str, DatasetModel] = {}
    for ds in qs:
        key = _dataset_export_key(ds)
        existing = datasets_by_key.get(key)
        if existing is None or _dataset_export_rank(ds, ic_ct.pk, ic.pk) < _dataset_export_rank(existing, ic_ct.pk, ic.pk):
            datasets_by_key[key] = ds
    return sorted(datasets_by_key.values(), key=_dataset_export_key)


def export_instance(ic: InstanceConfig) -> InstanceExport:
    """Serialize a DB-sourced InstanceConfig with dataset bodies included."""
    from django.contrib.contenttypes.models import ContentType

    from nodes.page_snapshot import build_instance_page_snapshots

    snapshot = build_instance_snapshot(ic)

    ic_ct = ContentType.objects.get_for_model(ic)
    datasets = [DatasetSnapshot.from_model_for_instance(ds, ic) for ds in _datasets_for_instance_export(ic, ic_ct)]

    return InstanceExport(instance=snapshot, datasets=datasets, pages=build_instance_page_snapshots(ic))


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


@transaction.atomic
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
                column_name=ds_snapshot.dimension_columns.get(dim_id),
            )

    # Create dataset
    dataset = DatasetModel(
        identifier=ds_snapshot.identifier,
        spec={'forecast_from': ds_snapshot.forecast_from} if ds_snapshot.forecast_from is not None else {},
        is_external_placeholder=ds_snapshot.is_external_placeholder,
        external_ref=ds_snapshot.external_ref,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        schema=schema,
    )
    dataset.save()

    # Create data points
    dp_map: dict[tuple[int, str, tuple[str, ...]], Any] = {}
    if ds_snapshot.data is not None:
        dp_map = _import_data_points(dataset, ds_snapshot, metrics_by_id, dim_lookup)

    # Recreate source references and comments (data points must exist first).
    _import_dataset_provenance(ic, ic_ct, ds_snapshot, dataset, dp_map)

    if not dataset.is_external_placeholder:
        from nodes.dataset_materialization import refresh_dataset_materialization

        refresh_dataset_materialization(dataset)

    return dataset


def _data_point_key_tuple(point: DataPointKey) -> tuple[int, str, tuple[str, ...]]:
    return (point.year, point.metric, tuple(point.categories))


def _import_dataset_provenance(  # noqa: C901
    ic: InstanceConfig,
    ic_ct: ContentType,
    ds_snapshot: DatasetSnapshot,
    dataset: DatasetModel,
    dp_map: dict[tuple[int, str, tuple[str, ...]], Any],
) -> None:
    """Recreate a dataset's DataSources, source references and data-point comments."""
    if not (ds_snapshot.data_sources or ds_snapshot.source_references or ds_snapshot.comments):
        return

    from django.contrib.auth import get_user_model

    from kausal_common.datasets.models import (
        DataPointComment,
        DataPointCommentReviewState,
        DatasetSourceReference,
        DataSource,
    )

    user_model = get_user_model()
    user_cache: dict[str, Any] = {}

    def resolve_user(user_uuid: str | None) -> Any:
        if not user_uuid:
            return None
        if user_uuid not in user_cache:
            user_cache[user_uuid] = user_model.objects.filter(uuid=user_uuid).first()
        return user_cache[user_uuid]

    # DataSources are scoped to the target instance and get fresh uuids (a same-DB
    # copy can't reuse the globally-unique source uuid); map old uuid → new object.
    src_map: dict[str, Any] = {}
    for s in ds_snapshot.data_sources:
        src_map[s.uuid] = DataSource.objects.create(
            scope_content_type=ic_ct,
            scope_id=ic.pk,
            name=s.name,
            edition=s.edition,
            authority=s.authority,
            description=s.description,
            url=s.url,
        )

    for ref in ds_snapshot.source_references:
        src_obj = src_map.get(ref.data_source)
        if src_obj is None:
            continue
        if ref.point is None:
            DatasetSourceReference.objects.create(dataset=dataset, data_source=src_obj)
            continue
        dp = dp_map.get(_data_point_key_tuple(ref.point))
        if dp is not None:
            DatasetSourceReference.objects.create(data_point=dp, data_source=src_obj)

    for c in ds_snapshot.comments:
        dp = dp_map.get(_data_point_key_tuple(c.point))
        if dp is None:
            continue
        DataPointComment.objects.create(
            data_point=dp,
            text=c.text,
            is_sticky=c.is_sticky,
            is_review=c.is_review,
            review_state=DataPointCommentReviewState(c.review_state) if c.review_state else None,
            resolved_at=datetime.fromisoformat(c.resolved_at) if c.resolved_at else None,
            created_by=resolve_user(c.created_by),
            last_modified_by=resolve_user(c.last_modified_by),
            resolved_by=resolve_user(c.resolved_by),
        )


def _resolve_metric_data_columns(
    ds_snapshot: DatasetSnapshot, metric_ids: list[str], dim_columns: dict[str, str]
) -> dict[str, str]:
    """
    Map each metric id to the data column that holds its value.

    The serialized data columns are named ``Coalesce(name, label, uuid)`` (see
    ``DBDataset.deserialize_df``), whereas a metric snapshot's identifier is
    ``name or uuid`` — so a metric with no ``name`` but a ``label`` is keyed by
    its uuid here while its data column is the label. ``deserialize_df`` builds
    that column from the metric's raw (base-language) ``label``, which need not
    equal ``str(label)`` under a different active Django language, so match
    against *all* of the label's translations. Fall back (for the common
    single-metric case) to the sole remaining value column.
    """
    fields = (ds_snapshot.data or {}).get('schema', {}).get('fields', [])
    all_columns = {f['name'] for f in fields}
    value_columns = all_columns - {'Year', 'id', 'uuid', *dim_columns.values()}
    labels_by_id = {m.identifier: (m.label.all() if m.label is not None else []) for m in ds_snapshot.metrics}

    columns: dict[str, str] = {}
    for metric_id in metric_ids:
        label_match = next((lbl for lbl in labels_by_id.get(metric_id, []) if lbl in value_columns), None)
        if metric_id in value_columns:
            columns[metric_id] = metric_id
        elif label_match is not None:
            columns[metric_id] = label_match
        elif len(metric_ids) == 1 and len(value_columns) == 1:
            columns[metric_id] = next(iter(value_columns))
        else:
            columns[metric_id] = metric_id
    return columns


def _import_data_points(
    dataset: DatasetModel,
    ds_snapshot: DatasetSnapshot,
    metrics_by_id: dict[str, DatasetMetric],
    dim_lookup: dict[str, DimensionCategory],
) -> dict[tuple[int, str, tuple[str, ...]], Any]:
    """Create DataPoints; return a natural-key → DataPoint map for provenance wiring."""
    from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory

    assert ds_snapshot.data is not None
    dim_ids = ds_snapshot.dimensions
    dim_columns = {dim_id: ds_snapshot.dimension_columns.get(dim_id, dim_id) for dim_id in dim_ids}
    metric_columns = _resolve_metric_data_columns(ds_snapshot, list(metrics_by_id), dim_columns)

    data_points: list[DataPoint] = []
    # (data_point_index, category) pairs for bulk M2M creation
    dp_categories: list[tuple[int, DimensionCategory]] = []
    # natural key per created data point, parallel to ``data_points``
    dp_keys: list[tuple[int, str, tuple[str, ...]]] = []

    for row in ds_snapshot.data['data']:
        year_val = row.get('Year')
        if year_val is None:
            continue
        dp_date = date(year=int(year_val), month=1, day=1)

        # Resolve dimension categories for this row (objects + their id strings,
        # which match the export-side natural key).
        row_cats: list[DimensionCategory] = []
        row_cat_ids: list[str] = []
        for dim_id in dim_ids:
            cat_id = row.get(dim_columns[dim_id])
            if cat_id:
                cat = dim_lookup.get(f'{dim_id}/{cat_id}')
                if cat:
                    row_cats.append(cat)
                    row_cat_ids.append(str(cat_id))
        cat_key = tuple(sorted(row_cat_ids))

        for metric_id, metric in metrics_by_id.items():
            value = row.get(metric_columns[metric_id])
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
            dp_keys.append((int(year_val), metric_id, cat_key))
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

    return {dp_keys[i]: created_dps[i] for i in range(len(created_dps))}


def _dimension_category_lookup_for_instance(ic: InstanceConfig, ic_ct: ContentType) -> dict[str, DimensionCategory]:
    from kausal_common.datasets.models import DimensionScope

    lookup: dict[str, DimensionCategory] = {}
    scopes = (
        DimensionScope.objects
        .filter(scope_content_type=ic_ct, scope_id=ic.pk, identifier__isnull=False)
        .select_related('dimension')
        .prefetch_related('dimension__categories')
    )
    for scope in scopes:
        assert scope.identifier is not None
        for category in scope.dimension.categories.all():
            if category.identifier is not None:
                lookup[f'{scope.identifier}/{category.identifier}'] = category
    return lookup


def _ensure_dataset_dimensions(
    ic: InstanceConfig,
    ds_snapshot: DatasetSnapshot,
    ic_ct: ContentType,
    dim_lookup: dict[str, DimensionCategory],
) -> None:
    from kausal_common.datasets.models import Dimension, DimensionCategory as DimensionCategoryModel, DimensionScope

    for dim_id in ds_snapshot.dimensions:
        dim_scope = (
            DimensionScope.objects
            .filter(
                identifier=dim_id,
                scope_content_type=ic_ct,
                scope_id=ic.pk,
            )
            .select_related('dimension')
            .first()
        )
        if dim_scope is None:
            dimension = Dimension.objects.create(name=_label_from_identifier(dim_id))
            DimensionScope.objects.create(
                dimension=dimension,
                identifier=dim_id,
                scope_content_type=ic_ct,
                scope_id=ic.pk,
            )
        else:
            dimension = dim_scope.dimension

        existing_categories = set(dimension.categories.values_list('identifier', flat=True))
        column_name = ds_snapshot.dimension_columns.get(dim_id, dim_id)
        category_ids = sorted({
            str(cat_id) for row in (ds_snapshot.data or {}).get('data', []) if (cat_id := row.get(column_name))
        })
        for cat_id in category_ids:
            if cat_id in existing_categories:
                continue
            cat = DimensionCategoryModel.objects.create(
                dimension=dimension,
                identifier=cat_id,
                label=_label_from_identifier(cat_id),
            )
            dim_lookup[f'{dim_id}/{cat_id}'] = cat
            existing_categories.add(cat_id)


def _validate_dataset_dimensions(
    ic: InstanceConfig,
    ds_snapshot: DatasetSnapshot,
    dim_lookup: dict[str, DimensionCategory],
) -> None:
    if ds_snapshot.data is None:
        return
    missing = {
        f'{dim_id}/{cat_id}'
        for row in ds_snapshot.data.get('data', [])
        for dim_id in ds_snapshot.dimensions
        if (cat_id := row.get(ds_snapshot.dimension_columns.get(dim_id, dim_id))) and f'{dim_id}/{cat_id}' not in dim_lookup
    }
    if missing:
        missing_str = ', '.join(sorted(missing)[:10])
        if len(missing) > 10:
            missing_str += ', ...'
        raise ValueError(
            f'Cannot import dataset {ds_snapshot.identifier!r} into {ic.identifier!r}; missing dimension categories: '
            + f'{missing_str}'
        )


def _rewire_dataset_ports(ic: InstanceConfig, datasets_by_id: dict[str, DatasetModel]) -> int:
    from nodes.models import DatasetPort

    rewired = 0
    ports = DatasetPort.objects.filter(instance=ic, dataset__identifier__in=datasets_by_id).select_related('dataset', 'metric')
    for port in ports:
        if port.dataset.identifier is None:
            continue
        dataset = datasets_by_id.get(port.dataset.identifier)
        if dataset is None or dataset.pk == port.dataset_id:
            continue
        assert dataset.schema is not None
        metric = dataset.schema.metrics.filter(name=port.metric.name).first()
        if metric is None:
            raise ValueError(
                f'Cannot rewire dataset port {port.pk} to dataset {dataset.identifier!r}; metric {port.metric.name!r} is missing'
            )
        port.dataset = dataset
        port.metric = metric
        port.save(update_fields=['dataset', 'metric'])
        rewired += 1
    return rewired


def _delete_superseded_placeholders(ic: InstanceConfig, dataset_ids: Iterable[str]) -> int:
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetSchemaScope

    ids = set(dataset_ids)
    if not ids:
        return 0

    ic_ct = ContentType.objects.get_for_model(ic)
    schema_scope_ids = DatasetSchemaScope.objects.filter(
        scope_content_type=ic_ct,
        scope_id=ic.pk,
    ).values('schema_id')
    placeholders = list(
        DatasetModel.objects
        .filter(
            schema_id__in=schema_scope_ids,
            identifier__in=ids,
            is_external_placeholder=True,
        )
        .exclude(scope_content_type=ic_ct, scope_id=ic.pk)
        .select_related('schema')
    )

    deleted = 0
    for placeholder in placeholders:
        schema = placeholder.schema
        placeholder.delete()
        deleted += 1
        if schema is not None and not schema.datasets.exists():
            schema.delete()
    return deleted


def import_instance_datasets(
    ic: InstanceConfig,
    dataset_snapshots: Iterable[DatasetSnapshot],
    *,
    rewire_dataset_ports: bool = False,
    delete_superseded_placeholders: bool = False,
    create_missing_dimensions: bool = False,
) -> list[DatasetModel]:
    """
    Import dataset bodies into an existing InstanceConfig without touching nodes.

    This is used when a template instance already has its node graph and
    dataset ports, but its datasets need to be promoted from external
    placeholders to real DB datasets with datapoints.
    """
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset as DatasetModel

    ic_ct = ContentType.objects.get_for_model(ic)
    dim_lookup = _dimension_category_lookup_for_instance(ic, ic_ct)
    imported: list[DatasetModel] = []
    datasets_by_id: dict[str, DatasetModel] = {}

    for ds_snapshot in dataset_snapshots:
        if ds_snapshot.identifier is not None:
            existing = DatasetModel.objects.filter(
                scope_content_type=ic_ct,
                scope_id=ic.pk,
                identifier=ds_snapshot.identifier,
                is_external_placeholder=False,
            ).first()
            if existing is not None:
                if ds_snapshot.data is None or existing.data_points.exists():
                    imported.append(existing)
                    datasets_by_id[ds_snapshot.identifier] = existing
                    continue
                raise ValueError(f'Dataset {ds_snapshot.identifier!r} already exists for {ic.identifier!r} but has no datapoints')

        if create_missing_dimensions:
            _ensure_dataset_dimensions(ic, ds_snapshot, ic_ct, dim_lookup)
        _validate_dataset_dimensions(ic, ds_snapshot, dim_lookup)
        dataset = _import_dataset(ic, ds_snapshot, ic_ct, dim_lookup)
        imported.append(dataset)
        if ds_snapshot.identifier is not None:
            datasets_by_id[ds_snapshot.identifier] = dataset

    if rewire_dataset_ports:
        _rewire_dataset_ports(ic, datasets_by_id)
    if delete_superseded_placeholders:
        _delete_superseded_placeholders(ic, datasets_by_id.keys())

    return imported


def import_instance_nodes(ic: InstanceConfig, export: InstanceExport) -> dict[UUID, NodeConfig]:
    """
    Create NodeConfig rows for ``ic`` from the snapshot's nodes.

    Materialises the node rows (all fields — ``name``/``short_description``/
    ``description``/``goal``/``color``/``order``/``is_visible`` — plus ``spec``
    and ``indicator_node`` links) from ``export.instance.nodes``, *without*
    touching the instance-level spec or ``config_source``. Node references are
    UUID-keyed in the snapshot, so no pk remapping is needed.

    Used by yaml-mode copies so admin-authored node fields (which the YAML
    can't express) are carried over, instead of rebuilding rows from the YAML
    via ``InstanceConfig.sync_nodes()``.
    """
    return _import_nodes(ic, export)


def import_instance_edges_and_ports(
    ic: InstanceConfig,
    export: InstanceExport,
    nodes_by_uuid: dict[UUID, NodeConfig],
    datasets_by_id: dict[str, DatasetModel],
) -> None:
    """
    Recreate the editor graph bindings (``NodeEdge`` + ``DatasetPort``) for ``ic``.

    Companion to :func:`import_instance_nodes` for callers that build the DB
    mirror piecemeal (yaml-mode copies) rather than through the full
    :func:`import_instance`. Edges and ports are matched by node UUID and dataset
    identifier, so references that don't resolve in ``ic`` (e.g. a DVC dataset
    not materialised in the DB) are skipped rather than erroring. Does not touch
    ``config_source`` or the instance spec — these tables are dormant for
    ``config_source='yaml'`` (the runtime loads the YAML) but are read by the
    Trailhead editor, so a copy should mirror whatever the source has.
    """
    from nodes.input_bindings import sync_input_bindings

    _import_edges(ic, export, nodes_by_uuid)
    _import_dataset_ports(ic, export, nodes_by_uuid, datasets_by_id)
    sync_input_bindings(ic)


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
) -> dict[UUID, NodeConfig]:
    """Create NodeConfig objects. Returns UUID → NodeConfig map."""
    from nodes.models import NodeConfig, NodeLayout, NodeLayoutSource

    primary_lang = ic.primary_language
    nodes_by_uuid: dict[UUID, NodeConfig] = {}
    for n in export.instance.nodes:
        if n.identifier is None:
            raise ValueError(f'Node {n.uuid} has no identifier; the legacy runtime still requires one')
        fields: dict[str, Any] = {}
        i18n_dict: dict[str, str] = {}
        _apply_translated(fields, i18n_dict, n.name, 'name', primary_lang)
        _apply_translated(fields, i18n_dict, n.short_name, 'short_name', primary_lang)
        _apply_translated(fields, i18n_dict, n.short_description, 'short_description', primary_lang)
        _apply_translated(fields, i18n_dict, n.description, 'description', primary_lang)
        _apply_translated(fields, i18n_dict, n.goal, 'goal', primary_lang)

        nc = NodeConfig.objects.create(
            instance=ic,
            identifier=n.identifier,
            color=n.color,
            order=n.order,
            is_visible=n.is_visible,
            body=n.body or [],
            i18n=i18n_dict,
            **fields,
        )
        # Write spec via queryset.update() to bypass ClusterableModel.save()
        if n.spec is not None:
            NodeConfig.objects.filter(pk=nc.pk).update(spec=n.spec)
            nc.spec = n.spec
        nodes_by_uuid[n.uuid] = nc

        if n.layout is not None:
            NodeLayout.objects.create(
                node=nc,
                x=n.layout.x,
                y=n.layout.y,
                source=NodeLayoutSource(n.layout.source),
            )

    # Resolve indicator_node references
    for n in export.instance.nodes:
        if n.indicator_node and n.indicator_node in nodes_by_uuid:
            nc = nodes_by_uuid[n.uuid]
            indicator = nodes_by_uuid[n.indicator_node]
            NodeConfig.objects.filter(pk=nc.pk).update(indicator_node=indicator)

    # Resolve copy_of references by uuid (restore fidelity; the source node may
    # live in another instance and be absent here, in which case it stays null).
    for n in export.instance.nodes:
        if not n.copy_of:
            continue
        src = NodeConfig.objects.filter(uuid=n.copy_of).first()
        if src is not None:
            NodeConfig.objects.filter(pk=nodes_by_uuid[n.uuid].pk).update(copy_of=src)

    return nodes_by_uuid


def _import_edges(
    ic: InstanceConfig,
    export: InstanceExport,
    nodes_by_uuid: dict[UUID, NodeConfig],
) -> None:
    from nodes.models import NodeEdge

    for e in export.instance.edges:
        from_node = nodes_by_uuid.get(e.from_node)
        to_node = nodes_by_uuid.get(e.to_node)
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
    nodes_by_uuid: dict[UUID, NodeConfig],
    datasets_by_id: dict[str, DatasetModel],
) -> None:
    from nodes.models import DatasetPort

    for p in export.instance.dataset_ports:
        node = nodes_by_uuid.get(p.node)
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
            dataset_index=p.dataset_index,
            spec=p.spec,
        )


def import_instance(ic: InstanceConfig, export: InstanceExport, framework_config: FrameworkConfig | None = None) -> None:
    """
    Populate an InstanceConfig with computation model objects from an InstanceExport.

    The InstanceConfig must already exist (with identifier, org, etc.).
    This function creates all related objects: nodes, edges, datasets, ports.
    """
    from django.contrib.contenttypes.models import ContentType

    ic_ct = ContentType.objects.get_for_model(ic)

    # Store the computation spec. Copy the template's language metadata onto
    # the InstanceConfig row so i18n-bearing data (ActionGroup names, etc.)
    # stays loadable — the spec's TranslatedStrings are authored under the
    # template's primary_language and would be filtered out if the
    # InstanceConfig used a different language.
    ic.spec = export.instance.spec.model_copy()
    meta = export.instance.metadata
    ic.primary_language = meta.primary_language
    ic.other_languages = list(meta.other_languages)
    ic.config_source = 'database'
    update_fields = ['spec', 'primary_language', 'other_languages', 'config_source']

    # Owner display name comes from the template (or the framework org) and is
    # written to the column; the instance keeps its own name.
    owner_src = meta.owner
    if framework_config is not None:
        owner_src = str(framework_config.organization_name)
        ic.uuid = framework_config.uuid
        update_fields.append('uuid')
    i18n = dict(ic.i18n or {})
    ic.owner = ''
    if owner_src:
        owner_val, owner_i18n = get_modeltrans_attrs_from_str(
            cast('str | TranslatedString', owner_src), 'owner', ic.primary_language
        )
        ic.owner = owner_val
        i18n.update(owner_i18n)
    for field_name, value in (
        ('lead_title', meta.lead_title),
        ('lead_paragraph', meta.lead_paragraph),
    ):
        if value is None:
            continue
        primary_val, translations = get_modeltrans_attrs_from_str(
            cast('str | TranslatedString', value), field_name, ic.primary_language, strict=False
        )
        setattr(ic, field_name, primary_val)
        i18n.update(translations)
        update_fields.append(field_name)
    ic.i18n = i18n
    update_fields += ['owner', 'i18n']
    ic.save(update_fields=update_fields)

    # Resolve copy_of by uuid (restore fidelity; absent source → stays null).
    if export.instance.copy_of:
        from nodes.models import InstanceConfig as _InstanceConfig

        src_ic = _InstanceConfig.objects.filter(uuid=export.instance.copy_of).first()
        if src_ic is not None:
            ic.copy_of = src_ic
            ic.save(update_fields=['copy_of'])

    # Dimensions first — datasets and data points reference them
    _import_dimensions(ic, export, ic_ct)

    # Datasets (with data points)
    datasets = import_instance_datasets(ic, export.datasets, create_missing_dimensions=True)
    # ``identifier`` may be None for datasets keyed only by uuid; skip those
    # here since node→dataset wiring goes through identifier.
    datasets_by_id = {ds.identifier: ds for ds in datasets if ds.identifier is not None}

    # Nodes
    nodes_by_uuid = _import_nodes(ic, export)

    # Edges
    _import_edges(ic, export, nodes_by_uuid)

    # Dataset ports
    _import_dataset_ports(ic, export, nodes_by_uuid, datasets_by_id)

    # Refresh the unified input-binding mirror from the imported rows.
    from nodes.input_bindings import sync_input_bindings

    sync_input_bindings(ic)
