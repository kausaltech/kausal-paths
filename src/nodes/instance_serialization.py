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
from uuid import UUID, uuid3

from django.db import transaction
from modeltrans.translator import get_i18n_field
from pydantic import BaseModel, Field, field_validator

from markdown_it import MarkdownIt

from kausal_common.datasets.category_domain import DatasetCategoryDomain
from kausal_common.i18n.pydantic import (
    I18nBaseModel,
    ModeltransModelProtocol,
    TranslatedString,
    get_modeltrans_attrs_from_str,
    get_translated_string_from_modeltrans,
)

from datasets.validation_rules import ValidationRule, validation_rule_adapter
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
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from django.contrib.contenttypes.models import ContentType
    from django.db.models import Model, QuerySet

    from kausal_common.datasets.models import (
        Dataset as DatasetModel,
        DatasetMetric,
        DimensionCategory,
    )

    from frameworks.models import FrameworkConfig
    from nodes.models import InstanceConfig, NodeConfig, NodeInputPortBinding, NodeLayout
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
#   v9: one discriminated ``bindings`` list with stored per-port positions
#       replaces the ``edges`` + ``dataset_ports`` arrays.
#   v10: action groups have stable UUIDs and action node group references
#        use those UUIDs instead of human-readable identifiers.
SNAPSHOT_SCHEMA_VERSION = 11

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
    i18n_field = get_i18n_field(obj)
    assert i18n_field is not None
    assert i18n_field.attname == 'i18n'
    mt_obj = cast('ModeltransModelProtocol', obj)
    i18n = cast('dict[str, str]', mt_obj.i18n or {})
    has_translation = any(k.startswith(f'{field_name}_') and v for k, v in i18n.items())
    if not val and not has_translation:
        return None
    return get_translated_string_from_modeltrans(mt_obj, field_name, primary_language)


class MetricValidationRuleSnapshot(ModelSnapshot):
    """
    One validation rule bound to a metric.

    ``rule`` parses the stored blob strictly against the schema in
    ``datasets.validation_rules``; ``uuid`` records the source row's identity.
    """

    uuid: UUID
    rule: ValidationRule

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        return cls(uuid=obj.uuid, rule=validation_rule_adapter.validate_python(obj.rule))


def metric_column_id(metric: DatasetMetric) -> str:
    """
    Resolve the dataframe column a metric maps to.

    ``DatasetMetric.name`` *is* the column name and ``label`` is display text, so the
    two agree for imported metrics: ``load_dvc_dataset`` and ``dataset_placeholders``
    both create them as ``name=label=<column>``. A metric authored in the editor has
    no ``name`` at all, and its column is built from the label — see the
    ``Coalesce(name, label, uuid)`` in ``DBDataset.deserialize_df``
    (``nodes/datasets.py``), which is the writer this function has to agree with.

    **Falling through to the uuid is never a working answer.** No dataframe column is
    ever named after a metric uuid, so a selector built from one cannot match: it
    fails in ``select_metric`` as "Column '<uuid>' not found". It is kept only so a
    metric carrying neither name nor label still yields a stable string instead of
    ``None``.
    """
    return metric.name or metric.label or str(metric.uuid)


class DatasetMetricSnapshot(ModelSnapshot):
    identifier: str
    label: TranslatedString | None = None
    unit: str
    quantity: str | None = None
    validation_rules: list[MetricValidationRuleSnapshot] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: Any) -> Self:
        # Metrics live under a DatasetSchema; primary language is the
        # schema's parent scope's instance-config primary language. For the
        # nested path the caller resolves the language and passes via
        # ``from_model_with_language`` below — the default path assumes
        # i18n-less data.
        return cls(
            identifier=metric_column_id(obj),
            label=_ts_from_modeltrans(obj, 'label', 'en') if obj.label or obj.i18n else None,
            unit=obj.unit,
            quantity=(obj.spec or {}).get('quantity'),
            validation_rules=cls._rules_from_model(obj),
        )

    @classmethod
    def from_model_with_language(cls, obj: Any, primary_language: str) -> Self:
        return cls(
            identifier=metric_column_id(obj),
            label=_ts_from_modeltrans(obj, 'label', primary_language),
            unit=obj.unit,
            quantity=(obj.spec or {}).get('quantity'),
            validation_rules=cls._rules_from_model(obj),
        )

    @staticmethod
    def _rules_from_model(obj: Any) -> list[MetricValidationRuleSnapshot]:
        return [MetricValidationRuleSnapshot.from_model(rule) for rule in obj.validation_rules.order_by('order')]


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
    is_editable: bool = True
    dimensions: list[str] = Field(default_factory=list)
    dimension_columns: dict[str, str] = Field(default_factory=dict)
    metrics: list[DatasetMetricSnapshot] = Field(default_factory=list)
    category_domain: DatasetCategoryDomain = Field(default_factory=DatasetCategoryDomain)
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
        is_editable = True
        primary_language = instance_config.primary_language if instance_config is not None else _primary_language_for_dataset(obj)

        if schema is not None:
            time_resolution = schema.time_resolution
            is_editable = schema.is_editable
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
            is_editable=is_editable,
            dimensions=dimensions,
            dimension_columns=dimension_columns,
            metrics=metrics,
            category_domain=schema.category_domain if schema is not None else DatasetCategoryDomain(),
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
    is_editable: bool | None = None
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
            is_editable=obj.is_editable,
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
            is_editable=obj.is_editable,
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
            'is_editable': source.is_editable if source.is_editable is not None else stored.is_editable,
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
    kind: Literal['edge'] = 'edge'
    uuid: UUID | None = None
    # Stable order among values delivered to the target port, shared with
    # dataset bindings. Assigned at snapshot production; ``None`` only on
    # transient pre-resolution (parse-side) snapshots.
    position: int | None = None
    from_node: UUID
    to_node: UUID
    from_port: UUID
    to_port: UUID
    transformations: list[EdgeTransformOp] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class DatasetPortSnapshot(ModelSnapshot):
    kind: Literal['dataset'] = 'dataset'
    uuid: UUID | None = None
    # See ``EdgeSnapshot.position`` — one order across both binding kinds.
    position: int | None = None
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


type BindingSnapshot = EdgeSnapshot | DatasetPortSnapshot
"""One entry of ``InstanceSnapshot.bindings``, discriminated by ``kind``."""


def _upgrade_bindings_v9(data: dict[str, Any]) -> None:
    """Merge the legacy ``edges`` + ``dataset_ports`` arrays into one positioned binding list."""
    edges = [EdgeSnapshot.model_validate(e) for e in data.pop('edges', [])]
    ports = [DatasetPortSnapshot.model_validate(p) for p in data.pop('dataset_ports', [])]
    bindings: list[dict[str, Any]] = []
    for item, position in ordered_binding_snapshots(edges, ports):
        item.position = position
        # Dumped so the v11 upgrader (which operates on raw dicts) sees them.
        bindings.append(item.model_dump(mode='json'))
    data['bindings'] = bindings


def _upgrade_bindings_v11(data: dict[str, Any]) -> None:
    """
    Convert the kind-discriminated edge/dataset binding entries to the unified form.

    v9/v10 stored ``EdgeSnapshot`` / ``DatasetPortSnapshot`` payloads; v11 stores
    ``InputBindingSnapshot``. The dataset entry's ``spec`` may still be in a
    pre-pipeline stored shape, so it is normalized through ``DatasetPortSpec``
    before its transformations and tags move onto the binding.
    """
    upgraded: list[dict[str, Any]] = []
    for entry in data.get('bindings', []):
        if not isinstance(entry, dict):
            upgraded.append(entry)
            continue
        if entry.get('kind') == 'edge':
            upgraded.append({
                'uuid': entry.get('uuid'),
                'node_id': entry['to_node'],
                'port_id': entry['to_port'],
                'position': entry.get('position') or 0,
                'source': {'kind': 'node', 'node_id': entry['from_node'], 'port_id': entry['from_port']},
                'transformations': entry.get('transformations') or [],
                'tags': entry.get('tags') or [],
            })
            continue
        spec = DatasetPortSpec.model_validate(entry.get('spec') or {})
        upgraded.append({
            'uuid': entry.get('uuid'),
            'node_id': entry['node'],
            'port_id': entry['port_id'],
            'position': entry.get('position') or 0,
            'source': {
                'kind': 'dataset',
                'dataset': entry['dataset'],
                'metric': entry['metric'],
                'dataset_uuid': entry.get('dataset_uuid'),
                'metric_uuid': entry.get('metric_uuid'),
                'dataset_revision': entry.get('dataset_revision'),
            },
            'transformations': [op.model_dump(mode='json') for op in spec.transformations],
            'tags': list(spec.tags),
        })
    data['bindings'] = upgraded


def _upgrade_action_group_references_v10(data: dict[str, Any]) -> None:  # noqa: C901
    """Give legacy action groups deterministic UUIDs and rewrite action references."""
    spec = data.get('spec')
    if not isinstance(spec, dict):
        return
    groups = spec.get('action_groups', [])
    if not groups:
        return

    metadata = data.get('metadata') or {}
    instance_uuid_raw = metadata.get('uuid') or spec.get('uuid')
    if instance_uuid_raw is None:
        raise ValueError('Legacy snapshots with action groups require an instance UUID')
    instance_uuid = UUID(str(instance_uuid_raw))

    group_uuids: dict[str, UUID] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        identifier = group.get('id')
        if identifier is None:
            raise ValueError('Legacy action groups require an identifier')
        group_uuid = (
            UUID(str(group['uuid']))
            if group.get('uuid') is not None
            else uuid3(
                instance_uuid,
                f'action-group:{identifier}',
            )
        )
        group['uuid'] = group_uuid
        group_uuids[str(identifier)] = group_uuid

    for node in data.get('nodes', []):
        if not isinstance(node, dict):
            continue
        node_spec = node.get('spec')
        if not isinstance(node_spec, dict):
            continue
        type_config = node_spec.get('type_config')
        if not isinstance(type_config, dict) or type_config.get('kind') != 'action':
            continue
        group_ref = type_config.get('group')
        if group_ref is None:
            continue
        group_uuid = group_uuids.get(str(group_ref))
        if group_uuid is None:
            # Preserve a dangling legacy reference as a deterministic UUID;
            # runtime validation will continue to report the missing group.
            group_uuid = uuid3(instance_uuid, f'action-group:{group_ref}')
        type_config['group'] = group_uuid


class NodePortSource(BaseModel):
    """An input-binding source: another node's output port."""

    kind: Literal['node'] = 'node'
    node_id: UUID
    port_id: UUID


class DatasetMetricSource(BaseModel):
    """
    An input-binding source: one metric of a dataset.

    Natural references keep portable exports restore-stable; the UUIDs pin
    the structural identity so published semantics survive renames, and the
    revision records what a published snapshot computed from.
    """

    kind: Literal['dataset'] = 'dataset'
    dataset: str
    metric: str
    dataset_uuid: UUID | None = None
    metric_uuid: UUID | None = None
    dataset_revision: int | None = None


type InputBindingSource = NodePortSource | DatasetMetricSource


class InputBindingSnapshot(ModelSnapshot):
    """
    Snapshot form of one ``NodeInputPortBinding`` row.

    The single binding form of ``InstanceSnapshot.bindings`` (v11) and the
    row-level ``snapshot_model`` (change history, revisions). Old payloads
    carrying the retired ``dataset_spec`` / ``dataset_index`` fields stay
    loadable; the keys are ignored (``I18nBaseModel`` ignores extra keys).
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
                metric=metric_column_id(obj.metric),
                dataset_uuid=obj.dataset.uuid,
                metric_uuid=obj.metric.uuid,
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

    @property
    def dataset_source(self) -> DatasetMetricSource | None:
        return self.source if isinstance(self.source, DatasetMetricSource) else None

    @property
    def node_source(self) -> NodePortSource | None:
        return self.source if isinstance(self.source, NodePortSource) else None

    def selects_metric(self) -> bool:
        """Whether this binding picks one metric column out of a wide frame."""
        return any(op.kind == 'select_metric' for op in self.transformations)


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
    snapshots: unified-row pk order, the authored order), then dataset ports
    sorted by ``(node, dataset_index, port, metric)``.
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


def unified_binding_snapshots(
    edges: Sequence[EdgeSnapshot],
    dataset_ports: Sequence[DatasetPortSnapshot],
) -> list[InputBindingSnapshot]:
    """
    Order production-side carriers canonically and convert to the unified form.

    ``EdgeSnapshot`` / ``DatasetPortSnapshot`` survive only as parse/sync-internal
    carriers (they hold the authored ordinal and pre-resolution spec state the
    production pipeline needs); everything persisted or consumed downstream is
    ``InputBindingSnapshot``.
    """
    result: list[InputBindingSnapshot] = []
    for item, position in ordered_binding_snapshots(edges, dataset_ports):
        if isinstance(item, EdgeSnapshot):
            result.append(
                InputBindingSnapshot(
                    uuid=item.uuid,
                    node_id=item.to_node,
                    port_id=item.to_port,
                    position=position,
                    source=NodePortSource(node_id=item.from_node, port_id=item.from_port),
                    transformations=list(item.transformations),
                    tags=list(item.tags),
                )
            )
            continue
        result.append(
            InputBindingSnapshot(
                uuid=item.uuid,
                node_id=item.node,
                port_id=item.port_id,
                position=position,
                source=DatasetMetricSource(
                    dataset=item.dataset,
                    metric=item.metric,
                    dataset_uuid=item.dataset_uuid,
                    metric_uuid=item.metric_uuid,
                    dataset_revision=item.dataset_revision,
                ),
                transformations=list(item.spec.transformations),
                tags=list(item.spec.tags),
            )
        )
    return result


def group_dataset_bindings(
    snapshot: InstanceSnapshot,
) -> dict[UUID, list[tuple[DatasetPortSpec, str, list[InputBindingSnapshot]]]]:
    """Recover per-node dataset binding groups from a snapshot; see ``group_unified_dataset_bindings``."""
    port_specs_by_node = {node.uuid: node.spec for node in snapshot.nodes if node.spec is not None}
    dataset_rows = [(item, position) for item, position in snapshot.bindings_with_positions() if item.dataset_source is not None]
    return group_unified_dataset_bindings(dataset_rows, port_specs_by_node)


def group_unified_dataset_bindings(
    dataset_rows: Sequence[tuple[InputBindingSnapshot, int]],
    port_specs_by_node: Mapping[UUID, NodeSpec | None],
) -> dict[UUID, list[tuple[DatasetPortSpec, str, list[InputBindingSnapshot]]]]:
    """
    Recover per-node dataset binding groups from native fields only.

    A row whose pipeline selects a metric is a single-metric binding and forms
    its own group; the column-less rows of one (node, dataset) are the
    per-metric fan-out of a single whole-frame binding and collapse back into
    one group — with one refinement: one fan-out enumerates each schema metric
    once, so a metric repeating in the open group can only start the next
    binding of the same dataset. Group order per node follows (input-port
    declaration order, per-port position), which is the authored order the
    fan-out was created in; a group's index is the authored ordinal the retired
    ``dataset_index`` column carried.

    Returns per node: (binding-level spec, dataset identifier, rows).
    """
    from collections import defaultdict

    rows_by_node: defaultdict[UUID, list[tuple[InputBindingSnapshot, int]]] = defaultdict(list)
    for item, position in dataset_rows:
        rows_by_node[item.node_id].append((item, position))

    groups_by_node: dict[UUID, list[tuple[DatasetPortSpec, str, list[InputBindingSnapshot]]]] = {}
    for node_uuid, node_rows in rows_by_node.items():
        node_spec = port_specs_by_node.get(node_uuid)
        port_order = {port.id: index for index, port in enumerate(node_spec.input_ports)} if node_spec is not None else {}
        node_rows.sort(key=lambda entry: (port_order.get(entry[0].port_id, len(port_order)), entry[1], str(entry[0].port_id)))
        grouped: list[tuple[str, list[InputBindingSnapshot]]] = []
        open_group: dict[str, tuple[int, set[str]]] = {}
        for row, _position in node_rows:
            row_source = row.dataset_source
            assert row_source is not None
            if row.selects_metric():
                grouped.append((row_source.dataset, [row]))
                continue
            current = open_group.get(row_source.dataset)
            if current is None or row_source.metric in current[1]:
                open_group[row_source.dataset] = (len(grouped), {row_source.metric})
                grouped.append((row_source.dataset, [row]))
            else:
                grouped[current[0]][1].append(row)
                current[1].add(row_source.metric)
        node_groups: list[tuple[DatasetPortSpec, str, list[InputBindingSnapshot]]] = []
        for dataset_id, group_rows in grouped:
            first = group_rows[0]
            first_source = first.dataset_source
            assert first_source is not None
            spec = DatasetPortSpec(
                transformations=list(first.transformations),
                column=first_source.metric if first.selects_metric() else None,
                tags=list(first.tags),
            )
            node_groups.append((spec, dataset_id, group_rows))
        groups_by_node[node_uuid] = node_groups
    return groups_by_node


def ordered_unified_bindings(
    edges: Sequence[InputBindingSnapshot],
    dataset_rows: Sequence[InputBindingSnapshot],
) -> list[tuple[InputBindingSnapshot, int]]:
    """
    Assign per-port positions over already-ordered unified rows.

    Mirrors ``ordered_binding_snapshots`` for post-resolution rows: on a shared
    port, edges come first in their given order, then dataset rows in theirs.
    Callers supply both sequences in canonical order (edges in snapshot order,
    dataset rows as the resolution step emits them).
    """
    from collections import defaultdict

    positions: defaultdict[tuple[UUID, UUID], int] = defaultdict(int)
    result: list[tuple[InputBindingSnapshot, int]] = []
    for row in [*edges, *dataset_rows]:
        key = (row.node_id, row.port_id)
        result.append((row, positions[key]))
        positions[key] += 1
    return result


def match_preserved_uuids(
    existing: Sequence[tuple[tuple[Hashable, ...], UUID]],
    replacements: Sequence[tuple[Hashable, ...]],
) -> list[UUID | None]:
    """
    Match replacement rows to existing rows through successive structural keys.

    The sync paths preserve authored order by deleting and recreating the
    ``NodeInputPortBinding`` rows (pk order is the authored order), but
    the rebuilt rows must keep their durable UUIDs: the row UUID is the
    binding identity, and it must survive a
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


def dataset_port_match_keys(node: UUID, dataset_pk: int, metric_pk: int, port_id: UUID) -> tuple[Hashable, ...]:
    """Structural match keys for one dataset-binding row (loose pass survives port changes)."""
    return ((node, dataset_pk, metric_pk, port_id), (node, dataset_pk, metric_pk))


def existing_edge_identities(ic: InstanceConfig) -> list[tuple[tuple[Hashable, ...], UUID]]:
    """Capture edge match keys and UUIDs, in authored (per-port position) order, before a sync rewrite."""
    from nodes.models import NodeInputPortBinding

    rows = (
        NodeInputPortBinding.objects
        .filter(instance=ic, source_node__isnull=False)
        .order_by('node_id', 'port_id', 'position')
        .values_list('source_node__uuid', 'source_port_id', 'node__uuid', 'port_id', 'uuid')
    )
    result: list[tuple[tuple[Hashable, ...], UUID]] = []
    for from_node, from_port, to_node, to_port, row_uuid in rows:
        assert from_port is not None  # one-source check constraint
        result.append((edge_match_keys(from_node, from_port, to_node, to_port), row_uuid))
    return result


def existing_dataset_port_identities(ic: InstanceConfig) -> list[tuple[tuple[Hashable, ...], UUID]]:
    """Capture dataset-binding match keys and UUIDs, in authored order, before a sync rewrite."""
    from nodes.models import NodeInputPortBinding

    return [
        (dataset_port_match_keys(node_uuid, dataset_pk, metric_pk, port_id), row_uuid)
        for node_uuid, dataset_pk, metric_pk, port_id, row_uuid in (
            NodeInputPortBinding.objects
            .filter(instance=ic, dataset__isnull=False)
            .order_by('node_id', 'port_id', 'position')
            .values_list('node__uuid', 'dataset_id', 'metric_id', 'port_id', 'uuid')
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

    Contains metadata + spec + nodes + input bindings (edge- and
    dataset-sourced, one discriminated list in stored ``position`` order).
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
    bindings: list[InputBindingSnapshot] = Field(default_factory=list)
    dataset_revisions: list[DatasetRevisionPinSnapshot] = Field(default_factory=list)
    dimensions: list[DimensionMeta] = Field(default_factory=list)
    datasets: list[DatasetMeta] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}

    @property
    def edge_bindings(self) -> list[InputBindingSnapshot]:
        return [b for b in self.bindings if isinstance(b.source, NodePortSource)]

    @property
    def dataset_bindings(self) -> list[InputBindingSnapshot]:
        return [b for b in self.bindings if isinstance(b.source, DatasetMetricSource)]

    def bindings_with_positions(self) -> list[tuple[InputBindingSnapshot, int]]:
        """Bindings with per-port positions, assigned at snapshot production."""
        return [(binding, binding.position) for binding in self.bindings]

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
        if schema_version < 9:
            _upgrade_bindings_v9(data)
        if schema_version < 10:
            _upgrade_action_group_references_v10(data)
        if schema_version < 11:
            _upgrade_bindings_v11(data)

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
    metric_id = metric_column_id(metric)
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


def _check_spec_is_not_yaml_minimal(ic: InstanceConfig, nodes: list[NodeSnapshot]) -> None:
    """
    Refuse a spec that ``ensure_spec()`` derived from YAML, which carries no dimensions.

    A yaml-sourced instance stores the *minimal* spec that ``make_minimal_instance_spec()``
    builds: identity, params, scenarios, pages — but no dimension catalogue, because the YAML
    runtime reads dimensions from the config file and never consults the spec. Flipping such an
    instance to ``config_source='database'`` therefore hands the snapshot path a spec with an
    empty dimension list, and the load dies far downstream on the first node that declares one,
    as ``NodeError: Dimension <x> not found``. Say what is actually wrong instead.

    An instance whose nodes are all dimensionless is left alone: an empty catalogue is correct
    there, and this must not become a reason to refuse a model that would load fine.
    """
    if ic.spec is None or ic.spec.dimensions:
        return
    wanted = next(
        (
            dim_id
            for node in nodes
            if node.spec is not None
            for dim_id in (list(node.spec.input_dimensions or []) + list(node.spec.output_dimensions or []))
        ),
        None,
    )
    if wanted is None:
        return
    msg = (
        f'Instance {ic.identifier} has a spec with no dimensions, but its nodes declare some '
        f"(e.g. '{wanted}'). This is the minimal spec derived from the YAML config "
        f"(config_source is '{ic.config_source}'), which does not carry a dimension catalogue. "
        f'Run `sync_instance_to_db {ic.identifier}` to store a full spec before loading from '
        f'the database.'
    )
    raise ValueError(msg)


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
    _check_spec_is_not_yaml_minimal(ic, nodes)

    bindings: list[InputBindingSnapshot] = []
    dataset_ids: set[int] = set()
    for row in binding_qs_for(ic):
        source_node = row.source_node
        source: NodePortSource | DatasetMetricSource
        if source_node is not None:
            assert row.source_port_id is not None
            source = NodePortSource(node_id=source_node.uuid, port_id=row.source_port_id)
        else:
            assert row.dataset is not None
            assert row.metric is not None
            dataset_ids.add(row.dataset.pk)
            if dataset_revision_pins is not None:
                pin = dataset_revision_pins.get(row.dataset.pk)
                dataset_revision = pin.revision_id if pin is not None else None
            else:
                # No explicit pins (drafts): record the dataset's current revision
                # so the snapshot is deterministically reconstructible.
                dataset_revision = getattr(row.dataset, 'latest_revision_id', None)
            source = DatasetMetricSource(
                dataset=row.dataset.identifier or str(row.dataset.uuid),
                metric=metric_column_id(row.metric),
                dataset_uuid=row.dataset.uuid,
                metric_uuid=row.metric.uuid,
                dataset_revision=dataset_revision,
            )
        bindings.append(
            InputBindingSnapshot(
                uuid=row.uuid,
                node_id=row.node.uuid,
                port_id=row.port_id,
                position=row.position,
                source=source,
                transformations=list(row.transformations or []),
                tags=list(row.tags or []),
            )
        )

    dimensions = _dimension_catalog_for(ic)
    datasets = _dataset_catalog_for(
        ic,
        dataset_ids=dataset_ids,
        dataset_revision_pins=dataset_revision_pins,
    )

    return InstanceSnapshot(
        metadata=InstanceMetadata.from_model(ic),
        spec=ic.spec,
        copy_of=str(ic.copy_of.uuid) if ic.copy_of else None,
        nodes=nodes,
        bindings=bindings,
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
            quantity=(metric.spec or {}).get('quantity'),
            order=metric.order,
            validation_rules=tuple(
                validation_rule_adapter.validate_python(rule.rule)
                # Meta.ordering is (metric, order), so .all() hits the
                # prefetch cache already in rule order.
                for rule in metric.validation_rules.all()
            ),
        )
        for metric in schema.metrics.all()
    )
    declared_dimension_ids = tuple(schema_dimension.dimension.uuid for schema_dimension in schema.dimensions.all())
    return DatasetMeta(
        id=dataset.uuid,
        identifier=dataset.identifier,
        schema_id=schema.uuid,
        is_editable=schema.is_editable,
        metrics=metrics,
        declared_dimension_ids=declared_dimension_ids,
        is_external_placeholder=dataset.is_external_placeholder,
        external_ref=dataset.external_ref,
        revision_id=pinned_revision_id if pinned_revision_id is not None else dataset.latest_revision_id,
        category_domain=schema.category_domain,
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
        .prefetch_related('schema__metrics__validation_rules', 'schema__dimensions__dimension')
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


def binding_qs_for(ic: InstanceConfig) -> QuerySet[NodeInputPortBinding]:
    """
    Unified bindings in canonical snapshot order: (node, port, position).

    Per-port ``position`` is the only order the loader and graph observe;
    global list order is normalized rather than inherited from row pks,
    because mirror rows keep their pk across resyncs (first-appearance
    order), which is not the authored order the positions encode.
    """
    from nodes.models import NodeInputPortBinding

    return (
        NodeInputPortBinding.objects
        .filter(instance=ic)
        .select_related('node', 'source_node', 'dataset', 'metric')
        # Snapshot production only reads identity fields off the related
        # nodes; hydrating every binding's NodeConfig.spec would parse the
        # heaviest column in the schema twice per edge for nothing.
        .defer('node__spec', 'source_node__spec')
        .order_by('node_id', 'port_id', 'position')
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
        DatasetMetricValidationRule,
        DatasetSchema as DatasetSchemaModel,
        DatasetSchemaDimension,
        DatasetSchemaScope,
        DimensionScope,
    )

    primary_lang = ic.primary_language

    # Resolve schema name from the TranslatedString snapshot.
    schema_fields: dict[str, Any] = {
        'time_resolution': ds_snapshot.time_resolution,
        'is_editable': ds_snapshot.is_editable,
        'category_domain': ds_snapshot.category_domain,
    }
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
            spec={'quantity': m_snap.quantity} if m_snap.quantity is not None else {},
            order=idx,
            i18n=metric_i18n,
            **metric_fields,
        )
        # Like the metric itself, a restored rule gets a fresh uuid; the
        # snapshot uuid records provenance only.
        for rule_idx, rule_snap in enumerate(m_snap.validation_rules):
            DatasetMetricValidationRule.objects.create(
                metric=metric,
                rule=rule_snap.rule.model_dump(mode='json'),
                order=rule_idx,
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
    from nodes.models import NodeInputPortBinding

    rewired = 0
    ports = NodeInputPortBinding.objects.filter(instance=ic, dataset__identifier__in=datasets_by_id).select_related(
        'dataset', 'metric'
    )
    for port in ports:
        assert port.dataset is not None
        assert port.metric is not None
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
    Recreate the editor graph bindings (``NodeInputPortBinding``) for ``ic``.

    Companion to :func:`import_instance_nodes` for callers that build the DB
    mirror piecemeal (yaml-mode copies) rather than through the full
    :func:`import_instance`. Edges and ports are matched by node UUID and dataset
    identifier, so references that don't resolve in ``ic`` (e.g. a DVC dataset
    not materialised in the DB) are skipped rather than erroring. Does not touch
    ``config_source`` or the instance spec — these rows are dormant for
    ``config_source='yaml'`` (the runtime loads the YAML) but are read by the
    Trailhead editor, so a copy should mirror whatever the source has.
    """
    _import_bindings(ic, export, nodes_by_uuid, datasets_by_id)


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
            is_editable=n.is_editable if n.is_editable is not None else True,
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


def _import_bindings(
    ic: InstanceConfig,
    export: InstanceExport,
    nodes_by_uuid: dict[UUID, NodeConfig],
    datasets_by_id: dict[str, DatasetModel],
) -> None:
    """
    Create the copy's ``NodeInputPortBinding`` rows from the export snapshot.

    References that don't resolve in ``ic`` (a node not copied, a DVC dataset
    not materialised in the DB) are skipped; that may leave position gaps on a
    port, which is harmless — only relative order is semantic. Fresh UUIDs are
    minted: a copy's bindings are new identities.
    """
    from nodes.models import NodeInputPortBinding

    rows: list[NodeInputPortBinding] = []
    for item, position in export.instance.bindings_with_positions():
        source = item.source
        if isinstance(source, NodePortSource):
            from_node = nodes_by_uuid.get(source.node_id)
            to_node = nodes_by_uuid.get(item.node_id)
            if from_node is None or to_node is None:
                continue
            rows.append(
                NodeInputPortBinding(
                    instance=ic,
                    node=to_node,
                    port_id=item.port_id,
                    position=position,
                    source_node=from_node,
                    source_port_id=source.port_id,
                    transformations=list(item.transformations),
                    tags=list(item.tags),
                )
            )
            continue
        node = nodes_by_uuid.get(item.node_id)
        dataset = datasets_by_id.get(source.dataset)
        if node is None or dataset is None:
            continue
        # Resolve metric by name within the dataset's schema
        assert dataset.schema is not None
        metric = dataset.schema.metrics.filter(name=source.metric).first()
        if metric is None:
            continue
        rows.append(
            NodeInputPortBinding(
                instance=ic,
                node=node,
                port_id=item.port_id,
                position=position,
                dataset=dataset,
                metric=metric,
                transformations=list(item.transformations),
                tags=list(item.tags),
            )
        )
    NodeInputPortBinding.objects.bulk_create(rows)


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

    # Input bindings (edges and dataset ports)
    _import_bindings(ic, export, nodes_by_uuid, datasets_by_id)
