"""
Serialize and deserialize DB-sourced instance configurations.

Two related Pydantic models define the serialization layers:

- ``InstanceSnapshot`` — structural state of an instance (spec + nodes +
  edges + dataset ports), plus the UUID catalogs needed to resolve those
  references without loading dataset bodies. This is the unit of revisioning.
- ``InstanceExport`` — ``InstanceSnapshot`` plus the dataset bodies as
  ``DatasetSnapshot`` objects. Used for portable export/import (e.g. when
  cloning a framework template into a new instance).

Dataset snapshot shapes live in ``datasets.snapshot``; ORM transfer lives in
``datasets.transfer``. The instance serializer coordinates both with node and
binding snapshots.
"""

from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, cast
from uuid import UUID, uuid3

from django.apps import apps
from django.contrib.contenttypes.models import ContentType
from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from wagtail.models import Revision

from markdown_it import MarkdownIt

from kausal_common.datasets.models import (
    Dataset,
    Dataset as DatasetModel,
    DatasetMetric,
    DatasetSchemaScope,
    Dimension,
    DimensionCategory as DimensionCategoryModel,
    DimensionScope,
)
from kausal_common.i18n.pydantic import (
    TranslatedString,
    get_modeltrans_attrs_from_str,
)

from paths.identifiers import BindingId, NodeId  # noqa: TC002 - Pydantic field
from paths.refs import (
    DatasetMetricRef,
    DatasetRef,
    FrameworkCategoryRef,
    FrameworkRef,
    InstanceCopyOf,
    MeasureTemplateRef,
    NodeCopyOf,
    NodeRef,
    PortRef,
)
from paths.rekey import rekeyed, survey
from paths.uuid_kinds import Token

from datasets.catalogue import dataset_meta_from_model
from datasets.legacy_snapshot import complete_catalog_entry, upgrade_dataset_snapshot_v1
from datasets.shape_domain import CategoryDomainResolver
from datasets.snapshot import DatasetSnapshot, metric_column_id
from datasets.transfer import import_instance_datasets, save_in_order
from frameworks.catalogue import dimension_scopes
from frameworks.models import Framework, FrameworkConfig, FrameworkDimensionCategory, Measure, MeasureDataPoint, MeasureTemplate
from nodes.defs.data_entry import data_entry_dataset_ids
from nodes.defs.graph import (
    DatasetMeta,
    DimensionCategoryMeta,
    DimensionMeta,
)
from nodes.defs.instance_defs import InstanceMetadata, InstanceModelSpec
from nodes.defs.node_defs import DatasetPortSpec, NodeSpec
from nodes.defs.transform_def import EdgeTransformOp, PortTransformOp
from nodes.goals import NodeGoals
from nodes.legacy_specs import upgrade_formula_specs_v13
from nodes.page_snapshot import PageSnapshot, build_instance_page_snapshots
from nodes.snapshot_base import ModelSnapshot, apply_translated, translated_string_from_model

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

    from django.db.models import QuerySet

    from kausal_common.i18n.pydantic import I18nString

    from paths.rekey import Rekeying
    from paths.uuid_kinds import UuidEntity

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
#   v11: bindings are unified ``InputBindingSnapshot`` entries instead of
#        kind-discriminated edge/dataset-port payloads.
#   v12: ``NodeSnapshot.datasets`` holds the catalog entries of the datasets a
#        node owns; ``InstanceSnapshot.datasets`` keeps the instance's own. Older
#        snapshots have no node-owned datasets, so the default ``[]`` upgrades them.
#   v13: formulas belong to type_config; scenarios carry only authored deviations.
#   v14: revisions retain local authoring inputs, a verified template pin and local selections.
#   v15: a dataset catalog entry is pure structure: its pinned revision lives only in
#        ``dataset_revisions``, it carries the dataset's name, forecast year and time
#        resolution, and each metric validation rule carries its row's uuid.
SNAPSHOT_SCHEMA_VERSION = 17

_MARKDOWN = MarkdownIt('commonmark', {'html': True})


# ---------------------------------------------------------------------------
# Snapshot base + models
# ---------------------------------------------------------------------------

# The ``i18n`` field on these models stores the raw modeltrans JSON dict
# (e.g. {"label_en": "...", "label_da": "..."}).  This allows lossless
# round-tripping of translations through export/import.


class NodeLayoutSnapshot(ModelSnapshot['NodeLayout']):
    x: float
    y: float
    source: Literal['auto', 'user'] = 'auto'

    @classmethod
    def from_model(cls, obj: NodeLayout) -> Self:
        return cls(x=obj.x, y=obj.y, source=cast("Literal['auto', 'user']", obj.source))


class InheritedNodeSettings(BaseModel):
    """Local selections for inherited nodes, retained in authoring snapshots."""

    node_uuid: NodeRef
    goals: NodeGoals | None = None
    layout: NodeLayoutSnapshot | None = None
    parameter_values: dict[str, bool | float | str | None] = Field(default_factory=dict)
    parameter_sources: dict[str, str] = Field(default_factory=dict)


class InputBindingOverrideSnapshot(BaseModel):
    node_uuid: NodeRef
    port_uuid: PortRef
    bindings: list['InputBindingSnapshot'] = Field(default_factory=list)


class NodeSnapshot(ModelSnapshot['NodeConfig']):
    uuid: NodeId
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
    template_revision_id: int | None = None
    indicator_node: NodeRef | None = None
    copy_of: NodeCopyOf | None = None
    body: list[Any] | None = None
    """Raw StreamField data of ``NodeConfig.body``. Admin-authored only, so
    parse-side snapshots never carry it; row-side snapshots preserve it so
    published serving doesn't lose (or leak drafts of) body content."""
    spec: NodeSpec | None = None
    layout: NodeLayoutSnapshot | None = None
    datasets: list[DatasetMeta] = Field(default_factory=list)
    """Catalog entries of the datasets this node owns; they are internal to it and travel with it."""

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
            name=translated_string_from_model(obj, 'name', primary_language),
            short_name=translated_string_from_model(obj, 'short_name', primary_language),
            short_description=translated_string_from_model(obj, 'short_description', primary_language),
            description=translated_string_from_model(obj, 'description', primary_language),
            goal=translated_string_from_model(obj, 'goal', primary_language),
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


class EdgeSnapshot(BaseModel):
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


class DatasetPortSnapshot(BaseModel):
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


_RETIRED_RULE_KINDS = ('required_combinations', 'allowed_combinations')


def upgrade_dataset_catalog_v15(data: dict[str, Any]) -> None:
    """
    Make each dataset catalog entry pure structure, in place.

    The pinned revision moves out (``dataset_revisions`` already holds it), the deprecated
    ``category_domain_spec`` goes, and a validation rule becomes ``{id, rule}``. The rule
    uuids, name, forecast year and time resolution that v15 adds are not in a v14
    snapshot; the migration that rewrites stored revisions fills them from the database.

    Rules of a retired kind are dropped. ``required_combinations`` and
    ``allowed_combinations`` became shapes (datasets migration 0003 refuses to drop a
    stored rule row of either kind), but revisions published before that froze them into
    their catalog, where nothing can evaluate them any more and they only make the
    revision unreadable.
    """
    entries = [*data.get('datasets', []), *(item for node in data.get('nodes', []) for item in node.get('datasets', []))]
    for entry in entries:
        entry.pop('revision_id', None)
        entry.pop('category_domain_spec', None)
        for metric in entry.get('metrics', []):
            rules = [
                rule if isinstance(rule, dict) and 'rule' in rule else {'rule': rule}
                for rule in metric.get('validation_rules', [])
            ]
            metric['validation_rules'] = [rule for rule in rules if rule['rule'].get('kind') not in _RETIRED_RULE_KINDS]


def upgrade_node_references_v16(nodes: Iterable[dict[str, Any]], uuid_by_identifier: Mapping[str, UUID | str]) -> bool:
    """
    Turn an action's parent and hook targets from node identifiers into uuids, in place.

    Returns whether anything changed. An identifier that ``uuid_by_identifier`` does not
    name raises: the target is not in this snapshot, so its uuid cannot be known here.
    """
    changed = False
    for node in nodes:
        type_config = (node.get('spec') or {}).get('type_config') or {}
        if type_config.get('kind') != 'action':
            continue
        refs = [type_config, *(type_config.get('hooks') or [])]
        for holder, key in ((ref, 'parent' if ref is type_config else 'node') for ref in refs):
            value = holder.get(key)
            if not value or _is_uuid(value):
                continue
            if value not in uuid_by_identifier:
                raise ValueError(f'Node {node.get("identifier")}: {key} {value!r} is not a node of this snapshot')
            holder[key] = str(uuid_by_identifier[value])
            changed = True
    return changed


def drop_dead_node_spec_fields_v17(spec: dict[str, Any]) -> bool:
    """
    Remove ``NodeSpec.pipeline`` and ``NodeSpecExtra.other`` from a stored spec, in place.

    Nothing wrote either field with content or read it, and ``NodeSpec`` forbids unknown
    keys. Returns whether anything changed.
    """
    changed = 'pipeline' in spec
    spec.pop('pipeline', None)
    extra = spec.get('extra')
    if isinstance(extra, dict) and 'other' in extra:
        del extra['other']
        changed = True
    return changed


def _is_uuid(value: object) -> bool:
    try:
        UUID(str(value))
    except ValueError:
        return False
    return True


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
    node_id: NodeRef
    port_id: PortRef


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
    dataset_uuid: DatasetRef | None = None
    metric_uuid: DatasetMetricRef | None = None
    dataset_revision: int | None = None


type InputBindingSource = NodePortSource | DatasetMetricSource


class InputBindingSnapshot(ModelSnapshot['NodeInputPortBinding']):
    """
    Snapshot form of one ``NodeInputPortBinding`` row.

    The single binding form of ``InstanceSnapshot.bindings`` (v11) and the
    row-level ``snapshot_model`` (change history, revisions). Old payloads
    carrying the retired ``dataset_spec`` / ``dataset_index`` fields stay
    loadable; the keys are ignored (``I18nBaseModel`` ignores extra keys).
    """

    uuid: BindingId | None = None
    node_id: NodeRef
    port_id: PortRef
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
    dataset_uuid: DatasetRef
    identifier: str | None = None
    revision_id: int
    content_hash: str
    generation: int
    forecast_from: int | None = None


class DefinitionOrigin(BaseModel):
    """The authoring instance and pinned revision supplying a declaration or value."""

    instance_uuid: UUID
    revision_id: int | None = None
    content_hash: str | None = None


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
    copy_of: InstanceCopyOf | None = None
    """The uuid of the InstanceConfig this was copied from."""
    nodes: list[NodeSnapshot] = Field(default_factory=list)
    bindings: list[InputBindingSnapshot] = Field(default_factory=list)
    dataset_revisions: list[DatasetRevisionPinSnapshot] = Field(default_factory=list)
    dimensions: list[DimensionMeta] = Field(default_factory=list)
    datasets: list[DatasetMeta] = Field(default_factory=list)
    template_revision_id: int | None = None
    template_content_hash: str | None = None
    snapshot_kind: Literal['authored', 'composed', 'legacy'] = 'authored'
    node_settings: list[InheritedNodeSettings] = Field(default_factory=list)
    binding_overrides: list[InputBindingOverrideSnapshot] = Field(default_factory=list)
    _template: 'InstanceSnapshot | None' = PrivateAttr(default=None)
    _provenance: dict[str, 'DefinitionOrigin'] = PrivateAttr(default_factory=dict)
    composition_errors: list[str] = Field(default_factory=list)

    model_config = {'arbitrary_types_allowed': True}

    def with_instance_role(self, config: InstanceConfig) -> Self:
        """Fill legacy/YAML role metadata on a copy at an ORM-aware loading boundary."""
        if 'is_template' in self.metadata.model_fields_set:
            return self
        return self.model_copy(
            update={
                'metadata': self.metadata.model_copy(update={'is_template': config.is_template}),
            }
        )

    def all_datasets(self) -> list[DatasetMeta]:
        """Return the instance's own catalog entries followed by those of the datasets its nodes own."""
        return [*self.datasets, *(dataset for node in self.nodes for dataset in node.datasets)]

    @property
    def edge_bindings(self) -> list[InputBindingSnapshot]:
        return [b for b in self.bindings if isinstance(b.source, NodePortSource)]

    @property
    def dataset_bindings(self) -> list[InputBindingSnapshot]:
        return [b for b in self.bindings if isinstance(b.source, DatasetMetricSource)]

    def bindings_with_positions(self) -> list[tuple[InputBindingSnapshot, int]]:
        """Bindings with per-port positions, assigned at snapshot production."""
        return [(binding, binding.position) for binding in self.bindings]

    @property
    def provenance(self) -> dict[str, DefinitionOrigin]:
        return dict(self._provenance)

    def parameter_value_origin(self, identifier: str, scenario_id: str) -> DefinitionOrigin | None:
        """Resolve the origin of a scenario's value, including its municipal-default fallback."""
        return self._provenance.get(
            f'scenarios/{scenario_id}/param_values/{identifier}',
            self._provenance.get(f'parameter_defaults/{identifier}'),
        )

    def rekey_finish(self, original: Self, rekeying: Rekeying) -> Self:
        """Record the instance a copy was copied from."""
        if self.metadata.uuid == original.metadata.uuid:
            return self
        return self.model_copy(update={'copy_of': original.metadata.uuid})

    def resolve(self, template: InstanceSnapshot | None = None) -> Self:
        """Compose frozen authoring inputs without consulting either instance's live draft."""
        if self.snapshot_kind != 'authored' or self.template_revision_id is None:
            return self
        # template_graph imports these snapshot types.
        from nodes.template_graph import resolve_template_snapshot

        return cast('Self', resolve_template_snapshot(self, template=template))

    @classmethod
    def from_serialized_data(cls, data: dict[str, Any], *, compose: bool = True) -> Self:
        """Load persisted snapshot data, upgrading older node metadata and references."""
        schema_version = data.get('schema_version', 1)
        if schema_version >= SNAPSHOT_SCHEMA_VERSION:
            snapshot = cls.model_validate(data)
            if snapshot.snapshot_kind == 'composed':
                snapshot.spec._is_composed = True
            return snapshot.resolve() if compose else snapshot

        data = deepcopy(data)
        _upgrade_snapshot_data(data, schema_version)
        data['schema_version'] = SNAPSHOT_SCHEMA_VERSION
        data['snapshot_kind'] = 'legacy'
        return cls.model_validate(data)


def _node_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.setdefault('nodes', [])


def _drop_dead_node_spec_fields_v17(data: dict[str, Any]) -> None:
    for node in _node_list(data):
        if isinstance(node.get('spec'), dict):
            drop_dead_node_spec_fields_v17(node['spec'])


_SNAPSHOT_UPGRADERS: tuple[tuple[int, Callable[[dict[str, Any]], object]], ...] = (
    (3, lambda data: _upgrade_node_references_v3(data, _node_list(data))),
    (4, lambda data: _upgrade_node_metadata_v4(_node_list(data))),
    (9, _upgrade_bindings_v9),
    (10, _upgrade_action_group_references_v10),
    (11, _upgrade_bindings_v11),
    (13, upgrade_formula_specs_v13),
    (15, upgrade_dataset_catalog_v15),
    (
        16,
        lambda data: upgrade_node_references_v16(
            _node_list(data), {n['identifier']: n['uuid'] for n in _node_list(data) if n.get('identifier')}
        ),
    ),
    (17, _drop_dead_node_spec_fields_v17),
)
"""Upgrader ``N`` brings serialized snapshot data older than version ``N`` up to it, in place."""


def _upgrade_snapshot_data(data: dict[str, Any], schema_version: int) -> None:
    """Upgrade serialized snapshot data from ``schema_version`` to the current one, in place."""
    for version, upgrade in _SNAPSHOT_UPGRADERS:
        if schema_version < version:
            upgrade(data)


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


class MeasureDataPointSnapshot(BaseModel):
    year: int
    value: float | None = None
    default_value: float | None = None
    probable_lower_bound: float | None = None
    probable_upper_bound: float | None = None


class MeasureSnapshot(BaseModel):
    """A municipality's value for one of its framework's measure templates."""

    template: MeasureTemplateRef
    unit: str | None = None
    internal_notes: str = ''
    data_points: list[MeasureDataPointSnapshot] = Field(default_factory=list)


class FrameworkMembershipSnapshot(BaseModel):
    """
    The instance's membership of a framework (`FrameworkConfig`), which the framework itself is not part of.

    An import finds the framework by uuid and refuses without it. The config's access token
    belongs to the database that issued it and is not carried.
    """

    framework: FrameworkRef
    framework_identifier: str
    organization_name: str | None = None
    organization_identifier: str | None = None
    organization_slug: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    categories: list[FrameworkCategoryRef] = Field(default_factory=list)
    measures: list[MeasureSnapshot] = Field(default_factory=list)

    @classmethod
    def from_model(cls, config: FrameworkConfig) -> Self:
        measures = (
            config.measures.select_related('measure_template').prefetch_related('data_points').order_by('measure_template__uuid')
        )
        return cls(
            framework=config.framework.uuid,
            framework_identifier=config.framework.identifier,
            organization_name=config.organization_name,
            organization_identifier=config.organization_identifier,
            organization_slug=config.organization_slug,
            extra=dict(config.extra or {}),
            categories=sorted(category.uuid for category in config.categories.all()),
            measures=[
                MeasureSnapshot(
                    template=measure.measure_template.uuid,
                    unit=str(measure.unit) if measure.unit is not None else None,
                    internal_notes=measure.internal_notes,
                    data_points=[
                        MeasureDataPointSnapshot(
                            year=point.year,
                            value=point.value,
                            default_value=point.default_value,
                            probable_lower_bound=point.probable_lower_bound,
                            probable_upper_bound=point.probable_upper_bound,
                        )
                        for point in sorted(measure.data_points.all(), key=lambda point: point.year)
                    ],
                )
                for measure in measures
            ],
        )


class InstanceExport(BaseModel):
    """
    Self-contained export: snapshot + dataset bodies.

    Used for cloning template instances and any standalone import/export
    flow where dataset data needs to travel with the model structure.
    Each ``DatasetSnapshot`` carries its data points, keyed by uuid.
    """

    # A copy refers to the bundled template rather than copying it, and pages name nodes
    # by identifier.
    __rekey_foreign__: ClassVar[frozenset[str]] = frozenset({'template', 'pages'})

    schema_version: int = SNAPSHOT_SCHEMA_VERSION
    instance: InstanceSnapshot
    template: 'InstanceExport | None' = None
    datasets: list[DatasetSnapshot] = Field(default_factory=list)
    framework: FrameworkMembershipSnapshot | None = None
    """The instance's framework membership, when it has one."""
    # Wagtail page tree, for verification only (not used on import — pages are
    # copied/restored via Wagtail's own machinery). Node references are by identifier.
    pages: list[PageSnapshot] = Field(default_factory=list)

    # Provenance of a standalone export document. All optional with null
    # defaults, so documents written before these existed validate without a
    # schema-version bump.
    exported_at: datetime | None = None
    exported_from: str | None = Field(
        default=None,
        description='Base URL of the backend that produced this document.',
    )
    draft_head_token: Annotated[UUID, Token()] | None = Field(
        default=None,
        description='Optimistic-locking token of the source draft at export time; null if it had no edits.',
    )

    model_config = {'arbitrary_types_allowed': True}

    def rekeyed(self, *, seed: Mapping[UUID, UUID] | None = None) -> tuple[Self, Rekeying]:
        """
        Return this export with new uuids for everything the instance owns, as a copy.

        The bundled template, the framework's dimensions and schemas, and the template's
        entities the instance overrides keep their uuids. ``seed`` fixes new uuids in
        advance, such as the instance's own when its row is created first.
        """
        if self.instance.template_revision_id is not None and self.template is None:
            raise ValueError('Rekeying an instance built on a template needs the template bundled, to tell overrides apart')
        return rekeyed(self, seed=seed)

    def rekey_finish(self, original: Self, rekeying: Rekeying) -> Self:
        """Restamp the pins of the instance's own datasets, whose content hashes cover their uuids."""
        from datasets.materialization import hash_dataset_content

        def content_hash(body: DatasetSnapshot) -> str:
            return hash_dataset_content(body.model_dump(mode='json', exclude_unset=True))

        bodies = {body.meta.id: body for body in self.datasets}
        pins = self.instance.dataset_revisions
        restamped = [
            pin.model_copy(update={'content_hash': content_hash(bodies[pin.dataset_uuid])}) if pin.dataset_uuid in bodies else pin
            for pin in pins
        ]
        if all(new.content_hash == old.content_hash for new, old in zip(restamped, pins, strict=True)):
            return self
        return self.model_copy(update={'instance': self.instance.model_copy(update={'dataset_revisions': restamped})})

    @classmethod
    def from_serialized_data(cls, data: dict[str, Any]) -> Self:
        """
        Load a persisted export document.

        Plain ``model_validate`` would validate the nested snapshot through
        Pydantic alone and skip its schema-version upgraders; route it through
        ``InstanceSnapshot.from_serialized_data`` so documents saved under an
        older snapshot schema still load.

        Dataset bodies written before `DatasetSnapshot` v2 are upgraded against the
        export's own catalog (`datasets.legacy_snapshot`); a deployment still on v1
        exports them. That changes their content, so the hashes covering it are
        recomputed: the template's pins, and the template hash the instance records.
        """
        data = dict(data)
        template_data = data.get('template')
        template = cls.from_serialized_data(template_data) if isinstance(template_data, dict) else None
        data['template'] = template
        instance_data = data.get('instance')
        if not isinstance(instance_data, dict):
            return cls.model_validate(data)
        instance = InstanceSnapshot.from_serialized_data(instance_data, compose=False)
        data['instance'] = instance
        bodies = data.get('datasets') or []
        if any(body.get('schema_version', 1) < 2 for body in bodies):
            data['datasets'] = _upgrade_export_bodies(instance, bodies)
        export = cls.model_validate(data)
        if template is not None and instance.template_content_hash is not None and instance_data.get('schema_version', 1) < 15:
            # template_graph imports this module.
            from nodes.template_graph import snapshot_content_hash

            instance.template_content_hash = snapshot_content_hash(template.instance)
        return export


def _upgrade_export_bodies(instance: InstanceSnapshot, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Upgrade an export's v1 dataset bodies, and the instance snapshot's records of them, in place.

    A deployment still on v1 exports such bodies; see `InstanceExport.from_serialized_data`.
    """
    from datasets.materialization import hash_dataset_content

    catalog = {entry.id: entry for entry in instance.all_datasets()}
    by_identifier = {entry.identifier: entry for entry in catalog.values() if entry.identifier is not None}
    upgraded_bodies: list[dict[str, Any]] = []
    upgraded: dict[UUID, DatasetSnapshot] = {}
    for body in bodies:
        if body.get('schema_version', 1) >= 2:
            upgraded_bodies.append(body)
            continue
        identifier = body.get('identifier')
        base = catalog.get(UUID(body['uuid'])) if body.get('uuid') else by_identifier.get(identifier) if identifier else None
        content = upgrade_dataset_snapshot_v1(body, base=base, dimensions=instance.dimensions)
        upgraded_bodies.append(content)
        snapshot = DatasetSnapshot.model_validate(content)
        upgraded[snapshot.meta.id] = snapshot
    for pin in instance.dataset_revisions:
        if pin.dataset_uuid in upgraded:
            pin.content_hash = hash_dataset_content(upgraded[pin.dataset_uuid].model_dump(mode='json'))
    instance.datasets = [complete_catalog_entry(entry, upgraded.get(entry.id)) for entry in instance.datasets]
    for node in instance.nodes:
        node.datasets = [complete_catalog_entry(entry, upgraded.get(entry.id)) for entry in node.datasets]
    return upgraded_bodies


# ---------------------------------------------------------------------------
# Export / Import helpers
# ---------------------------------------------------------------------------


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
    there, and this must not become a reason to refuse a model that would load fine. So is one
    that inherits from a template: its local spec is sparse by design, and the dimensions arrive
    with the template when the snapshot is composed.
    """
    if ic.spec is None or ic.spec.dimensions or ic.template_revision_id is not None:
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
    *,
    compose: bool = True,
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
    node_configs = list(node_qs)
    nodes = [NodeSnapshot.from_model(nc, primary_language=ic.primary_language) for nc in node_configs]
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
    # A node's inputs go in the order its ports are declared, which a copy keeps; its port
    # uuids are new there, so their order is not.
    declared = {(n.uuid, port.id): idx for n in nodes if n.spec is not None for idx, port in enumerate(n.spec.input_ports)}
    bindings.sort(key=lambda b: (str(b.node_id), declared.get((b.node_id, b.port_id), len(declared)), str(b.port_id), b.position))

    dimensions = _dimension_catalog_for(ic)
    datasets, owned_datasets = _dataset_catalog_for(
        ic,
        dataset_ids=dataset_ids,
        node_uuids={nc.pk: nc.uuid for nc in node_configs},
    )
    nodes = [node.model_copy(update={'datasets': owned_datasets.get(node.uuid, [])}) for node in nodes]

    snapshot = InstanceSnapshot(
        metadata=InstanceMetadata.from_model(ic),
        spec=ic.spec,
        copy_of=ic.copy_of.uuid if ic.copy_of else None,
        nodes=nodes,
        bindings=bindings,
        dataset_revisions=list(dataset_revision_pins.values()) if dataset_revision_pins is not None else [],
        dimensions=dimensions,
        datasets=datasets,
    )

    if ic.template_revision_id is not None:
        from nodes.template_graph import compose_template_snapshot

        authored = compose_template_snapshot(ic, snapshot, dataset_revision_pins=dataset_revision_pins, compose=False)
        return authored.resolve() if compose else authored
    return snapshot


def _dimension_catalog_for(ic: InstanceConfig) -> list[DimensionMeta]:
    scopes = (
        dimension_scopes(ic)
        .select_related('dimension', 'scope_content_type')
        .prefetch_related('dimension__categories')
        .order_by('order')
    )
    framework_scoped = {scope.dimension_id for scope in scopes if scope.scope_content_type.model == 'framework'}
    dimensions: list[DimensionMeta] = []
    for scope in scopes:
        dimension = scope.dimension
        if scope.identifier is None:
            raise ValueError(f'Dimension {dimension.uuid} has no identifier in instance {ic.identifier}')
        categories = tuple(
            DimensionCategoryMeta(
                id=category.uuid,
                identifier=category.identifier,
                label=translated_string_from_model(category, 'label', ic.primary_language),
                short_label=translated_string_from_model(category, 'short_label', ic.primary_language),
                order=category.order,
                spec=dict(category.spec or {}),
            )
            for category in dimension.categories.all()
        )
        dimensions.append(
            DimensionMeta(
                id=dimension.uuid,
                identifier=scope.identifier,
                label=translated_string_from_model(dimension, 'name', ic.primary_language),
                order=scope.order,
                spec=dict(dimension.spec or {}),
                categories=categories,
                scope='framework' if dimension.pk in framework_scoped else 'instance',
            )
        )
    return dimensions


def _dataset_catalog_for(
    ic: InstanceConfig,
    *,
    dataset_ids: set[int],
    node_uuids: dict[int, UUID],
) -> tuple[list[DatasetMeta], dict[UUID, list[DatasetMeta]]]:
    """
    Catalog the bound datasets and every dataset the nodes own.

    Returns the instance-level entries and, per node uuid, the node-owned ones. A node's
    datasets are listed whether bound or not: they belong to the node.
    """
    from nodes.models import NodeConfig

    node_ct = ContentType.objects.get_for_model(NodeConfig)
    datasets = (
        DatasetModel.objects
        .filter(
            Q(pk__in=dataset_ids)
            | Q(scope_content_type=node_ct, scope_id__in=node_uuids.keys())
            | (
                Q(uuid__in=data_entry_dataset_ids(ic.spec.data_entry if ic.spec else None))
                & Q(pk__in=DatasetModel.objects.get_queryset().for_instance_config(ic).values('pk'))
            )
        )
        .select_related('schema')
        .prefetch_related('schema__metrics__validation_rules', 'schema__dimensions__dimension', 'schema__scopes')
        # Not by pk, which differs between databases: an export and its import list alike.
        .order_by('identifier', 'uuid')
    )
    instance_level: list[DatasetMeta] = []
    owned: dict[UUID, list[DatasetMeta]] = {}
    domains = CategoryDomainResolver()
    for dataset in datasets:
        meta = dataset_meta_from_model(dataset, primary_language=ic.primary_language, domains=domains)
        owner = node_uuids.get(dataset.scope_id) if dataset.scope_content_type_id == node_ct.pk else None
        if owner is not None:
            owned.setdefault(owner, []).append(meta)
        else:
            instance_level.append(meta)
    return instance_level, owned


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
        # By uuid, not pk: the same in every database the model is imported into.
        .order_by('node__uuid', 'port_id', 'position')
    )


def _dataset_export_key(ds: DatasetModel) -> str:
    return ds.identifier or str(ds.uuid)


def _dataset_export_rank(ds: DatasetModel, ic_ct_id: int, ic_id: int) -> tuple[bool, bool, int]:
    is_direct = ds.scope_content_type_id == ic_ct_id and ds.scope_id == ic_id
    return (not is_direct, ds.is_external_placeholder, ds.pk)


def _datasets_for_instance_export(ic: InstanceConfig, ic_ct: ContentType) -> list[DatasetModel]:
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


def export_instance(ic: InstanceConfig, *, exported_from: str | None = None) -> InstanceExport:
    """
    Serialize a DB-sourced InstanceConfig with dataset bodies included.

    ``exported_from`` is recorded as provenance when the document is meant to
    leave this database (the GraphQL ``instance.export`` field passes the
    backend's base URL); in-process uses such as ``copy_instance`` leave it out.
    """
    snapshot = build_instance_snapshot(ic, compose=False)

    ic_ct = ContentType.objects.get_for_model(ic)
    source_datasets = {item.uuid: item for item in _datasets_for_instance_export(ic, ic_ct)}
    source_datasets.update({
        item.uuid: item
        for item in DatasetModel.objects.filter(
            uuid__in=[dataset.id for node in snapshot.nodes for dataset in node.datasets],
        )
    })
    datasets = [DatasetSnapshot.from_model(ds, ic) for ds in source_datasets.values()]

    template_export = None
    if snapshot.template_revision_id is not None:
        from nodes.template_graph import template_snapshot

        base = template_snapshot(ic)
        pinned_bodies = Revision.objects.in_bulk([pin.revision_id for pin in base.dataset_revisions])
        template_export = InstanceExport(
            instance=base,
            datasets=_template_dataset_bodies(base, pinned_bodies),
            exported_at=timezone.now(),
            exported_from=exported_from,
        )
    return InstanceExport(
        instance=snapshot,
        template=template_export,
        datasets=datasets,
        framework=FrameworkMembershipSnapshot.from_model(ic.framework_config) if ic.has_framework_config() else None,
        pages=build_instance_page_snapshots(ic),
        exported_at=timezone.now(),
        exported_from=exported_from,
        draft_head_token=ic.draft_head_token,
    )


# ---------------------------------------------------------------------------
# Import (from_dict)
# ---------------------------------------------------------------------------


def _template_dataset_bodies(base: InstanceSnapshot, revisions: dict[int, Revision]) -> list[DatasetSnapshot]:
    """Return the template's pinned dataset revisions, and an empty body for each of its other datasets."""
    bodies = {
        pin.dataset_uuid: DatasetSnapshot.model_validate(revisions[pin.revision_id].content) for pin in base.dataset_revisions
    }
    for dataset in base.all_datasets():
        bodies.setdefault(dataset.id, DatasetSnapshot(meta=dataset))
    return list(bodies.values())


def _import_dimensions(ic: InstanceConfig, export: InstanceExport) -> None:
    """
    Create the instance's own dimensions and categories from the export's catalog, with their uuids.

    A framework's dimension is not created but checked: it must be here, under the same uuid,
    with every category the export names under the same uuid and identifier. Labels and order
    may differ, since a framework's presentation may have moved on.
    """
    ic_ct = ContentType.objects.get_for_model(ic)
    language = ic.primary_language
    for dimension_meta in export.instance.dimensions:
        if dimension_meta.scope == 'framework':
            _check_framework_dimension(dimension_meta)
            continue
        fields: dict[str, Any] = {}
        i18n: dict[str, str] = {}
        _apply_i18n_value(fields, i18n, dimension_meta.label, 'name', language)
        dimension = Dimension.objects.create(
            uuid=dimension_meta.id,
            name=fields.get('name') or dimension_meta.identifier,
            spec=dict(dimension_meta.spec),
            i18n=i18n,
        )
        save_in_order(
            DimensionScope(dimension=dimension, identifier=dimension_meta.identifier, scope_content_type=ic_ct, scope_id=ic.pk),
            dimension_meta.order or 0,
        )
        for idx, category_meta in enumerate(dimension_meta.categories):
            fields = {}
            i18n = {}
            _apply_i18n_value(fields, i18n, category_meta.label, 'label', language)
            _apply_i18n_value(fields, i18n, category_meta.short_label, 'short_label', language)
            category = DimensionCategoryModel(
                uuid=category_meta.id,
                dimension=dimension,
                identifier=category_meta.identifier,
                label=fields.get('label') or category_meta.identifier or '',
                short_label=fields.get('short_label'),
                spec=dict(category_meta.spec),
                i18n=i18n,
            )
            save_in_order(category, category_meta.order if category_meta.order is not None else idx)


def _check_framework_dimension(dimension_meta: DimensionMeta) -> None:
    dimension = Dimension.objects.filter(uuid=dimension_meta.id).prefetch_related('categories').first()
    if dimension is None:
        raise ValueError(f'Framework dimension {dimension_meta.identifier} ({dimension_meta.id}) is not here')
    here = {category.uuid: category.identifier for category in dimension.categories.all()}
    differing = [
        category.identifier or str(category.id)
        for category in dimension_meta.categories
        if here.get(category.id, ...) != category.identifier
    ]
    if differing:
        raise ValueError(f'Framework dimension {dimension_meta.identifier} differs here in categories {differing}')


def _apply_i18n_value(fields: dict[str, Any], i18n: dict[str, str], value: I18nString | None, name: str, language: str) -> None:
    """Set a modeltrans field from a snapshot value, which may be a translated string or a plain one."""
    if value is None or isinstance(value, TranslatedString):
        apply_translated(fields, i18n, value, name, language)
    else:
        fields[name] = str(value)


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
) -> None:
    """
    Recreate the editor graph bindings (``NodeInputPortBinding``) for ``ic``.

    Companion to :func:`import_instance_nodes` for callers that build the DB
    mirror piecemeal (yaml-mode copies) rather than through the full
    :func:`import_instance`. A binding whose node or dataset is not here (a DVC
    dataset not materialised in the DB) is skipped rather than erroring. Does not
    touch ``config_source`` or the instance spec — these rows are dormant for
    ``config_source='yaml'`` (the runtime loads the YAML) but are read by the
    Trailhead editor, so a copy should mirror whatever the source has.
    """
    _import_bindings(ic, export, nodes_by_uuid)


def _import_nodes(ic: InstanceConfig, export: InstanceExport) -> dict[UUID, NodeConfig]:
    """Create the NodeConfig rows under their uuids. Returns UUID → NodeConfig map."""
    from nodes.models import NodeConfig, NodeLayout, NodeLayoutSource

    primary_lang = ic.primary_language
    nodes_by_uuid: dict[UUID, NodeConfig] = {}
    for n in export.instance.nodes:
        if n.identifier is None:
            raise ValueError(f'Node {n.uuid} has no identifier; the legacy runtime still requires one')
        fields: dict[str, Any] = {}
        i18n_dict: dict[str, str] = {}
        apply_translated(fields, i18n_dict, n.name, 'name', primary_lang)
        apply_translated(fields, i18n_dict, n.short_name, 'short_name', primary_lang)
        apply_translated(fields, i18n_dict, n.short_description, 'short_description', primary_lang)
        apply_translated(fields, i18n_dict, n.description, 'description', primary_lang)
        apply_translated(fields, i18n_dict, n.goal, 'goal', primary_lang)

        nc = NodeConfig.objects.create(
            uuid=n.uuid,
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


def _import_bindings(ic: InstanceConfig, export: InstanceExport, nodes_by_uuid: dict[UUID, NodeConfig]) -> None:
    """
    Create the instance's ``NodeInputPortBinding`` rows from the export snapshot, under their uuids.

    A binding whose node or dataset is not here (a node not copied, a DVC dataset not
    materialised in the DB) is skipped; that may leave position gaps on a port, which is
    harmless, since only relative order is semantic.
    """
    from nodes.models import NodeInputPortBinding

    items = list(export.instance.bindings_with_positions())
    dataset_ids = {item.source.dataset_uuid for item, _ in items if isinstance(item.source, DatasetMetricSource)}
    metric_ids = {item.source.metric_uuid for item, _ in items if isinstance(item.source, DatasetMetricSource)}
    datasets = Dataset.objects.in_bulk(dataset_ids - {None}, field_name='uuid')
    metrics = DatasetMetric.objects.in_bulk(metric_ids - {None}, field_name='uuid')
    rows: list[NodeInputPortBinding] = []
    for item, position in items:
        node = nodes_by_uuid.get(item.node_id)
        if node is None:
            continue
        common: dict[str, Any] = {
            **({'uuid': item.uuid} if item.uuid is not None else {}),
            'instance': ic,
            'node': node,
            'port_id': item.port_id,
            'position': position,
            'transformations': list(item.transformations),
            'tags': list(item.tags),
        }
        source = item.source
        if isinstance(source, NodePortSource):
            from_node = nodes_by_uuid.get(source.node_id)
            if from_node is not None:
                rows.append(NodeInputPortBinding(**common, source_node=from_node, source_port_id=source.port_id))
            continue
        dataset = datasets.get(source.dataset_uuid) if source.dataset_uuid is not None else None
        metric = metrics.get(source.metric_uuid) if source.metric_uuid is not None else None
        if dataset is not None and metric is not None:
            rows.append(NodeInputPortBinding(**common, dataset=dataset, metric=metric))
    NodeInputPortBinding.objects.bulk_create(rows)


def _import_template_revision(export: InstanceExport, *, organization_id: int) -> int | None:
    from nodes.models import InstanceConfig
    from nodes.template_graph import snapshot_content_hash

    snapshot = export.instance
    if snapshot.snapshot_kind != 'authored' or snapshot.template_revision_id is None:
        return None
    if export.template is None:
        revision = Revision.objects.get(pk=snapshot.template_revision_id)
        base = InstanceSnapshot.from_serialized_data(revision.content['model_snapshot']['structured'], compose=False)
        if snapshot_content_hash(base) != snapshot.template_content_hash:
            raise ValueError('Template export is required to import this pinned revision')
        return revision.pk
    base = export.template.instance
    if snapshot_content_hash(base) != snapshot.template_content_hash:
        raise ValueError('Bundled template does not match the pinned content hash')
    ct = ContentType.objects.get_for_model(InstanceConfig)
    template = InstanceConfig.objects.filter(uuid=base.metadata.uuid).first()
    installed_template = template is None
    if template is not None:
        for revision in Revision.objects.filter(content_type=ct, object_id=str(template.pk)):
            payload = (revision.content.get('model_snapshot') or {}).get('structured')
            if (
                payload
                and snapshot_content_hash(InstanceSnapshot.from_serialized_data(payload, compose=False))
                == snapshot.template_content_hash
            ):
                return revision.pk
    else:
        template = InstanceConfig.objects.create(
            uuid=base.metadata.uuid,
            organization_id=organization_id,
            identifier=base.metadata.identifier,
            name=str(base.metadata.name),
            primary_language=base.metadata.primary_language,
            other_languages=base.metadata.other_languages,
            config_source='database',
            spec=base.spec,
        )
        import_instance(template, export.template)
    base = _import_template_dataset_pins(template, export.template)
    snapshot.template_content_hash = snapshot_content_hash(base)
    revision = Revision.objects.create(
        content_type=ct,
        base_content_type=ct,
        object_id=str(template.pk),
        object_str=template.identifier,
        content={
            'pk': template.pk,
            'identifier': template.identifier,
            'config_source': 'database',
            'model_snapshot': {'schema_version': SNAPSHOT_SCHEMA_VERSION, 'structured': base.model_dump(mode='json')},
        },
    )

    from nodes.models import InstanceRevisionDatasetPin

    datasets = {
        dataset.uuid: dataset for dataset in Dataset.objects.filter(uuid__in=[pin.dataset_uuid for pin in base.dataset_revisions])
    }
    InstanceRevisionDatasetPin.objects.bulk_create([
        InstanceRevisionDatasetPin(
            instance_config=template,
            instance_revision=revision,
            dataset=datasets[pin.dataset_uuid],
            dataset_revision_id=pin.revision_id,
            dataset_uuid=pin.dataset_uuid,
            identifier=pin.identifier,
            forecast_from=pin.forecast_from,
            shape_profiles=None,
        )
        for pin in base.dataset_revisions
    ])
    if installed_template:
        type(template).objects.filter(pk=template.pk).update(live_revision=revision, latest_revision=revision, live=True)
    return revision.pk


def _import_template_dataset_pins(template: InstanceConfig, export: InstanceExport) -> InstanceSnapshot:
    from datasets.materialization import hash_dataset_content

    base = export.instance.model_copy(deep=True)
    pins = []
    for pin in base.dataset_revisions:
        dataset = Dataset.objects.get(uuid=pin.dataset_uuid)
        body = next(item for item in export.datasets if item.meta.id == pin.dataset_uuid)
        if hash_dataset_content(body.model_dump(mode='json', exclude_unset=True)) != pin.content_hash:
            raise ValueError('Bundled template dataset does not match its pinned content hash')
        revision = Revision.objects.create(
            content_type=ContentType.objects.get_for_model(Dataset),
            base_content_type=ContentType.objects.get_for_model(Dataset),
            object_id=str(dataset.pk),
            object_str=pin.identifier or str(pin.dataset_uuid),
            content=body.model_dump(mode='json', exclude_unset=True),
        )
        pins.append(pin.model_copy(update={'revision_id': revision.pk}))
    base = base.model_copy(update={'dataset_revisions': pins})
    # Revision IDs are database-local. The pins are retargeted above; retarget the input payload references too.
    ids = {pin.dataset_uuid: pin.revision_id for pin in pins}
    base.bindings = [
        item.model_copy(update={'source': item.source.model_copy(update={'dataset_revision': ids[item.source.dataset_uuid]})})
        if isinstance(item.source, DatasetMetricSource) and item.source.dataset_uuid in ids
        else item
        for item in base.bindings
    ]
    return base


@transaction.atomic
def import_instance(
    ic: InstanceConfig,
    export: InstanceExport,
    framework_config: FrameworkConfig | None = None,
) -> None:
    """
    Populate an InstanceConfig with the model of an InstanceExport, keeping every uuid.

    ``ic`` must already exist under the export's instance uuid. A copy within this database
    rekeys the export first (`InstanceExport.rekeyed`), seeded with the new row's uuid. The
    import refuses an export whose own entities are already here, or that refers to
    something that is not (`check_import`). ``framework_config``, when given, is the
    membership the caller created; otherwise the export's own is recreated.
    """
    meta = export.instance.metadata
    if ic.uuid != meta.uuid:
        raise ValueError(f'Instance {ic.identifier} is {ic.uuid}, the export is {meta.uuid}; rekey the export to copy it')
    if export.instance.snapshot_kind == 'composed':
        raise ValueError('Import requires an authoring snapshot, not a composed runtime model')
    check_import(export, instance=ic)
    template_revision_id = _import_template_revision(export, organization_id=ic.organization_id)

    _import_instance_metadata(ic, export, template_revision_id, framework_config)

    # Resolve copy_of by uuid (restore fidelity; absent source → stays null).
    if export.instance.copy_of:
        from nodes.models import InstanceConfig as _InstanceConfig

        src_ic = _InstanceConfig.objects.filter(uuid=export.instance.copy_of).first()
        if src_ic is not None:
            ic.copy_of = src_ic
            ic.save(update_fields=['copy_of'])

    # Membership first: it is what makes the framework's dimensions and schemas visible here.
    if framework_config is None and export.framework is not None:
        _import_framework_membership(ic, export.framework)
    _import_dimensions(ic, export)
    import_instance_datasets(
        ic, export.datasets, dimensions={dimension.id: dimension for dimension in export.instance.dimensions}
    )
    nodes_by_uuid = _import_nodes(ic, export)
    _import_dataset_ownership(export.instance, nodes_by_uuid)
    _import_bindings(ic, export, nodes_by_uuid)
    if template_revision_id is not None:
        _import_binding_overrides(ic, export.instance)


def _import_instance_metadata(
    ic: InstanceConfig,
    export: InstanceExport,
    template_revision_id: int | None,
    framework_config: FrameworkConfig | None,
) -> None:
    meta = export.instance.metadata
    # Store the computation spec. Copy the template's language metadata onto
    # the InstanceConfig row so i18n-bearing data (ActionGroup names, etc.)
    # stays loadable — the spec's TranslatedStrings are authored under the
    # template's primary_language and would be filtered out if the
    # InstanceConfig used a different language.
    ic.spec = export.instance.spec.model_copy()
    ic.primary_language = meta.primary_language
    ic.other_languages = list(meta.other_languages)
    ic.config_source = 'database'
    ic.template_revision_id = template_revision_id
    ic.node_settings = export.instance.node_settings
    update_fields = ['spec', 'primary_language', 'other_languages', 'config_source', 'template_revision', 'node_settings']

    # Owner display name comes from the template (or the framework org) and is
    # written to the column; the instance keeps its own name.
    owner_src = meta.owner
    if framework_config is not None:
        owner_src = str(framework_config.organization_name)
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


def import_instance_copy(ic: InstanceConfig, export: InstanceExport, framework_config: FrameworkConfig | None = None) -> None:
    """
    Populate ``ic`` with a copy of the exported instance, under new uuids and ``ic``'s own.

    The copy refers to the same template, framework and other shared entities as the source,
    and records the source as what it was copied from.
    """
    copy, _ = export.rekeyed(seed={export.instance.metadata.uuid: ic.uuid})
    import_instance(ic, copy, framework_config)


_ROW_MODELS: dict[UuidEntity, tuple[str, str]] = {
    'instance': ('nodes', 'InstanceConfig'),
    'node': ('nodes', 'NodeConfig'),
    'binding': ('nodes', 'NodeInputPortBinding'),
    'dimension': ('datasets', 'Dimension'),
    'category': ('datasets', 'DimensionCategory'),
    'dataset': ('datasets', 'Dataset'),
    'dataset_schema': ('datasets', 'DatasetSchema'),
    'metric': ('datasets', 'DatasetMetric'),
    'validation_rule': ('datasets', 'DatasetMetricValidationRule'),
    'data_point': ('datasets', 'DataPoint'),
    'comment': ('datasets', 'DataPointComment'),
    'data_source': ('datasets', 'DataSource'),
    'source_reference': ('datasets', 'DatasetSourceReference'),
    'framework': ('frameworks', 'Framework'),
    'framework_category': ('frameworks', 'FrameworkDimensionCategory'),
    'measure_template': ('frameworks', 'MeasureTemplate'),
}
"""The entities that are database rows, which an import can check; the rest live inside specs."""


def check_import(export: InstanceExport, *, instance: InstanceConfig | None = None) -> None:
    """
    Refuse an export this database cannot take as it is, before anything is written.

    Every entity the export owns must be new here, and every entity it refers to without
    bundling it (a framework's dimensions and schemas, a template revision it does not
    carry) must already be here. ``instance`` is the row created for the import, which
    holds the export's instance uuid already.
    """
    rekeying = survey(export)
    problems: list[str] = []
    for entity, ids in rekeying.owned.items():
        if entity not in _ROW_MODELS:
            continue
        model = apps.get_model(*_ROW_MODELS[entity])
        taken = model._default_manager.filter(uuid__in=ids)
        if entity == 'instance' and instance is not None:
            taken = taken.exclude(pk=instance.pk)
        if found := sorted(str(uuid) for uuid in taken.values_list('uuid', flat=True)):
            problems.append(f'{len(found)} {entity} already here, e.g. {found[0]}')
    for entity, ids in rekeying.outside().items():
        if entity not in _ROW_MODELS:
            continue
        model = apps.get_model(*_ROW_MODELS[entity])
        missing = ids - set(model._default_manager.filter(uuid__in=ids).values_list('uuid', flat=True))
        if missing:
            problems.append(f'{len(missing)} {entity} referred to but not here, e.g. {min(map(str, missing))}')
    if problems:
        raise ValueError('Cannot import this export here: ' + '; '.join(problems))


def _import_framework_membership(ic: InstanceConfig, membership: FrameworkMembershipSnapshot) -> None:
    """Make ``ic`` a member of the export's framework again, with its measures."""
    framework = Framework.objects.filter(uuid=membership.framework).first()
    if framework is None:
        raise ValueError(f'Framework {membership.framework_identifier} ({membership.framework}) is not here')
    config = FrameworkConfig.objects.create(
        uuid=ic.uuid,
        framework=framework,
        instance_config=ic,
        organization_name=membership.organization_name,
        organization_identifier=membership.organization_identifier,
        organization_slug=membership.organization_slug,
        extra=dict(membership.extra),
    )
    config.categories.set(FrameworkDimensionCategory.objects.filter(uuid__in=membership.categories))
    templates = MeasureTemplate.objects.in_bulk([measure.template for measure in membership.measures], field_name='uuid')
    for measure_snapshot in membership.measures:
        measure = Measure.objects.create(
            framework_config=config,
            measure_template=templates[measure_snapshot.template],
            unit=measure_snapshot.unit,
            internal_notes=measure_snapshot.internal_notes,
        )
        MeasureDataPoint.objects.bulk_create([
            MeasureDataPoint(measure=measure, **point.model_dump()) for point in measure_snapshot.data_points
        ])


def _import_dataset_ownership(snapshot: InstanceSnapshot, nodes_by_uuid: dict[UUID, NodeConfig]) -> None:
    """Scope each node's own datasets to the node, as the export records it."""
    from nodes.models import NodeConfig

    node_ct = ContentType.objects.get_for_model(NodeConfig)
    for node in snapshot.nodes:
        if node.datasets:
            Dataset.objects.filter(uuid__in=[owned.id for owned in node.datasets]).update(
                scope_content_type=node_ct, scope_id=nodes_by_uuid[node.uuid].pk
            )


def _import_binding_overrides(ic: InstanceConfig, snapshot: InstanceSnapshot) -> None:
    """Record the instance's selections for the template's input ports, against this database's dataset revisions."""
    from nodes.models import InputPortBindingSet, NodeInputPortBinding
    from nodes.template_graph import template_snapshot

    own_nodes = {node.uuid for node in snapshot.nodes}
    selections = list(snapshot.binding_overrides)
    inherited_ports = {
        (binding.node_id, binding.port_id)
        for binding in snapshot.bindings
        if isinstance(binding.source, NodePortSource) and binding.source.node_id not in own_nodes
    }
    selections.extend(
        InputBindingOverrideSnapshot(
            node_uuid=node_id,
            port_uuid=port_id,
            bindings=[item for item in snapshot.bindings if (item.node_id, item.port_id) == (node_id, port_id)],
        )
        for node_id, port_id in inherited_ports
    )
    # Revision ids are database-local: point pinned template datasets at this database's revisions.
    pins = {pin.dataset_uuid: pin.revision_id for pin in template_snapshot(ic).dataset_revisions}
    for override in selections:
        bindings = []
        for item in override.bindings:
            source = item.source
            if isinstance(source, DatasetMetricSource):
                source = source.model_copy(
                    update={'dataset_revision': pins.get(source.dataset_uuid)} if source.dataset_uuid else {}
                )
            bindings.append(item.model_copy(update={'source': source}))
        NodeInputPortBinding.objects.filter(instance=ic, node__uuid=override.node_uuid, port_id=override.port_uuid).delete()
        InputPortBindingSet.objects.create(
            instance=ic, node_uuid=override.node_uuid, port_uuid=override.port_uuid, bindings=bindings
        )
