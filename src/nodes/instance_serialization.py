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

from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Self, cast
from uuid import UUID, uuid3, uuid4

from django.db import transaction
from django.db.models import Q
from django.utils import timezone
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from markdown_it import MarkdownIt

from kausal_common.i18n.pydantic import (
    TranslatedString,
    get_modeltrans_attrs_from_str,
)

from datasets.catalogue import dataset_meta_from_model
from datasets.snapshot import DatasetMetricSnapshot, DatasetSnapshot, metric_column_id
from datasets.transfer import import_instance_datasets
from nodes.defs.data_entry import DataEntrySpec, data_entry_dataset_ids, remap_data_entry
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
from nodes.page_snapshot import PageSnapshot
from nodes.snapshot_base import ModelSnapshot, apply_translated, translated_string_from_model

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from django.contrib.contenttypes.models import ContentType
    from django.db.models import QuerySet
    from wagtail.models import Revision

    from kausal_common.datasets.models import (
        Dataset as DatasetModel,
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
#   v11: bindings are unified ``InputBindingSnapshot`` entries instead of
#        kind-discriminated edge/dataset-port payloads.
#   v12: ``NodeSnapshot.datasets`` holds the catalog entries of the datasets a
#        node owns; ``InstanceSnapshot.datasets`` keeps the instance's own. Older
#        snapshots have no node-owned datasets, so the default ``[]`` upgrades them.
#   v13: formulas belong to type_config; scenarios carry only authored deviations.
#   v14: revisions retain local authoring inputs, a verified template pin and local selections.
SNAPSHOT_SCHEMA_VERSION = 14

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

    node_uuid: UUID
    goals: NodeGoals | None = None
    layout: NodeLayoutSnapshot | None = None
    parameter_values: dict[str, bool | float | str | None] = Field(default_factory=dict)
    parameter_sources: dict[str, str] = Field(default_factory=dict)


class InputBindingOverrideSnapshot(BaseModel):
    node_uuid: UUID
    port_uuid: UUID
    bindings: list['InputBindingSnapshot'] = Field(default_factory=list)


class NodeSnapshot(ModelSnapshot['NodeConfig']):
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
    template_revision_id: int | None = None
    indicator_node: UUID | None = None
    copy_of: UUID | None = None
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


class InputBindingSnapshot(ModelSnapshot['NodeInputPortBinding']):
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
    copy_of: str | None = None  # uuid of the InstanceConfig this was copied from
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

        if schema_version < 13:
            upgrade_formula_specs_v13(data)

        data['schema_version'] = SNAPSHOT_SCHEMA_VERSION
        data['snapshot_kind'] = 'legacy'
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
    template: 'InstanceExport | None' = None
    datasets: list[DatasetSnapshot] = Field(default_factory=list)
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
    draft_head_token: UUID | None = Field(
        default=None,
        description='Optimistic-locking token of the source draft at export time; null if it had no edits.',
    )

    model_config = {'arbitrary_types_allowed': True}

    @classmethod
    def from_serialized_data(cls, data: dict[str, Any]) -> Self:
        """
        Load a persisted export document.

        Plain ``model_validate`` would validate the nested snapshot through
        Pydantic alone and skip its schema-version upgraders; route it through
        ``InstanceSnapshot.from_serialized_data`` so documents saved under an
        older snapshot schema still load.
        """
        data = dict(data)
        instance_data = data.get('instance')
        if isinstance(instance_data, dict):
            data['instance'] = InstanceSnapshot.from_serialized_data(instance_data, compose=False)
        return cls.model_validate(data)


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

    dimensions = _dimension_catalog_for(ic)
    datasets, owned_datasets = _dataset_catalog_for(
        ic,
        dataset_ids=dataset_ids,
        dataset_revision_pins=dataset_revision_pins,
        node_uuids={nc.pk: nc.uuid for nc in node_configs},
    )
    nodes = [node.model_copy(update={'datasets': owned_datasets.get(node.uuid, [])}) for node in nodes]

    snapshot = InstanceSnapshot(
        metadata=InstanceMetadata.from_model(ic),
        spec=ic.spec,
        copy_of=str(ic.copy_of.uuid) if ic.copy_of else None,
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
    from frameworks.catalogue import dimension_scopes

    scopes = dimension_scopes(ic).select_related('dimension').prefetch_related('dimension__categories').order_by('order')
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
            )
        )
    return dimensions


def _dataset_catalog_for(
    ic: InstanceConfig,
    *,
    dataset_ids: set[int],
    dataset_revision_pins: dict[int, DatasetRevisionPinSnapshot] | None,
    node_uuids: dict[int, UUID],
) -> tuple[list[DatasetMeta], dict[UUID, list[DatasetMeta]]]:
    """
    Catalog the bound datasets and every dataset the nodes own.

    Returns the instance-level entries and, per node uuid, the node-owned ones. A node's
    datasets are listed whether bound or not: they belong to the node.
    """
    from django.contrib.contenttypes.models import ContentType

    from kausal_common.datasets.models import Dataset as DatasetModel

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
        .prefetch_related('schema__metrics__validation_rules', 'schema__dimensions__dimension')
        .order_by('pk')
    )
    instance_level: list[DatasetMeta] = []
    owned: dict[UUID, list[DatasetMeta]] = {}
    for dataset in datasets:
        pin = dataset_revision_pins.get(dataset.pk) if dataset_revision_pins is not None else None
        meta = dataset_meta_from_model(
            dataset,
            primary_language=ic.primary_language,
            pinned_revision_id=pin.revision_id if pin is not None else None,
        )
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
        .order_by('node_id', 'port_id', 'position')
    )


def _dataset_export_key(ds: DatasetModel) -> str:
    return ds.identifier or str(ds.uuid)


def _dataset_export_rank(ds: DatasetModel, ic_ct_id: int, ic_id: int) -> tuple[bool, bool, int]:
    is_direct = ds.scope_content_type_id == ic_ct_id and ds.scope_id == ic_id
    return (not is_direct, ds.is_external_placeholder, ds.pk)


def _datasets_for_instance_export(ic: InstanceConfig, ic_ct: ContentType) -> list[DatasetModel]:
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


def export_instance(ic: InstanceConfig, *, exported_from: str | None = None) -> InstanceExport:
    """
    Serialize a DB-sourced InstanceConfig with dataset bodies included.

    ``exported_from`` is recorded as provenance when the document is meant to
    leave this database (the GraphQL ``instance.export`` field passes the
    backend's base URL); in-process uses such as ``copy_instance`` leave it out.
    """
    from django.contrib.contenttypes.models import ContentType

    from nodes.page_snapshot import build_instance_page_snapshots

    snapshot = build_instance_snapshot(ic, compose=False)

    ic_ct = ContentType.objects.get_for_model(ic)
    from kausal_common.datasets.models import Dataset as DatasetModel

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
        from wagtail.models import Revision

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
        pages=build_instance_page_snapshots(ic),
        exported_at=timezone.now(),
        exported_from=exported_from,
        draft_head_token=ic.draft_head_token,
    )


# ---------------------------------------------------------------------------
# Import (from_dict)
# ---------------------------------------------------------------------------


def _template_dataset_bodies(base: InstanceSnapshot, revisions: dict[int, Revision]) -> list[DatasetSnapshot]:
    bodies = {
        pin.dataset_uuid: DatasetSnapshot.model_validate(revisions[pin.revision_id].content) for pin in base.dataset_revisions
    }
    dimensions = {item.id: item.identifier for item in base.dimensions}
    for dataset in base.all_datasets():
        if dataset.id in bodies:
            continue
        bodies[dataset.id] = DatasetSnapshot(
            identifier=dataset.identifier,
            is_external_placeholder=dataset.is_external_placeholder,
            external_ref=dataset.external_ref,
            is_editable=dataset.is_editable if dataset.is_editable is not None else True,
            dimensions=[dimensions[item] for item in dataset.declared_dimension_ids],
            category_domain=dataset.category_domain,
            metrics=[
                DatasetMetricSnapshot(
                    identifier=item.identifier or str(item.id),
                    unit=item.unit,
                    label=(
                        item.label
                        if isinstance(item.label, TranslatedString)
                        else TranslatedString(str(item.label), default_language=base.metadata.primary_language)
                        if item.label is not None
                        else None
                    ),
                    quantity=item.quantity,
                )
                for item in dataset.metrics
            ],
        )
    return list(bodies.values())


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


def _import_nodes(
    ic: InstanceConfig,
    export: InstanceExport,
    *,
    preserve_uuids: bool = False,
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
        apply_translated(fields, i18n_dict, n.name, 'name', primary_lang)
        apply_translated(fields, i18n_dict, n.short_name, 'short_name', primary_lang)
        apply_translated(fields, i18n_dict, n.short_description, 'short_description', primary_lang)
        apply_translated(fields, i18n_dict, n.description, 'description', primary_lang)
        apply_translated(fields, i18n_dict, n.goal, 'goal', primary_lang)

        fields.update({'uuid': n.uuid} if preserve_uuids else {})
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


def _import_template_revision(export: InstanceExport, *, organization_id: int) -> int | None:
    from django.contrib.contenttypes.models import ContentType
    from wagtail.models import Revision

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
        import_instance(template, export.template, preserve_node_uuids=True)
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

    from kausal_common.datasets.models import Dataset

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
    from django.contrib.contenttypes.models import ContentType
    from wagtail.models import Revision

    from kausal_common.datasets.models import Dataset

    from datasets.materialization import hash_dataset_content

    base = export.instance.model_copy(deep=True)
    # Dataset UUIDs in the frozen catalog must address imported bodies on this backend.
    catalog = {item.identifier: item for item in base.all_datasets()}
    for dataset in Dataset.objects.for_instance_config(template):
        meta = catalog.get(dataset.identifier)
        if meta is not None and dataset.uuid != meta.id:
            if Dataset.objects.filter(uuid=meta.id).exists():
                raise ValueError('Imported template dataset UUID conflicts with an existing dataset')
            Dataset.objects.filter(pk=dataset.pk).update(uuid=meta.id)
    pins = []
    for pin in base.dataset_revisions:
        dataset = Dataset.objects.get(uuid=pin.dataset_uuid)
        body = next(item for item in export.datasets if item.identifier == pin.identifier)
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
    # Revision IDs are database-local. Retarget both the catalogs and input payload references.
    ids = {pin.dataset_uuid: pin.revision_id for pin in pins}
    base.datasets = [item.model_copy(update={'revision_id': ids.get(item.id, item.revision_id)}) for item in base.datasets]
    for node in base.nodes:
        node.datasets = [item.model_copy(update={'revision_id': ids.get(item.id, item.revision_id)}) for item in node.datasets]
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
    *,
    preserve_node_uuids: bool = False,
) -> None:
    """
    Populate an InstanceConfig with computation model objects from an InstanceExport.

    The InstanceConfig must already exist (with identifier, org, etc.).
    This function creates all related objects: nodes, edges, datasets, ports.
    """
    from django.contrib.contenttypes.models import ContentType

    ic_ct = ContentType.objects.get_for_model(ic)
    if export.instance.snapshot_kind == 'composed':
        raise ValueError('Import requires an authoring snapshot, not a composed runtime model')
    template_revision_id = _import_template_revision(export, organization_id=ic.organization_id)

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
    ic.template_revision_id = template_revision_id
    ic.node_settings = export.instance.node_settings
    update_fields = ['spec', 'primary_language', 'other_languages', 'config_source', 'template_revision', 'node_settings']

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
    nodes_by_uuid = _import_nodes(ic, export, preserve_uuids=preserve_node_uuids)
    _remap_imported_data_entry(ic, export, nodes_by_uuid, datasets, preserve_node_uuids=preserve_node_uuids)

    _import_dataset_ownership(ic, export.instance, nodes_by_uuid, datasets_by_id)

    # Input bindings (edges and dataset ports)
    _import_bindings(ic, export, nodes_by_uuid, datasets_by_id)
    if template_revision_id is not None:
        _import_binding_overrides(ic, export.instance, nodes_by_uuid, datasets_by_id)


def _remap_imported_data_entry(
    ic: InstanceConfig,
    export: InstanceExport,
    nodes_by_uuid: dict[UUID, NodeConfig],
    datasets: list[DatasetModel],
    *,
    preserve_node_uuids: bool,
) -> None:
    if ic.spec is not None and isinstance(ic.spec.data_entry, DataEntrySpec):
        identities = {original: node.uuid for original, node in nodes_by_uuid.items()}
        new_dimensions = {dim.identifier: dim for dim in _dimension_catalog_for(ic)}
        for original in export.instance.dimensions:
            target = new_dimensions.get(original.identifier)
            if target is None:
                continue
            identities[original.id] = target.id
            categories = {cat.identifier: cat.id for cat in target.categories}
            identities.update({cat.id: categories[cat.identifier] for cat in original.categories if cat.identifier in categories})
        targets = {
            original.uuid: dataset
            for original, dataset in zip(export.datasets, datasets, strict=True)
            if original.uuid is not None
        }
        targets_by_identifier = {dataset.identifier: dataset for dataset in datasets if dataset.identifier is not None}
        for original in export.instance.all_datasets():
            target_dataset = targets.get(original.id) or (
                targets_by_identifier.get(original.identifier) if original.identifier is not None else None
            )
            if target_dataset is None:
                continue
            identities[original.id] = target_dataset.uuid
            target_metrics = (
                {metric.name: metric.uuid for metric in target_dataset.schema.metrics.all()} if target_dataset.schema else {}
            )
            identities.update({
                metric.id: target_metrics[metric.identifier] for metric in original.metrics if metric.identifier in target_metrics
            })
        if not preserve_node_uuids:
            identities.update({section.id: uuid4() for section in ic.spec.data_entry.sections})
            identities.update({table.id: uuid4() for section in ic.spec.data_entry.sections for table in section.tables})
            # An amendment may replace an inherited entry or add a locally owned one.
            inherited_layout = export.template.instance.spec.data_entry if export.template else None
            inherited_tables = (
                {table.id for section in inherited_layout.sections for table in section.tables}
                if isinstance(inherited_layout, DataEntrySpec)
                else set()
            )
            identities.update({
                table.id: uuid4()
                for amendment in ic.spec.data_entry.amendments
                for table in amendment.tables
                if table.id not in inherited_tables
            })
        ic.spec.data_entry = remap_data_entry(ic.spec.data_entry, identities)
        ic.save(update_fields=['spec'])


def _import_dataset_ownership(
    ic: InstanceConfig,
    snapshot: InstanceSnapshot,
    nodes_by_uuid: dict[UUID, NodeConfig],
    datasets_by_id: dict[str, DatasetModel],
) -> None:
    from django.contrib.contenttypes.models import ContentType

    from nodes.models import NodeConfig

    node_ct = ContentType.objects.get_for_model(NodeConfig)
    for node in snapshot.nodes:
        for owned in node.datasets:
            dataset = datasets_by_id.get(owned.identifier) if owned.identifier is not None else None
            if dataset is not None:
                type(dataset).objects.filter(pk=dataset.pk).update(
                    scope_content_type=node_ct, scope_id=nodes_by_uuid[node.uuid].pk
                )


def _import_binding_overrides(
    ic: InstanceConfig,
    snapshot: InstanceSnapshot,
    nodes_by_uuid: dict[UUID, NodeConfig],
    datasets_by_id: dict[str, DatasetModel],
) -> None:
    from nodes.models import InputPortBindingSet

    def remap_node(identifier: UUID) -> UUID:
        node = nodes_by_uuid.get(identifier)
        return node.uuid if node is not None else identifier

    selections = list(snapshot.binding_overrides)
    inherited_ports = {
        (binding.node_id, binding.port_id)
        for binding in snapshot.bindings
        if isinstance(binding.source, NodePortSource) and binding.source.node_id not in nodes_by_uuid
    }
    selections.extend(
        InputBindingOverrideSnapshot(
            node_uuid=node_id,
            port_uuid=port_id,
            bindings=[item for item in snapshot.bindings if (item.node_id, item.port_id) == (node_id, port_id)],
        )
        for node_id, port_id in inherited_ports
    )
    from nodes.models import NodeInputPortBinding
    from nodes.template_graph import template_snapshot

    pins = {pin.dataset_uuid: pin.revision_id for pin in template_snapshot(ic).dataset_revisions}
    for override in selections:
        bindings = []
        for item in override.bindings:
            source = item.source
            if isinstance(source, NodePortSource):
                source = source.model_copy(update={'node_id': remap_node(source.node_id)})
            else:
                dataset = datasets_by_id.get(source.dataset)
                if dataset is not None:
                    assert dataset.schema is not None
                    metric = dataset.schema.metrics.get(name=source.metric)
                    source = source.model_copy(
                        update={'dataset_uuid': dataset.uuid, 'metric_uuid': metric.uuid, 'dataset_revision': None}
                    )
            if isinstance(source, DatasetMetricSource) and source.dataset_uuid in pins:
                source = source.model_copy(update={'dataset_revision': pins[source.dataset_uuid]})
            bindings.append(item.model_copy(update={'node_id': remap_node(item.node_id), 'source': source}))
        NodeInputPortBinding.objects.filter(
            instance=ic, node__uuid=remap_node(override.node_uuid), port_id=override.port_uuid
        ).delete()
        InputPortBindingSet.objects.create(
            instance=ic, node_uuid=remap_node(override.node_uuid), port_uuid=override.port_uuid, bindings=bindings
        )
