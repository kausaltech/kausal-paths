"""
Model-editor card positions of the nodes an instance inherits from its template.

An instance's own node keeps its position in a `NodeLayout` row. An inherited node's
`NodeConfig` belongs to the template, so its position is one of the instance's choices about
inherited nodes and is kept in `InstanceConfig.node_settings`, beside the goals and parameter
values. Conversion moves positions there too (`frameworks.conversion`).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nodes.instance_serialization import InheritedNodeSettings, NodeLayoutSnapshot
from nodes.models import NodeLayoutSource

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from nodes.instance_graph import InstanceGraph
    from nodes.models import InstanceConfig
    from users.models import User


@dataclass
class InheritedNodeLayout:
    """
    The position of an inherited node, shaped like a `NodeLayout` row for the GraphQL type.

    Node settings keep no modification history, so the timestamps and users are always None.
    """

    node_identifier: str
    x: float
    y: float
    source: NodeLayoutSource
    created_at: datetime | None = None
    created_by: User | None = None
    last_modified_at: datetime | None = None
    last_modified_by: User | None = None


def inherited_node_layouts(instance: InstanceConfig, graph: InstanceGraph) -> list[InheritedNodeLayout]:
    """Return the stored positions of the inherited nodes that are in `graph`."""
    layouts: list[InheritedNodeLayout] = []
    for settings in instance.node_settings:
        if settings.layout is None:
            continue
        node = graph.node_by_id.get(settings.node_uuid)
        if node is None or node.identifier is None:
            # Composition refuses settings for a node outside the pinned revision, so this
            # is only a graph that has not caught up with a settings change yet.
            continue
        layouts.append(
            InheritedNodeLayout(
                node_identifier=node.identifier,
                x=settings.layout.x,
                y=settings.layout.y,
                source=NodeLayoutSource(settings.layout.source),
            )
        )
    return sorted(layouts, key=lambda layout: layout.node_identifier)


def set_inherited_node_layouts(instance: InstanceConfig, layouts: dict[UUID, NodeLayoutSnapshot]) -> None:
    """Store positions of inherited nodes. The caller holds the instance's row lock."""
    if not layouts:
        return
    remaining = dict(layouts)
    settings_list: list[InheritedNodeSettings] = []
    for settings in instance.node_settings:
        layout = remaining.pop(settings.node_uuid, None)
        settings_list.append(settings.model_copy(update={'layout': layout}) if layout is not None else settings)
    settings_list.extend(InheritedNodeSettings(node_uuid=node_uuid, layout=layout) for node_uuid, layout in remaining.items())
    instance.node_settings = settings_list
    instance.save(update_fields=['node_settings'])


def clear_inherited_node_layouts(instance: InstanceConfig) -> None:
    """Forget the positions of inherited nodes, dropping settings that held nothing else."""
    if not any(settings.layout is not None for settings in instance.node_settings):
        return
    settings_list: list[InheritedNodeSettings] = []
    for settings in instance.node_settings:
        cleared = settings.model_copy(update={'layout': None})
        if cleared != InheritedNodeSettings(node_uuid=settings.node_uuid):
            settings_list.append(cleared)
    instance.node_settings = settings_list
    instance.save(update_fields=['node_settings'])
