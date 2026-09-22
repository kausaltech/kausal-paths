"""Editing capabilities for objects in the selected effective graph."""

from typing import TYPE_CHECKING

from nodes.instance_graph import NodeMeta

if TYPE_CHECKING:
    from paths import gql

    from nodes.models import PreferredInstanceSource
    from nodes.node import Node


def runtime_source(info: gql.Info, node: Node) -> PreferredInstanceSource:
    resources = info.context.instance_resources
    assert resources is not None
    instance = node.context.instance
    for key, runtime in resources.instances.items():
        if runtime is instance:
            return key.source
    return resources.resolve_source(instance.config)[1]


def port_editable(info: gql.Info, node: Node | None, *, binding_owner: str | None = None, definition_flag: bool = True) -> bool:
    resources = info.context.instance_resources
    if resources is None or node is None or node.context.instance.config is None:
        return False
    context = resources.node_edit_context(info, node.context.instance.config, runtime_source(info, node))
    return NodeMeta.can_edit(
        context,
        inherited=node.source_snapshot is not None and node.source_snapshot.template_revision_id is not None,
        node_is_editable=node.db_obj.is_editable if node.db_obj is not None else True,
        definition_is_editable=definition_flag,
        binding_owner=binding_owner,
    )
