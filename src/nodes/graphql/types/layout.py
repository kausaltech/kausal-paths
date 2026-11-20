from datetime import datetime
from typing import TYPE_CHECKING, Annotated

import strawberry as sb
import strawberry_django
from strawberry import auto

from kausal_common.strawberry.registry import register_strawberry_type

from nodes.models import NodeLayout
from nodes.node_layouts import InheritedNodeLayout

if TYPE_CHECKING:
    from users.schema import UserType


@register_strawberry_type
@strawberry_django.type(
    NodeLayout,
    name='NodeLayout',
    description=(
        "A model-editor card position. An inherited node's position is kept with the instance's "
        'node settings, which record no modification history, so its timestamps and users are null.'
    ),
)
class NodeLayoutType:
    x: auto
    y: auto
    source: auto
    created_at: datetime | None
    created_by: Annotated['UserType', sb.lazy('users.schema')] | None
    last_modified_at: datetime | None
    last_modified_by: Annotated['UserType', sb.lazy('users.schema')] | None

    @classmethod
    def is_type_of(cls, obj: object, _info: object) -> bool:
        return isinstance(obj, (NodeLayout, InheritedNodeLayout))

    @strawberry_django.field
    @staticmethod
    def node_id(root: sb.Parent[NodeLayout | InheritedNodeLayout]) -> sb.ID:
        if isinstance(root, InheritedNodeLayout):
            return sb.ID(root.node_identifier)
        return sb.ID(root.node.identifier)


@sb.type
class UpdateNodeLayoutsResult:
    layouts: list[NodeLayoutType]
