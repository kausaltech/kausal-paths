from typing import TYPE_CHECKING, Annotated

import strawberry as sb
import strawberry_django
from strawberry import auto

from kausal_common.strawberry.registry import register_strawberry_type

from nodes.models import NodeLayout

if TYPE_CHECKING:
    from users.schema import UserType


@register_strawberry_type
@strawberry_django.type(NodeLayout, name='NodeLayout')
class NodeLayoutType:
    x: auto
    y: auto
    source: auto
    created_at: auto
    created_by: Annotated['UserType', sb.lazy('users.schema')] | None
    last_modified_at: auto
    last_modified_by: Annotated['UserType', sb.lazy('users.schema')] | None

    @strawberry_django.field
    @staticmethod
    def node_id(root: sb.Parent[NodeLayout]) -> sb.ID:
        return sb.ID(root.node.identifier)


@sb.type
class UpdateNodeLayoutsResult:
    layouts: list[NodeLayoutType]
