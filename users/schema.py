from __future__ import annotations

from typing import TYPE_CHECKING, Annotated
from uuid import UUID

import strawberry as sb
import strawberry_django
from django.db.models import Q
from strawberry import auto

from kausal_common.users import user_or_none

from paths import gql

from frameworks.roles import FrameworkRoleDef
from nodes.models import InstanceConfig

from .models import User

if TYPE_CHECKING:
    from nodes.graphql.types.instance import InstanceType


@sb.type(name='UserFrameworkRole')
class UserFrameworkRoleType:
    framework_id: sb.ID
    role_id: str | None
    org_slug: str | None
    org_id: str | None


@strawberry_django.type(User, name='User')
class UserType:
    email: auto
    first_name: auto
    last_name: auto

    @strawberry_django.field
    @staticmethod
    def id(root: sb.Parent[User]) -> sb.ID:
        return sb.ID(str(root.uuid))

    @strawberry_django.field(graphql_type=list[UserFrameworkRoleType])
    @staticmethod
    def framework_roles(root: sb.Parent[User]) -> list[FrameworkRoleDef]:
        # FrameworkRoleDef shares the same field names as UserFrameworkRoleType,
        # so Strawberry serializes the pydantic instances directly by attribute.
        return list(root.extra.framework_roles)

    @strawberry_django.field(
        graphql_type=list[Annotated['InstanceType', sb.lazy('nodes.graphql.types.instance')]],
        description=(
            'Instances the user can edit (admin or owner). Pass `frameworkId` to limit the '
            'list to instances under the given framework (either by identifier or UUID).'
        ),
    )
    @staticmethod
    def editable_instances(
        root: sb.Parent[User],
        framework_id: sb.ID | None = None,
    ) -> list['InstanceType']:
        admin_group_ids = list(root.groups.values_list('pk', flat=True))
        qs = InstanceConfig.objects.qs.filter(
            Q(owned_by=root) | Q(admin_group__in=admin_group_ids) | Q(super_admin_group__in=admin_group_ids)
        ).distinct()
        if framework_id is not None:
            framework_filter = Q(framework_config__framework__identifier=str(framework_id))
            try:
                fw_uuid = UUID(str(framework_id))
            except ValueError, AttributeError:
                pass
            else:
                framework_filter |= Q(framework_config__framework__uuid=fw_uuid)
            qs = qs.filter(framework_filter)
        from nodes.graphql.types.instance import InstanceType

        return [InstanceType.from_model(ic) for ic in qs]


@sb.type
class UsersQuery:
    @sb.field(graphql_type=UserType | None)
    @staticmethod
    def me(info: gql.Info) -> User | None:
        return user_or_none(info.context.user)
