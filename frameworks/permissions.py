from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

from django.db.models import Q

from kausal_common.models.permission_policy import BaseObjectAction, ModelPermissionPolicy, ModelReadOnlyPolicy

from paths.const import INSTANCE_ADMIN_ROLE

from users.models import User

if TYPE_CHECKING:
    from collections.abc import Sequence

    from kausal_common.models.permissions import PermissionedModel
    from kausal_common.users import UserOrAnon

    from frameworks.models import (  # noqa: F401
        Framework,
        FrameworkConfig,
        FrameworkConfigQuerySet,
        FrameworkQuerySet,
        Section,
        SectionQuerySet,
    )
    from users.models import User


class FrameworkPermissionPolicy(ModelReadOnlyPolicy['Framework', 'FrameworkQuerySet']):
    def creatable_child_models(self, user: UserOrAnon, obj: Framework) -> Sequence[type[PermissionedModel]]:
        from .models import FrameworkConfig

        if not self.user_is_authenticated(user):
            return []
        fwc_pp = FrameworkConfig.permission_policy()
        if fwc_pp.user_can_create(user, obj):
            return [FrameworkConfig]
        return []

    def __init__(self):
        from .models import Framework
        super().__init__(Framework)


class SectionPermissionPolicy(ModelReadOnlyPolicy['Section', 'SectionQuerySet']):
    def __init__(self):
        from .models import Section
        super().__init__(Section)


class FrameworkConfigPermissionPolicy(
    ModelPermissionPolicy['FrameworkConfig', 'FrameworkConfigQuerySet', 'Framework'],
):
    def is_create_context_valid(self, context: Any) -> TypeGuard[Framework]:  # noqa: ANN401
        fw_pp = FrameworkPermissionPolicy()
        return isinstance(context, fw_pp.model)

    def __init__(self):
        from nodes.roles import instance_admin_role, instance_viewer_role

        from .models import FrameworkConfig
        from .roles import framework_admin_role

        self.framework_admin_role = framework_admin_role
        self.realm_admin_role = instance_admin_role
        self.realm_viewer_role = instance_viewer_role
        super().__init__(FrameworkConfig)

    def construct_perm_q(self, user: User, action: BaseObjectAction) -> Q | None:
        fw_admin_q = Q(framework__admin_group__in=user.cgroups)
        if action == 'delete':
            return fw_admin_q
        instance_admin_q = Q(instance_config__admin_group__in=user.cgroups)
        return fw_admin_q | instance_admin_q

    def construct_perm_q_anon(self, action: BaseObjectAction) -> Q | None:
        return None

    def anon_has_perm(self, action: BaseObjectAction, obj: FrameworkConfig) -> bool:
        return False

    def user_has_perm(self, user: User, action: BaseObjectAction, obj: FrameworkConfig) -> bool:
        fw = obj.framework
        is_fw_admin = user.has_instance_role(self.framework_admin_role, fw)
        if is_fw_admin:
            return True
        if action == 'delete':
            return False
        ic = obj.instance_config
        is_realm_admin = user.has_instance_role(self.realm_admin_role, ic)
        is_realm_viewer = user.has_instance_role(self.realm_viewer_role, ic)
        if action == 'view':
            return is_realm_viewer or is_realm_admin
        return is_realm_admin

    def get_create_defaults(self, user: User, context: Framework) -> dict[str, str | None]:
        for role in user.extra.framework_roles:
            if role.framework_id == context.identifier:
                break
        else:
            return {}
        return dict(organization_identifier=role.org_id, organization_slug=role.org_slug)

    def creatable_child_models(self, user: UserOrAnon, obj: FrameworkConfig) -> Sequence[type[PermissionedModel]]:
        if not self.user_is_authenticated(user):
            return []
        return []

    def user_can_create(self, user: User, context: Framework) -> bool:
        if user.has_instance_role(self.framework_admin_role, context):
            return True
        for role in user.extra.framework_roles:
            if role.framework_id == context.identifier:
                break
        else:
            return False

        if role.role_id == INSTANCE_ADMIN_ROLE:
            return True
        return False
