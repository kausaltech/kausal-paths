from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from paths.const import FRAMEWORK_ADMIN_ROLE, INSTANCE_ADMIN_ROLE

from frameworks.models import FrameworkConfig

if TYPE_CHECKING:
    from social_django import BaseAuth

    from frameworks.roles import FrameworkRoleDef
    from users.models import User


def assign_roles(
    *, backend: type[BaseAuth], user: User | None, details: dict[str, Any], **kwargs,
) -> None:
    from frameworks.models import Framework
    from frameworks.roles import framework_admin_role
    from nodes.roles import instance_admin_role

    fw_roles: list[FrameworkRoleDef] = details.get('framework_roles', [])
    if user is None or not fw_roles:
        return

    for role in fw_roles:
        fw_obj = Framework.objects.filter(identifier=role.framework_id).first()
        if not fw_obj:
            logger.error("Framework '%s' not found" % role.framework_id)
            continue
        user.extra.set_framework_role(role)
        user.save(update_fields=['extra'])
        if role.role_id == FRAMEWORK_ADMIN_ROLE:
            framework_admin_role.assign_user(fw_obj, user)
        elif role.role_id == INSTANCE_ADMIN_ROLE:
            fwc = FrameworkConfig.objects.filter(framework=fw_obj, organization_identifier=role.org_id).first()
            if fwc is not None:
                ic = fwc.instance_config
                instance_admin_role.assign_user(ic, user)
            else:
                logger.info("Framework config for org UID '%s' does not yet exist" % (role.org_id))


def update_role_permissions(*, user: User | None, **kwargs):
    if user is None:
        return
    user.perms.refresh_role_permissions()
