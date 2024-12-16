from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from paths.const import FRAMEWORK_ADMIN_ROLE, FRAMEWORK_VIEWER_ROLE, INSTANCE_ADMIN_ROLE, INSTANCE_VIEWER_ROLE

from frameworks.models import FrameworkConfig

if TYPE_CHECKING:
    from social_django import BaseAuth

    from frameworks.roles import FrameworkRoleDef
    from users.models import User


def assign_roles(
    *, backend: type[BaseAuth], user: User | None, details: dict[str, Any], **kwargs,
) -> None:
    from frameworks.models import Framework
    from frameworks.roles import framework_admin_role, framework_viewer_role
    from nodes.roles import instance_admin_role, instance_viewer_role

    fw_roles: list[FrameworkRoleDef] = details.get('framework_roles', [])
    if user is None or not fw_roles:
        return

    for role in fw_roles:
        if not role.role_id:
            continue

        fw_obj = Framework.objects.filter(identifier=role.framework_id).first()
        if not fw_obj:
            logger.error("Framework '%s' not found" % role.framework_id)
            continue
        user.extra.set_framework_role(role)
        user.save(update_fields=['extra'])
        if role.role_id == FRAMEWORK_ADMIN_ROLE:
            framework_admin_role.assign_user(fw_obj, user)
        elif role.role_id == FRAMEWORK_VIEWER_ROLE:
            framework_viewer_role.assign_user(fw_obj, user)
        elif role.role_id in (INSTANCE_ADMIN_ROLE, INSTANCE_VIEWER_ROLE):
            fwc = FrameworkConfig.objects.filter(framework=fw_obj, organization_identifier=role.org_id).first()
            if fwc is not None:
                ic = fwc.instance_config
                if role.role_id == INSTANCE_ADMIN_ROLE:
                    instance_admin_role.assign_user(ic, user)
                elif role.role_id == INSTANCE_VIEWER_ROLE:
                    instance_viewer_role.assign_user(ic, user)
            else:
                logger.info("Framework config for org UID '%s' does not yet exist" % (role.org_id))
            if user.perms.has_instance_role(framework_viewer_role, fw_obj):
                framework_viewer_role.unassign_user(fw_obj, user)
            if user.perms.has_instance_role(framework_admin_role, fw_obj):
                framework_admin_role.unassign_user(fw_obj, user)
        else:
            logger.error("Unknown role '%s'" % role.role_id)


def update_role_permissions(*, user: User | None, **kwargs):
    if user is None:
        return
    user.perms.refresh_role_permissions()
