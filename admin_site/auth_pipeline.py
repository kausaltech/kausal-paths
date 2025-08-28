from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from frameworks.models import FrameworkConfig
from paths.const import (
    FRAMEWORK_ADMIN_ROLE,
    FRAMEWORK_VIEWER_ROLE,
    INSTANCE_ADMIN_ROLE,
    INSTANCE_REVIEWER_ROLE,
    INSTANCE_VIEWER_ROLE,
)

if TYPE_CHECKING:
    from social_django import BaseAuth

    from frameworks.roles import FrameworkRoleDef
    from users.models import User


def assign_roles(
    *, backend: type[BaseAuth], user: User | None, details: dict[str, Any], **kwargs,
) -> None:
    from frameworks.models import Framework
    from frameworks.roles import framework_admin_role, framework_viewer_role
    from nodes.roles import (
        instance_admin_role,
        instance_reviewer_role,
        instance_viewer_role,
    )

    framework_roles: list[FrameworkRoleDef] = details.get('framework_roles', [])
    if user is None or not framework_roles:
        return

    for role in framework_roles:
        if not role.role_id:
            continue

        framework_object = Framework.objects.filter(identifier=role.framework_id).first()
        if not framework_object:
            logger.error("Framework '%s' not found" % role.framework_id)
            continue

        user.extra.set_framework_role(role)
        user.save(update_fields=['extra'])

        if role.role_id == FRAMEWORK_ADMIN_ROLE:
            framework_admin_role.assign_user(framework_object, user)
        elif role.role_id == FRAMEWORK_VIEWER_ROLE:
            framework_viewer_role.assign_user(framework_object, user)
        elif role.role_id in (INSTANCE_ADMIN_ROLE, INSTANCE_VIEWER_ROLE, INSTANCE_REVIEWER_ROLE):
            framework_configs = FrameworkConfig.objects.filter(
                framework=framework_object, organization_identifier=role.org_id
            ).exclude(organization_identifier__isnull=True)

            for framework_config in framework_configs:
                instance_config = framework_config.instance_config
                if role.role_id == INSTANCE_ADMIN_ROLE:
                    instance_admin_role.assign_user(instance_config, user)
                elif role.role_id == INSTANCE_VIEWER_ROLE:
                    instance_viewer_role.assign_user(instance_config, user)
                elif role.role_id == INSTANCE_REVIEWER_ROLE:
                    instance_reviewer_role.assign_user(instance_config, user)

            if not framework_configs:
                logger.info("Framework config for org UID '%s' does not yet exist" % (role.org_id))
        else:
            logger.error("Unknown role '%s'" % role.role_id)


def update_role_permissions(*, user: User | None, **kwargs):
    if user is None:
        return
    user.perms.refresh_role_permissions()
