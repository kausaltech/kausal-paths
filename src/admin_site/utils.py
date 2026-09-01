from __future__ import annotations

from typing import TYPE_CHECKING, cast

from wagtail.admin.menu import MenuItem

from paths.context import realm_context

from nodes.roles import instance_super_admin_role

if TYPE_CHECKING:
    from django.http import HttpRequest

    from users.models import User


class SuperAdminOnlyMenuItem(MenuItem):
    def is_shown(self, request: HttpRequest):
        user = request.user
        if not user.is_authenticated:
            return False
        user = cast('User', user)
        active_instance = realm_context.get().realm
        return user.is_superuser or user.has_instance_role(instance_super_admin_role, active_instance)
