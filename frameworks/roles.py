from __future__ import annotations

from typing import TYPE_CHECKING

from django.utils.translation import gettext_lazy as _
from pydantic import BaseModel

from kausal_common.models.roles import ALL_MODEL_PERMS, AdminRole, register_role

from paths.const import FRAMEWORK_ADMIN_ROLE

from nodes.roles import InstanceAdminRole

if TYPE_CHECKING:
    from django.contrib.auth.models import Group
    from wagtail.models.sites import Site

    from frameworks.models import Framework, FrameworkQuerySet
    from users.models import User



class FrameworkAdminRole(AdminRole['Framework']):
    id = FRAMEWORK_ADMIN_ROLE
    name = _("Framework admins")
    group_name = "Framework admins"

    model_perms = InstanceAdminRole.model_perms + [
        ('frameworks', ('measuretemplate',), ALL_MODEL_PERMS),
    ]
    page_perms = InstanceAdminRole.page_perms

    def __init__(self):
        from .models import Framework
        super().__init__(Framework)

    def get_instance_group_name(self, obj: Framework) -> str:
        assert obj is not None
        return 'Framework %s admins' % obj.name

    def get_existing_instance_group(self, obj: Framework) -> Group | None:
        return obj.admin_group

    def update_instance_group(self, obj: Framework, group: Group):
        obj.admin_group = group
        obj.save(update_fields=['admin_group'])

    def get_instance_site(self, obj: Framework) -> Site | None:
        return None

    def get_instances_for_user(self, user: User) -> FrameworkQuerySet:
        return self.model.objects.get_queryset().filter(admin_group__in=user.cgroups).distinct()


framework_admin_role = FrameworkAdminRole()
register_role(framework_admin_role)

class FrameworkRoleDef(BaseModel):
    framework_id: str
    role_id: str | None
    org_slug: str | None
    org_id: str | None
