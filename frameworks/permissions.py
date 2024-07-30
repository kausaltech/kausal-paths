from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib.auth.models import Group
from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _
from wagtail.models.sites import Site

from admin_site.roles import InstanceAdminRole
from frameworks.models import Framework
from kausal_common.models.roles import ALL_MODEL_PERMS, AdminRole
from paths.permissions import PathsPermissionPolicy

if TYPE_CHECKING:
    from frameworks.models import Framework, MeasureTemplate


class FrameworkAdminRole(AdminRole[Framework]):
    id = 'framework_admin'
    name = _("Framework admins")
    group_name = "Framework admins"

    model_perms = InstanceAdminRole.model_perms + [
        ('frameworks', ('measuretemplate',), ALL_MODEL_PERMS),
    ]

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


class MeasureTemplatePermissionPolicy(PathsPermissionPolicy[MeasureTemplate, QuerySet[MeasureTemplate]]):
    pass


framework_admin_role = FrameworkAdminRole()
