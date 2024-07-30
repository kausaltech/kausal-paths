from __future__ import annotations

from typing import TYPE_CHECKING

from django.contrib.auth.models import Group
from django.utils.translation import gettext_lazy as _
from wagtail.models.sites import Site

from kausal_common.models.roles import ALL_MODEL_PERMS, AdminRole
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from nodes.models import InstanceConfig


class InstanceAdminRole(AdminRole[InstanceConfig]):
    id = 'instance_admin'
    name = _("General admin")
    group_name = "General admins"

    model_perms = AdminRole.model_perms + [
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view', 'change')),
        ('datasets', (
            'dataset', 'datasetcomment', 'datasetdimension', 'datasetdimensionselectedcategory',
            'datasetmetric', 'datasetsourcereference', 'dimension', 'dimensioncategory',
        ), ALL_MODEL_PERMS),
    ]

    def get_instance_group_name(self, obj: InstanceConfig) -> str:
        assert obj is not None
        return '%s %s' % (obj.name, self.group_name)

    def get_existing_instance_group(self, obj: InstanceConfig) -> Group | None:
        return obj.admin_group

    def update_instance_group(self, obj: InstanceConfig, group: Group):
        obj.admin_group = group
        obj.save(update_fields=['admin_group'])

    def get_instance_site(self, obj: InstanceConfig) -> Site | None:
        return obj.site


instance_admin_role = InstanceAdminRole()
