from __future__ import annotations

from typing import TYPE_CHECKING

from django.utils.translation import gettext_lazy as _

from kausal_common.models.roles import ALL_MODEL_PERMS, AdminRole, register_role

from paths.const import INSTANCE_ADMIN_ROLE

if TYPE_CHECKING:
    from django.contrib.auth.models import Group
    from django.db.models import QuerySet
    from wagtail.models.sites import Site

    from nodes.models import InstanceConfig
    from users.models import User


class InstanceAdminRole(AdminRole['InstanceConfig']):
    id = INSTANCE_ADMIN_ROLE
    name = _("General admin")
    group_name = "General admins"

    model_perms = AdminRole.model_perms + [
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view', 'change')),
        ('datasets', (
            'dataset', 'datasetcomment', 'datasetdimension', 'datasetdimensionselectedcategory',
            'datasetmetric', 'datasetsourcereference', 'dimension', 'dimensioncategory',
        ), ALL_MODEL_PERMS),
    ]

    def __init__(self):
        from .models import InstanceConfig
        super().__init__(InstanceConfig)

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

    def get_instances_for_user(self, user: User) -> QuerySet[InstanceConfig, InstanceConfig]:
        user_groups = user.groups
        return self.model.objects.filter(admin_group__in=user_groups).distinct()


instance_admin_role = InstanceAdminRole()

register_role(instance_admin_role)
