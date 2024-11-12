from __future__ import annotations

from typing import TYPE_CHECKING

from django.utils.translation import gettext_lazy as _

from kausal_common.models.roles import ALL_MODEL_PERMS, AdminRole, InstanceFieldGroupRole, InstanceSpecificRole, register_role

from paths.const import INSTANCE_ADMIN_ROLE, INSTANCE_VIEWER_ROLE

if TYPE_CHECKING:
    from django.contrib.auth.models import Group
    from wagtail.models.sites import Site

    from nodes.models import InstanceConfig


class InstanceGroupMembershipRole(InstanceFieldGroupRole['InstanceConfig']):
    def __init__(self):
        from .models import InstanceConfig
        super().__init__(InstanceConfig)

    def get_instance_group_name(self, obj: InstanceConfig) -> str:
        assert obj is not None
        return '%s %s' % (obj.name, self.group_name)

    def get_instance_site(self, obj: InstanceConfig) -> Site | None:
        return obj.site


class InstanceAdminRole(InstanceGroupMembershipRole, AdminRole['InstanceConfig']):
    id = INSTANCE_ADMIN_ROLE
    name = _("General admin")
    group_name = "General admins"
    instance_group_field_name = 'admin_group'

    model_perms = AdminRole.model_perms + [
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view', 'change')),
        ('datasets', (
            'dataset', 'datasetcomment', 'datasetdimension', 'datasetdimensionselectedcategory',
            'datasetmetric', 'datasetsourcereference', 'dimension', 'dimensioncategory',
        ), ALL_MODEL_PERMS),
        ('frameworks', (
            'framework',
        ), ('view',)),
        ('frameworks', (
            'frameworkconfig', 'measure', 'measuredatapoint',
        ), ALL_MODEL_PERMS),
    ]

    def get_existing_instance_group(self, obj: InstanceConfig) -> Group | None:
        return obj.admin_group

    def update_instance_group(self, obj: InstanceConfig, group: Group | None):
        obj.admin_group = group
        obj.save(update_fields=[self.instance_group_field_name])


class InstanceViewerRole(InstanceGroupMembershipRole, InstanceSpecificRole['InstanceConfig']):
    id = INSTANCE_VIEWER_ROLE
    name = _('Viewer')
    group_name = "Viewer"
    instance_group_field_name = 'viewer_group'

    model_perms = [
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view',)),
        ('datasets', (
            'dataset', 'datasetcomment', 'datasetdimension', 'datasetdimensionselectedcategory',
            'datasetmetric', 'datasetsourcereference', 'dimension', 'dimensioncategory',
        ), ('view',)),
        ('frameworks', (
            'framework', 'frameworkconfig', 'measure', 'measuredatapoint',
        ), ('view',)),
    ]

    def get_existing_instance_group(self, obj: InstanceConfig) -> Group | None:
        return obj.viewer_group

    def update_instance_group(self, obj: InstanceConfig, group: Group | None):
        obj.viewer_group = group
        obj.save(update_fields=[self.instance_group_field_name])


instance_admin_role = InstanceAdminRole()
instance_viewer_role = InstanceViewerRole()

register_role(instance_admin_role)
register_role(instance_viewer_role)
