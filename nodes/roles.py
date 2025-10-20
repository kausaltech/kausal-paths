from __future__ import annotations

from abc import ABCMeta
from typing import TYPE_CHECKING

from django.utils.translation import gettext_lazy as _

from kausal_common.models.roles import (
    ALL_MODEL_PERMS,
    AdminRole,
    InstanceFieldGroupRole,
    InstanceSpecificRole,
    Role,
    register_role,
)

from paths.const import (
    INSTANCE_ADMIN_ROLE,
    INSTANCE_REVIEWER_ROLE,
    INSTANCE_SUPER_ADMIN_ROLE,
    INSTANCE_VIEWER_ROLE,
    SUBSECTOR_ADMIN_GROUP_NAME,
)

if TYPE_CHECKING:
    from django.contrib.auth.models import Group
    from wagtail.models.sites import Site

    from nodes.models import InstanceConfig


class SubsectorAdminRole(Role):
    id = 'subsector-admin'
    name = _('Subsector Admin')
    description = _('A Paths administrator with specifically set per-model-instance permissions.')
    group_name = SUBSECTOR_ADMIN_GROUP_NAME

    model_perms = [
        ('wagtailadmin', 'admin', ('access',)),
        ('datasets', (
            'datasetschema',
            'dataset',
            'datapoint',
            'datasource',
            'datasetsourcereference',
            'datapointcomment',
            'datasetmetric',
        ), ALL_MODEL_PERMS),
    ]


class InstanceGroupMembershipRole(InstanceFieldGroupRole['InstanceConfig'], metaclass=ABCMeta):
    def __init__(self):
        from .models import InstanceConfig
        super().__init__(InstanceConfig)

    def get_instance_group_name(self, obj: InstanceConfig) -> str:
        assert obj is not None
        return '%s %s' % (obj.name, self.group_name)

    def get_instance_site(self, obj: InstanceConfig) -> Site | None:
        return obj.site


class InstanceSuperAdminRole(InstanceGroupMembershipRole, AdminRole['InstanceConfig']):
    id = INSTANCE_SUPER_ADMIN_ROLE
    name = _("Super Admin")
    description = _(
        'Full administrative access to the instance, also for managing people and organizations'
    )
    group_name = "Super Admins"
    instance_group_field_name = 'super_admin_group'

    model_perms = AdminRole.model_perms + [
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view', 'change')),
        ('datasets', (
            'datasetschema',
            'dataset',
            'datapoint',
            'datasource',
            'datasetsourcereference',
            'datapointcomment',
            'datasetmetric',
        ), ALL_MODEL_PERMS),
        ('frameworks', (
            'framework',
        ), ('view',)),
        ('frameworks', (
            'frameworkconfig', 'measure', 'measuredatapoint',
        ), ALL_MODEL_PERMS),
        ('people', (
            'person',
        ), ALL_MODEL_PERMS),
        ('orgs', (
            'organization',
        ), ALL_MODEL_PERMS),
    ]

    def get_existing_instance_group(self, obj: InstanceConfig) -> Group | None:
        return obj.super_admin_group

    def update_instance_group(self, obj: InstanceConfig, group: Group | None):
        obj.super_admin_group = group
        obj.save(update_fields=[self.instance_group_field_name])


class InstanceAdminRole(InstanceGroupMembershipRole, AdminRole['InstanceConfig']):
    id = INSTANCE_ADMIN_ROLE
    name = _("Admin")
    description = _(
        'Administrative access to the instance without permissions to manage people and organizations'
    )
    group_name = "Admins"
    instance_group_field_name = 'admin_group'

    model_perms = AdminRole.model_perms + [
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view', 'change')),
        ('datasets', (
            'datasetschema',
            'dataset',
            'datapoint',
            'datasource',
            'datasetsourcereference',
            'datapointcomment',
            'datasetmetric',
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
    description = _(
        'Read-only access to instance data'
    )
    group_name = "Viewers"
    instance_group_field_name = 'viewer_group'

    model_perms = [
        ('wagtailadmin', 'admin', ('access',)),
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view',)),
        ('datasets', (
            'datasetschema',
            'dataset',
            'datapoint',
            'datasource',
            'datasetsourcereference',
            'datapointcomment',
            'datasetmetric',
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


class InstanceReviewerRole(InstanceGroupMembershipRole, InstanceSpecificRole['InstanceConfig']):
    id = INSTANCE_REVIEWER_ROLE
    name = _('Reviewer')
    description = _('Ability to write review comments and read-only access to instance data')
    group_name = "Reviewers"
    instance_group_field_name = 'reviewer_group'

    model_perms = [
        ('wagtailadmin', 'admin', ('access',)),
        ('nodes', ('instanceconfig', 'nodeconfig'), ('view',)),
        ('datasets', ('datapointcomment'), ('add',)),
        ('datasets', (
            'datasetschema',
            'dataset',
            'datapoint',
            'datasource',
            'datasetsourcereference',
            'datapointcomment',
            'datasetmetric',
        ), ('view',)),
        ('frameworks', (
            'framework', 'frameworkconfig', 'measure', 'measuredatapoint',
        ), ('view',)),
    ]

    def get_existing_instance_group(self, obj: InstanceConfig) -> Group | None:
        return obj.reviewer_group

    def update_instance_group(self, obj: InstanceConfig, group: Group | None):
        obj.reviewer_group = group
        obj.save(update_fields=[self.instance_group_field_name])


instance_super_admin_role = InstanceSuperAdminRole()
instance_admin_role = InstanceAdminRole()
instance_viewer_role = InstanceViewerRole()
instance_reviewer_role = InstanceReviewerRole()

# Roles registered from most permissions to least permissions to have them in sensible order when listed
register_role(instance_super_admin_role)
register_role(instance_admin_role)
register_role(instance_reviewer_role)
register_role(instance_viewer_role)
