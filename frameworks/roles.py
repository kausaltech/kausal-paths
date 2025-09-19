from __future__ import annotations

from typing import TYPE_CHECKING

from django.utils.translation import gettext_lazy as _
from pydantic import BaseModel

from kausal_common.models.roles import ALL_MODEL_PERMS, AdminRole, InstanceFieldGroupRole, InstanceSpecificRole, register_role

from paths.const import FRAMEWORK_ADMIN_ROLE, FRAMEWORK_VIEWER_ROLE

from nodes.roles import InstanceAdminRole, InstanceViewerRole

if TYPE_CHECKING:
    from django.contrib.auth.models import Group
    from wagtail.models.sites import Site

    from frameworks.models import Framework, FrameworkQuerySet
    from users.models import User


class FrameworkGroupMembershipRole(InstanceFieldGroupRole['Framework', 'FrameworkQuerySet']):
    def __init__(self):
        from .models import Framework

        super().__init__(Framework)

    def get_instance_site(self, obj: Framework) -> Site | None:
        return None


class FrameworkAdminRole(FrameworkGroupMembershipRole, AdminRole['Framework']):
    id = FRAMEWORK_ADMIN_ROLE
    name = _('Framework admins')
    description = _('Administrative access to the instance without permissions to manage people and organizations')
    group_name = 'Framework admins'
    instance_group_field_name = 'admin_group'

    model_perms = InstanceAdminRole.model_perms + [
        ('frameworks', ('measuretemplate', 'frameworkconfig'), ALL_MODEL_PERMS),
        ('frameworks', ('framework',), ('view', 'change')),
    ]
    page_perms = InstanceAdminRole.page_perms

    def get_instance_group_name(self, obj: Framework) -> str:
        assert obj is not None
        return 'Framework %s admins' % obj.name

    def get_existing_instance_group(self, obj: Framework) -> Group | None:
        return obj.admin_group

    def update_instance_group(self, obj: Framework, group: Group):
        obj.admin_group = group
        obj.save(update_fields=['admin_group'])

    def get_instances_for_user(self, user: User) -> FrameworkQuerySet:
        return self.model.objects.get_queryset().filter(admin_group__in=user.cgroups).distinct()


class FrameworkViewerRole(FrameworkGroupMembershipRole, InstanceSpecificRole['Framework']):
    id = FRAMEWORK_VIEWER_ROLE
    name = _('Framework viewers')
    description = _('Read-only access to instance data')
    group_name = 'Framework viewers'
    instance_group_field_name = 'viewer_group'

    model_perms = [
        *InstanceViewerRole.model_perms,
        ('frameworks', ('measuretemplate', 'frameworkconfig'), ('view',)),
        ('frameworks', ('framework',), ('view',)),
    ]

    def get_instance_group_name(self, obj: Framework) -> str:
        assert obj is not None
        return 'Framework %s viewers' % obj.name

    def get_existing_instance_group(self, obj: Framework) -> Group | None:
        return obj.viewer_group

    def update_instance_group(self, obj: Framework, group: Group):
        obj.viewer_group = group
        obj.save(update_fields=['viewer_group'])

    def get_instances_for_user(self, user: User) -> FrameworkQuerySet:
        return self.model.objects.get_queryset().filter(viewer_group__in=user.cgroups).distinct()


framework_admin_role = FrameworkAdminRole()
register_role(framework_admin_role)
framework_viewer_role = FrameworkViewerRole()
register_role(framework_viewer_role)


class FrameworkRoleDef(BaseModel):
    framework_id: str
    role_id: str | None
    org_slug: str | None
    org_id: str | None
