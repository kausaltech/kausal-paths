"""Framework-scoped organization access, including delegated subtree grants."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from django.db.models import Exists, F, OuterRef, Q, QuerySet
from django.db.models.functions import Left, Length

from kausal_common.people.models import ObjectRole

from frameworks.models import Framework, FrameworkOrganizationRoot, OrganizationAccessGrant
from frameworks.roles import framework_admin_role, framework_viewer_role
from orgs.models import Organization

if TYPE_CHECKING:
    from nodes.models import InstanceConfig
    from users.models import User

type OrganizationAction = Literal['view', 'change', 'delete']


def accessible_organizations(
    framework: Framework,
    user: User,
    *,
    action: OrganizationAction = 'view',
) -> QuerySet[Organization]:
    """
    Return only organizations covered by this framework and a user's roles.

    A grant on a district covers its municipalities but not its parent state or
    neighboring districts. Framework administrators can manage the whole scope;
    framework viewers can read it. The old instance-based organization policy
    remains separate for instance-local administration.
    """
    roots = list(framework.organization_roots.select_related('organization').values_list('organization__path', flat=True))
    if not roots or not user.is_authenticated:
        return Organization.objects.none()
    scope_q = Q(pk__in=[])
    for path in roots:
        scope_q |= Q(path__startswith=path)
    if user.is_superuser or user.has_instance_role(framework_admin_role, framework):
        return Organization.objects.filter(scope_q)
    if action == 'view' and user.has_instance_role(framework_viewer_role, framework):
        return Organization.objects.filter(scope_q)

    roles = ObjectRole.get_roles_for_action(action)
    grants = OrganizationAccessGrant.objects.filter(framework=framework, user=user, role__in=roles, suspended_at__isnull=True)
    access_q = Q(pk__in=[])
    for path in grants.values_list('organization__path', flat=True):
        if any(path.startswith(root_path) for root_path in roots):
            access_q |= Q(path__startswith=path)
    return Organization.objects.filter(scope_q & access_q)


@dataclass(frozen=True)
class ViewerScope:
    """
    The subtrees of a framework's organization tree that one user may read.

    Unlike `accessible_organizations`, this also knows what lies *above* those
    subtrees: a territory that contains something the user may read is not itself
    readable, but its totals may be shown.
    """

    everything: bool
    paths: tuple[str, ...]

    def covers(self, path: str) -> bool:
        """Whether the organization at `path` lies in a subtree the user may read."""
        return self.everything or any(path.startswith(own) for own in self.paths)

    def is_above(self, path: str) -> bool:
        """Whether the organization at `path` contains, without being, something the user may read."""
        return any(own.startswith(path) and own != path for own in self.paths)


def viewer_scope(framework: Framework, user: User) -> ViewerScope:
    """
    Resolve what a user may read in a framework's organization tree.

    Subtree grants count, and so does the organization of every instance the
    user holds an active membership in: a municipal account has no grant, but it
    may read its own organization.
    """
    from nodes.models import InstanceMemberAssignment

    roots = list(framework.organization_roots.values_list('organization__path', flat=True))
    if not roots or not user.is_authenticated:
        return ViewerScope(everything=False, paths=())
    if (
        user.is_superuser
        or user.has_instance_role(framework_admin_role, framework)
        or user.has_instance_role(framework_viewer_role, framework)
    ):
        return ViewerScope(everything=True, paths=tuple(roots))
    grants = OrganizationAccessGrant.objects.filter(
        framework=framework, user=user, role__in=ObjectRole.get_roles_for_action('view'), suspended_at__isnull=True
    ).values_list('organization__path', flat=True)
    memberships = InstanceMemberAssignment.objects.filter(
        user=user, suspended_at__isnull=True, instance_config__framework_config__framework=framework
    ).values_list('instance_config__organization__path', flat=True)
    paths = {path for path in (*grants, *memberships) if path and any(path.startswith(root) for root in roots)}
    return ViewerScope(everything=False, paths=tuple(sorted(paths)))


def organization_is_in_framework(framework: Framework, organization: Organization) -> bool:
    return any(
        organization.path.startswith(path) for path in framework.organization_roots.values_list('organization__path', flat=True)
    )


def instance_grant_q(user: User, *, action: OrganizationAction) -> Q:
    """Match instances covered by active organization grants, preserving framework scope."""
    roles = ObjectRole.get_roles_for_action(action)
    if action == 'delete' or not roles:
        return Q(pk__in=[])
    from nodes.models import InstanceMemberAssignment

    valid_root = (
        FrameworkOrganizationRoot.objects
        .filter(framework_id=OuterRef('framework_id'))
        .annotate(grant_root_prefix=Left(OuterRef('organization__path'), Length(F('organization__path'))))
        .filter(grant_root_prefix=F('organization__path'))
    )
    grants = (
        OrganizationAccessGrant.objects
        .filter(
            user=user,
            role__in=roles,
            suspended_at__isnull=True,
            framework_id=OuterRef('framework_config__framework_id'),
        )
        .annotate(
            instance_grant_prefix=Left(OuterRef('organization__path'), Length(F('organization__path'))),
            valid_root=Exists(valid_root),
        )
        .filter(instance_grant_prefix=F('organization__path'), valid_root=True)
    )
    suspended_ids = InstanceMemberAssignment.objects.filter(user=user, suspended_at__isnull=False).values('instance_config_id')
    return Q(Exists(grants)) & ~Q(pk__in=suspended_ids)


def user_has_instance_grant(user: User, instance: InstanceConfig, *, action: OrganizationAction) -> bool:
    from nodes.models import InstanceMemberAssignment

    if InstanceMemberAssignment.objects.filter(instance_config=instance, user=user, suspended_at__isnull=False).exists():
        return False
    if not instance.has_framework_config():
        return False
    roles = ObjectRole.get_roles_for_action(action)
    if action == 'delete' or not roles:
        return False
    framework = instance.framework_config.framework
    grants = OrganizationAccessGrant.objects.filter(framework=framework, user=user, role__in=roles, suspended_at__isnull=True)
    return any(
        organization_is_in_framework(framework, grant.organization)
        and instance.organization.path.startswith(grant.organization.path)
        for grant in grants.select_related('organization')
    )


def user_is_organization_admin(user: User, instance: InstanceConfig) -> bool:
    from nodes.models import InstanceMemberAssignment

    if InstanceMemberAssignment.objects.filter(instance_config=instance, user=user, suspended_at__isnull=False).exists():
        return False
    if not instance.has_framework_config():
        return False
    framework = instance.framework_config.framework
    grants = OrganizationAccessGrant.objects.filter(
        framework=framework, user=user, role=ObjectRole.ADMIN, suspended_at__isnull=True
    ).select_related('organization')
    return any(
        organization_is_in_framework(framework, grant.organization)
        and instance.organization.path.startswith(grant.organization.path)
        for grant in grants
    )


def user_can_access_organization(
    framework: Framework,
    user: User,
    organization: Organization,
    *,
    action: OrganizationAction = 'view',
) -> bool:
    return accessible_organizations(framework, user, action=action).filter(pk=organization.pk).exists()
