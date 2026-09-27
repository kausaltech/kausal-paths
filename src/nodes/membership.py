"""Municipal account roles, seat reservations, and suspension lifecycle."""

import calendar
from datetime import date, datetime
from typing import TYPE_CHECKING

from django.db import transaction
from django.utils import timezone

from nodes.models import InstanceInvitation, InstanceMemberAssignment, InstanceMemberEvent, InstanceMemberRole
from nodes.roles import instance_admin_role, instance_editor_role, instance_reviewer_role, instance_viewer_role

if TYPE_CHECKING:
    from nodes.models import InstanceConfig
    from users.models import User


class MembershipError(ValueError):
    pass


MEMBER_ROLES = {
    InstanceMemberRole.ADMIN: instance_admin_role,
    InstanceMemberRole.EDITOR: instance_editor_role,
    InstanceMemberRole.REVIEWER: instance_reviewer_role,
    InstanceMemberRole.VIEWER: instance_viewer_role,
}


def current_group_role(instance: InstanceConfig, user: User) -> InstanceMemberRole | None:
    for role_name, role in MEMBER_ROLES.items():
        if user.has_instance_role(role, instance):
            return role_name
    return None


def retention_date(start: date) -> date:
    year = start.year + 5
    return date(year, start.month, min(start.day, calendar.monthrange(year, start.month)[1]))


def _active_account_ids(instance: InstanceConfig) -> set[int]:
    ids: set[int] = set(
        InstanceMemberAssignment.objects.filter(instance_config=instance, suspended_at__isnull=True).values_list(
            'user_id', flat=True
        )
    )
    for role in MEMBER_ROLES.values():
        group = role.get_existing_instance_group(instance)
        if group is not None:
            ids.update(group.user_set.values_list('pk', flat=True))
    suspended = InstanceMemberAssignment.objects.filter(instance_config=instance, suspended_at__isnull=False).values_list(
        'user_id', flat=True
    )
    ids.difference_update(suspended)
    operator_group = instance.permission_policy().super_admin_role.get_existing_instance_group(instance)
    if operator_group is not None:
        ids.difference_update(operator_group.user_set.values_list('pk', flat=True))
    from users.models import User

    ids.difference_update(User.objects.filter(pk__in=ids, is_superuser=True).values_list('pk', flat=True))
    return ids


def seats_in_use(instance: InstanceConfig) -> int:
    return (
        len(_active_account_ids(instance))
        + InstanceInvitation.objects.filter(
            instance_config=instance, accepted_at__isnull=True, expires_at__gt=timezone.now()
        ).count()
    )


def require_free_seat(instance: InstanceConfig, *, user: User | None = None) -> None:
    if user is not None and user.pk in _active_account_ids(instance):
        return
    if not instance.has_framework_config():
        return
    limit = instance.framework_config.framework.max_user_accounts_per_instance
    if limit is not None and seats_in_use(instance) >= limit:
        raise MembershipError(f'This licence has reached its limit of {limit} active accounts and invitations')


def _replace_group_role(instance: InstanceConfig, user: User, role: InstanceMemberRole | None) -> None:
    for role_obj in MEMBER_ROLES.values():
        role_obj.unassign_user(instance, user)
    if role is not None:
        MEMBER_ROLES[role].assign_user(instance, user)
    user.invalidate_adminable_instances_cache()


def _record(
    assignment: InstanceMemberAssignment,
    action: str,
    old_role: InstanceMemberRole | None,
    actor: User | None,
    *,
    suspended_at: datetime | None = None,
    retention_until: date | None = None,
) -> None:
    InstanceMemberEvent.objects.create(
        assignment=assignment,
        action=action,
        old_role=old_role,
        new_role=assignment.role,
        changed_by=actor,
        suspended_at=suspended_at if suspended_at is not None else assignment.suspended_at,
        retention_until=retention_until if retention_until is not None else assignment.retention_until,
    )


@transaction.atomic
def set_member_role(
    instance: InstanceConfig,
    user: User,
    role: InstanceMemberRole,
    *,
    actor: User | None,
    require_existing: bool = False,
) -> InstanceMemberAssignment:
    if role not in MEMBER_ROLES:
        raise MembershipError('Invalid municipal role')
    locked = type(instance).objects.select_for_update().get(pk=instance.pk)
    if user.has_instance_role(locked.permission_policy().super_admin_role, locked):
        raise MembershipError('Operator roles are managed separately')
    if locked.owned_by_id == user.pk and role != InstanceMemberRole.ADMIN:
        raise MembershipError('The instance owner must remain an admin')
    assignment = InstanceMemberAssignment.objects.select_for_update().filter(instance_config=locked, user=user).first()
    legacy_role = current_group_role(locked, user)
    if require_existing and assignment is None and legacy_role is None:
        raise MembershipError('User is not a member of this instance')
    if assignment is None:
        if legacy_role is None:
            require_free_seat(locked, user=user)
        assignment = InstanceMemberAssignment.objects.create(
            instance_config=locked, user=user, role=role, created_by=actor, last_modified_by=actor
        )
        action = 'added'
        old_role = legacy_role
    else:
        old_role = InstanceMemberRole(assignment.role)
        if assignment.suspended_at is not None:
            assignment.role = role
            assignment.last_modified_by = actor
            assignment.save(update_fields=['role', 'last_modified_by', 'last_modified_at'])
            if old_role != role:
                _record(assignment, 'role_changed', old_role, actor)
            return assignment
        assignment.role = role
        assignment.last_modified_by = actor
        assignment.save(update_fields=['role', 'last_modified_by', 'last_modified_at'])
        action = 'role_changed'
    _replace_group_role(locked, user, role)
    if action == 'added' or old_role != role:
        _record(assignment, action, old_role, actor)
    return assignment


@transaction.atomic
def suspend_member(instance: InstanceConfig, user: User, *, actor: User | None) -> InstanceMemberAssignment:
    locked = type(instance).objects.select_for_update().get(pk=instance.pk)
    if locked.owned_by_id == user.pk:
        raise MembershipError('Cannot suspend the instance owner')
    if user.has_instance_role(locked.permission_policy().super_admin_role, locked):
        raise MembershipError('Cannot suspend an operator account')
    assignment = InstanceMemberAssignment.objects.select_for_update().filter(instance_config=locked, user=user).first()
    if assignment is None:
        role = current_group_role(locked, user)
        if role is None:
            raise MembershipError('User is not a member of this instance')
        assignment = InstanceMemberAssignment.objects.create(
            instance_config=locked, user=user, role=role, created_by=actor, last_modified_by=actor
        )
    if assignment.suspended_at is not None:
        raise MembershipError('Account is already suspended')
    now = timezone.now()
    assignment.suspended_at = now
    assignment.retention_until = retention_date(now.date())
    assignment.last_modified_by = actor
    assignment.save(update_fields=['suspended_at', 'retention_until', 'last_modified_by', 'last_modified_at'])
    _replace_group_role(locked, user, None)
    type(user).objects.filter(pk=user.pk, selected_instance=locked).update(selected_instance=None)
    _record(assignment, 'suspended', InstanceMemberRole(assignment.role), actor)
    return assignment


@transaction.atomic
def reactivate_member(instance: InstanceConfig, user: User, *, actor: User | None) -> InstanceMemberAssignment:
    locked = type(instance).objects.select_for_update().get(pk=instance.pk)
    assignment = InstanceMemberAssignment.objects.select_for_update().filter(instance_config=locked, user=user).first()
    if assignment is None or assignment.suspended_at is None:
        raise MembershipError('Account is not suspended')
    require_free_seat(locked, user=user)
    prior_suspended_at = assignment.suspended_at
    prior_retention_until = assignment.retention_until
    assignment.suspended_at = None
    assignment.retention_until = None
    assignment.last_modified_by = actor
    assignment.save(update_fields=['suspended_at', 'retention_until', 'last_modified_by', 'last_modified_at'])
    _replace_group_role(locked, user, InstanceMemberRole(assignment.role))
    _record(
        assignment,
        'reactivated',
        InstanceMemberRole(assignment.role),
        actor,
        suspended_at=prior_suspended_at,
        retention_until=prior_retention_until,
    )
    return assignment
