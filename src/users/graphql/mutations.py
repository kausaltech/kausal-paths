"""
Strawberry mutations for user management.

Top-level mutations exposed:

- ``registerUser``: anonymous registration. Accepts an optional
  ``invitationToken`` to bypass framework-level ``allow_user_registration``
  and grant the role stored on that invitation.
- ``instanceAdmin``: an instance-scoped namespace gated on admin/owner/superuser.
  Children live in :class:`InstanceUserMutation` (and future siblings merged
  via :func:`strawberry.tools.merge_types`).
"""

from typing import TYPE_CHECKING, Annotated
from uuid import UUID, uuid4

import strawberry as sb
import strawberry_django
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Q
from graphql import GraphQLError
from strawberry import auto
from strawberry.tools import merge_types

from kausal_common.strawberry.errors import (
    AuthenticationRequiredError,
    GraphQLValidationError,
    NotFoundError,
    PermissionDeniedError,
)
from kausal_common.users import user_or_none

from paths import gql

from frameworks.models import Framework
from nodes.graphql.types.instance import InstanceMemberRole
from nodes.membership import MembershipError, reactivate_member, require_free_seat, set_member_role, suspend_member
from nodes.models import InstanceConfig, InstanceInvitation, InstanceMemberAssignment, InstanceMemberRole as MemberRole
from nodes.notifications import send_instance_invitation
from users.base import uuid_to_username
from users.models import User

if TYPE_CHECKING:
    from users.schema import UserType


# ----------------------------------------------------------------------
# Strawberry types for InstanceInvitation / user-not-found error
# ----------------------------------------------------------------------


@strawberry_django.type(InstanceInvitation, name='InstanceInvitation')
class InstanceInvitationType:
    email: auto
    role: InstanceMemberRole
    expires_at: auto
    accepted_at: auto
    created_at: auto
    created_by: Annotated['UserType', sb.lazy('users.schema')] | None

    @strawberry_django.field
    @staticmethod
    def id(root: sb.Parent[InstanceInvitation]) -> sb.ID:
        return sb.ID(str(root.uuid))


@sb.type(
    name='UserNotFoundError',
    description=(
        'Returned by `addUserToInstance` when no user exists for the given email. '
        'The UI is expected to follow up with an `inviteUserToInstance` call.'
    ),
)
class UserNotFoundError:
    email: str


AddUserToInstancePayload = Annotated[
    Annotated['UserType', sb.lazy('users.schema')] | UserNotFoundError,
    sb.union('AddUserToInstancePayload'),
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _require_authenticated_user(info: gql.Info) -> User:
    user = user_or_none(info.context.user)
    if user is None:
        raise AuthenticationRequiredError(info)
    return user


def _resolve_framework(framework_id: str) -> Framework:
    qs = Framework.objects.filter(Q(identifier=framework_id))
    try:
        as_uuid = UUID(framework_id)
    except ValueError, AttributeError:
        pass
    else:
        qs = Framework.objects.filter(Q(identifier=framework_id) | Q(uuid=as_uuid))
    fw = qs.first()
    if fw is None:
        raise GraphQLError(f'Framework "{framework_id}" not found')
    return fw


def _resolve_instance(info: gql.Info, instance_id: str) -> InstanceConfig:
    ic = InstanceConfig.objects.qs.by_all_identifiers(instance_id).first()
    if ic is None:
        raise NotFoundError(info, f'Instance "{instance_id}" not found')
    return ic


def _user_is_admin_for(user: User, ic: InstanceConfig) -> bool:
    if user.is_superuser:
        return True
    if ic.owned_by_id == user.pk:
        return True
    return ic.permission_policy().is_admin(user, ic)


def _user_is_owner_for(user: User, ic: InstanceConfig) -> bool:
    if user.is_superuser:
        return True
    return ic.owned_by_id == user.pk


def _require_user_management_enabled(info: gql.Info, ic: InstanceConfig) -> None:
    if ic.spec is None or not ic.spec.features.enable_user_management:
        raise PermissionDeniedError(
            info,
            f'User management is not enabled for instance "{ic.identifier}"',
            code='user_management_disabled',
        )


def _resolve_user_by_id(info: gql.Info, user_id: str) -> User:
    try:
        user_uuid = UUID(str(user_id))
    except ValueError, AttributeError:
        raise NotFoundError(info, 'User not found') from None
    user = User.objects.filter(uuid=user_uuid).first()
    if user is None:
        raise NotFoundError(info, 'User not found')
    return user


def _municipal_role(info: gql.Info, role: InstanceMemberRole) -> MemberRole:
    if role == InstanceMemberRole.SUPER_ADMIN:
        raise GraphQLValidationError(
            info,
            'Operator access cannot be assigned through municipal member management.',
            code='invalid_role',
        )
    return MemberRole(role.value)


def _resolve_invitation(
    info: gql.Info,
    instance: InstanceConfig,
    invitation_id: str,
) -> InstanceInvitation:
    try:
        inv_uuid = UUID(str(invitation_id))
    except ValueError, AttributeError:
        raise NotFoundError(info, 'Invitation not found') from None
    inv = InstanceInvitation.objects_including_soft_deleted.filter(
        instance_config=instance,
        uuid=inv_uuid,
    ).first()
    if inv is None:
        raise NotFoundError(info, 'Invitation not found')
    return inv


# ----------------------------------------------------------------------
# Inputs / result types
# ----------------------------------------------------------------------


@sb.input
class RegisterUserInput:
    email: str
    password: str
    framework_id: sb.ID | None = None
    invitation_token: str | None = None
    first_name: str | None = None
    last_name: str | None = None


@sb.type
class RegisterUserResult:
    user_id: sb.ID
    email: str


# ----------------------------------------------------------------------
# InstanceUserMutation: instance-admin-gated operations
# ----------------------------------------------------------------------


@sb.type
class InstanceUserMutation:
    instance: sb.Private[InstanceConfig]

    @gql.mutation(
        description=(
            'Add an existing user to the instance with the requested role. Returns `UserNotFoundError` '
            'if no user has the given email — the UI may then offer to send an invitation.'
        ),
        graphql_type=AddUserToInstancePayload,
    )
    @staticmethod
    def add_user_to_instance(
        info: gql.Info,
        root: sb.Parent['InstanceUserMutation'],
        email: str,
        role: InstanceMemberRole = InstanceMemberRole.ADMIN,
    ) -> User | UserNotFoundError:
        ic = root.instance
        _require_user_management_enabled(info, ic)
        normalized = email.strip().lower()
        user = User.objects.filter(email__iexact=normalized).first()
        if user is None:
            return UserNotFoundError(email=normalized)
        if InstanceMemberAssignment.objects.filter(instance_config=ic, user=user, suspended_at__isnull=False).exists():
            raise GraphQLValidationError(info, 'Account is suspended; reactivate it instead.', code='member_suspended')
        try:
            set_member_role(ic, user, _municipal_role(info, role), actor=_require_authenticated_user(info))
        except MembershipError as error:
            raise GraphQLValidationError(info, str(error), code='membership_invalid') from error
        return user

    @gql.mutation(
        description='Invite a user with a chosen role. Sends an email with a single-use token.',
    )
    @staticmethod
    def invite_user_to_instance(
        info: gql.Info,
        root: sb.Parent['InstanceUserMutation'],
        email: str,
        role: InstanceMemberRole = InstanceMemberRole.ADMIN,
    ) -> InstanceInvitationType:
        ic = root.instance
        _require_user_management_enabled(info, ic)
        normalized = email.strip().lower()
        if User.objects.filter(email__iexact=normalized).exists():
            raise GraphQLValidationError(
                info,
                'A user with this email already exists; use addUserToInstance instead.',
                code='user_exists',
            )
        existing = InstanceInvitation.objects.filter(
            instance_config=ic,
            email=normalized,
            accepted_at__isnull=True,
        ).first()
        if existing is not None and existing.is_valid():
            raise GraphQLValidationError(
                info,
                'An active invitation for this email already exists.',
                code='invitation_exists',
            )
        actor = _require_authenticated_user(info)
        try:
            with transaction.atomic():
                locked = InstanceConfig.objects.select_for_update().get(pk=ic.pk)
                require_free_seat(locked)
                inv = InstanceInvitation.objects.create(
                    instance_config=locked,
                    email=normalized,
                    role=_municipal_role(info, role),
                    created_by=actor,
                    last_modified_by=actor,
                )
        except MembershipError as error:
            raise GraphQLValidationError(info, str(error), code='seat_limit') from error
        send_instance_invitation(inv)
        return inv  # type: ignore[return-value]

    @gql.mutation(description='Change an active or suspended member role.')
    @staticmethod
    def change_user_role(
        info: gql.Info,
        root: sb.Parent['InstanceUserMutation'],
        user_id: sb.ID,
        role: InstanceMemberRole,
    ) -> None:
        _require_user_management_enabled(info, root.instance)
        actor = _require_authenticated_user(info)
        target = _resolve_user_by_id(info, str(user_id))
        try:
            set_member_role(root.instance, target, _municipal_role(info, role), actor=actor, require_existing=True)
        except MembershipError as error:
            raise GraphQLValidationError(info, str(error), code='membership_invalid') from error

    @gql.mutation(description='Suspend one account on this instance, preserving attribution and a five-year retention date.')
    @staticmethod
    def suspend_user(info: gql.Info, root: sb.Parent['InstanceUserMutation'], user_id: sb.ID) -> None:
        _require_user_management_enabled(info, root.instance)
        actor = _require_authenticated_user(info)
        target = _resolve_user_by_id(info, str(user_id))
        if actor.pk == target.pk:
            raise GraphQLValidationError(info, 'Cannot suspend your own account.', code='cannot_suspend_self')
        try:
            suspend_member(root.instance, target, actor=actor)
        except MembershipError as error:
            raise GraphQLValidationError(info, str(error), code='membership_invalid') from error

    @gql.mutation(description='Reactivate a suspended account, subject to the licence seat limit.')
    @staticmethod
    def reactivate_user(info: gql.Info, root: sb.Parent['InstanceUserMutation'], user_id: sb.ID) -> None:
        _require_user_management_enabled(info, root.instance)
        actor = _require_authenticated_user(info)
        target = _resolve_user_by_id(info, str(user_id))
        try:
            reactivate_member(root.instance, target, actor=actor)
        except MembershipError as error:
            raise GraphQLValidationError(info, str(error), code='membership_invalid') from error

    @gql.mutation(description='Remove a user from this instance. Only the instance owner or a superuser may call this.')
    @staticmethod
    def remove_user_from_instance(
        info: gql.Info,
        root: sb.Parent['InstanceUserMutation'],
        user_id: sb.ID,
    ) -> None:
        ic = root.instance
        _require_user_management_enabled(info, ic)
        actor = _require_authenticated_user(info)
        if not _user_is_owner_for(actor, ic):
            raise PermissionDeniedError(info, 'Only the instance owner or a superuser can remove users.')

        target = _resolve_user_by_id(info, str(user_id))
        if ic.owned_by_id is not None and ic.owned_by_id == target.pk:
            raise GraphQLValidationError(
                info,
                'Cannot remove the instance owner. Transfer ownership first.',
                code='cannot_remove_owner',
            )

        try:
            suspend_member(ic, target, actor=actor)
        except MembershipError as error:
            raise GraphQLValidationError(info, str(error), code='membership_invalid') from error

    @gql.mutation(description='Revoke an active invitation. The row is kept (soft-deleted) for audit.')
    @staticmethod
    def remove_invitation(
        info: gql.Info,
        root: sb.Parent['InstanceUserMutation'],
        invitation_id: sb.ID,
    ) -> None:
        ic = root.instance
        _require_user_management_enabled(info, ic)
        actor = _require_authenticated_user(info)
        inv = _resolve_invitation(info, ic, str(invitation_id))
        if inv.is_soft_deleted or inv.accepted_at is not None:
            raise GraphQLValidationError(
                info,
                'This invitation is no longer active.',
                code='invitation_inactive',
            )
        inv.soft_delete(actor)


InstanceAdminMutation = merge_types('InstanceAdminMutation', (InstanceUserMutation,))


# ----------------------------------------------------------------------
# Top-level mutation type (registerUser + instanceAdmin namespace)
# ----------------------------------------------------------------------


def _consume_invitation(info: gql.Info, token: str, email: str) -> InstanceInvitation:
    inv = InstanceInvitation.objects.filter(token=token).first()
    if inv is None or not inv.is_valid():
        raise GraphQLValidationError(info, 'Invitation is invalid or has expired.', code='invitation_invalid')
    if inv.email.lower() != email.strip().lower():
        raise GraphQLValidationError(
            info,
            'The email address does not match the invitation.',
            code='invitation_email_mismatch',
        )
    return inv


@sb.type
class UsersMutation:
    @gql.mutation(description='Register a new user. Pass `invitationToken` to redeem an instance invitation.')
    @staticmethod
    def register_user(info: gql.Info, input: RegisterUserInput) -> RegisterUserResult:
        normalized_email = input.email.strip().lower()
        if User.objects.filter(email__iexact=normalized_email).exists():
            raise GraphQLValidationError(info, 'A user with this email already exists.', code='user_exists')

        invitation: InstanceInvitation | None = None
        if input.invitation_token is not None:
            invitation = _consume_invitation(info, input.invitation_token, normalized_email)
        else:
            if input.framework_id is None:
                raise GraphQLValidationError(
                    info,
                    'Either invitationToken or frameworkId must be provided.',
                    code='missing_registration_context',
                )
            fw = _resolve_framework(str(input.framework_id))
            if not fw.allow_user_registration:
                raise PermissionDeniedError(
                    info,
                    f'User registration is not allowed for framework "{fw.identifier}".',
                    code='registration_disabled',
                )

        try:
            validate_password(input.password)
        except ValidationError as e:
            raise GraphQLValidationError(info, f'Invalid password: {"; ".join(e.messages)}', code='invalid_password') from None

        user_uuid = uuid4()
        with transaction.atomic():
            user = User.objects.create_user(
                username=uuid_to_username(user_uuid),
                email=normalized_email,
                password=input.password,
                first_name=input.first_name or '',
                last_name=input.last_name or '',
                uuid=user_uuid,
                is_staff=False,
                is_active=True,
            )

            if invitation is not None:
                locked = InstanceConfig.objects.select_for_update().get(pk=invitation.instance_config_id)
                invitation = InstanceInvitation.objects.select_for_update().get(pk=invitation.pk)
                if not invitation.is_valid():
                    raise GraphQLValidationError(info, 'Invitation is invalid or has expired.', code='invitation_invalid')
                invitation.mark_accepted(user)
                set_member_role(locked, user, MemberRole(invitation.role), actor=invitation.created_by)

        return RegisterUserResult(user_id=sb.ID(str(user.uuid)), email=user.email)

    @sb.field(
        description='Instance-admin namespace for the given instance. Requires admin or owner permissions.',
        graphql_type=InstanceAdminMutation,
    )
    @staticmethod
    def instance_admin(info: gql.Info, instance_id: sb.ID) -> InstanceUserMutation:
        actor = _require_authenticated_user(info)
        ic = _resolve_instance(info, str(instance_id))
        if not _user_is_admin_for(actor, ic):
            raise PermissionDeniedError(info, 'Permission denied for instance admin actions.')
        return InstanceAdminMutation(instance=ic)
