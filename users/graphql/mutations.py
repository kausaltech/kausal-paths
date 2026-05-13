"""
Strawberry mutations for user management.

Top-level mutations exposed:

- ``registerUser``: anonymous registration. Accepts an optional
  ``invitationToken`` to bypass framework-level ``allow_user_registration``
  and auto-grant instance admin role on the inviting instance.
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
from nodes.models import InstanceConfig, InstanceInvitation
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
    spec = ic.spec
    if spec is None or not spec.features.enable_user_management:
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
            'Add an existing user to the instance as admin. Returns `UserNotFoundError` '
            'if no user has the given email — the UI may then offer to send an invitation.'
        ),
        graphql_type=AddUserToInstancePayload,
    )
    @staticmethod
    def add_user_to_instance(
        info: gql.Info,
        root: sb.Parent['InstanceUserMutation'],
        email: str,
    ) -> User | UserNotFoundError:
        ic = root.instance
        _require_user_management_enabled(info, ic)
        normalized = email.strip().lower()
        user = User.objects.filter(email__iexact=normalized).first()
        if user is None:
            return UserNotFoundError(email=normalized)
        ic.permission_policy().admin_role.assign_user(ic, user)
        user.invalidate_adminable_instances_cache()
        return user

    @gql.mutation(
        description='Invite a user (by email) to administer this instance. Sends an email with a single-use token.',
    )
    @staticmethod
    def invite_user_to_instance(
        info: gql.Info,
        root: sb.Parent['InstanceUserMutation'],
        email: str,
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
        inv = InstanceInvitation(
            instance_config=ic,
            email=normalized,
            created_by=actor,
            last_modified_by=actor,
        )
        inv.save()
        send_instance_invitation(inv)
        return inv  # type: ignore[return-value]

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

        pp = ic.permission_policy()
        pp.admin_role.unassign_user(ic, target)
        pp.super_admin_role.unassign_user(ic, target)
        pp.viewer_role.unassign_user(ic, target)
        pp.reviewer_role.unassign_user(ic, target)

        User.objects.filter(pk=target.pk, selected_instance=ic).update(selected_instance=None)

        target.invalidate_adminable_instances_cache()

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
            invitation.instance_config.permission_policy().admin_role.assign_user(invitation.instance_config, user)
            invitation.mark_accepted(user)
            user.invalidate_adminable_instances_cache()

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
