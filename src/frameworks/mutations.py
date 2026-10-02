"""Strawberry mutations for framework instance management (CADS self-service)."""

from __future__ import annotations

import enum
from datetime import date, datetime
from uuid import UUID

import strawberry as sb
from django.db import transaction
from django.utils import timezone
from graphql import GraphQLError

from kausal_common.people.models import ObjectRole
from kausal_common.strawberry.registry import register_strawberry_type

from paths import gql

from frameworks.bisko.activation import ActivationError, activate_bisko_municipality
from frameworks.models import (
    Framework,
    FrameworkConfig,
    OrganizationAccessGrant,
    OrganizationAccessGrantEvent,
)
from frameworks.organization_access import organization_is_in_framework, user_can_access_organization
from frameworks.roles import framework_admin_role
from nodes.membership import retention_date
from nodes.models import InstanceConfig
from orgs.models import Organization
from users.models import User


@sb.input
class CreateInstanceInput:
    framework_id: str
    name: str
    identifier: str
    organization_name: str


@sb.type
class CreateInstanceResult:
    instance: sb.Private[InstanceConfig]

    @sb.field
    def instance_id(self) -> sb.ID:
        return sb.ID(str(self.instance.identifier))

    @sb.field
    def instance_name(self) -> str:
        return self.instance.get_name()


@sb.type
class ActivateOrganizationResult:
    organization_id: sb.ID
    framework_config_id: sb.ID
    instance_identifier: str
    created: bool


@sb.enum(name='OrganizationAccessRole')
class OrganizationAccessRole(enum.Enum):
    VIEWER = 'viewer'
    EDITOR = 'editor'
    ADMIN = 'admin'


@register_strawberry_type
@sb.type(name='OrganizationAccessGrant')
class OrganizationAccessGrantType:
    user_id: sb.ID
    user_email: str
    organization_id: sb.ID
    role: OrganizationAccessRole
    suspended_at: datetime | None
    retention_until: date | None

    @classmethod
    def from_model(cls, grant: OrganizationAccessGrant) -> OrganizationAccessGrantType:
        return cls(
            user_id=sb.ID(str(grant.user.uuid)),
            user_email=grant.user.email,
            organization_id=sb.ID(str(grant.organization.uuid)),
            role=OrganizationAccessRole(grant.role),
            suspended_at=grant.suspended_at,
            retention_until=grant.retention_until,
        )


def _get_authenticated_user(info: gql.Info) -> User:
    user = info.context.user
    if user is None or not user.is_authenticated:
        raise GraphQLError('Authentication required')
    assert isinstance(user, User)
    return user


def _get_framework(framework_id: str) -> Framework:
    try:
        return Framework.objects.get(identifier=framework_id)
    except Framework.DoesNotExist:
        raise GraphQLError(f'Framework "{framework_id}" not found') from None


def _organization_grant_context(
    info: gql.Info, framework_id: sb.ID, organization_id: sb.ID
) -> tuple[User, Framework, Organization]:
    actor = _get_authenticated_user(info)
    framework = _get_framework(str(framework_id))
    try:
        organization = Organization.objects.get(uuid=UUID(str(organization_id)))
    except Organization.DoesNotExist, ValueError:
        raise GraphQLError('Organization not found') from None
    if not organization_is_in_framework(framework, organization):
        raise GraphQLError('Organization is outside this framework')
    if actor.is_superuser or actor.has_instance_role(framework_admin_role, framework):
        return actor, framework, organization
    admin_grants = OrganizationAccessGrant.objects.filter(
        framework=framework, user=actor, role=ObjectRole.ADMIN, suspended_at__isnull=True
    )
    if not any(organization.path.startswith(path) for path in admin_grants.values_list('organization__path', flat=True)):
        raise GraphQLError('Permission denied for organization access management')
    return actor, framework, organization


def _grant_for_user(
    info: gql.Info, framework: Framework, organization: Organization, user_id: sb.ID
) -> tuple[User, OrganizationAccessGrant | None]:
    try:
        user = User.objects.get(uuid=UUID(str(user_id)))
    except User.DoesNotExist, ValueError:
        raise GraphQLError('User not found') from None
    return user, OrganizationAccessGrant.objects.filter(framework=framework, organization=organization, user=user).first()


@sb.type
class FrameworkMutation:
    @gql.mutation(description='Activate a BISKO municipality with a local instance pinned to the published template.')
    @staticmethod
    def activate_framework_organization(
        info: gql.Info, framework_id: sb.ID, organization_id: sb.ID
    ) -> ActivateOrganizationResult:
        actor = _get_authenticated_user(info)
        framework = _get_framework(str(framework_id))
        try:
            organization = Organization.objects.get(uuid=UUID(str(organization_id)))
        except Organization.DoesNotExist, ValueError:
            raise GraphQLError('Organization not found') from None
        if not user_can_access_organization(framework, actor, organization, action='change'):
            raise GraphQLError('Permission denied for organization activation')
        try:
            config, created = activate_bisko_municipality(framework, organization, actor=actor)
        except ActivationError as error:
            raise GraphQLError(str(error)) from error
        return ActivateOrganizationResult(
            organization_id=sb.ID(str(organization.uuid)),
            framework_config_id=sb.ID(str(config.uuid)),
            instance_identifier=config.instance_config.identifier,
            created=created,
        )

    @gql.mutation(description='Assign or update an organization subtree role for an existing user.')
    @staticmethod
    def assign_organization_role(
        info: gql.Info,
        framework_id: sb.ID,
        organization_id: sb.ID,
        user_id: sb.ID,
        role: OrganizationAccessRole,
    ) -> OrganizationAccessGrantType:
        actor, framework, organization = _organization_grant_context(info, framework_id, organization_id)
        user, _ = _grant_for_user(info, framework, organization, user_id)
        with transaction.atomic():
            grant, created = OrganizationAccessGrant.objects.select_for_update().get_or_create(
                framework=framework,
                organization=organization,
                user=user,
                defaults={'role': role.value, 'created_by': actor, 'last_modified_by': actor},
            )
            old_role = None if created else grant.role
            if not created:
                grant.role = role.value
                grant.last_modified_by = actor
                grant.save(update_fields=['role', 'last_modified_by', 'last_modified_at'])
            if created or old_role != role.value:
                OrganizationAccessGrantEvent.objects.create(
                    grant=grant,
                    action='assigned' if created else 'role_changed',
                    old_role=old_role,
                    new_role=role.value,
                    changed_by=actor,
                )
        return OrganizationAccessGrantType.from_model(grant)

    @gql.mutation(description='Suspend an organization grant while retaining its history and attribution.')
    @staticmethod
    def suspend_organization_role(
        info: gql.Info, framework_id: sb.ID, organization_id: sb.ID, user_id: sb.ID
    ) -> OrganizationAccessGrantType:
        actor, framework, organization = _organization_grant_context(info, framework_id, organization_id)
        user, grant = _grant_for_user(info, framework, organization, user_id)
        if actor.pk == user.pk:
            raise GraphQLError('Cannot suspend your own organization grant')
        if grant is None or grant.suspended_at is not None:
            raise GraphQLError('Active organization grant not found')
        with transaction.atomic():
            grant = OrganizationAccessGrant.objects.select_for_update().get(pk=grant.pk)
            now = timezone.now()
            grant.suspended_at = now
            grant.retention_until = retention_date(now.date())
            grant.last_modified_by = actor
            grant.save(update_fields=['suspended_at', 'retention_until', 'last_modified_by', 'last_modified_at'])
            OrganizationAccessGrantEvent.objects.create(
                grant=grant,
                action='suspended',
                old_role=grant.role,
                new_role=grant.role,
                changed_by=actor,
                suspended_at=grant.suspended_at,
                retention_until=grant.retention_until,
            )
        return OrganizationAccessGrantType.from_model(grant)

    @gql.mutation(description='Reactivate a suspended organization grant.')
    @staticmethod
    def reactivate_organization_role(
        info: gql.Info, framework_id: sb.ID, organization_id: sb.ID, user_id: sb.ID
    ) -> OrganizationAccessGrantType:
        actor, framework, organization = _organization_grant_context(info, framework_id, organization_id)
        _, grant = _grant_for_user(info, framework, organization, user_id)
        if grant is None or grant.suspended_at is None:
            raise GraphQLError('Suspended organization grant not found')
        with transaction.atomic():
            grant = OrganizationAccessGrant.objects.select_for_update().get(pk=grant.pk)
            prior_suspended_at = grant.suspended_at
            prior_retention_until = grant.retention_until
            grant.suspended_at = None
            grant.retention_until = None
            grant.last_modified_by = actor
            grant.save(update_fields=['suspended_at', 'retention_until', 'last_modified_by', 'last_modified_at'])
            OrganizationAccessGrantEvent.objects.create(
                grant=grant,
                action='reactivated',
                old_role=grant.role,
                new_role=grant.role,
                changed_by=actor,
                suspended_at=prior_suspended_at,
                retention_until=prior_retention_until,
            )
        return OrganizationAccessGrantType.from_model(grant)

    @gql.mutation(description='Create a new model instance under a framework, cloning from the framework template')
    @staticmethod
    def create_instance(info: gql.Info, input: CreateInstanceInput) -> CreateInstanceResult:
        user = _get_authenticated_user(info)
        fw = _get_framework(input.framework_id)

        if not fw.allow_instance_creation:
            raise GraphQLError(f'Instance creation is not allowed for framework "{fw.identifier}"')

        if InstanceConfig.objects.filter(identifier=input.identifier).exists():
            raise GraphQLError(f'Instance with identifier "{input.identifier}" already exists')

        # Export template from the framework's configured template instance
        template_export = None
        if fw.template_instance is not None:
            from nodes.instance_serialization import export_instance

            template_export = export_instance(fw.template_instance)

        with transaction.atomic():
            from orgs.models import Organization

            org = Organization.objects.filter(name=input.organization_name).first()
            if org is None:
                org = Organization.add_root(name=input.organization_name)
            ic = InstanceConfig.objects.create(
                name=input.name,
                identifier=input.identifier,
                primary_language='en',
                other_languages=[],
                organization=org,
                config_source='database',
                created_by=user,
                last_modified_by=user,
                owned_by=user,
            )
            fwc = FrameworkConfig.objects.create(
                framework=fw,
                instance_config=ic,
                organization_name=input.organization_name,
                uuid=ic.uuid,
                created_by=user,
                last_modified_by=user,
            )
            ic.owned_by = user
            ic.save(update_fields=['owned_by'])

            pp = ic.permission_policy()
            pp.admin_role.assign_user(ic, user)

            if template_export is not None:
                from nodes.instance_serialization import import_instance

                import_instance(ic, export=template_export, framework_config=fwc)
            else:
                spec = ic.spec
                assert spec is not None
                reference_year = fw.defaults.baseline_year.default or fw.defaults.baseline_year.min or 2020
                target_year = fw.defaults.target_year.default or fw.defaults.target_year.min
                spec.years.reference = reference_year
                spec.years.min_historical = reference_year
                spec.years.max_historical = reference_year
                spec.years.target = target_year
                # Owner display name lives on the column now.
                ic.owner = fwc.organization_name or ''
                ic.save(update_fields=['spec', 'owner'])

            spec = ic.spec
            assert spec is not None
            if ic.template_revision_id is not None:
                spec.features.enable_user_management = fw.enable_user_management
                ic.spec = spec
            else:
                features = spec.features.model_copy(update={'enable_user_management': fw.enable_user_management})
                ic.spec = spec.model_copy(update={'features': features})
            ic.save(update_fields=['spec'])

            ic.refresh_from_db()
            ic.create_default_content()
            fwc.setup_instance_pages()

        return CreateInstanceResult(instance=ic)
