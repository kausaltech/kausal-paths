"""Strawberry mutations for framework instance management (CADS self-service)."""

from __future__ import annotations

import strawberry as sb
from django.db import transaction
from graphql import GraphQLError

from paths import gql

from frameworks.models import Framework, FrameworkConfig
from nodes.models import InstanceConfig
from users.models import User


@sb.input
class CreateInstanceInput:
    framework_id: str
    name: str
    identifier: str
    organization_name: str


@sb.input
class RegisterUserInput:
    framework_id: str
    email: str
    password: str
    first_name: str | None = None
    last_name: str | None = None


@sb.type
class RegisterUserResult:
    user_id: sb.ID
    email: str


@sb.type
class CreateInstanceResult:
    instance: sb.Private[InstanceConfig]

    @sb.field
    def instance_id(self) -> sb.ID:
        return sb.ID(str(self.instance.identifier))

    @sb.field
    def instance_name(self) -> str:
        return self.instance.get_name()


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


@sb.type
class FrameworkMutation:
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

            from nodes.models import make_empty_instance_spec

            ic = InstanceConfig.objects.create(
                name=input.name,
                identifier=input.identifier,
                primary_language='en',
                other_languages=[],
                organization=org,
                config_source='database',
                spec=make_empty_instance_spec(),
            )

            FrameworkConfig.objects.create(
                framework=fw,
                instance_config=ic,
                organization_name=input.organization_name,
                baseline_year=fw.defaults.baseline_year.default or fw.defaults.baseline_year.min or 2020,
                target_year=fw.defaults.target_year.default or fw.defaults.target_year.min,
            )

            ic.create_or_update_instance_groups()

            pp = ic.permission_policy()
            pp.admin_role.assign_user(ic, user)

            if template_export is not None:
                from nodes.instance_serialization import import_instance

                import_instance(ic, template_export)

        return CreateInstanceResult(instance=ic)

    @gql.mutation(description='Register a new user with email and password')
    @staticmethod
    def register_user(info: gql.Info, input: RegisterUserInput) -> RegisterUserResult:
        fw = _get_framework(input.framework_id)

        if not fw.allow_user_registration:
            raise GraphQLError(f'User registration is not allowed for framework "{fw.identifier}"')

        if User.objects.filter(email__iexact=input.email).exists():
            raise GraphQLError('A user with this email already exists')

        from django.contrib.auth.password_validation import validate_password
        from django.core.exceptions import ValidationError

        try:
            validate_password(input.password)
        except ValidationError as e:
            raise GraphQLError(f'Invalid password: {"; ".join(e.messages)}') from None

        from uuid import uuid4

        from users.base import uuid_to_username

        user_uuid = uuid4()
        user = User.objects.create_user(
            username=uuid_to_username(user_uuid),
            email=input.email,
            password=input.password,
            first_name=input.first_name or '',
            last_name=input.last_name or '',
            uuid=user_uuid,
            is_staff=False,
            is_active=True,
        )
        return RegisterUserResult(user_id=sb.ID(str(user.uuid)), email=user.email)
