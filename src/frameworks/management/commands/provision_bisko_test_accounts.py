"""Create password-login accounts for testing BISKO organization access."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from kausal_common.people.models import ObjectRole

from frameworks.models import Framework, OrganizationAccessGrant
from frameworks.organization_access import organization_is_in_framework
from orgs.models import Organization, OrganizationIdentifier
from users.base import uuid_to_username
from users.models import User

if TYPE_CHECKING:
    from argparse import ArgumentParser


STATES = ('03', '07', '12')
ACCOUNT_SECRET_ENV = 'BISKO_TEST_ACCOUNT_PASSWORD'  # noqa: S105 -- This names an environment variable, not a secret.


@dataclass(frozen=True)
class AccountSpec:
    email: str
    ars: str
    organization: Organization
    role: ObjectRole


def _organization_by_ars(ars: str) -> Organization:
    try:
        return (
            OrganizationIdentifier.objects
            .select_related('organization', 'organization__classification')
            .get(namespace__identifier='ars', identifier=ars)
            .organization
        )
    except OrganizationIdentifier.DoesNotExist as error:
        raise CommandError(f'No organization with ARS {ars}; import the BKG tree first.') from error


def _is_classified_as(organization: Organization, identifier: str) -> bool:
    classification = organization.classification
    return classification is not None and classification.identifier == identifier


def _select_district_and_municipality(
    state: Organization, state_ars: str
) -> tuple[tuple[str, Organization], tuple[str, Organization]]:
    identifiers = list(
        OrganizationIdentifier.objects
        .filter(namespace__identifier='ars', organization__path__startswith=state.path)
        .select_related('organization', 'organization__classification')
        .order_by('identifier')
    )
    districts = [
        ident for ident in identifiers if len(ident.identifier) == 5 and _is_classified_as(ident.organization, 'de_district')
    ]
    municipalities = [
        ident for ident in identifiers if len(ident.identifier) == 12 and _is_classified_as(ident.organization, 'de_municipality')
    ]
    for district in districts:
        for municipality in municipalities:
            if municipality.organization.path.startswith(district.organization.path):
                return (district.identifier, district.organization), (municipality.identifier, municipality.organization)
    raise CommandError(f'No Landkreis with a municipality beneath state {state_ars}; import the BKG tree first.')


def _account_specs(framework: Framework) -> list[AccountSpec]:
    specs: list[AccountSpec] = []
    for state_ars in STATES:
        state = _organization_by_ars(state_ars)
        if not _is_classified_as(state, 'de_state') or not organization_is_in_framework(framework, state):
            raise CommandError(f'State {state_ars} is not an attached BISKO state root.')
        district, municipality = _select_district_and_municipality(state, state_ars)
        for level, ars, organization, role in (
            ('state-admin', state_ars, state, ObjectRole.ADMIN),
            ('state-editor', state_ars, state, ObjectRole.EDITOR),
            ('district-editor', district[0], district[1], ObjectRole.EDITOR),
            ('municipality-editor', municipality[0], municipality[1], ObjectRole.EDITOR),
        ):
            specs.append(
                AccountSpec(
                    email=f'bisko-{state_ars}-{level}.fake@kausal.tech',
                    ars=ars,
                    organization=organization,
                    role=role,
                )
            )
    return specs


def _provision_one(framework: Framework, spec: AccountSpec, password: str) -> str:
    user = User.objects.select_for_update().filter(email__iexact=spec.email).first()
    if user is None:
        user_uuid = uuid4()
        user = User.objects.create_user(username=uuid_to_username(user_uuid), uuid=user_uuid, email=spec.email, password=password)
        status = 'created'
    else:
        if not user.is_active:
            raise CommandError(f'{spec.email} is inactive; reactivate it explicitly before rerunning.')
        user.set_password(password)
        user.save(update_fields=['password'])
        status = 'existing'

    grant, created = OrganizationAccessGrant.objects.get_or_create(
        framework=framework,
        organization=spec.organization,
        user=user,
        defaults={'role': spec.role},
    )
    if not created and (grant.role != spec.role or grant.suspended_at is not None):
        raise CommandError(f'{spec.email} has a changed or suspended grant; resolve it explicitly before rerunning.')
    other_grants = OrganizationAccessGrant.objects.filter(framework=framework, user=user).exclude(pk=grant.pk)
    if other_grants.exists():
        raise CommandError(f'{spec.email} has another BISKO grant; resolve it explicitly before rerunning.')
    return f'{status}: {spec.email} | username={user.username} | {spec.role} | {spec.ars} {spec.organization.name}'


class Command(BaseCommand):
    help = 'Create 12 password-login BISKO test accounts and grant state, Landkreis, and municipality access.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            '--password-env',
            default=ACCOUNT_SECRET_ENV,
            help=f'Environment variable containing the password (default: {ACCOUNT_SECRET_ENV}).',
        )
        parser.add_argument('--dry-run', action='store_true', help='Show the account plan without writing to the database.')

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            framework = Framework.objects.get(identifier='bisko')
        except Framework.DoesNotExist as error:
            raise CommandError('BISKO is not provisioned; run python -m tools.setup_bisko first.') from error
        specs = _account_specs(framework)
        if options['dry_run']:
            for spec in specs:
                self.stdout.write(f'{spec.email} | {spec.role} | {spec.ars} {spec.organization.name}')
            self.stdout.write('Dry run: no accounts or grants created.')
            return

        password = os.environ.get(options['password_env'])
        if not password:
            raise CommandError(f'Set {options["password_env"]} to the password for the 12 test accounts.')

        with transaction.atomic():
            results = [_provision_one(framework, spec, password) for spec in specs]
        for result in results:
            self.stdout.write(result)
