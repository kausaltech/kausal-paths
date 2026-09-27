"""Assign a framework organization subtree grant to an existing user."""

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from kausal_common.people.models import ObjectRole

from frameworks.models import Framework, OrganizationAccessGrant
from frameworks.organization_access import organization_is_in_framework
from orgs.models import OrganizationIdentifier
from users.models import User

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Grant an existing user access to an administrative subtree within a framework.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('username')
        parser.add_argument('ars', help='Official ARS of the grant root')
        parser.add_argument('--framework', default='bisko')
        parser.add_argument('--role', choices=ObjectRole.values, default=ObjectRole.VIEWER)
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args: Any, **options: Any) -> None:
        try:
            user = User.objects.get(username=options['username'])
            framework = Framework.objects.get(identifier=options['framework'])
            organization = OrganizationIdentifier.objects.get(namespace__identifier='ars', identifier=options['ars']).organization
        except (User.DoesNotExist, Framework.DoesNotExist, OrganizationIdentifier.DoesNotExist) as error:
            raise CommandError(f'Unknown user, framework, or ARS: {error}') from error
        if not organization_is_in_framework(framework, organization):
            raise CommandError(f'ARS {options["ars"]} is outside framework {framework.identifier}')
        with transaction.atomic():
            grant, created = OrganizationAccessGrant.objects.update_or_create(
                framework=framework,
                organization=organization,
                user=user,
                defaults={'role': options['role']},
            )
            self.stdout.write(f'{"Created" if created else "Updated"} {grant}')
            if options['dry_run']:
                transaction.set_rollback(True)
                self.stdout.write('Dry run: database changes rolled back.')
