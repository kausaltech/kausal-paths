"""
Create or update every instance's role groups and their permissions.

    python manage.py update_instance_role_groups --dry-run      # report only, roll back
    python manage.py update_instance_role_groups                # all instances
    python manage.py update_instance_role_groups INSTANCE_ID ...

The groups are normally brought up to date only when an instance's default content is created or
a user logs in through the SSO pipeline, so a change to what a role grants does not reach existing
groups by itself. Running this re-applies the current role definitions to all of them. It is
idempotent: a group that already has the right permissions is not touched.
"""

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from wagtail.models import GroupPagePermission

from nodes.models import InstanceConfig
from nodes.signals import get_instance_config_role_group_fields

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from django.db.models import QuerySet


def _page_perm_rows(ic: InstanceConfig, group_fields: list[str]) -> set[tuple[int, int, int]]:
    group_ids = [getattr(ic, field) for field in group_fields]
    perms = GroupPagePermission.objects.filter(group_id__in=[gid for gid in group_ids if gid is not None])
    return set(perms.values_list('group_id', 'page_id', 'permission_id'))


class Command(BaseCommand):
    help = "Create or update instances' role groups, including their Wagtail page permissions."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='*', metavar='INSTANCE_ID', help='Instances to update (default: all).')
        parser.add_argument('--dry-run', action='store_true', help='Report what would change and roll back.')

    def handle(self, *args: Any, **options: Any) -> None:
        instances = self._resolve_instances(options['instances'])
        dry_run: bool = options['dry_run']

        group_fields = get_instance_config_role_group_fields()
        changed = 0
        with transaction.atomic():
            for ic in instances:
                before = _page_perm_rows(ic, group_fields)
                ic.create_or_update_instance_groups()
                after = _page_perm_rows(ic, group_fields)
                if before == after:
                    continue
                changed += 1
                self.stdout.write(f'{ic.identifier}: page permission rows +{len(after - before)} -{len(before - after)}')
            if dry_run:
                transaction.set_rollback(True)

        summary = f'{changed} of {len(instances)} instances had page permission changes.'
        if dry_run:
            self.stdout.write(self.style.WARNING(f'{summary} Dry run, nothing written.'))
        else:
            self.stdout.write(self.style.SUCCESS(summary))

    def _resolve_instances(self, identifiers: list[str]) -> list[InstanceConfig]:
        qs: QuerySet[InstanceConfig] = InstanceConfig.objects.order_by('identifier')
        if not identifiers:
            return list(qs)
        instances = list(qs.filter(identifier__in=identifiers))
        missing = set(identifiers) - {ic.identifier for ic in instances}
        if missing:
            raise CommandError(f'No such instances: {", ".join(sorted(missing))}')
        return instances
