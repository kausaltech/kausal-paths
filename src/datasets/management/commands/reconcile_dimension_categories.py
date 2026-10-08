"""
Bring an instance's stored dimension categories in line with what its model declares.

    python manage.py reconcile_dimension_categories mainz-bisko                  # plan only
    python manage.py reconcile_dimension_categories mainz-bisko augsburg-bisko --dimension sector --apply

A category the model no longer declares is merged into the declared category that lists its id
as an alias, or else deleted; see ``datasets.category_reconcile`` for what refuses a change.
Nothing is written without ``--apply``, and a refusal stops the whole set.

**Run it right after ``sync_instance_to_db``.** The plan compares the stored rows with the
model the instance loads, which for a database-sourced instance is its stored spec -- so until
the sync, a category removed from ``configs/`` is still declared and nothing is planned. And
between the sync and this command, data in database datasets still carries the old id, which no
alias resolves, so a node reading it fails with ``Unknown categories in dimension column``.
Published revisions are unaffected throughout: they compute from frozen dataset revisions.
"""

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from datasets.category_reconcile import CategoryChange, apply_changes, plan_instance
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Merge or delete stored dimension categories that the model no longer declares'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='+', metavar='INSTANCE')
        parser.add_argument('--dimension', action='append', help='Only this dimension (repeatable)')
        parser.add_argument('--apply', action='store_true', help='Write the changes; without it, only the plan is shown')

    def handle(self, *args: Any, **options: Any) -> None:
        instances = list(InstanceConfig.objects.filter(identifier__in=options['instances']))
        missing = set(options['instances']) - {ic.identifier for ic in instances}
        if missing:
            raise CommandError(f'Unknown instances: {", ".join(sorted(missing))}')

        plans: list[tuple[InstanceConfig, list[CategoryChange]]] = []
        for instance in instances:
            changes = plan_instance(instance, options['dimension'])
            plans.append((instance, changes))
            self.stdout.write(f'{instance.identifier}: {len(changes) or "nothing"} to change')
            for change in changes:
                self.stdout.write(f'  {change.describe()}')

        if any(change.refusals for _, changes in plans for change in changes):
            raise CommandError('Refused; nothing was written.')
        if not options['apply']:
            self.stdout.write('Plan only; pass --apply to write it.')
            return
        with transaction.atomic():
            for instance, changes in plans:
                if not changes:
                    continue
                touched = apply_changes(changes)
                names = ', '.join(sorted(str(ds.identifier) for ds in touched)) or 'none'
                self.stdout.write(f'{instance.identifier}: applied; data points moved in {names}')
        for instance, changes in plans:
            if changes:
                instance.invalidate_cache()
