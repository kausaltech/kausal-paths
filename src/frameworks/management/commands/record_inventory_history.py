r"""
Record past inventories already held in an instance's data as final submissions.

A municipality that arrives with earlier balances (a migration, an import) has the
values but no record that those years were ever delivered. This command closes the
gap: every year in which the instance's final-energy dataset holds a value, and
which has no submission at all, is opened, moved into review with the given note
and finalised. A year that already has a submission, open or final, is left alone,
so the command is safe to repeat and never touches work in progress.

    python manage.py record_inventory_history bisko-03404000 \
        --note 'Aus Klimaschutz-Planer übernommen' --apply

Finalising publishes the instance, so every recorded year pins the revision
current when the command ran. The note is the provenance and is kept in each
submission's events; it is required for that reason.
"""

from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from kausal_common.datasets.models import Dataset

from frameworks import submissions as ops
from frameworks.models import SubmissionKind
from nodes.models import InstanceConfig
from users.models import User

if TYPE_CHECKING:
    from argparse import ArgumentParser


def history_years(ic: InstanceConfig, dataset_identifier: str) -> list[int]:
    """Years with a value in the dataset and no inventory submission, within the historical span."""
    dataset = Dataset.objects.for_instance_config(ic).filter(identifier=dataset_identifier).first()
    if dataset is None:
        raise CommandError(f'{ic.identifier}: no dataset {dataset_identifier}')
    valued = set(dataset.data_points.filter(value__isnull=False).values_list('date__year', flat=True))
    submitted = set(ic.submissions.filter(kind=SubmissionKind.INVENTORY).values_list('period_start', flat=True))
    years = ic.ensure_spec().years
    return sorted(
        year
        for year in valued - submitted
        if (years.min_historical is None or year >= years.min_historical)
        and (years.max_historical is None or year <= years.max_historical)
    )


class Command(BaseCommand):
    help = 'Finalise a submission for every past year that has data but no submission; dry run by default.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='+', metavar='INSTANCE')
        parser.add_argument('--note', required=True, help='Where the values came from; kept in every recorded submission.')
        parser.add_argument('--dataset', default='kommune/endenergieverbrauch', help='The dataset whose years count.')
        parser.add_argument('--user', help='Email of the user to record as actor; none by default.')
        parser.add_argument('--apply', action='store_true')

    def handle(self, *args: Any, **options: Any) -> None:
        note = options['note'].strip()
        if not note:
            raise CommandError('--note must say where the values came from')
        user = None
        if options['user']:
            user = User.objects.filter(email__iexact=options['user']).first()
            if user is None:
                raise CommandError(f'No user {options["user"]}')
        instances = {ic.identifier: ic for ic in InstanceConfig.objects.filter(identifier__in=options['instances'])}
        missing = set(options['instances']) - instances.keys()
        if missing:
            raise CommandError(f'Unknown instances: {", ".join(sorted(missing))}')

        results: list[str] = []
        with transaction.atomic():
            for identifier in options['instances']:
                ic = instances[identifier]
                years = history_years(ic, options['dataset'])
                for year in years:
                    submission = ops.create_submission(ic, period_start=year, user=user)
                    submission = ops.request_review(submission, user=user, note=note)
                    ops.finalise(submission, user=user)
                span = ', '.join(str(year) for year in years) or 'nothing to record'
                results.append(f'{identifier}: {span}')
            if not options['apply']:
                transaction.set_rollback(True)
        for line in results:
            self.stdout.write(line)
        if not options['apply']:
            self.stdout.write('Dry run: database changes rolled back.')
