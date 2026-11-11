"""Compute and store the figures of submissions that lack them, or all of them again."""

from typing import TYPE_CHECKING, cast

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from frameworks.indicators import IndicatorComputationError, materialise_final_submission, materialise_open_submissions
from frameworks.models import Submission, SubmissionStatus
from frameworks.models.submission import OPEN_STATUSES
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = (
        'Store the declared figures on submissions. Open submissions are computed from the draft; final ones from '
        'the revision they pin, which need not be the live one. Dry run unless --apply is given.'
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='*', help='Instance identifiers; all instances with submissions if none')
        parser.add_argument('--all', action='store_true', help='Recompute submissions that already have figures too')
        parser.add_argument('--apply', action='store_true')

    def handle(self, *args: object, **options: object) -> None:
        requested = cast('list[str]', options['instances'])
        recompute = bool(options['all'])
        apply = bool(options['apply'])
        submissions = Submission.objects.filter(status__in=(*OPEN_STATUSES, SubmissionStatus.FINAL))
        if not recompute:
            submissions = submissions.filter(indicators__isnull=True)
        configs = InstanceConfig.objects.filter(pk__in=submissions.values('instance_config_id'), config_source='database')
        if requested:
            configs = configs.filter(identifier__in=requested)
            unknown = set(requested) - set(
                InstanceConfig.objects.filter(identifier__in=requested).values_list('identifier', flat=True)
            )
            if unknown:
                raise CommandError(f'Unknown instances: {sorted(unknown)}')
        failed = 0
        for ic in configs.order_by('identifier'):
            own = submissions.filter(instance_config=ic)
            years = sorted(own.filter(status__in=OPEN_STATUSES).values_list('period_start', flat=True))
            finals = list(own.filter(status=SubmissionStatus.FINAL).select_related('instance_config', 'instance_revision'))
            if not apply:
                final_years = sorted(submission.period_start for submission in finals)
                self.stdout.write(f'{ic.identifier}: would compute open {years}, final {final_years}')
                continue
            if years:
                with transaction.atomic():
                    written = materialise_open_submissions(ic)
                self.stdout.write(f'{ic.identifier}: open submissions written: {written}')
            for submission in finals:
                try:
                    with transaction.atomic():
                        materialise_final_submission(submission)
                except IndicatorComputationError as error:
                    failed += 1
                    self.stderr.write(f'{ic.identifier} {submission.period_start} (final): {error}')
                    continue
                state = 'written' if submission.indicators is not None else 'its revision declares no figures'
                self.stdout.write(f'{ic.identifier} {submission.period_start} (final): {state}')
        if failed:
            raise CommandError(f'{failed} final submissions could not be computed; their figures were left as they were')
