"""
Re-populate the comparable-city defaults and probable bounds of existing cities.

``populate_measure_defaults`` is only ever called when a FrameworkConfig is created or
its baseline year changes (``frameworks/schema.py``), so an import that rewrites
``MeasureTemplateDefaultDataPoint`` -- new central values, widened bounds -- reaches new
cities and no one else. This command applies the current template defaults to cities
that already exist.

Nothing is written without ``--apply``: a dry run does the work inside a transaction,
reports what moved, and rolls back.

Note the deletion. With a year selected (the default, matching city creation) the
underlying method drops *default-only* data points at every other year -- rows the city
never entered a value into. Rows carrying a city value are untouched. The report counts
them before you commit, because for a city created under the older yearly-defaults
system that prunes a 2018-2024 series back to the baseline year alone; ``--all-years``
populates every year instead and deletes nothing.

Cities whose ``categories`` are empty are reported and skipped: cluster-specific
defaults cannot be selected for them, so they would be silently repopulated with
nothing. Those need ``repair_nzc_missing_categories`` first.
"""

from typing import Any

from django.core.management.base import BaseCommand
from django.db import transaction

from frameworks.models import FrameworkConfig, MeasureDataPoint


class Command(BaseCommand):
    help = 'Re-apply current MeasureTemplate defaults to existing framework configs'

    def add_arguments(self, parser) -> None:
        parser.add_argument('--framework', default='nzc', help='Framework identifier (default: nzc)')
        parser.add_argument(
            '--instance',
            dest='instances',
            metavar='IDENTIFIER',
            action='append',
            default=[],
            help='Limit to this instance identifier (repeatable)',
        )
        parser.add_argument(
            '--all-years',
            action='store_true',
            help='Populate defaults for every year instead of the baseline year only; deletes nothing',
        )
        parser.add_argument('--apply', action='store_true', help='Write the changes (default: dry run)')

    def _counts(self, fwc: FrameworkConfig, only_year: int | None) -> tuple[int, int]:
        """Return (data points carrying a default, default-only points outside only_year)."""
        qs = MeasureDataPoint.objects.filter(measure__framework_config=fwc)
        with_default = qs.filter(default_value__isnull=False).count()
        if only_year is None:
            return with_default, 0
        doomed = qs.filter(default_value__isnull=False, value__isnull=True).exclude(year=only_year).count()
        return with_default, doomed

    def handle(self, *args: Any, **options: Any) -> None:
        apply_changes: bool = options['apply']
        qs = FrameworkConfig.objects.filter(framework__identifier=options['framework']).select_related('instance_config')
        if options['instances']:
            qs = qs.filter(instance_config__identifier__in=options['instances'])

        total_selected = 0
        total_deleted = 0
        touched = 0
        with transaction.atomic():
            for fwc in qs.order_by('pk'):
                ident = fwc.instance_config.identifier if fwc.instance_config else f'fc:{fwc.pk}'
                if not fwc.categories.exists():
                    self.stdout.write(
                        self.style.WARNING(f'  SKIP {ident}: no categories; run repair_nzc_missing_categories first'),
                    )
                    continue
                only_year = None if options['all_years'] else fwc.reference_year
                before, doomed = self._counts(fwc, only_year)
                selected = fwc.populate_measure_defaults(only_year=only_year)
                after, _ = self._counts(fwc, only_year)
                fwc.notify_change(save=True)
                touched += 1
                total_selected += selected
                total_deleted += doomed
                note = f' (-{doomed} default-only points at other years)' if doomed else ''
                self.stdout.write(f'  {ident}: {selected} defaults applied, {before} -> {after} points{note}')
            if not apply_changes:
                transaction.set_rollback(True)

        verb = 'Applied' if apply_changes else 'Would apply'
        self.stdout.write(
            self.style.SUCCESS(f'{verb} defaults to {touched} configs; {total_selected} selected, {total_deleted} deleted'),
        )
        if not apply_changes:
            self.stdout.write('Dry run: rolled back. Re-run with --apply to write.')
