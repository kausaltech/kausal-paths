"""
Repair NZC framework configs created without categories.

Cities created via CreateNZCFrameworkConfigMutation between April 29 and ~May 18, 2026
had their categories silently dropped because the FrameworkDimension records did not
exist in production at creation time.  The GraphQLError from get_category() was caught
by Graphene before rolling back the transaction, leaving `extra['create_context']` saved
but `categories` empty and `populate_measure_defaults` never called.

This command fixes those cities by:
  1. Reading temperature and renewable_mix from extra['create_context']
  2. Adding the matching FrameworkDimensionCategory records to fwc.categories
  3. Calling populate_measure_defaults(only_year=fwc.baseline_year)
"""

from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction


class Command(BaseCommand):
    help = 'Repair NZC framework configs that have create_context but no categories'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Print what would be done without making any changes',
        )
        parser.add_argument(
            '--instance',
            dest='instances',
            metavar='IDENTIFIER',
            action='append',
            default=[],
            help='Limit repair to this instance identifier (can be repeated)',
        )

    @transaction.atomic
    def handle(self, *args, **options):
        from frameworks.models import FrameworkConfig, FrameworkDimension, FrameworkDimensionCategory  # noqa: TC001

        dry_run: bool = options['dry_run']
        limit_to: list[str] = options['instances']

        # Find FrameworkDimension categories once
        try:
            dim_renewable = FrameworkDimension.objects.get(framework__identifier='nzc', identifier='renewable_mix')
            dim_temperature = FrameworkDimension.objects.get(framework__identifier='nzc', identifier='temperature')
        except FrameworkDimension.DoesNotExist as exc:
            raise CommandError('NZC FrameworkDimension records not found — run convert_nzc_yearly_placeholders first') from exc

        cat_by_dim: dict[str, dict[str, FrameworkDimensionCategory]] = {
            'renewable_mix': {cat.name.lower(): cat for cat in dim_renewable.categories.all()},
            'temperature': {cat.name.lower(): cat for cat in dim_temperature.categories.all()},
        }

        qs = FrameworkConfig.objects.filter(framework__identifier='nzc').prefetch_related('categories')
        if limit_to:
            qs = qs.filter(instance_config__identifier__in=limit_to)

        fixed = 0
        skipped = 0
        for fwc in qs:
            if list(fwc.categories.all()):
                skipped += 1
                continue
            extra = fwc.extra or {}
            ctx = extra.get('create_context')
            if not ctx:
                self.stdout.write(
                    self.style.WARNING(f'  SKIP {fwc.instance_config.identifier}: no create_context, manual fix needed')
                )
                skipped += 1
                continue

            renewable_mix = ctx.get('renewable_mix', '').lower()
            temperature = ctx.get('temperature', '').lower()

            cat_renew = cat_by_dim['renewable_mix'].get(renewable_mix)
            cat_temp = cat_by_dim['temperature'].get(temperature)

            if cat_renew is None or cat_temp is None:
                self.stdout.write(
                    self.style.ERROR(
                        f'  ERROR {fwc.instance_config.identifier}: unknown category '
                        f'renewable_mix={renewable_mix!r} temperature={temperature!r}'
                    )
                )
                continue

            ic_id = fwc.instance_config.identifier if fwc.instance_config else f'fc:{fwc.id}'
            pop = ctx.get('population')
            self.stdout.write(f'  FIX {ic_id}: pop={pop} temp={temperature} renew={renewable_mix}')
            if not dry_run:
                fwc.categories.add(cat_renew, cat_temp)
                count = fwc.populate_measure_defaults(only_year=fwc.baseline_year)
                self.stdout.write(f'       → {count} defaults populated')
            fixed += 1

        verb = 'Would fix' if dry_run else 'Fixed'
        self.stdout.write(self.style.SUCCESS(f'{verb} {fixed} configs, skipped {skipped}'))
