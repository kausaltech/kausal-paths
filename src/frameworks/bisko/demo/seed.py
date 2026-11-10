"""Seed complete synthetic district inventories through the real submission lifecycle."""

from typing import TYPE_CHECKING, Any

from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from frameworks.bisko.demo.aggregation import DEFAULT_DISTRICTS, DEFAULT_YEARS, district_municipalities, seed_municipality
from frameworks.models import Framework

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Seed final BISKO inventories for every municipality in Ammerland and Teltow-Fläming; dry run by default.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('--district', action='append', help='Landkreis ARS; repeat for several.')
        parser.add_argument('--year', action='append', type=int, help='Inventory year; default: 2022 and 2023.')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--apply', action='store_true', help='Persist inventories; otherwise roll everything back.')

    def handle(self, *args: Any, **options: Any) -> None:
        framework = Framework.objects.filter(identifier='bisko').first()
        if framework is None:
            raise CommandError('Set up BISKO, import the BKG tree and publish the template first.')
        districts = sorted(set(options['district'] or DEFAULT_DISTRICTS))
        years = sorted(set(options['year'] or DEFAULT_YEARS))
        if any(year not in DEFAULT_YEARS for year in years):
            raise CommandError('The demo supports 2022 and 2023 only.')
        count = 0
        try:
            with transaction.atomic():
                # Resolve every root before writing any municipality.
                targets = [(district, district_municipalities(framework, district)) for district in districts]
                for district, municipalities in targets:
                    self.stdout.write(f'{district}: {len(municipalities)} municipalities; years {years}')
                    for municipality in municipalities:
                        created = seed_municipality(framework, municipality, district=district, years=years, seed=options['seed'])
                        count += 1
                        self.stdout.write(
                            f'{municipality.name}: {"created" if created else "unchanged"}, {len(years)} final inventories'
                        )
                if not options['apply']:
                    transaction.set_rollback(True)
        except (ValueError, ValidationError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f'{count} municipalities, {count * len(years)} final inventories.')
        if not options['apply']:
            self.stdout.write('Dry run: all database changes rolled back.')
