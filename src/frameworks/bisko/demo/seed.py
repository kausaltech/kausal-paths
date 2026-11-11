"""Seed complete synthetic district inventories through the real submission lifecycle."""

from typing import TYPE_CHECKING, Any

from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from frameworks.bisko.demo.accounts import grant_demo_city_access
from frameworks.bisko.demo.aggregation import (
    DEFAULT_DISTRICTS,
    DEFAULT_YEARS,
    district_municipalities,
    municipality_by_ars,
    seed_municipality,
)
from frameworks.bisko.demo.inventories import DEMO_MUNICIPALITIES
from frameworks.models import Framework

if TYPE_CHECKING:
    from argparse import ArgumentParser


class Command(BaseCommand):
    help = 'Seed 2022/2023 inventories for Ammerland, Teltow-Fläming, Osnabrück, Bendorf and Rathenow; dry run by default.'

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('--district', action='append', help='Landkreis ARS; repeat for several.')
        parser.add_argument('--municipality', action='append', help='Municipal ARS; repeat for several.')
        parser.add_argument('--year', action='append', type=int, help='Inventory year; default: 2022 and 2023.')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--apply', action='store_true', help='Persist inventories; otherwise roll everything back.')
        parser.add_argument(
            '--refresh-submissions',
            action='store_true',
            help='Reissue existing demo finals from current inputs and the published template; preserve superseded finals.',
        )

    def handle(self, *args: Any, **options: Any) -> None:
        framework = Framework.objects.filter(identifier='bisko').first()
        if framework is None:
            raise CommandError('Set up BISKO, import the BKG tree and publish the template first.')
        districts = sorted(set(options['district'] or (() if options['municipality'] else DEFAULT_DISTRICTS)))
        cities = sorted(set(options['municipality'] or (() if options['district'] else DEMO_MUNICIPALITIES)))
        years = sorted(set(options['year'] or DEFAULT_YEARS))
        if any(year not in DEFAULT_YEARS for year in years):
            raise CommandError('The demo supports 2022 and 2023 only.')
        count = 0
        try:
            with transaction.atomic():
                # Resolve every root before writing any municipality.
                targets = [(district, district_municipalities(framework, district)) for district in districts]
                targets.extend((ars[:5], [municipality_by_ars(framework, ars)]) for ars in cities)
                visited: set[int] = set()
                for district, municipalities in targets:
                    self.stdout.write(f'{district}: {len(municipalities)} municipalities; years {years}')
                    for municipality in municipalities:
                        if municipality.pk in visited:
                            continue
                        visited.add(municipality.pk)
                        created = seed_municipality(
                            framework,
                            municipality,
                            district=district,
                            years=years,
                            seed=options['seed'],
                            refresh_submissions=options['refresh_submissions'],
                        )
                        count += 1
                        self.stdout.write(
                            f'{municipality.name}: {"finalised" if created else "unchanged"}, {len(years)} final inventories'
                        )
                grant_demo_city_access(framework)
                if not options['apply']:
                    transaction.set_rollback(True)
        except (ValueError, ValidationError) as error:
            raise CommandError(str(error)) from error
        self.stdout.write(f'{count} municipalities, {count * len(years)} final inventories.')
        if not options['apply']:
            self.stdout.write('Dry run: all database changes rolled back.')
