from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandParser

from frameworks.models import Framework, MeasurePriority, MeasureTemplate, Section
from nodes.units import unit_registry

UNIT_CONVERSION_MAP = {
    'Capita': 'cap',
    'km2': 'km^2',
    'Million passenger-kilometers / year': 'Mpkm/a',
    'passengers / car + motorcycle': 'passenger/vehicle',
    'passengers / metro train': 'passenger/vehicle',
    'Number of cars + motorcycles': 'vehicle',
    'Nmbr of buses': 'vehicle',
    'Million tonne-kilometers/year': 'tonne * Mkm/a',
    'Number of trucks': 'vehicle',
    'Thousand squaremeters': 'ksqm',
    'kWh/m2 & year': 'kWh/m^2/a',
    'kWh / m2 & year': 'kWh/m^2/a',
    'GWh / y': 'GWh/a',
    'kWh/m2': 'kWh/m^2',
    'Share': '%',
    'Year': 'a',
    'Number of trees': 'pcs',
}


class Command(BaseCommand):
    help = 'Load Framework measures data from JSON file'

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('file', metavar='FILE', type=str)

    def handle(self, *args, **options):
        # Load JSON data
        with open(options['file']) as file:
            data = json.load(file)

        # Create or get the Framework
        identifier = 'nzc'
        name = 'NetZeroCities'

        fw = Framework.objects.filter(identifier=identifier).first()
        if fw is not None:
            fw.delete()
        fw = Framework.objects.create(identifier=identifier, name=name)
        root_section = Section.add_root(instance=Section(framework=fw, name='%s Root' % name))
        fw.root_section = root_section
        fw.save()

        dc_section = root_section.add_child(
            instance=Section(framework=fw, name='%s Data Collection' % name, identifier='data_collection')
        )
        fa_section = root_section.add_child(
            instance=Section(framework=fw, name='%s Future Assumptions' % name, identifier='future_assumptions')
        )
        # Process the data
        self.process_items(data['dataCollection']['items'], dc_section)
        self.process_items(data['futureAssumptions']['items'], fa_section)

        self.stdout.write(self.style.SUCCESS('Successfully loaded measures and sections'))

    def process_measure(self, data: dict, section: Section):
        unit_str = data['unit']
        try:
            unit = unit_registry.parse_units(unit_str)
        except Exception:
            if unit_str in UNIT_CONVERSION_MAP:
                unit = unit_registry.parse_units(UNIT_CONVERSION_MAP[unit_str])
            elif '%' in unit_str:
                unit = unit_registry.parse_units('%')
            else:
                self.stderr.write(self.style.ERROR('Unable to parse unit: %s' % unit_str))

        src = data.get('fallbackSource', '').strip()
        measure = MeasureTemplate.objects.create(
            section=section,
            name=data['label'],
            unit=str(unit),
            priority=self.get_priority(data.get('priority')),
            default_value_source=src,
        )

        val_str = data.get('fallbackValue', data['value'])
        try:
            val = float(val_str.replace(',', '').replace('%', ''))
        except Exception:
            return
        measure.default_data_points.create(year=2022, value=val)

    def process_items(self, items, parent: Section):
        for item in items:
            ref = item.get('referenceType', None)
            if ref:
                assert ref == 'SECTION'
                if item['label'] == 'Key assumptions for levers':
                    section = parent
                else:
                    # Create a Section
                    section = parent.add_child(
                        instance=Section(
                            framework=parent.framework,
                            name=item['label'],
                            available_years=None,
                        )
                    )
                if 'items' in item:
                    self.process_items(item['items'], section)
            else:
                self.process_measure(item, parent)

    def get_priority(self, priority_str):
        if priority_str == 'HIGH':
            return MeasurePriority.HIGH
        if priority_str == 'LOW':
            return MeasurePriority.LOW
        return MeasurePriority.MEDIUM
