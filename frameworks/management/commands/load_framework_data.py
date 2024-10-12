from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction

from frameworks.models import Framework, MeasurePriority, MeasureTemplate, MeasureTemplateDefaultDataPoint, Section
from nodes.units import unit_registry


class Command(BaseCommand):
    help = 'Import or export Framework measures data'

    def add_arguments(self, parser: CommandParser):
        parser.add_argument('action', type=str, choices=['import', 'export'], help='Action to perform')
        parser.add_argument('file', type=str, help='JSON file to read from or write to')
        parser.add_argument('--framework', type=str, help='Framework identifier (required for export)')
        parser.add_argument('--force', action='store_true', help='Force import by deleting existing Framework')

    @transaction.atomic
    def handle(self, *args, **options):
        action = options['action']
        file_path = Path(options['file'])
        force = options['force']

        if action == 'import':
            self.import_data(file_path, force)
        elif action == 'export':
            framework_identifier = options['framework']
            if not framework_identifier:
                self.stderr.write(self.style.ERROR('Framework identifier is required for export'))
                return
            self.export_data(file_path, framework_identifier)

    def import_data(self, file_path: Path, force: bool):
        with file_path.open('r') as file:
            data = json.load(file)

        framework_data = data['framework']
        identifier = framework_data['identifier']

        existing_framework = Framework.objects.filter(identifier=identifier).first()

        if existing_framework:
            if force:
                self.stdout.write(self.style.WARNING(f"Deleting existing framework: {existing_framework.name}"))
                existing_framework.delete()
            else:
                self.stderr.write(
                    self.style.ERROR(f"Framework with identifier '{identifier}' already exists. Use --force to override."),
                )
                return

        fw = Framework.objects.create(
            identifier=identifier,
            name=framework_data['name'],
            description=framework_data.get('description', ''),
            public_base_fqdn=framework_data.get('public_base_fqdn'),
            result_excel_url=framework_data.get('result_excel_url'),
            result_excel_node_ids=framework_data.get('result_excel_node_ids'),
        )
        self.stdout.write(self.style.SUCCESS(f"Created new framework: {fw.name}"))
        root_section = fw.create_root_section()

        # Import sections and measures
        all_sections: dict[str, Section] = {}
        for sd in data['sections']:
            self.import_section(sd, root_section, all_sections)

        self.stdout.write(self.style.SUCCESS('Successfully imported framework data'))

    def import_section(self, section_data: dict, root_section: Section, all_sections: dict[str, Section]):
        parent_uuid = section_data['parent']
        if parent_uuid is None:
            parent = root_section
        else:
            parent = all_sections[parent_uuid]
        obj = Section(
            framework=root_section.framework,
            identifier=section_data.get('identifier', ''),
            uuid=section_data['uuid'],
            name=section_data['name'],
            description=section_data.get('description', ''),
            available_years=section_data.get('available_years'),
        )
        section = parent.add_child(instance=obj)
        all_sections[str(section.uuid)] = section

        dp_list: list[MeasureTemplateDefaultDataPoint] = []

        # Import measure templates
        for mt_data in section_data.get('measure_templates', []):
            mt = MeasureTemplate.objects.create(
                section=section,
                uuid=mt_data['uuid'],
                name=mt_data['name'],
                unit=str(unit_registry.parse_units(mt_data['unit'])),
                priority=MeasurePriority(mt_data['priority']),
                min_value=mt_data.get('min_value'),
                max_value=mt_data.get('max_value'),
                time_series_max=mt_data.get('time_series_max'),
                default_value_source=mt_data.get('default_value_source', ''),
            )
            ddps = mt_data.get('default_data_points', [])
            dp_list += [MeasureTemplateDefaultDataPoint(template=mt, year=ddp['year'], value=ddp['value']) for ddp in ddps]

        if dp_list:
            MeasureTemplateDefaultDataPoint.objects.bulk_create(dp_list)

    def export_data(self, file_path: Path, framework_identifier: str):
        try:
            framework = Framework.objects.get(identifier=framework_identifier)
        except Framework.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Framework with identifier '{framework_identifier}' not found"))
            return

        data = {
            'framework': framework.to_dict(),
            'sections': framework.export_sections(),
        }

        with file_path.open('w') as file:
            json.dump(data, file, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Successfully exported framework data to {file_path}"))
