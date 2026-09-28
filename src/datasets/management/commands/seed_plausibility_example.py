"""Create one clearly labelled, instance-local plausibility finding for UI development."""

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from kausal_common.datasets.models import DataPoint, Dataset

from datasets.coordinates import DatasetCoordinateIndex
from datasets.models import DatasetMetricPlausibilityRange, PlausibilitySource
from datasets.plausibility import _invalid_cell, evaluate_dataset_plausibility
from datasets.snapshot import metric_column_id
from datasets.validation import evaluate_dataset_rules
from nodes.models import InstanceConfig


def _point_coordinates(point: DataPoint) -> dict[str, str]:
    return {str(category.dimension.uuid): str(category.uuid) for category in point.dimension_categories.all()}


class Command(BaseCommand):
    help = 'Seed one illustrative, instance-scoped plausibility range for local UI development.'

    def add_arguments(self, parser) -> None:
        parser.add_argument('instance')
        parser.add_argument('dataset')
        parser.add_argument('--metric', default='Value')
        parser.add_argument('--year', type=int)

    @transaction.atomic
    def handle(self, *args, **options) -> None:
        if not settings.DEBUG:
            raise CommandError('Example plausibility ranges may only be seeded in DEBUG mode.')
        instance = InstanceConfig.objects.filter(identifier=options['instance']).first()
        if instance is None:
            raise CommandError(f'Unknown instance: {options["instance"]}')
        dataset = (
            Dataset.objects
            .filter(
                scope_content_type=ContentType.objects.get_for_model(InstanceConfig),
                scope_id=instance.pk,
                identifier=options['dataset'],
            )
            .select_related('schema')
            .first()
        )
        if dataset is None or dataset.schema is None:
            raise CommandError(f'Unknown instance dataset: {options["dataset"]}')
        metric = dataset.schema.metrics.filter(name=options['metric']).first()
        if metric is None:
            raise CommandError(f'Unknown metric: {options["metric"]}')
        coordinate_index = DatasetCoordinateIndex(dataset)
        points = (
            DataPoint.objects
            .filter(dataset=dataset, metric=metric, value__gt=0)
            .prefetch_related('dimension_categories__dimension')
            .order_by('date', 'pk')
        )
        if options['year'] is not None:
            points = points.filter(date__year=options['year'])
        violations = evaluate_dataset_rules(dataset)
        source, _ = PlausibilitySource.objects.update_or_create(
            identifier='ui-example',
            defaults={
                'name': 'Local UI example',
                'url': '',
                'revision': 'local-example',
                'method': 'Illustrative UI fixture derived from one entered cell; not a reference benchmark.',
                'is_example': True,
            },
        )
        source.full_clean()
        for point in points:
            coordinates = _point_coordinates(point)
            legacy_coordinates = {
                coordinate.dimension: coordinate.category for coordinate in coordinate_index.resolve_uuids(coordinates)
            }
            year = point.date.year
            if _invalid_cell(violations, metric.uuid, year, legacy_coordinates):
                continue
            assert point.value is not None
            observed = float(point.value)
            rule, _ = DatasetMetricPlausibilityRange.objects.update_or_create(
                instance_config=instance,
                metric=metric,
                identifier='ui-example',
                defaults={
                    'source': source,
                    'framework': None,
                    'selection': {dimension: [category] for dimension, category in coordinates.items()},
                    'aggregation': DatasetMetricPlausibilityRange.Aggregation.CELL,
                    'denominator': DatasetMetricPlausibilityRange.Denominator.NONE,
                    'reference': DatasetMetricPlausibilityRange.Reference.ABSOLUTE,
                    'max_gap_years': None,
                    'lower': 0.0,
                    'upper': observed * 0.8,
                    'first_year': year,
                    'last_year': year,
                    'sample_size': None,
                    'revision': 1,
                    'enabled': True,
                },
            )
            rule.full_clean()
            if any(finding.rule_uuid == rule.uuid for finding in evaluate_dataset_plausibility(dataset)):
                self.stdout.write(
                    self.style.SUCCESS(
                        f'Seeded example finding on {dataset.identifier}, {metric_column_id(metric)}, {year}: {coordinates}'
                    )
                )
                return
        raise CommandError('No positive, valid data point could produce an example finding.')
