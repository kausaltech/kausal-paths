from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from kausal_common.datasets.models import DataPoint, Dataset, DatasetMetric

from frameworks.evidence import QUALITY_OF_SPEC_KEY, get_evidence, quality_schemes_for_dataset, set_evidence
from frameworks.models import DataQualityLevel
from nodes.dataset_materialization import refresh_dataset_materialization
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from decimal import Decimal


type PointKey = tuple[Any, frozenset[int]]


def point_key(dp: DataPoint) -> PointKey:
    return (dp.date, frozenset(cat.pk for cat in dp.dimension_categories.all()))


def grade_for(value: Decimal, levels: list[DataQualityLevel], *, snap_down: bool) -> tuple[DataQualityLevel | None, bool]:
    """Return the one grade scored exactly `value` or, with `snap_down`, the best one scored below it."""
    exact = [level for level in levels if level.score == value]
    if exact or not snap_down:
        return (exact[0] if len(exact) == 1 else None), False
    below = [level for level in levels if level.score <= value]
    best = max((level.score for level in below), default=None)
    candidates = [level for level in below if level.score == best]
    return (candidates[0] if len(candidates) == 1 else None), True


class Command(BaseCommand):
    help = (
        'Convert a legacy numeric quality metric into data-point evidence and mark the metric as its projection. '
        "Each quality value must equal the score of exactly one grade of the dataset's quality scheme; values that "
        'do not, and quality points with no value point at the same coordinates, are reported and left alone, and '
        'the metric is then not marked as a projection, since that would silently drop those values from the '
        'calculation. Evidence that already carries a grade is never overwritten.'
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            'instance', nargs='+', help='Instance identifier(s); every dataset with the quality metric is converted.'
        )
        parser.add_argument('--quality-metric', default='quality', help='Name of the legacy quality metric.')
        parser.add_argument(
            '--value-metric',
            help='Name of the graded metric; defaults to the only other metric of the schema.',
        )
        parser.add_argument(
            '--delete-legacy',
            action='store_true',
            help='Delete the converted legacy quality data points. They are ignored once the metric is projected.',
        )
        parser.add_argument(
            '--snap-down',
            action='store_true',
            help='Map a value between grades to the best grade whose score does not exceed it, and report it.',
        )
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args: Any, **options: Any) -> None:
        with transaction.atomic():
            for identifier in options['instance']:
                ic = InstanceConfig.objects.filter(identifier=identifier).first()
                if ic is None:
                    raise CommandError('No instance %r' % identifier)
                datasets = Dataset.objects.filter(scope_content_type=ContentType.objects.get_for_model(ic), scope_id=ic.pk)
                for dataset in datasets.select_related('schema').filter(schema__metrics__name=options['quality_metric']):
                    self._convert(dataset, options)
            if options['dry_run']:
                transaction.set_rollback(True)
                self.stdout.write('Dry run: changes rolled back.')

    def _convert(self, dataset: Dataset, options: dict[str, Any]) -> None:  # noqa: C901
        assert dataset.schema is not None
        metrics = list(dataset.schema.metrics.all())
        quality_metric = next(m for m in metrics if m.name == options['quality_metric'])
        value_metric = self._value_metric(dataset, metrics, quality_metric, options['value_metric'])
        if value_metric is None:
            return
        levels = list(DataQualityLevel.objects.filter(scheme__in=quality_schemes_for_dataset(dataset)))
        if not levels:
            self.stdout.write(self.style.WARNING('%s: no quality scheme applies; skipped' % dataset.identifier))
            return

        points = (
            DataPoint.objects
            .filter(dataset=dataset)
            .select_related('metric', 'evidence__quality_level')
            .prefetch_related('dimension_categories')
        )
        value_points = {point_key(dp): dp for dp in points if dp.metric.pk == value_metric.pk}
        quality_points = [dp for dp in points if dp.metric.pk == quality_metric.pk]

        stats: Counter[str] = Counter()
        unmapped: Counter[str] = Counter()
        snapped: Counter[str] = Counter()
        converted: list[DataPoint] = []
        for qdp in quality_points:
            if qdp.value is None:
                stats['empty'] += 1
                converted.append(qdp)
                continue
            level, was_snapped = grade_for(qdp.value, levels, snap_down=options['snap_down'])
            if level is None:
                unmapped[str(qdp.value.normalize())] += 1
                continue
            if was_snapped:
                snapped[f'{qdp.value.normalize()} -> {level.identifier}'] += 1
            target = value_points.get(point_key(qdp))
            if target is None:
                stats['orphaned'] += 1
                continue
            existing = get_evidence(target)
            if existing is not None and existing.quality_level_id is not None:
                stats['same' if existing.quality_level_id == level.pk else 'conflicting'] += 1
            else:
                set_evidence(target, quality_level=level, user=None)
                stats['graded'] += 1
            converted.append(qdp)

        complete = not unmapped and not stats['orphaned']
        if complete:
            quality_metric.spec = {**(quality_metric.spec or {}), QUALITY_OF_SPEC_KEY: str(value_metric.uuid)}
            quality_metric.save(update_fields=['spec'])
            if options['delete_legacy']:
                DataPoint.objects.filter(pk__in=[dp.pk for dp in converted]).delete()
                stats['deleted'] = len(converted)
        refresh_dataset_materialization(dataset)
        self._report(dataset, quality_metric, stats, snapped, unmapped, complete=complete)

    def _report(
        self,
        dataset: Dataset,
        quality_metric: DatasetMetric,
        stats: Counter[str],
        snapped: Counter[str],
        unmapped: Counter[str],
        *,
        complete: bool,
    ) -> None:
        warn = self.style.WARNING
        summary = ', '.join('%s %d' % item for item in sorted(stats.items())) or 'nothing to convert'
        self.stdout.write('%s: %s' % (dataset.identifier, summary))
        for mapping, count in sorted(snapped.items()):
            self.stdout.write(warn('  %d value(s) snapped down: %s' % (count, mapping)))
        for value, count in sorted(unmapped.items()):
            self.stdout.write(warn('  %d value(s) %s match no single grade; left as is' % (count, value)))
        if not complete:
            self.stdout.write(warn('  %s not marked as projection: legacy values remain authoritative' % quality_metric.name))
        if stats['conflicting']:
            self.stdout.write(warn('  %d point(s) already had a different grade; kept it' % stats['conflicting']))

    def _value_metric(
        self, dataset: Dataset, metrics: list[DatasetMetric], quality_metric: DatasetMetric, name: str | None
    ) -> DatasetMetric | None:
        others = [m for m in metrics if m.pk != quality_metric.pk and (name is None or m.name == name)]
        if len(others) != 1:
            self.stdout.write(
                self.style.WARNING(
                    '%s: cannot determine the graded metric among %s; skipped (pass --value-metric)'
                    % (dataset.identifier, [m.name for m in metrics])
                )
            )
            return None
        return others[0]
