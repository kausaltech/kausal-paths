"""
Report what the plausibility checks find in instances' datasets, for tuning them.

Per dataset metric it prints the band derived from the dataset's own history
(its bounds and how many ratios it rests on), how many findings it and the
curated ranges produce, and a few examples. The closing summary is the number
to watch while tuning ``datasets.plausibility_history``: how many cells a
derived band flags out of those it judges.

    python manage.py plausibility_report espoo
    python manage.py plausibility_report --all --examples 3
"""

from collections import Counter
from typing import TYPE_CHECKING, Any

from django.core.management.base import BaseCommand, CommandError

from kausal_common.datasets.models import Dataset

from datasets.plausibility import applicable_plausibility_ranges, evaluate_dataset_plausibility, history_ranges
from datasets.plausibility_history import SOURCE_IDENTIFIER
from nodes.models import InstanceConfig

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from datasets.plausibility import PlausibilityFinding


class Command(BaseCommand):
    help = "Report plausibility findings, and the bands derived from each dataset's history."

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('instances', nargs='*', help='Instance identifiers')
        parser.add_argument('--all', action='store_true', help='Every instance')
        parser.add_argument('--examples', type=int, default=2, help='Example findings per metric (default 2)')

    def handle(self, *args: Any, **options: Any) -> None:
        if options['all']:
            instances = list(InstanceConfig.objects.order_by('identifier'))
        elif options['instances']:
            instances = list(InstanceConfig.objects.filter(identifier__in=options['instances']).order_by('identifier'))
            missing = set(options['instances']) - {ic.identifier for ic in instances}
            if missing:
                raise CommandError(f'Unknown instances: {", ".join(sorted(missing))}')
        else:
            raise CommandError('Name instances, or pass --all.')

        totals: Counter[str] = Counter()
        for ic in instances:
            for dataset in Dataset.objects.for_instance_config(ic).select_related('schema').order_by('identifier'):
                self._report_dataset(ic, dataset, options['examples'], totals)

        judged = totals['judged']
        flagged = totals['history']
        rate = f'{flagged / judged:.2%}' if judged else '-'
        self.stdout.write(
            f'\n{totals["datasets"]} datasets checked; {totals["banded"]} metrics with a derived band over '
            f'{judged} judged ratios; {flagged} history findings ({rate}); {totals["curated"]} curated findings.'
        )
        if totals['failed']:
            self.stdout.write(self.style.WARNING(f'{totals["failed"]} datasets could not be read.'))

    def _report_dataset(self, ic: InstanceConfig, dataset: Dataset, examples: int, totals: Counter[str]) -> None:
        try:
            curated = applicable_plausibility_ranges(dataset)
            derived = history_ranges(dataset, curated)
            if not curated and not derived:
                return
            findings = evaluate_dataset_plausibility(dataset)
        except Exception as exc:
            totals['failed'] += 1
            self.stdout.write(self.style.ERROR(f'{ic.identifier} {dataset.identifier}: {type(exc).__name__}: {exc}'))
            return
        totals['datasets'] += 1
        by_rule: dict[object, list[PlausibilityFinding]] = {}
        for finding in findings:
            by_rule.setdefault(finding.rule_uuid, []).append(finding)
        curated_count = sum(1 for finding in findings if finding.source_identifier != SOURCE_IDENTIFIER)
        totals['curated'] += curated_count

        self.stdout.write(self.style.MIGRATE_HEADING(f'{ic.identifier} {dataset.identifier}'))
        if curated:
            self.stdout.write(f'  curated: {len(curated)} ranges, {curated_count} findings')
        for item in derived:
            rule = item.rule
            flagged = by_rule.get(rule.uuid, [])
            totals['banded'] += 1
            totals['judged'] += rule.sample_size or 0
            totals['history'] += len(flagged)
            self.stdout.write(
                f'  {rule.metric.name}: band {rule.lower:g}-{rule.upper:g} from {rule.sample_size} ratios, '
                f'floor {item.min_reference:.3g}; {len(flagged)} findings'
            )
            for finding in flagged[:examples]:
                cell = ', '.join(f'{dim}={cat}' for dim, cat in finding.categories.items()) or '-'
                spike = f' (return from {finding.attribution.year})' if finding.attribution else ''
                self.stdout.write(
                    f'    {finding.years[0]} {cell}: {finding.reference_value:.4g} -> {finding.observed:.4g} '
                    f'(x{finding.normalized:.3g}){spike}'
                )
