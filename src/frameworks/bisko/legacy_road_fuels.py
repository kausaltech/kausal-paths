"""
One-off repair: road-transport energy entered with the biofuel split, before the shape closed.

BISKO's road-transport fuels are petrol, diesel, natural gas (CNG + LNG), LPG and electricity, with
the biofuel blend inside each fuel's figure (Methodenpapier, Tabelle 10); the model splits a
measured fuel into its fossil and biogenic parts itself. Before `road_transport_energy` was a
closed shape, data entry offered every carrier, and a municipality could enter the split form --
a Klimaschutz-Planer transcription, say. The closed shape refuses those cells, which then block an
upgrade to the template that closed it. `setup_bisko` moves them back:

    biodiesel -> diesel, bio_petrol -> petrol, cng_fossil + cng_bio -> natural_gas

A value is added to the target cell of the same vehicle class, year and metric; a grade on a
folded cell is dropped, since the target keeps its own. Totals do not change. Each municipality's
writes are one CLI operation in its change history. A municipality with nothing to fold is left
alone, so a second run does nothing.

Remove this module once staging and production have run it.
"""

from decimal import Decimal

from django.contrib.contenttypes.models import ContentType

from kausal_common.datasets.models import DataPoint, DataPointDimensionCategory, Dataset, DimensionCategory

from datasets.change_snapshots import data_point_snapshot
from datasets.materialization import refresh_dataset_materialization
from frameworks.models import FrameworkConfig
from nodes.change_ops import change_operation, record_change
from nodes.models import InstanceChangeSource, InstanceConfig

ROAD_TRANSPORT_DATASET = 'kommune/energieverbrauch_strassenverkehr'
FOLDED_CARRIERS = {'biodiesel': 'diesel', 'bio_petrol': 'petrol', 'cng_fossil': 'natural_gas', 'cng_bio': 'natural_gas'}


def fold_road_biofuels(framework_identifier: str = 'bisko') -> list[str]:
    """Fold the split road-transport fuels of every municipality; return what was done, one line per cell."""
    report: list[str] = []
    configs = FrameworkConfig.objects.filter(framework__identifier=framework_identifier).select_related('instance_config')
    for config in configs:
        instance = config.instance_config
        dataset = Dataset.objects.filter(
            scope_content_type=ContentType.objects.get_for_model(InstanceConfig),
            scope_id=instance.pk,
            identifier=ROAD_TRANSPORT_DATASET,
        ).first()
        if dataset is not None:
            report.extend(f'{instance.identifier}: {line}' for line in _fold(instance, dataset))
    return report


def _fold(instance: InstanceConfig, dataset: Dataset) -> list[str]:
    points = list(DataPoint.objects.filter(dataset=dataset).select_related('metric').prefetch_related('dimension_categories'))
    carriers = {c.dimension_id for p in points for c in p.dimension_categories.all() if c.identifier in FOLDED_CARRIERS}
    if not carriers:
        return []
    if len(carriers) != 1:
        raise ValueError(f'{instance.identifier}: {ROAD_TRANSPORT_DATASET} has several dimensions with fuel categories')
    carrier_dimension = carriers.pop()

    def carrier_of(point: DataPoint) -> DimensionCategory:
        return next(c for c in point.dimension_categories.all() if c.dimension_id == carrier_dimension)

    def key_of(point: DataPoint, carrier: str | None) -> tuple[object, ...]:
        others = tuple(sorted(c.pk for c in point.dimension_categories.all() if c.dimension_id != carrier_dimension))
        return (point.metric.pk, point.date, others, carrier)

    def label_of(point: DataPoint) -> str:
        names = sorted(c.identifier or '' for c in point.dimension_categories.all())
        return f'{point.date.year} {"/".join(names)} [{point.metric.name}]'

    by_key = {key_of(p, carrier_of(p).identifier): p for p in points}
    folded = [p for p in points if carrier_of(p).identifier in FOLDED_CARRIERS]
    # cng_fossil first, so that it becomes the natural_gas cell cng_bio is then added to.
    folded.sort(key=lambda p: carrier_of(p).identifier != 'cng_fossil')
    report: list[str] = []
    with change_operation(instance, user=None, action='dataset.datapoint.fold_biofuels', source=InstanceChangeSource.CLI):
        for point in folded:
            old = carrier_of(point)
            assert old.identifier is not None
            target_id = FOLDED_CARRIERS[old.identifier]
            target = by_key.get(key_of(point, target_id))
            label = label_of(point)
            if target is None:
                # No cell to add to: the point itself becomes the target cell.
                category = DimensionCategory.objects.get(dimension_id=carrier_dimension, identifier=target_id)
                before = data_point_snapshot(point)
                links = DataPointDimensionCategory.objects.filter(data_point=point, dimension_category=old)
                links.update(dimension_category=category)
                relabelled = DataPoint.objects.select_related('metric').prefetch_related('dimension_categories').get(pk=point.pk)
                record_change(relabelled, action='dataset.datapoint.update', before=before, after=data_point_snapshot(relabelled))
                by_key[key_of(relabelled, target_id)] = relabelled
                report.append(f'{label}: relabelled as {target_id}')
                continue
            is_grade = 'quality_of' in (point.metric.spec or {})
            if point.value is not None and not is_grade:
                before = data_point_snapshot(target)
                target.value = (target.value or Decimal(0)) + point.value
                target.save(update_fields=['value'])
                record_change(target, action='dataset.datapoint.update', before=before, after=data_point_snapshot(target))
                report.append(f'{label}: {point.value:f} added to {target_id}')
            else:
                report.append(f'{label}: {"grade" if is_grade else "empty cell"} dropped')
            record_change(point, action='dataset.datapoint.delete', before=data_point_snapshot(point), after=None)
            point.delete()
        refresh_dataset_materialization(dataset)
    instance.invalidate_cache()
    return report
