"""
Upgrade dataset payloads written before `DatasetSnapshot` schema version 2.

A v1 payload described a dataset by identifiers: metric columns, dimension and category
identifiers in a wide pandas table, and comments, evidence and source references located
by natural key. Version 2 refers to everything by uuid (docs/plans/node-owned-datasets.md,
step 4). This module turns the one into the other, and is used by:

- the migration that rewrites the stored payloads (`nodes` migration 0085), which borrows
  the uuids of the live rows a payload still describes, and
- `InstanceExport.from_serialized_data`, for export documents written by a deployment
  that still runs v1, which has nothing to borrow from and derives the uuids instead.

It goes when the migrations are next squashed, and with it the reading of v1 exports.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid3

from datasets.snapshot import (
    DataPointCommentSnapshot,
    DataPointEvidenceSnapshot,
    DataPointSnapshot,
    DatasetSnapshot,
    DataSourceSnapshot,
    SourceReferenceSnapshot,
)
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, ValidationRuleMeta

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from nodes.defs.graph import DimensionMeta

type Cell = tuple[UUID, int, frozenset[UUID]]
"""A data point's cell: metric uuid, year and category uuids."""


@dataclass
class LiveIdentities:
    """The uuids of the rows a payload still describes, so that its upgrade keeps them."""

    points: dict[Cell, UUID] = field(default_factory=dict)
    comments: dict[tuple[UUID, str], list[UUID]] = field(default_factory=lambda: defaultdict(list))
    """Comment uuids by data point uuid and text."""
    references: dict[tuple[UUID, UUID], UUID] = field(default_factory=dict)
    """Source reference uuids by cited-by (data point or dataset) uuid and data source uuid."""

    @classmethod
    def load(cls, dataset_pk: int, dataset_uuid: UUID, models: Mapping[str, Any]) -> LiveIdentities:
        """Read them from the database; ``models`` are the (historical) model classes by name."""
        identities = cls()
        categories: defaultdict[int, set[UUID]] = defaultdict(set)
        for point_pk, category_uuid in (
            models['DataPointDimensionCategory']
            .objects.filter(data_point__dataset_id=dataset_pk)
            .values_list('data_point_id', 'dimension_category__uuid')
        ):
            categories[point_pk].add(category_uuid)
        uuids: dict[int, UUID] = {}
        for pk, uuid, point_date, metric_uuid in (
            models['DataPoint'].objects.filter(dataset_id=dataset_pk).values_list('pk', 'uuid', 'date', 'metric__uuid')
        ):
            uuids[pk] = uuid
            identities.points.setdefault((metric_uuid, point_date.year, frozenset(categories[pk])), uuid)
        for point_pk, uuid, text in (
            models['DataPointComment']
            .objects.filter(data_point__dataset_id=dataset_pk, is_soft_deleted=False)
            .order_by('pk')
            .values_list('data_point_id', 'uuid', 'text')
        ):
            identities.comments[uuids[point_pk], text].append(uuid)
        references = models['DatasetSourceReference'].objects.filter(dataset_id=dataset_pk) | models[
            'DatasetSourceReference'
        ].objects.filter(data_point__dataset_id=dataset_pk)
        for point_pk, uuid, source_uuid in references.values_list('data_point_id', 'uuid', 'data_source__uuid'):
            cited_by = uuids[point_pk] if point_pk is not None else dataset_uuid
            identities.references[cited_by, source_uuid] = uuid
        return identities


def _metric_columns(content: dict[str, Any], dim_columns: Iterable[str]) -> dict[str, str]:
    """Each v1 metric identifier's column in the v1 table (see the v1 `resolve_metric_data_columns`)."""
    fields = (content.get('data') or {}).get('schema', {}).get('fields', [])
    value_columns = {f['name'] for f in fields} - {'Year', 'id', 'uuid', 'index', *dim_columns}
    metric_ids = [metric['identifier'] for metric in content.get('metrics', [])]
    columns: dict[str, str] = {}
    for metric in content.get('metrics', []):
        identifier = metric['identifier']
        labels = list((metric.get('label') or {}).values()) if isinstance(metric.get('label'), dict) else []
        label_match = next((label for label in labels if label in value_columns), None)
        if identifier in value_columns:
            columns[identifier] = identifier
        elif label_match is not None:
            columns[identifier] = label_match
        elif len(metric_ids) == 1 and len(value_columns) == 1:
            columns[identifier] = next(iter(value_columns))
    return columns


def _structure(content: dict[str, Any], base: DatasetMeta | None, dataset_id: UUID) -> DatasetMeta:
    """Return the v2 structure: ``base`` where known, completed and reconciled with what the v1 payload says."""
    metrics = {metric.identifier: metric for metric in base.metrics} if base is not None else {}
    for order, metric in enumerate(content.get('metrics', [])):
        identifier = metric['identifier']
        existing = metrics.get(identifier)
        rules = tuple(ValidationRuleMeta(id=UUID(rule['uuid']), rule=rule['rule']) for rule in metric.get('validation_rules', []))
        if existing is None:
            metrics[identifier] = DatasetMetricMeta(
                id=uuid3(dataset_id, f'metric:{identifier}'),
                identifier=identifier,
                label=metric.get('label'),
                unit=metric.get('unit', ''),
                quantity=metric.get('quantity'),
                order=order,
                validation_rules=rules,
            )
        elif rules:
            metrics[identifier] = existing.model_copy(update={'validation_rules': rules})
    structure = {
        'id': dataset_id,
        'identifier': content.get('identifier'),
        'name': content.get('name'),
        'schema_id': base.schema_id if base is not None else uuid3(dataset_id, 'schema'),
        'is_editable': content.get('is_editable', True),
        'metrics': tuple(metrics.values()),
        'time_resolution': content.get('time_resolution', 'yearly'),
        'forecast_from': content.get('forecast_from'),
        'is_external_placeholder': content.get('is_external_placeholder', False),
        'external_ref': content.get('external_ref'),
        'category_domain': content.get('category_domain') or {},
    }
    if base is not None:
        structure = {**base.model_dump(mode='json'), **structure}
    return DatasetMeta.model_validate(structure)


def upgrade_dataset_snapshot_v1(  # noqa: C901, PLR0912, PLR0915
    content: dict[str, Any],
    *,
    base: DatasetMeta | None,
    dimensions: Iterable[DimensionMeta],
    identities: LiveIdentities | None = None,
    scores: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    Return the v2 form of a v1 dataset payload.

    ``base`` is the dataset's structure where it is known: the catalog entry of the
    instance revision that pinned it, or the live rows. ``dimensions`` resolves the
    payload's dimension and category identifiers. A uuid that ``identities`` does not
    supply is derived from the dataset's uuid and what it names, so the same payload
    always upgrades to the same content. ``scores`` gives quality-level scores by uuid.
    """
    if content.get('schema_version', 1) >= 2:
        return content
    dataset_id = UUID(content['uuid']) if content.get('uuid') else base.id if base is not None else None
    if dataset_id is None:
        raise ValueError(f'Dataset payload {content.get("identifier")!r} has no uuid to upgrade it by')
    meta = _structure(content, base, dataset_id)
    by_identifier = {dimension.identifier: dimension for dimension in dimensions}
    declared: list[DimensionMeta] = []
    for identifier in content.get('dimensions', []):
        dimension = by_identifier.get(identifier)
        if dimension is None:
            raise ValueError(f'Dataset {meta.identifier or dataset_id}: no dimension {identifier!r} to upgrade its payload by')
        declared.append(dimension)
    meta = meta.model_copy(update={'declared_dimension_ids': tuple(dimension.id for dimension in declared)})
    dim_columns = {
        dimension.identifier: content.get('dimension_columns', {}).get(dimension.identifier, dimension.identifier)
        for dimension in declared
    }
    labels = {
        dimension.identifier: {category.identifier or str(category.id): category.id for category in dimension.categories}
        for dimension in declared
    }
    metrics = {metric.identifier: metric for metric in meta.metrics}
    columns = _metric_columns(content, dim_columns.values())
    projected = {metric.identifier for metric in meta.metrics if metric.quality_of is not None}
    live = identities or LiveIdentities()

    points: dict[Cell, DataPointSnapshot] = {}
    by_key: dict[tuple[int, str, tuple[str, ...]], DataPointSnapshot] = {}
    for row in (content.get('data') or {}).get('data', []):
        if row.get('Year') is None:
            continue
        year = int(row['Year'])
        categories: dict[UUID, UUID] = {}
        row_labels: list[str] = []
        for dimension in declared:
            label = row.get(dim_columns[dimension.identifier])
            if not label:
                continue
            category = labels[dimension.identifier].get(str(label))
            if category is None:
                raise ValueError(
                    f'Dataset {meta.identifier or dataset_id}: no category {label!r} in dimension {dimension.identifier!r}'
                )
            categories[dimension.id] = category
            row_labels.append(str(label))
        for identifier, column in columns.items():
            if identifier in projected or column not in row or identifier not in metrics:
                continue  # a projected quality column is derived from evidence, not stored
            metric = metrics[identifier]
            cell = (metric.id, year, frozenset(categories.values()))
            value = row[column]
            existing = points.get(cell)
            if existing is not None and (value is None or existing.value is not None):
                continue  # a split row repeats its cell empty; the valued one wins
            point = DataPointSnapshot(
                id=live.points.get(cell) or uuid3(dataset_id, f'point:{metric.id}:{year}:{sorted(map(str, cell[2]))}'),
                date=date(year, 1, 1),
                metric=metric.id,
                categories=categories,
                value=float(value) if value is not None else None,
            )
            points[cell] = point
            by_key[year, identifier, tuple(sorted(row_labels))] = point

    def locate(key: dict[str, Any]) -> DataPointSnapshot | None:
        return by_key.get((key['year'], key['metric'], tuple(sorted(key.get('categories', [])))))

    for item in content.get('evidence', []):
        point = locate(item['point'])
        if point is None:
            continue
        level = item.get('quality_level')
        if level is not None and scores is not None and level.get('uuid') in scores:
            level = {**level, 'score': scores[level['uuid']]}
        point.evidence = DataPointEvidenceSnapshot.model_validate({**_without_point(item), 'quality_level': level})
    for index, item in enumerate(content.get('comments', [])):
        point = locate(item['point'])
        if point is None:
            continue
        borrowed = live.comments.get((point.id, item['text']))
        comment_id = borrowed.pop(0) if borrowed else uuid3(point.id, f'comment:{index}:{item["text"]}')
        point.comments.append(DataPointCommentSnapshot.model_validate({**_without_point(item), 'id': comment_id}))
    dataset_references: list[SourceReferenceSnapshot] = []
    for item in content.get('source_references', []):
        source_id = UUID(item['data_source'])
        point = locate(item['point']) if item.get('point') is not None else None
        if item.get('point') is not None and point is None:
            continue
        cited_by = point.id if point is not None else dataset_id
        reference = SourceReferenceSnapshot(
            id=live.references.get((cited_by, source_id)) or uuid3(cited_by, f'source:{source_id}'),
            data_source=source_id,
        )
        (point.sources if point is not None else dataset_references).append(reference)

    snapshot = DatasetSnapshot(
        meta=meta,
        dimension_columns={
            dimension.id: column
            for dimension in declared
            if (column := dim_columns[dimension.identifier]) != dimension.identifier
        },
        points=list(points.values()),
        data_sources=[
            DataSourceSnapshot.model_validate({**{k: v for k, v in source.items() if k != 'uuid'}, 'id': source['uuid']})
            for source in content.get('data_sources', [])
        ],
        source_references=dataset_references,
    )
    return snapshot.model_dump(mode='json')


def _without_point(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != 'point'}


def complete_catalog_entry(entry: DatasetMeta, body: DatasetSnapshot | None) -> DatasetMeta:
    """Give a v14 catalog entry what v15 adds to it, from the upgraded dataset body it describes."""
    if body is None:
        return entry
    rules = {metric.id: metric.validation_rules for metric in body.meta.metrics}
    return entry.model_copy(
        update={
            'name': body.meta.name,
            'forecast_from': body.meta.forecast_from,
            'time_resolution': body.meta.time_resolution,
            'metrics': tuple(
                metric.model_copy(update={'validation_rules': rules.get(metric.id, metric.validation_rules)})
                for metric in entry.metrics
            ),
        }
    )
