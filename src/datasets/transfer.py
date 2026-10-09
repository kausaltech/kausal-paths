"""
Transfer dataset bodies, provenance, and dimensions between database scopes.

A `DatasetSnapshot` refers to dimensions, categories and metrics by uuid. The
import still gives everything it creates a new uuid, and finds the target's
dimension categories by identifier, so it translates the snapshot's uuids into
identifiers first (`SourceDimensions`). Keeping the uuids instead is step 5 of
docs/plans/node-owned-datasets.md.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from django.contrib.contenttypes.models import ContentType
from django.db import transaction

from loguru import logger

from kausal_common.datasets.models import (
    DataPoint,
    DataPointComment,
    DataPointCommentReviewState,
    DataPointDimensionCategory,
    Dataset as DatasetModel,
    DatasetMetric as DatasetMetricModel,
    DatasetMetricValidationRule,
    DatasetSchema as DatasetSchemaModel,
    DatasetSchemaDimension,
    DatasetSchemaScope,
    DatasetSourceReference,
    DataSource,
    Dimension as DimensionModel,
    DimensionCategory as DimensionCategoryModel,
    DimensionScope,
)
from kausal_common.i18n.pydantic import TranslatedString

from nodes.snapshot_base import apply_translated
from users.models import User

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    from kausal_common.i18n.pydantic import I18nString

    from datasets.snapshot import DataPointSnapshot, DatasetSnapshot
    from nodes.defs.graph import DimensionMeta
    from nodes.models import InstanceConfig, NodeInputPortBinding


def _label_from_identifier(identifier: str) -> str:
    return identifier.replace('_', ' ').replace('-', ' ').title()


def _apply_i18n(fields: dict[str, Any], i18n: dict[str, str], value: I18nString | None, name: str, language: str) -> None:
    if value is None or isinstance(value, TranslatedString):
        apply_translated(fields, i18n, value, name, language)
    else:
        fields[name] = str(value)


class SourceDimensions:
    """
    The identifiers of the dimensions and categories that dataset snapshots refer to by uuid.

    From ``catalog`` (an export's dimensions) where it has them, else from this
    database, where a category uuid names its row -- which is the case for a copy.
    A category without an identifier is named by its uuid, as frames name it.
    """

    def __init__(self, catalog: Mapping[UUID, DimensionMeta] | None = None) -> None:
        self.dimensions: dict[UUID, str] = {}
        self.categories: dict[UUID, tuple[str, str]] = {}
        for dimension in (catalog or {}).values():
            self.dimensions[dimension.id] = dimension.identifier
            for category in dimension.categories:
                self.categories[category.id] = (dimension.identifier, category.identifier or str(category.id))

    def resolve(self, snapshots: Iterable[DatasetSnapshot]) -> None:
        """Look up in the database whatever the snapshots refer to and the catalog does not name."""
        dimension_ids = {
            d for s in snapshots for d in (*s.meta.declared_dimension_ids, *(c for p in s.points for c in p.categories))
        }
        category_ids = {c for s in snapshots for p in s.points for c in p.categories.values()}
        missing_dimensions = dimension_ids - self.dimensions.keys()
        missing_categories = category_ids - self.categories.keys()
        if not (missing_dimensions or missing_categories):
            return
        scopes = DimensionScope.objects.filter(
            dimension__uuid__in=missing_dimensions
            | set(DimensionCategoryModel.objects.filter(uuid__in=missing_categories).values_list('dimension__uuid', flat=True)),
            identifier__isnull=False,
        ).select_related('dimension')
        for scope in scopes.order_by('pk'):
            assert scope.identifier is not None
            self.dimensions.setdefault(scope.dimension.uuid, scope.identifier)
        for category in DimensionCategoryModel.objects.filter(uuid__in=missing_categories).select_related('dimension'):
            dimension = self.dimensions.get(category.dimension.uuid)
            if dimension is not None:
                self.categories[category.uuid] = (dimension, category.identifier or str(category.uuid))

    def dimension(self, dimension_id: UUID) -> str:
        try:
            return self.dimensions[dimension_id]
        except KeyError:
            raise ValueError(f'Dimension {dimension_id} is neither in the export nor in this database') from None

    def category(self, category_id: UUID) -> tuple[str, str]:
        try:
            return self.categories[category_id]
        except KeyError:
            raise ValueError(f'Dimension category {category_id} is neither in the export nor in this database') from None

    def point_categories(self, point: DataPointSnapshot) -> list[tuple[str, str]]:
        return [self.category(category_id) for category_id in point.categories.values()]


def comparable_cells(
    snapshot: DatasetSnapshot, source: SourceDimensions
) -> dict[tuple[str, int, frozenset[tuple[str, str]]], float | None]:
    """
    Return the dataset's values by metric identifier, year and category identifiers.

    For telling whether two datasets hold the same data when their metrics and categories
    are different rows, as a municipality's copy of a template dataset is.
    """
    metrics = {metric.id: metric.identifier or str(metric.id) for metric in snapshot.meta.metrics}
    return {
        (metrics[point.metric], point.date.year, frozenset(source.point_categories(point))): point.value
        for point in snapshot.points
        if point.metric in metrics
    }


def _import_dataset_evidence(
    ds_snapshot: DatasetSnapshot,
    dataset: DatasetModel,
    points: Mapping[UUID, DataPoint],
) -> None:
    """
    Recreate data-point evidence.

    A grade resolves by UUID, else by authored identity within the schemes that
    apply to the target dataset. An unresolvable grade is dropped with a warning
    rather than failing the import; the evidence kind is kept.
    """
    graded = [(point, point.evidence) for point in ds_snapshot.points if point.evidence is not None]
    if not graded:
        return

    from frameworks.evidence import quality_schemes_for_dataset
    from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel

    levels = list(DataQualityLevel.objects.filter(scheme__in=quality_schemes_for_dataset(dataset)).select_related('scheme'))
    by_uuid = {str(level.uuid): level for level in levels}
    by_identity = {(level.scheme.identifier, level.scheme.version, level.identifier): level for level in levels}
    users = {
        str(u.uuid): u
        for u in User.objects.filter(uuid__in={uid for _, ev in graded for uid in (ev.created_by, ev.last_modified_by) if uid})
    }

    rows = []
    for point, ev in graded:
        dp = points.get(point.id)
        if dp is None:
            continue
        level = None
        ref = ev.quality_level
        if ref is not None:
            scheme_identifier = ref.scheme
            if scheme_identifier == 'bisko':
                # The UUID path preserves deployed evidence; this alias only
                # restores portable payloads written before the scheme rename.
                from frameworks.evidence import framework_for_dataset

                framework = framework_for_dataset(dataset)
                if framework is not None and framework.identifier == 'bisko':
                    scheme_identifier = 'quality'
            level = by_uuid.get(ref.uuid) or by_identity.get((scheme_identifier, ref.scheme_version, ref.level))
            if level is None:
                logger.warning(
                    'Dropping unresolvable quality level %s/%s/%s on dataset %s'
                    % (
                        ref.scheme,
                        ref.scheme_version,
                        ref.level,
                        dataset.identifier,
                    )
                )
        kind = DataEvidenceKind(ev.kind) if ev.kind else None
        if kind is None and level is None:
            continue
        rows.append(
            DataPointEvidence(
                data_point=dp,
                kind=kind,
                quality_level=level,
                created_by=users.get(ev.created_by or ''),
                last_modified_by=users.get(ev.last_modified_by or ''),
            )
        )
    DataPointEvidence.objects.bulk_create(rows)


@transaction.atomic
def import_dataset(
    ic: InstanceConfig,
    ds_snapshot: DatasetSnapshot,
    ic_ct: ContentType,
    dim_lookup: dict[str, DimensionCategoryModel],
    source: SourceDimensions | None = None,
) -> DatasetModel:
    """Create DatasetSchema, Dataset, DatasetMetric, DatasetSchemaDimension, and DataPoints."""
    if source is None:
        source = SourceDimensions()
        source.resolve([ds_snapshot])
    meta = ds_snapshot.meta
    primary_lang = ic.primary_language

    schema_fields: dict[str, Any] = {
        'time_resolution': meta.time_resolution,
        'is_editable': meta.is_editable if meta.is_editable is not None else True,
        'category_domain': meta.category_domain,
    }
    schema_i18n: dict[str, str] = {}
    _apply_i18n(schema_fields, schema_i18n, meta.name, 'name', primary_lang)
    if schema_fields.get('name') is None:
        # DatasetSchema.name is required; fall back to empty if the source
        # snapshot had no name in any language.
        schema_fields['name'] = ''

    schema = DatasetSchemaModel.objects.create(
        i18n=schema_i18n,
        **schema_fields,
    )
    DatasetSchemaScope.objects.create(
        schema=schema,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
    )

    metrics: dict[UUID, DatasetMetricModel] = {}
    for idx, m_meta in enumerate(meta.metrics):
        metric_fields: dict[str, Any] = {}
        metric_i18n: dict[str, str] = {}
        _apply_i18n(metric_fields, metric_i18n, m_meta.label, 'label', primary_lang)
        if metric_fields.get('label') is None:
            metric_fields['label'] = m_meta.identifier
        metric = DatasetMetricModel.objects.create(
            schema=schema,
            name=m_meta.identifier,
            unit=m_meta.unit,
            spec={'quantity': m_meta.quantity} if m_meta.quantity is not None else {},
            order=idx,
            i18n=metric_i18n,
            **metric_fields,
        )
        # Like the metric itself, a restored rule gets a fresh uuid; the
        # snapshot uuid records provenance only.
        for rule_idx, rule_meta in enumerate(m_meta.validation_rules):
            DatasetMetricValidationRule.objects.create(
                metric=metric,
                rule=rule_meta.rule.model_dump(mode='json'),
                order=rule_idx,
            )
        metrics[m_meta.id] = metric

    for idx, dimension_id in enumerate(meta.declared_dimension_ids):
        dim_id = source.dimension(dimension_id)
        dim_scope = DimensionScope.objects.filter(
            identifier=dim_id,
            scope_content_type=ic_ct,
            scope_id=ic.pk,
        ).first()
        if dim_scope:
            DatasetSchemaDimension.objects.create(
                schema=schema,
                dimension=dim_scope.dimension,
                order=idx,
                column_name=ds_snapshot.dimension_columns.get(dimension_id),
            )

    dataset = DatasetModel(
        identifier=meta.identifier,
        spec={'forecast_from': meta.forecast_from} if meta.forecast_from is not None else {},
        is_external_placeholder=meta.is_external_placeholder,
        external_ref=meta.external_ref,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        schema=schema,
    )
    dataset.save()

    points = _import_data_points(dataset, ds_snapshot, metrics, dim_lookup, source)
    _import_dataset_provenance(ic, ic_ct, ds_snapshot, dataset, points)
    _import_dataset_evidence(ds_snapshot, dataset, points)

    if not dataset.is_external_placeholder:
        from datasets.materialization import refresh_dataset_materialization

        refresh_dataset_materialization(dataset)

    return dataset


class _UserLookup:
    """Users by uuid, each looked up once; a user this database does not have is None."""

    def __init__(self) -> None:
        self._users: dict[str, User | None] = {}

    def __call__(self, user_uuid: str | None) -> User | None:
        if not user_uuid:
            return None
        if user_uuid not in self._users:
            self._users[user_uuid] = User.objects.filter(uuid=user_uuid).first()
        return self._users[user_uuid]


def _import_data_sources(ic: InstanceConfig, ic_ct: ContentType, ds_snapshot: DatasetSnapshot) -> dict[UUID, DataSource]:
    """
    Create the snapshot's DataSources in the target instance's scope; return them by the snapshot's uuid.

    They get fresh uuids: a same-database copy cannot reuse the globally unique source uuid.
    """
    return {
        s.id: DataSource.objects.create(
            scope_content_type=ic_ct,
            scope_id=ic.pk,
            name=s.name,
            edition=s.edition,
            authority=s.authority,
            description=s.description,
            url=s.url,
        )
        for s in ds_snapshot.data_sources
    }


def _import_dataset_provenance(
    ic: InstanceConfig,
    ic_ct: ContentType,
    ds_snapshot: DatasetSnapshot,
    dataset: DatasetModel,
    points: Mapping[UUID, DataPoint],
) -> None:
    """Recreate a dataset's DataSources, source references and data-point comments."""
    resolve_user = _UserLookup()
    src_map = _import_data_sources(ic, ic_ct, ds_snapshot)

    for ref in ds_snapshot.source_references:
        src_obj = src_map.get(ref.data_source)
        if src_obj is not None:
            DatasetSourceReference.objects.create(dataset=dataset, data_source=src_obj)

    for point in ds_snapshot.points:
        dp = points.get(point.id)
        if dp is None:
            continue
        for ref in point.sources:
            src_obj = src_map.get(ref.data_source)
            if src_obj is not None:
                DatasetSourceReference.objects.create(data_point=dp, data_source=src_obj)
        for c in point.comments:
            DataPointComment.objects.create(
                data_point=dp,
                text=c.text,
                is_sticky=c.is_sticky,
                is_review=c.is_review,
                review_state=DataPointCommentReviewState(c.review_state) if c.review_state else None,
                resolved_at=datetime.fromisoformat(c.resolved_at) if c.resolved_at else None,
                created_by=resolve_user(c.created_by),
                last_modified_by=resolve_user(c.last_modified_by),
                resolved_by=resolve_user(c.resolved_by),
            )


def _import_data_points(
    dataset: DatasetModel,
    ds_snapshot: DatasetSnapshot,
    metrics: Mapping[UUID, DatasetMetricModel],
    dim_lookup: dict[str, DimensionCategoryModel],
    source: SourceDimensions,
) -> dict[UUID, DataPoint]:
    """Create the DataPoints; return them by the snapshot's data point uuid, for wiring provenance."""
    declared = set(ds_snapshot.meta.declared_dimension_ids)
    created: list[tuple[UUID, DataPoint, list[DimensionCategoryModel]]] = []
    for point in ds_snapshot.points:
        metric = metrics.get(point.metric)
        if metric is None:
            continue
        categories = [
            dim_lookup[f'{dim_id}/{cat_id}']
            for dimension_id, category_id in point.categories.items()
            if dimension_id in declared
            for dim_id, cat_id in [source.category(category_id)]
            if f'{dim_id}/{cat_id}' in dim_lookup
        ]
        # An empty cell is kept as a null-valued data point, as `load_dvc_dataset` creates
        # it: a template dataset is all empty cells, and dropping them leaves a dataset with
        # no data points and so no payload, which the runtime cannot load.
        created.append((point.id, DataPoint(dataset=dataset, date=point.date, metric=metric, value=point.value), categories))

    DataPoint.objects.bulk_create([dp for _, dp, _ in created])
    DataPointDimensionCategory.objects.bulk_create([
        DataPointDimensionCategory(data_point=dp, dimension_category=category)
        for _, dp, categories in created
        for category in categories
    ])
    return {point_id: dp for point_id, dp, _ in created}


def _dimension_category_lookup_for_instance(ic: InstanceConfig, ic_ct: ContentType) -> dict[str, DimensionCategoryModel]:
    lookup: dict[str, DimensionCategoryModel] = {}
    scopes = (
        DimensionScope.objects
        .filter(scope_content_type=ic_ct, scope_id=ic.pk, identifier__isnull=False)
        .select_related('dimension')
        .prefetch_related('dimension__categories')
    )
    for scope in scopes:
        assert scope.identifier is not None
        for category in scope.dimension.categories.all():
            if category.identifier is not None:
                lookup[f'{scope.identifier}/{category.identifier}'] = category
    return lookup


def _ensure_dataset_dimensions(
    ic: InstanceConfig,
    ds_snapshot: DatasetSnapshot,
    ic_ct: ContentType,
    dim_lookup: dict[str, DimensionCategoryModel],
    source: SourceDimensions,
) -> None:
    for dimension_id in ds_snapshot.meta.declared_dimension_ids:
        dim_id = source.dimension(dimension_id)
        dim_scope = (
            DimensionScope.objects
            .filter(
                identifier=dim_id,
                scope_content_type=ic_ct,
                scope_id=ic.pk,
            )
            .select_related('dimension')
            .first()
        )
        if dim_scope is None:
            dimension = DimensionModel.objects.create(name=_label_from_identifier(dim_id))
            DimensionScope.objects.create(
                dimension=dimension,
                identifier=dim_id,
                scope_content_type=ic_ct,
                scope_id=ic.pk,
            )
        else:
            dimension = dim_scope.dimension

        existing_categories = set(dimension.categories.values_list('identifier', flat=True))
        category_ids = sorted({
            cat_id
            for point in ds_snapshot.points
            if (category_id := point.categories.get(dimension_id)) is not None
            for _, cat_id in [source.category(category_id)]
        })
        for cat_id in category_ids:
            if cat_id in existing_categories:
                continue
            cat = DimensionCategoryModel.objects.create(
                dimension=dimension,
                identifier=cat_id,
                label=_label_from_identifier(cat_id),
            )
            dim_lookup[f'{dim_id}/{cat_id}'] = cat
            existing_categories.add(cat_id)


def _validate_dataset_dimensions(
    ic: InstanceConfig,
    ds_snapshot: DatasetSnapshot,
    dim_lookup: dict[str, DimensionCategoryModel],
    source: SourceDimensions,
) -> None:
    declared = set(ds_snapshot.meta.declared_dimension_ids)
    missing = {
        f'{dim_id}/{cat_id}'
        for point in ds_snapshot.points
        for dimension_id, category_id in point.categories.items()
        if dimension_id in declared
        for dim_id, cat_id in [source.category(category_id)]
        if f'{dim_id}/{cat_id}' not in dim_lookup
    }
    if missing:
        missing_str = ', '.join(sorted(missing)[:10])
        if len(missing) > 10:
            missing_str += ', ...'
        raise ValueError(
            f'Cannot import dataset {ds_snapshot.meta.identifier!r} into {ic.identifier!r}; missing dimension categories: '
            + f'{missing_str}'
        )


def _rewire_port(port: NodeInputPortBinding, dataset: DatasetModel) -> None:
    """Point a dataset port at `dataset`, keeping the metric by name."""
    assert port.metric is not None
    assert dataset.schema is not None
    metric = dataset.schema.metrics.filter(name=port.metric.name).first()
    if metric is None:
        raise ValueError(
            f'Cannot rewire dataset port {port.pk} to dataset {dataset.identifier!r}; metric {port.metric.name!r} is missing'
        )
    port.dataset = dataset
    port.metric = metric
    port.save(update_fields=['dataset', 'metric'])


def _rewire_dataset_ports(ic: InstanceConfig, datasets_by_id: dict[str, DatasetModel]) -> int:
    from nodes.models import NodeInputPortBinding

    rewired = 0
    ports = NodeInputPortBinding.objects.filter(instance=ic, dataset__identifier__in=datasets_by_id).select_related(
        'dataset', 'metric'
    )
    for port in ports:
        assert port.dataset is not None
        if port.dataset.identifier is None:
            continue
        dataset = datasets_by_id.get(port.dataset.identifier)
        if dataset is None or dataset.pk == port.dataset_id:
            continue
        _rewire_port(port, dataset)
        rewired += 1
    return rewired


def _set_aside_placeholder(ic: InstanceConfig, ic_ct: ContentType, identifier: str) -> DatasetModel | None:
    """Free the identifier of the instance's own placeholder for it, so a real dataset can take it."""
    placeholder = DatasetModel.objects.filter(
        scope_content_type=ic_ct, scope_id=ic.pk, identifier=identifier, is_external_placeholder=True
    ).first()
    if placeholder is None:
        return None
    placeholder.identifier = None
    placeholder.save(update_fields=['identifier'])
    return placeholder


def _replace_placeholder(placeholder: DatasetModel, dataset: DatasetModel) -> None:
    """Move the placeholder's ports to `dataset`, then delete it and its schema if nothing else uses it."""
    from nodes.models import NodeInputPortBinding

    for port in NodeInputPortBinding.objects.filter(dataset=placeholder).select_related('metric'):
        _rewire_port(port, dataset)
    schema = placeholder.schema
    placeholder.delete()
    if schema is not None and not schema.datasets.exists():
        schema.delete()


def _already_imported(ic: InstanceConfig, ic_ct: ContentType, ds_snapshot: DatasetSnapshot) -> DatasetModel | None:
    """Return the instance's real dataset for this snapshot's identifier, if it already holds the data."""
    identifier = ds_snapshot.meta.identifier
    if identifier is None:
        return None
    existing = DatasetModel.objects.filter(
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        identifier=identifier,
        is_external_placeholder=False,
    ).first()
    if existing is None:
        return None
    if not ds_snapshot.points or existing.data_points.exists():
        return existing
    raise ValueError(f'Dataset {identifier!r} already exists for {ic.identifier!r} but has no datapoints')


def import_instance_datasets(
    ic: InstanceConfig,
    dataset_snapshots: Iterable[DatasetSnapshot],
    *,
    rewire_dataset_ports: bool = False,
    replace_placeholders: bool = False,
    create_missing_dimensions: bool = False,
    dimensions: Mapping[UUID, DimensionMeta] | None = None,
) -> list[DatasetModel]:
    """
    Import dataset bodies into an existing InstanceConfig without touching nodes.

    This is used when a template instance already has its node graph and
    dataset ports, but its datasets need to be promoted from external
    placeholders to real DB datasets with datapoints. With `replace_placeholders`,
    an imported dataset takes over the instance's own placeholder of the same
    identifier: its ports, then its place.

    ``dimensions`` is the catalog the snapshots' dimension and category uuids are
    looked up in; whatever it does not name is looked up in this database.
    """
    dataset_snapshots = list(dataset_snapshots)
    source = SourceDimensions(dimensions)
    source.resolve(dataset_snapshots)
    ic_ct = ContentType.objects.get_for_model(ic)
    dim_lookup = _dimension_category_lookup_for_instance(ic, ic_ct)
    imported: list[DatasetModel] = []
    datasets_by_id: dict[str, DatasetModel] = {}
    replaced: list[tuple[DatasetModel, DatasetModel]] = []

    for ds_snapshot in dataset_snapshots:
        identifier = ds_snapshot.meta.identifier
        existing = _already_imported(ic, ic_ct, ds_snapshot)
        if existing is not None:
            assert identifier is not None
            imported.append(existing)
            datasets_by_id[identifier] = existing
            continue

        if create_missing_dimensions:
            _ensure_dataset_dimensions(ic, ds_snapshot, ic_ct, dim_lookup, source)
        _validate_dataset_dimensions(ic, ds_snapshot, dim_lookup, source)
        placeholder = None
        if replace_placeholders and identifier is not None:
            placeholder = _set_aside_placeholder(ic, ic_ct, identifier)
        dataset = import_dataset(ic, ds_snapshot, ic_ct, dim_lookup, source)
        imported.append(dataset)
        if identifier is not None:
            datasets_by_id[identifier] = dataset
        if placeholder is not None:
            replaced.append((placeholder, dataset))

    for placeholder, dataset in replaced:
        _replace_placeholder(placeholder, dataset)
    if rewire_dataset_ports:
        _rewire_dataset_ports(ic, datasets_by_id)

    return imported
