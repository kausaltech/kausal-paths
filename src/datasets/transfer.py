"""
Transfer dataset bodies, provenance, and dimensions between database scopes.

A `DatasetSnapshot` refers to dimensions, categories and metrics by uuid, and the import
keeps every uuid it holds: an import into another database reproduces the dataset. A copy
within one database rekeys the snapshot first (`paths.rekey`). The target's dimensions are
found by uuid, else by identifier (`TargetDimensions`), so a copy can be merged into an
instance whose dimensions are rows of its own.
"""

from dataclasses import dataclass, field
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
from kausal_common.models.ordered import OrderedModel

from datasets.catalogue import dataset_spec_from_meta
from frameworks.evidence import QUALITY_OF_SPEC_KEY
from nodes.snapshot_base import apply_translated
from users.models import User

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    from kausal_common.i18n.pydantic import I18nString

    from datasets.snapshot import DataPointSnapshot, DatasetSnapshot, DataSourceSnapshot
    from nodes.defs.graph import DatasetMeta, DimensionMeta
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


def save_in_order[M: OrderedModel](obj: M, order: int) -> M:
    """Create ``obj`` at ``order``, which `OrderedModel` would otherwise replace with the next free one."""
    obj.order_on_create = order
    obj.save()
    return obj


class TargetDimensions:
    """
    The target instance's dimension and category rows for the uuids a dataset snapshot names.

    By uuid among the dimensions the instance sees, its own and its framework's, which is
    every case once an import keeps uuids. Else by identifier, through the names the export
    gives them (`SourceDimensions`): a copy merged into an instance that already has
    dimensions of its own, such as a template taking datasets from another template.
    """

    def __init__(self, ic: InstanceConfig, source: SourceDimensions) -> None:
        self.ic = ic
        self.source = source
        self.reload()

    def reload(self) -> None:
        from frameworks.catalogue import dimension_scopes

        self.dimensions_by_uuid: dict[UUID, DimensionModel] = {}
        self.categories_by_uuid: dict[UUID, DimensionCategoryModel] = {}
        self.dimensions_by_identifier: dict[str, DimensionModel] = {}
        self.categories_by_identifier: dict[tuple[str, str], DimensionCategoryModel] = {}
        scopes = dimension_scopes(self.ic).select_related('dimension').prefetch_related('dimension__categories')
        for scope in scopes.order_by('pk'):
            dimension = scope.dimension
            self.dimensions_by_uuid[dimension.uuid] = dimension
            if scope.identifier is not None:
                self.dimensions_by_identifier.setdefault(scope.identifier, dimension)
            for category in dimension.categories.all():
                self.categories_by_uuid[category.uuid] = category
                if scope.identifier is not None and category.identifier is not None:
                    self.categories_by_identifier.setdefault((scope.identifier, category.identifier), category)

    def dimension(self, dimension_id: UUID) -> DimensionModel | None:
        found = self.dimensions_by_uuid.get(dimension_id)
        if found is None and (identifier := self.source.dimensions.get(dimension_id)) is not None:
            found = self.dimensions_by_identifier.get(identifier)
        return found

    def category(self, category_id: UUID) -> DimensionCategoryModel | None:
        found = self.categories_by_uuid.get(category_id)
        if found is None and (names := self.source.categories.get(category_id)) is not None:
            found = self.categories_by_identifier.get(names)
        return found


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


@dataclass
class _ImportState:
    """What one import has created so far, for datasets that share it."""

    schemas: dict[UUID, DatasetSchemaModel] = field(default_factory=dict)
    data_sources: dict[UUID, DataSource] = field(default_factory=dict)
    users: _UserLookup = field(default_factory=_UserLookup)


@transaction.atomic
def import_dataset(
    ic: InstanceConfig,
    ds_snapshot: DatasetSnapshot,
    *,
    dimensions: TargetDimensions | None = None,
    state: _ImportState | None = None,
) -> DatasetModel:
    """
    Create the dataset, its schema and its data points, keeping every uuid.

    A framework's schema is not created but found, with its metrics, by uuid. A copy within
    one database rekeys the snapshot first (`paths.rekey.rekeyed`), since every uuid it
    holds is taken.
    """
    if dimensions is None:
        source = SourceDimensions()
        source.resolve([ds_snapshot])
        dimensions = TargetDimensions(ic, source)
    state = state or _ImportState()
    ic_ct = ContentType.objects.get_for_model(ic)
    meta = ds_snapshot.meta
    schema, metrics = _import_schema(ic, ic_ct, ds_snapshot, dimensions, state)

    dataset = DatasetModel(
        uuid=meta.id,
        identifier=meta.identifier,
        spec=dataset_spec_from_meta(meta),
        is_external_placeholder=meta.is_external_placeholder,
        external_ref=meta.external_ref.model_dump() if meta.external_ref is not None else None,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        schema=schema,
    )
    dataset.save()

    points = _import_data_points(dataset, ds_snapshot, metrics, dimensions)
    _import_dataset_provenance(ic, ic_ct, ds_snapshot, dataset, points, state)
    _import_dataset_evidence(ds_snapshot, dataset, points)

    if not dataset.is_external_placeholder:
        from datasets.materialization import refresh_dataset_materialization

        refresh_dataset_materialization(dataset)

    return dataset


def _import_schema(
    ic: InstanceConfig,
    ic_ct: ContentType,
    ds_snapshot: DatasetSnapshot,
    dimensions: TargetDimensions,
    state: _ImportState,
) -> tuple[DatasetSchemaModel, dict[UUID, DatasetMetricModel]]:
    """Return the dataset's schema and its metrics by uuid: found when shared, created otherwise."""
    meta = ds_snapshot.meta
    schema = _shared_schema(meta, state)
    if schema is not None:
        metrics = {metric.uuid: metric for metric in schema.metrics.all()}
        missing = [metric.identifier or str(metric.id) for metric in meta.metrics if metric.id not in metrics]
        if missing:
            raise ValueError(f'Schema {schema.uuid} of dataset {meta.identifier or meta.id} lacks metrics {missing}')
        return schema, metrics

    schema_fields: dict[str, Any] = {
        'time_resolution': meta.time_resolution,
        'is_editable': meta.is_editable if meta.is_editable is not None else True,
        'category_domain': meta.category_domain,
    }
    schema_i18n: dict[str, str] = {}
    _apply_i18n(schema_fields, schema_i18n, meta.name, 'name', ic.primary_language)
    if schema_fields.get('name') is None:
        # DatasetSchema.name is required; fall back to empty if the source
        # snapshot had no name in any language.
        schema_fields['name'] = ''
    schema = DatasetSchemaModel.objects.create(uuid=meta.schema_id, i18n=schema_i18n, **schema_fields)
    DatasetSchemaScope.objects.create(schema=schema, scope_content_type=ic_ct, scope_id=ic.pk)
    state.schemas[schema.uuid] = schema
    metrics = {metric.uuid: metric for metric in _create_metrics(schema, meta, ic.primary_language)}

    for idx, dimension_id in enumerate(meta.declared_dimension_ids):
        dimension = dimensions.dimension(dimension_id)
        if dimension is None:
            raise ValueError(f'Dataset {meta.identifier or meta.id} declares dimension {dimension_id}, which is not here')
        DatasetSchemaDimension.objects.create(
            schema=schema, dimension=dimension, order=idx, column_name=ds_snapshot.dimension_columns.get(dimension_id)
        )
    return schema, metrics


def _shared_schema(meta: DatasetMeta, state: _ImportState) -> DatasetSchemaModel | None:
    """Return the schema this import already created for ``meta``, or the framework's; None to create it."""
    schema = state.schemas.get(meta.schema_id)
    if schema is None and meta.schema_scope == 'framework':
        schema = DatasetSchemaModel.objects.filter(uuid=meta.schema_id).first()
        if schema is None:
            raise ValueError(f'Dataset {meta.identifier or meta.id} uses framework schema {meta.schema_id}, which is not here')
    return schema


def _create_metrics(schema: DatasetSchemaModel, meta: DatasetMeta, language: str) -> list[DatasetMetricModel]:
    metrics: list[DatasetMetricModel] = []
    for idx, m_meta in enumerate(meta.metrics):
        metric_fields: dict[str, Any] = {}
        metric_i18n: dict[str, str] = {}
        _apply_i18n(metric_fields, metric_i18n, m_meta.label, 'label', language)
        if metric_fields.get('label') is None:
            metric_fields['label'] = m_meta.identifier
        spec: dict[str, Any] = {'quantity': m_meta.quantity} if m_meta.quantity is not None else {}
        if m_meta.quality_of is not None:
            spec[QUALITY_OF_SPEC_KEY] = str(m_meta.quality_of)
        metric = save_in_order(
            DatasetMetricModel(
                uuid=m_meta.id,
                schema=schema,
                name=m_meta.identifier,
                unit=m_meta.unit,
                spec=spec,
                i18n=metric_i18n,
                **metric_fields,
            ),
            m_meta.order if m_meta.order is not None else idx,
        )
        for rule_idx, rule_meta in enumerate(m_meta.validation_rules):
            DatasetMetricValidationRule.objects.create(
                **({'uuid': rule_meta.id} if rule_meta.id is not None else {}),
                metric=metric,
                rule=rule_meta.rule.model_dump(mode='json'),
                order=rule_idx,
            )
        metrics.append(metric)
    return metrics


def _data_source(ic: InstanceConfig, ic_ct: ContentType, source: DataSourceSnapshot, state: _ImportState) -> DataSource:
    """Return the snapshot's DataSource in the target instance's scope, created once per import."""
    found = state.data_sources.get(source.id)
    if found is None:
        found = DataSource.objects.create(
            uuid=source.id,
            scope_content_type=ic_ct,
            scope_id=ic.pk,
            name=source.name,
            edition=source.edition,
            authority=source.authority,
            description=source.description,
            url=source.url,
        )
        state.data_sources[source.id] = found
    return found


def _import_dataset_provenance(
    ic: InstanceConfig,
    ic_ct: ContentType,
    ds_snapshot: DatasetSnapshot,
    dataset: DatasetModel,
    points: Mapping[UUID, DataPoint],
    state: _ImportState,
) -> None:
    """Recreate a dataset's DataSources, source references and data-point comments, keeping their uuids."""
    sources = {source.id: _data_source(ic, ic_ct, source, state) for source in ds_snapshot.data_sources}

    for ref in ds_snapshot.source_references:
        if (data_source := sources.get(ref.data_source)) is not None:
            DatasetSourceReference.objects.create(uuid=ref.id, dataset=dataset, data_source=data_source)

    resolve_user = state.users
    for point in ds_snapshot.points:
        dp = points.get(point.id)
        if dp is None:
            continue
        for ref in point.sources:
            if (data_source := sources.get(ref.data_source)) is not None:
                DatasetSourceReference.objects.create(uuid=ref.id, data_point=dp, data_source=data_source)
        for c in point.comments:
            DataPointComment.objects.create(
                uuid=c.id,
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
    dimensions: TargetDimensions,
) -> dict[UUID, DataPoint]:
    """Create the DataPoints with their uuids; return them by uuid, for wiring provenance."""
    declared = set(ds_snapshot.meta.declared_dimension_ids)
    created: list[tuple[DataPoint, list[DimensionCategoryModel]]] = []
    for point in ds_snapshot.points:
        metric = metrics.get(point.metric)
        if metric is None:
            continue
        categories = [
            category
            for dimension_id, category_id in point.categories.items()
            if dimension_id in declared and (category := dimensions.category(category_id)) is not None
        ]
        # An empty cell is kept as a null-valued data point, as `load_dvc_dataset` creates
        # it: a template dataset is all empty cells, and dropping them leaves a dataset with
        # no data points and so no payload, which the runtime cannot load.
        created.append((DataPoint(uuid=point.id, dataset=dataset, date=point.date, metric=metric, value=point.value), categories))

    DataPoint.objects.bulk_create([dp for dp, _ in created])
    DataPointDimensionCategory.objects.bulk_create([
        DataPointDimensionCategory(data_point=dp, dimension_category=category)
        for dp, categories in created
        for category in categories
    ])
    return {dp.uuid: dp for dp, _ in created}


def _ensure_dataset_dimensions(ic: InstanceConfig, ds_snapshot: DatasetSnapshot, dimensions: TargetDimensions) -> None:
    """
    Create the dimensions and categories the snapshot names and the instance lacks.

    Under the snapshot's uuids where they are free. A copy of a dataset alone still names its
    source instance's dimensions, and the target gets dimensions of its own, with new uuids.
    """
    ic_ct = ContentType.objects.get_for_model(ic)
    source = dimensions.source
    created = False
    for dimension_id in ds_snapshot.meta.declared_dimension_ids:
        dimension = dimensions.dimension(dimension_id)
        if dimension is None:
            identifier = source.dimension(dimension_id)
            dimension = DimensionModel.objects.create(
                **_free_uuid(DimensionModel, dimension_id), name=_label_from_identifier(identifier)
            )
            DimensionScope.objects.create(dimension=dimension, identifier=identifier, scope_content_type=ic_ct, scope_id=ic.pk)
            created = True
        for category_id in sorted({c for point in ds_snapshot.points if (c := point.categories.get(dimension_id)) is not None}):
            if dimensions.category(category_id) is not None:
                continue
            _, identifier = source.category(category_id)
            DimensionCategoryModel.objects.create(
                **_free_uuid(DimensionCategoryModel, category_id),
                dimension=dimension,
                identifier=identifier,
                label=_label_from_identifier(identifier),
            )
            created = True
    if created:
        dimensions.reload()


def _free_uuid(model: type[DimensionModel | DimensionCategoryModel], uuid: UUID) -> dict[str, UUID]:
    return {} if model._default_manager.filter(uuid=uuid).exists() else {'uuid': uuid}


def _validate_dataset_dimensions(ic: InstanceConfig, ds_snapshot: DatasetSnapshot, dimensions: TargetDimensions) -> None:
    declared = set(ds_snapshot.meta.declared_dimension_ids)
    missing = {
        '/'.join(dimensions.source.categories.get(category_id, (str(dimension_id), str(category_id))))
        for point in ds_snapshot.points
        for dimension_id, category_id in point.categories.items()
        if dimension_id in declared and dimensions.category(category_id) is None
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
    Import dataset bodies into an existing InstanceConfig without touching nodes, keeping their uuids.

    Used by the instance import, and by callers that add datasets to an instance that already
    has its node graph, such as a template promoting its placeholders to real datasets. With
    `replace_placeholders`, an imported dataset takes over the instance's own placeholder of the
    same identifier: its ports, then its place. Datasets copied from this database must be
    rekeyed first.

    ``dimensions`` is the catalog the snapshots' dimension and category uuids are named in;
    whatever it does not name is looked up in this database.
    """
    dataset_snapshots = list(dataset_snapshots)
    source = SourceDimensions(dimensions)
    source.resolve(dataset_snapshots)
    target = TargetDimensions(ic, source)
    state = _ImportState()
    ic_ct = ContentType.objects.get_for_model(ic)
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
            _ensure_dataset_dimensions(ic, ds_snapshot, target)
        _validate_dataset_dimensions(ic, ds_snapshot, target)
        placeholder = None
        if replace_placeholders and identifier is not None:
            placeholder = _set_aside_placeholder(ic, ic_ct, identifier)
        dataset = import_dataset(ic, ds_snapshot, dimensions=target, state=state)
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
