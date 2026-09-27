"""Transfer dataset bodies, provenance, and dimensions between database scopes."""

from datetime import date, datetime
from itertools import chain
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

from datasets.snapshot import (
    DataPointCommentSnapshot,
    DataPointEvidenceSnapshot,
    DataPointKey,
    DatasetSnapshot,
    DataSourceSnapshot,
    SourceReferenceSnapshot,
    metric_column_id,
)
from nodes.snapshot_base import apply_translated
from users.models import User

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nodes.models import InstanceConfig, NodeInputPortBinding


def _label_from_identifier(identifier: str) -> str:
    return identifier.replace('_', ' ').replace('-', ' ').title()


def export_dataset_provenance(
    ds: DatasetModel,
) -> tuple[list[DataSourceSnapshot], list[SourceReferenceSnapshot], list[DataPointCommentSnapshot]]:
    """Serialize a dataset's source references and (non-soft-deleted) data-point comments."""
    sources: dict[str, DataSourceSnapshot] = {}

    def add_source(src: DataSource) -> None:
        key = str(src.uuid)
        if key not in sources:
            sources[key] = DataSourceSnapshot.from_model(src)

    dataset_refs = DatasetSourceReference.objects.filter(dataset=ds).select_related('data_source')
    dp_refs = (
        DatasetSourceReference.objects
        .filter(data_point__dataset=ds)
        .select_related('data_source', 'data_point__metric')
        .prefetch_related('data_point__dimension_categories')
    )
    references: list[SourceReferenceSnapshot] = []
    for reference in chain(dataset_refs, dp_refs):
        add_source(reference.data_source)
        references.append(SourceReferenceSnapshot.from_model(reference))

    comment_qs = (
        DataPointComment.objects  # default manager excludes soft-deleted
        .filter(data_point__dataset=ds)
        .select_related('data_point__metric', 'created_by', 'last_modified_by', 'resolved_by')
        .prefetch_related('data_point__dimension_categories')
    )
    comments = [DataPointCommentSnapshot.from_model(comment) for comment in comment_qs]
    return list(sources.values()), references, comments


def export_dataset_evidence(ds: DatasetModel) -> list[DataPointEvidenceSnapshot]:
    from frameworks.models import DataPointEvidence

    evidence_qs = (
        DataPointEvidence.objects
        .filter(data_point__dataset=ds)
        .select_related('data_point__metric', 'quality_level__scheme', 'created_by', 'last_modified_by')
        .prefetch_related('data_point__dimension_categories')
        .order_by('data_point_id')
    )
    return [DataPointEvidenceSnapshot.from_model(evidence) for evidence in evidence_qs]


def _import_dataset_evidence(
    ds_snapshot: DatasetSnapshot,
    dataset: DatasetModel,
    dp_map: dict[tuple[int, str, tuple[str, ...]], Any],
) -> None:
    """
    Recreate data-point evidence.

    A grade resolves by UUID, else by authored identity within the schemes that
    apply to the target dataset. An unresolvable grade is dropped with a warning
    rather than failing the import; the evidence kind is kept.
    """
    if not ds_snapshot.evidence:
        return

    from frameworks.evidence import quality_schemes_for_dataset
    from frameworks.models import DataEvidenceKind, DataPointEvidence, DataQualityLevel

    levels = list(DataQualityLevel.objects.filter(scheme__in=quality_schemes_for_dataset(dataset)).select_related('scheme'))
    by_uuid = {str(level.uuid): level for level in levels}
    by_identity = {(level.scheme.identifier, level.scheme.version, level.identifier): level for level in levels}
    users = {
        str(u.uuid): u
        for u in User.objects.filter(
            uuid__in={uid for ev in ds_snapshot.evidence for uid in (ev.created_by, ev.last_modified_by) if uid}
        )
    }

    rows = []
    for ev in ds_snapshot.evidence:
        dp = dp_map.get(_data_point_key_tuple(ev.point))
        if dp is None:
            continue
        level = None
        ref = ev.quality_level
        if ref is not None:
            level = by_uuid.get(ref.uuid) or by_identity.get((ref.scheme, ref.scheme_version, ref.level))
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


def _export_dataset_data(ds: DatasetModel) -> dict[str, Any]:
    """Serialize dataset DataPoints into JSON Table Schema format."""
    from nodes.datasets import DBDataset, JSONDataset

    df = DBDataset.deserialize_df(ds)
    return JSONDataset.serialize_df(df)


def export_dataset_data_safe(ds: DatasetModel) -> dict[str, Any] | None:
    """
    Serialize DataPoints.

    An empty dataset still needs a typed dataframe payload so inherited model
    inputs can evaluate it as missing municipal data. Returns ``None`` when
    deserialization fails (e.g. mis-seeded datasets during tests). Robustness
    matters here because ``serializable_data()`` is
    called on every ``save_revision`` and must not crash on edge cases.
    """
    if not ds.data_points.exists():
        data = _export_dataset_data(ds)
        fields = data['schema']['fields']
        for field in fields:
            if field['type'] == 'any':
                field['type'] = 'string'
                field.pop('constraints', None)
                field.pop('ordered', None)
        present = {field['name'] for field in fields}
        if ds.schema is not None:
            for metric in ds.schema.metrics.all():
                name = metric_column_id(metric)
                if name in present:
                    continue
                field = {'name': name, 'type': 'number', 'unit': metric.unit}
                fields.append(field)
        return data
    return _export_dataset_data(ds)


@transaction.atomic
def import_dataset(
    ic: InstanceConfig,
    ds_snapshot: DatasetSnapshot,
    ic_ct: ContentType,
    dim_lookup: dict[str, DimensionCategoryModel],
) -> DatasetModel:
    """Create DatasetSchema, Dataset, DatasetMetric, DatasetSchemaDimension, and DataPoints."""

    primary_lang = ic.primary_language

    # Resolve schema name from the TranslatedString snapshot.
    schema_fields: dict[str, Any] = {
        'time_resolution': ds_snapshot.time_resolution,
        'is_editable': ds_snapshot.is_editable,
        'category_domain': ds_snapshot.category_domain,
    }
    schema_i18n: dict[str, str] = {}
    apply_translated(schema_fields, schema_i18n, ds_snapshot.name, 'name', primary_lang)
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

    # Create metrics (metric.label is TranslatedString in the snapshot)
    metrics_by_id: dict[str, DatasetMetricModel] = {}
    for idx, m_snap in enumerate(ds_snapshot.metrics):
        metric_fields: dict[str, Any] = {}
        metric_i18n: dict[str, str] = {}
        apply_translated(metric_fields, metric_i18n, m_snap.label, 'label', primary_lang)
        if metric_fields.get('label') is None:
            metric_fields['label'] = m_snap.identifier
        metric = DatasetMetricModel.objects.create(
            schema=schema,
            name=m_snap.identifier,
            unit=m_snap.unit,
            spec={'quantity': m_snap.quantity} if m_snap.quantity is not None else {},
            order=idx,
            i18n=metric_i18n,
            **metric_fields,
        )
        # Like the metric itself, a restored rule gets a fresh uuid; the
        # snapshot uuid records provenance only.
        for rule_idx, rule_snap in enumerate(m_snap.validation_rules):
            DatasetMetricValidationRule.objects.create(
                metric=metric,
                rule=rule_snap.rule.model_dump(mode='json'),
                order=rule_idx,
            )
        metrics_by_id[m_snap.identifier] = metric

    # Link dimensions to schema
    for idx, dim_id in enumerate(ds_snapshot.dimensions):
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
                column_name=ds_snapshot.dimension_columns.get(dim_id),
            )

    # Create dataset
    dataset = DatasetModel(
        identifier=ds_snapshot.identifier,
        spec={'forecast_from': ds_snapshot.forecast_from} if ds_snapshot.forecast_from is not None else {},
        is_external_placeholder=ds_snapshot.is_external_placeholder,
        external_ref=ds_snapshot.external_ref,
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        schema=schema,
    )
    dataset.save()

    # Create data points
    dp_map: dict[tuple[int, str, tuple[str, ...]], Any] = {}
    if ds_snapshot.data is not None:
        dp_map = _import_data_points(dataset, ds_snapshot, metrics_by_id, dim_lookup)

    # Recreate source references and comments (data points must exist first).
    _import_dataset_provenance(ic, ic_ct, ds_snapshot, dataset, dp_map)
    _import_dataset_evidence(ds_snapshot, dataset, dp_map)

    if not dataset.is_external_placeholder:
        from datasets.materialization import refresh_dataset_materialization

        refresh_dataset_materialization(dataset)

    return dataset


def _data_point_key_tuple(point: DataPointKey) -> tuple[int, str, tuple[str, ...]]:
    return (point.year, point.metric, tuple(point.categories))


def _import_dataset_provenance(  # noqa: C901
    ic: InstanceConfig,
    ic_ct: ContentType,
    ds_snapshot: DatasetSnapshot,
    dataset: DatasetModel,
    dp_map: dict[tuple[int, str, tuple[str, ...]], Any],
) -> None:
    """Recreate a dataset's DataSources, source references and data-point comments."""
    if not (ds_snapshot.data_sources or ds_snapshot.source_references or ds_snapshot.comments):
        return

    user_cache: dict[str, Any] = {}

    def resolve_user(user_uuid: str | None) -> Any:
        if not user_uuid:
            return None
        if user_uuid not in user_cache:
            user_cache[user_uuid] = User.objects.filter(uuid=user_uuid).first()
        return user_cache[user_uuid]

    # DataSources are scoped to the target instance and get fresh uuids (a same-DB
    # copy can't reuse the globally-unique source uuid); map old uuid → new object.
    src_map: dict[str, Any] = {}
    for s in ds_snapshot.data_sources:
        src_map[s.uuid] = DataSource.objects.create(
            scope_content_type=ic_ct,
            scope_id=ic.pk,
            name=s.name,
            edition=s.edition,
            authority=s.authority,
            description=s.description,
            url=s.url,
        )

    for ref in ds_snapshot.source_references:
        src_obj = src_map.get(ref.data_source)
        if src_obj is None:
            continue
        if ref.point is None:
            DatasetSourceReference.objects.create(dataset=dataset, data_source=src_obj)
            continue
        dp = dp_map.get(_data_point_key_tuple(ref.point))
        if dp is not None:
            DatasetSourceReference.objects.create(data_point=dp, data_source=src_obj)

    for c in ds_snapshot.comments:
        dp = dp_map.get(_data_point_key_tuple(c.point))
        if dp is None:
            continue
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


def resolve_metric_data_columns(
    ds_snapshot: DatasetSnapshot, metric_ids: list[str], dim_columns: dict[str, str]
) -> dict[str, str]:
    """
    Map each metric id to the data column that holds its value.

    The serialized data columns are named ``Coalesce(name, label, uuid)`` (see
    ``DBDataset.deserialize_df``), whereas a metric snapshot's identifier is
    ``name or uuid`` — so a metric with no ``name`` but a ``label`` is keyed by
    its uuid here while its data column is the label. ``deserialize_df`` builds
    that column from the metric's raw (base-language) ``label``, which need not
    equal ``str(label)`` under a different active Django language, so match
    against *all* of the label's translations. Fall back (for the common
    single-metric case) to the sole remaining value column.
    """
    fields = (ds_snapshot.data or {}).get('schema', {}).get('fields', [])
    all_columns = {f['name'] for f in fields}
    value_columns = all_columns - {'Year', 'id', 'uuid', *dim_columns.values()}
    labels_by_id = {m.identifier: (m.label.all() if m.label is not None else []) for m in ds_snapshot.metrics}

    columns: dict[str, str] = {}
    for metric_id in metric_ids:
        label_match = next((lbl for lbl in labels_by_id.get(metric_id, []) if lbl in value_columns), None)
        if metric_id in value_columns:
            columns[metric_id] = metric_id
        elif label_match is not None:
            columns[metric_id] = label_match
        elif len(metric_ids) == 1 and len(value_columns) == 1:
            columns[metric_id] = next(iter(value_columns))
        else:
            columns[metric_id] = metric_id
    return columns


def _import_data_points(
    dataset: DatasetModel,
    ds_snapshot: DatasetSnapshot,
    metrics_by_id: dict[str, DatasetMetricModel],
    dim_lookup: dict[str, DimensionCategoryModel],
) -> dict[tuple[int, str, tuple[str, ...]], Any]:
    """Create DataPoints; return a natural-key → DataPoint map for provenance wiring."""
    assert ds_snapshot.data is not None
    dim_ids = ds_snapshot.dimensions
    dim_columns = {dim_id: ds_snapshot.dimension_columns.get(dim_id, dim_id) for dim_id in dim_ids}
    metric_columns = resolve_metric_data_columns(ds_snapshot, list(metrics_by_id), dim_columns)

    data_points: list[DataPoint] = []
    # (data_point_index, category) pairs for bulk M2M creation
    dp_categories: list[tuple[int, DimensionCategoryModel]] = []
    # natural key per created data point, parallel to ``data_points``
    dp_keys: list[tuple[int, str, tuple[str, ...]]] = []

    for row in ds_snapshot.data['data']:
        year_val = row.get('Year')
        if year_val is None:
            continue
        dp_date = date(year=int(year_val), month=1, day=1)

        # Resolve dimension categories for this row (objects + their id strings,
        # which match the export-side natural key).
        row_cats: list[DimensionCategoryModel] = []
        row_cat_ids: list[str] = []
        for dim_id in dim_ids:
            cat_id = row.get(dim_columns[dim_id])
            if cat_id:
                cat = dim_lookup.get(f'{dim_id}/{cat_id}')
                if cat:
                    row_cats.append(cat)
                    row_cat_ids.append(str(cat_id))
        cat_key = tuple(sorted(row_cat_ids))

        for metric_id, metric in metrics_by_id.items():
            value = row.get(metric_columns[metric_id])
            if value is None:
                continue
            dp_idx = len(data_points)
            data_points.append(
                DataPoint(
                    dataset=dataset,
                    date=dp_date,
                    metric=metric,
                    value=value,
                )
            )
            dp_keys.append((int(year_val), metric_id, cat_key))
            dp_categories.extend((dp_idx, cat) for cat in row_cats)

    # Bulk create data points
    created_dps = DataPoint.objects.bulk_create(data_points)

    # Bulk create M2M links
    if dp_categories:
        m2m_objs = [
            DataPointDimensionCategory(
                data_point=created_dps[dp_idx],
                dimension_category=cat,
            )
            for dp_idx, cat in dp_categories
        ]
        DataPointDimensionCategory.objects.bulk_create(m2m_objs)

    return {dp_keys[i]: created_dps[i] for i in range(len(created_dps))}


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
) -> None:
    for dim_id in ds_snapshot.dimensions:
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
        column_name = ds_snapshot.dimension_columns.get(dim_id, dim_id)
        category_ids = sorted({
            str(cat_id) for row in (ds_snapshot.data or {}).get('data', []) if (cat_id := row.get(column_name))
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
) -> None:
    if ds_snapshot.data is None:
        return
    missing = {
        f'{dim_id}/{cat_id}'
        for row in ds_snapshot.data.get('data', [])
        for dim_id in ds_snapshot.dimensions
        if (cat_id := row.get(ds_snapshot.dimension_columns.get(dim_id, dim_id))) and f'{dim_id}/{cat_id}' not in dim_lookup
    }
    if missing:
        missing_str = ', '.join(sorted(missing)[:10])
        if len(missing) > 10:
            missing_str += ', ...'
        raise ValueError(
            f'Cannot import dataset {ds_snapshot.identifier!r} into {ic.identifier!r}; missing dimension categories: '
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
    if ds_snapshot.identifier is None:
        return None
    existing = DatasetModel.objects.filter(
        scope_content_type=ic_ct,
        scope_id=ic.pk,
        identifier=ds_snapshot.identifier,
        is_external_placeholder=False,
    ).first()
    if existing is None:
        return None
    if ds_snapshot.data is None or existing.data_points.exists():
        return existing
    raise ValueError(f'Dataset {ds_snapshot.identifier!r} already exists for {ic.identifier!r} but has no datapoints')


def import_instance_datasets(
    ic: InstanceConfig,
    dataset_snapshots: Iterable[DatasetSnapshot],
    *,
    rewire_dataset_ports: bool = False,
    replace_placeholders: bool = False,
    create_missing_dimensions: bool = False,
) -> list[DatasetModel]:
    """
    Import dataset bodies into an existing InstanceConfig without touching nodes.

    This is used when a template instance already has its node graph and
    dataset ports, but its datasets need to be promoted from external
    placeholders to real DB datasets with datapoints. With `replace_placeholders`,
    an imported dataset takes over the instance's own placeholder of the same
    identifier: its ports, then its place.
    """
    ic_ct = ContentType.objects.get_for_model(ic)
    dim_lookup = _dimension_category_lookup_for_instance(ic, ic_ct)
    imported: list[DatasetModel] = []
    datasets_by_id: dict[str, DatasetModel] = {}
    replaced: list[tuple[DatasetModel, DatasetModel]] = []

    for ds_snapshot in dataset_snapshots:
        existing = _already_imported(ic, ic_ct, ds_snapshot)
        if existing is not None:
            assert ds_snapshot.identifier is not None
            imported.append(existing)
            datasets_by_id[ds_snapshot.identifier] = existing
            continue

        if create_missing_dimensions:
            _ensure_dataset_dimensions(ic, ds_snapshot, ic_ct, dim_lookup)
        _validate_dataset_dimensions(ic, ds_snapshot, dim_lookup)
        placeholder = None
        if replace_placeholders and ds_snapshot.identifier is not None:
            placeholder = _set_aside_placeholder(ic, ic_ct, ds_snapshot.identifier)
        dataset = import_dataset(ic, ds_snapshot, ic_ct, dim_lookup)
        imported.append(dataset)
        if ds_snapshot.identifier is not None:
            datasets_by_id[ds_snapshot.identifier] = dataset
        if placeholder is not None:
            replaced.append((placeholder, dataset))

    for placeholder, dataset in replaced:
        _replace_placeholder(placeholder, dataset)
    if rewire_dataset_ports:
        _rewire_dataset_ports(ic, datasets_by_id)

    return imported
