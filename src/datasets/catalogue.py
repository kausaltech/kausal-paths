"""Build dataset entries for instance graph catalogs."""

from typing import TYPE_CHECKING, Any, Final

from django.contrib.contenttypes.models import ContentType

from datasets.shape_domain import SHAPE_SPEC_KEY, CategoryDomainResolver, dataset_shape_id
from datasets.snapshot import metric_column_id
from datasets.validation_rules import validation_rule_adapter
from frameworks.evidence import DEFAULT_QUALITY_SPEC_KEY, QUALITY_OF_SPEC_KEY
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta, QualityLevelKey, ValidationRuleMeta
from nodes.snapshot_base import translated_string_from_model

FORECAST_FROM_SPEC_KEY: Final = 'forecast_from'

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DatasetModel, DatasetSchemaScope


def _is_framework(scope: DatasetSchemaScope) -> bool:
    return ContentType.objects.get_for_id(scope.scope_content_type_id).model == 'framework'


def dataset_meta_from_model(
    dataset: DatasetModel,
    *,
    primary_language: str,
    domains: CategoryDomainResolver | None = None,
) -> DatasetMeta:
    """Build the graph catalog entry for one dataset, exactly as snapshots record it."""
    schema = dataset.schema
    if schema is None:
        raise ValueError(f'Dataset {dataset.uuid} has no schema')
    metrics = tuple(
        DatasetMetricMeta(
            id=metric.uuid,
            identifier=metric_column_id(metric),
            label=translated_string_from_model(metric, 'label', primary_language),
            unit=metric.unit,
            quantity=(metric.spec or {}).get('quantity'),
            order=metric.order,
            validation_rules=tuple(
                ValidationRuleMeta(id=rule.uuid, rule=validation_rule_adapter.validate_python(rule.rule))
                # Meta.ordering is (metric, order), so .all() hits the
                # prefetch cache already in rule order.
                for rule in metric.validation_rules.all()
            ),
            quality_of=(metric.spec or {}).get(QUALITY_OF_SPEC_KEY),
        )
        for metric in schema.metrics.all()
    )
    declared_dimension_ids = tuple(schema_dimension.dimension.uuid for schema_dimension in schema.dimensions.all())
    return DatasetMeta(
        id=dataset.uuid,
        identifier=dataset.identifier,
        name=translated_string_from_model(schema, 'name', primary_language),
        schema_id=schema.uuid,
        schema_scope='framework' if any(_is_framework(scope) for scope in schema.scopes.all()) else 'instance',
        is_editable=schema.is_editable,
        metrics=metrics,
        declared_dimension_ids=declared_dimension_ids,
        time_resolution=schema.time_resolution,
        forecast_from=(dataset.spec or {}).get(FORECAST_FROM_SPEC_KEY),
        is_external_placeholder=dataset.is_external_placeholder,
        external_ref=dataset.external_ref,
        category_domain=(domains or CategoryDomainResolver()).for_dataset(dataset),
        shape_id=dataset_shape_id(dataset),
        default_quality=(
            QualityLevelKey.model_validate(default)
            if (default := (dataset.spec or {}).get(DEFAULT_QUALITY_SPEC_KEY)) is not None
            else None
        ),
        extra_spec={key: value for key, value in (dataset.spec or {}).items() if key not in _MODELED_SPEC_KEYS},
    )


_MODELED_SPEC_KEYS = frozenset({FORECAST_FROM_SPEC_KEY, SHAPE_SPEC_KEY, DEFAULT_QUALITY_SPEC_KEY})


def dataset_spec_from_meta(meta: DatasetMeta) -> dict[str, Any]:
    """Return the `Dataset.spec` that `dataset_meta_from_model` reads ``meta`` from."""
    spec: dict[str, Any] = dict(meta.extra_spec)
    if meta.forecast_from is not None:
        spec[FORECAST_FROM_SPEC_KEY] = meta.forecast_from
    if meta.shape_id is not None:
        spec[SHAPE_SPEC_KEY] = str(meta.shape_id)
    if meta.default_quality is not None:
        spec[DEFAULT_QUALITY_SPEC_KEY] = meta.default_quality.model_dump(mode='json')
    return spec
