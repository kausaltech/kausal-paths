"""Build dataset entries for instance graph catalogs."""

from typing import TYPE_CHECKING

from datasets.validation_rules import validation_rule_adapter
from nodes.defs.graph import DatasetMeta, DatasetMetricMeta
from nodes.snapshot_base import translated_string_from_model

if TYPE_CHECKING:
    from kausal_common.datasets.models import Dataset as DatasetModel


def dataset_meta_from_model(
    dataset: DatasetModel,
    *,
    primary_language: str,
    pinned_revision_id: int | None = None,
) -> DatasetMeta:
    """Build the graph catalog entry for one dataset, exactly as snapshots record it."""
    schema = dataset.schema
    if schema is None:
        raise ValueError(f'Dataset {dataset.uuid} has no schema')
    metrics = tuple(
        DatasetMetricMeta(
            id=metric.uuid,
            identifier=metric.name,
            label=translated_string_from_model(metric, 'label', primary_language),
            unit=metric.unit,
            quantity=(metric.spec or {}).get('quantity'),
            order=metric.order,
            validation_rules=tuple(
                validation_rule_adapter.validate_python(rule.rule)
                # Meta.ordering is (metric, order), so .all() hits the
                # prefetch cache already in rule order.
                for rule in metric.validation_rules.all()
            ),
        )
        for metric in schema.metrics.all()
    )
    declared_dimension_ids = tuple(schema_dimension.dimension.uuid for schema_dimension in schema.dimensions.all())
    return DatasetMeta(
        id=dataset.uuid,
        identifier=dataset.identifier,
        schema_id=schema.uuid,
        is_editable=schema.is_editable,
        metrics=metrics,
        declared_dimension_ids=declared_dimension_ids,
        is_external_placeholder=dataset.is_external_placeholder,
        external_ref=dataset.external_ref,
        revision_id=pinned_revision_id if pinned_revision_id is not None else dataset.latest_revision_id,
        category_domain=schema.category_domain,
    )
