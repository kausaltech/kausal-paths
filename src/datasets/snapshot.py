"""Dataset revision and portable export snapshot shapes."""

from typing import TYPE_CHECKING, Any, Self
from uuid import UUID  # noqa: TC003 - Pydantic field

from django.db.models import CharField, F, Value
from django.db.models.functions import Cast, Coalesce, NullIf
from pydantic import BaseModel, Field

from kausal_common.datasets.category_domain import DatasetCategoryDomain
from kausal_common.i18n.pydantic import TranslatedString  # noqa: TC002 - Pydantic field

from datasets.validation_rules import (
    ValidationRule,
    validation_rule_adapter,
)
from nodes.snapshot_base import ModelSnapshot, translated_string_from_model

if TYPE_CHECKING:
    from kausal_common.datasets.models import (
        DataPoint,
        DataPointComment,
        Dataset,
        DatasetMetric,
        DatasetMetricValidationRule,
        DatasetSourceReference,
        DataSource,
    )

    from frameworks.models import DataPointEvidence
    from nodes.models import InstanceConfig


class MetricValidationRuleSnapshot(ModelSnapshot['DatasetMetricValidationRule']):
    """
    One validation rule bound to a metric.

    ``rule`` parses the stored blob strictly against the schema in
    ``datasets.validation_rules``; ``uuid`` records the source row's identity.
    """

    uuid: UUID
    rule: ValidationRule

    @classmethod
    def from_model(cls, obj: DatasetMetricValidationRule) -> Self:
        return cls(uuid=obj.uuid, rule=validation_rule_adapter.validate_python(obj.rule))


def metric_column_id(metric: DatasetMetric) -> str:
    """
    Resolve the dataframe column a metric maps to.

    ``DatasetMetric.name`` *is* the column name and ``label`` is display text, so the
    two agree for imported metrics: ``load_dvc_dataset`` and ``dataset_placeholders``
    both create them as ``name=label=<column>``. A metric authored in the editor has
    no ``name`` at all, and its column is built from the label — see the
    ``Coalesce(name, label, uuid)`` in ``DBDataset.deserialize_df``
    (``nodes/datasets.py``), which is the writer this function has to agree with.
    Build SQL selectors with ``metric_column_id_expr()``, its twin below.

    **Falling through to the uuid is never a working answer.** No dataframe column is
    ever named after a metric uuid, so a selector built from one cannot match: it
    fails in ``select_metric`` as "Column '<uuid>' not found". It is kept only so a
    metric carrying neither name nor label still yields a stable string instead of
    ``None``.
    """
    return metric.name or metric.label or str(metric.uuid)


def metric_column_id_expr(prefix: str = '') -> Coalesce:
    """
    Return the SQL twin of ``metric_column_id()`` for a ``DatasetMetric`` queryset.

    ``prefix`` is the lookup path to the metric (``'metric__'`` from a binding row).
    Empty strings fall through as in the Python version, which tests truthiness.
    """
    return Coalesce(
        NullIf(F(f'{prefix}name'), Value('')),
        NullIf(F(f'{prefix}label'), Value('')),
        Cast(f'{prefix}uuid', output_field=CharField()),
    )


class DatasetMetricSnapshot(ModelSnapshot['DatasetMetric']):
    identifier: str
    label: TranslatedString | None = None
    unit: str
    quantity: str | None = None
    validation_rules: list[MetricValidationRuleSnapshot] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: DatasetMetric, primary_language: str = 'en') -> Self:
        return cls(
            identifier=metric_column_id(obj),
            label=translated_string_from_model(obj, 'label', primary_language),
            unit=obj.unit,
            quantity=(obj.spec or {}).get('quantity'),
            validation_rules=[MetricValidationRuleSnapshot.from_model(rule) for rule in obj.validation_rules.order_by('order')],
        )


class DataPointKey(BaseModel):
    """Natural key locating a DataPoint within its dataset (id-free, restore-stable)."""

    year: int
    metric: str  # metric identifier (name or uuid)
    categories: list[str] = Field(default_factory=list)  # sorted dimension-category ids

    @classmethod
    def from_model(cls, obj: DataPoint) -> Self:
        return cls(
            year=obj.date.year,
            metric=metric_column_id(obj.metric),
            categories=sorted(category.identifier or str(category.uuid) for category in obj.dimension_categories.all()),
        )


class DataSourceSnapshot(ModelSnapshot['DataSource']):
    """A published data source referenced by a dataset or its data points."""

    uuid: str  # source DataSource uuid; the join key for references within the snapshot
    name: str
    edition: str | None = None
    authority: str | None = None
    description: str | None = None
    url: str | None = None

    @classmethod
    def from_model(cls, obj: DataSource) -> Self:
        return cls(
            uuid=str(obj.uuid),
            name=obj.name,
            edition=obj.edition,
            authority=obj.authority,
            description=obj.description,
            url=obj.url,
        )


class SourceReferenceSnapshot(ModelSnapshot['DatasetSourceReference']):
    """Links a data source to the dataset (``point`` is None) or to one data point."""

    data_source: str  # DataSourceSnapshot.uuid
    point: DataPointKey | None = None

    @classmethod
    def from_model(cls, obj: DatasetSourceReference) -> Self:
        return cls(
            data_source=str(obj.data_source.uuid),
            point=DataPointKey.from_model(obj.data_point) if obj.data_point is not None else None,
        )


class DataPointCommentSnapshot(ModelSnapshot['DataPointComment']):
    """A (non-soft-deleted) comment on a data point. Users are referenced by uuid."""

    point: DataPointKey
    text: str
    is_sticky: bool = False
    is_review: bool = False
    review_state: str | None = None
    resolved_at: str | None = None  # ISO 8601
    created_by: str | None = None  # user uuid
    last_modified_by: str | None = None  # user uuid
    resolved_by: str | None = None  # user uuid

    @classmethod
    def from_model(cls, obj: DataPointComment) -> Self:
        assert obj.data_point is not None
        return cls(
            point=DataPointKey.from_model(obj.data_point),
            text=obj.text,
            is_sticky=obj.is_sticky,
            is_review=obj.is_review,
            review_state=obj.review_state,
            resolved_at=obj.resolved_at.isoformat() if obj.resolved_at else None,
            created_by=str(obj.created_by.uuid) if obj.created_by else None,
            last_modified_by=str(obj.last_modified_by.uuid) if obj.last_modified_by else None,
            resolved_by=str(obj.resolved_by.uuid) if obj.resolved_by else None,
        )


class QualityLevelRef(BaseModel):
    """
    A framework quality grade, by UUID and by authored identity.

    The UUID is exact within one deployment; the identifiers let a snapshot
    resolve against a framework provisioned elsewhere.
    """

    uuid: str
    scheme: str
    scheme_version: str
    level: str


class DataPointEvidenceSnapshot(ModelSnapshot['DataPointEvidence']):
    """What is asserted about one data point's value. Users are referenced by uuid."""

    point: DataPointKey
    kind: str | None = None
    quality_level: QualityLevelRef | None = None
    created_by: str | None = None  # user uuid
    last_modified_by: str | None = None  # user uuid

    @classmethod
    def from_model(cls, obj: DataPointEvidence) -> Self:
        level = obj.quality_level
        return cls(
            point=DataPointKey.from_model(obj.data_point),
            kind=obj.kind,
            quality_level=QualityLevelRef(
                uuid=str(level.uuid),
                scheme=level.scheme.identifier,
                scheme_version=level.scheme.version,
                level=level.identifier,
            )
            if level is not None
            else None,
            created_by=str(obj.created_by.uuid) if obj.created_by else None,
            last_modified_by=str(obj.last_modified_by.uuid) if obj.last_modified_by else None,
        )


class DatasetSnapshot(ModelSnapshot['Dataset']):
    """
    Pydantic representation of a ``Dataset`` ORM row.

    Includes its DataPoints. Used both as the Wagtail revision payload for Dataset
    (via ``Dataset.serializable_data`` bridged in Paths) and as the
    dataset-body carrier inside ``InstanceExport``.
    """

    schema_version: int = 1
    uuid: UUID | None = None  # source identity for remapping layout references when copied
    identifier: str | None = None
    name: TranslatedString | None = None
    forecast_from: int | None = None
    is_external_placeholder: bool = False
    external_ref: dict[str, Any] | None = None
    time_resolution: str = 'yearly'
    is_editable: bool = True
    dimensions: list[str] = Field(default_factory=list)
    dimension_columns: dict[str, str] = Field(default_factory=dict)
    metrics: list[DatasetMetricSnapshot] = Field(default_factory=list)
    category_domain: DatasetCategoryDomain = Field(default_factory=DatasetCategoryDomain)
    data: dict[str, Any] | None = None
    data_sources: list[DataSourceSnapshot] = Field(default_factory=list)
    source_references: list[SourceReferenceSnapshot] = Field(default_factory=list)
    comments: list[DataPointCommentSnapshot] = Field(default_factory=list)
    evidence: list[DataPointEvidenceSnapshot] = Field(default_factory=list)

    @classmethod
    def from_model(cls, obj: Dataset, instance_config: InstanceConfig | None = None) -> Self:
        from kausal_common.datasets.models import DatasetSchemaDimension, DimensionScope

        from datasets.transfer import export_dataset_data_safe, export_dataset_evidence, export_dataset_provenance

        schema = obj.schema
        metrics: list[DatasetMetricSnapshot] = []
        dimensions: list[str] = []
        dimension_columns: dict[str, str] = {}
        name_ts: TranslatedString | None = None
        time_resolution = 'yearly'
        is_editable = True
        primary_language = instance_config.primary_language if instance_config is not None else _primary_language_for_dataset(obj)

        if schema is not None:
            time_resolution = schema.time_resolution
            is_editable = schema.is_editable
            name_ts = translated_string_from_model(schema, 'name', primary_language)
            metrics = [
                DatasetMetricSnapshot.from_model(metric, primary_language) for metric in schema.metrics.all().order_by('order')
            ]
            dimension_instance = instance_config or obj.scope_instance
            for schema_dimension in (
                DatasetSchemaDimension.objects.filter(schema=schema).select_related('dimension').order_by('order')
            ):
                scope = (
                    DimensionScope.objects
                    .for_instance_config(dimension_instance)
                    .filter(dimension=schema_dimension.dimension)
                    .first()
                )
                if scope and scope.identifier:
                    dimensions.append(scope.identifier)
                    if schema_dimension.column_name and schema_dimension.column_name != scope.identifier:
                        dimension_columns[scope.identifier] = schema_dimension.column_name

        data: dict[str, Any] | None = None
        data_sources: list[DataSourceSnapshot] = []
        source_references: list[SourceReferenceSnapshot] = []
        comments: list[DataPointCommentSnapshot] = []
        evidence: list[DataPointEvidenceSnapshot] = []
        if not obj.is_external_placeholder:
            data = export_dataset_data_safe(obj, has_metrics=bool(metrics))
            data_sources, source_references, comments = export_dataset_provenance(obj)
            evidence = export_dataset_evidence(obj)

        return cls(
            uuid=obj.uuid,
            identifier=obj.identifier,
            name=name_ts,
            forecast_from=(obj.spec or {}).get('forecast_from'),
            is_external_placeholder=obj.is_external_placeholder,
            external_ref=obj.external_ref,
            time_resolution=time_resolution,
            is_editable=is_editable,
            dimensions=dimensions,
            dimension_columns=dimension_columns,
            metrics=metrics,
            category_domain=schema.category_domain if schema is not None else DatasetCategoryDomain(),
            data=data,
            data_sources=data_sources,
            source_references=source_references,
            comments=comments,
            evidence=evidence,
        )


def _primary_language_for_dataset(obj: Dataset) -> str:
    """Resolve the primary language for a Dataset via its scope's InstanceConfig."""
    return obj.scope_instance.primary_language or 'en'
